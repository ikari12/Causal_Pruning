# British English comments.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import japanize_matplotlib
from adjustText import adjust_text
from pathlib import Path

# ============================================================================
# 1. Configuration
# ============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DATASET_PATH = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Analysis Parameters ---
CALIBRATION_SAMPLES = 1400
ATTN_NEURON_BATCH_SIZE = 16
TOP_K_NEURONS_GLOBAL = 200
TOP_K_WORDS_FOR_LABEL = 3
BATCH_SIZE = 32
RESULTS_DIR = "/app/results"

print(f"Using device: {DEVICE}")

# ============================================================================
# 2. Helper Functions
# ============================================================================
def get_sentence_embedding(outputs, attention_mask):
    """Calculates the mean-pooled sentence embedding."""
    last_hidden_state = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings_batched(model, tokenizer, texts, batch_size):
    """Gets sentence embeddings for a list of texts using batch processing."""
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = get_sentence_embedding(outputs, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# ============================================================================
# 3. Main Analysis Function
# ============================================================================
def run_neuron_cluster_analysis():
    """Orchestrates the entire pipeline using native PyTorch hooks."""
    
    # --- Setup ---
    print("Setting up model and data...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="test", trust_remote_code=True)
    calib_dataset = dataset.select(range(min(CALIBRATION_SAMPLES, len(dataset))))
    
    calib_texts = [ex['sentence1'] for ex in calib_dataset]
    calib_tokens = tokenizer(calib_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    
    # Identify target modules (attention output layers)
    attn_output_modules = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear) and "attention.output.dense" in name}
    
    # Get model configuration details
    n_layers = model.config.num_hidden_layers
    
    # Get model configuration details
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads

    # ========================================================================
    # Phase 1: Batched Causal Importance Calculation (Attention Neurons)
    # ========================================================================
    print(f"\n--- Phase 1: Calculating Causal Importance for Attention Neurons ---")
    
    with torch.no_grad():
        baseline_outputs = model(**calib_tokens)
        baseline_embedding = get_sentence_embedding(baseline_outputs, calib_tokens['attention_mask'])

    all_attn_neuron_scores = torch.zeros(n_layers, n_heads, d_head).to(DEVICE)
    
    # This analysis targets the output of each attention head *before* the final projection (W_o).
    # We find the `BertSelfAttention` module for this.
    attn_modules = {name: module for name, module in model.named_modules() if "attention.self" in name}

    for layer_idx in tqdm(range(n_layers), desc="Analysing Layers"):
        target_name = f"encoder.layer.{layer_idx}.attention.self"
        module_to_hook = attn_modules[target_name]

        for head_idx in tqdm(range(n_heads), desc=f"  L{layer_idx} Heads", leave=False):
            for dim_batch_start in range(0, d_head, ATTN_NEURON_BATCH_SIZE):
                dim_batch_end = min(dim_batch_start + ATTN_NEURON_BATCH_SIZE, d_head)
                current_batch_size = dim_batch_end - dim_batch_start
                
                expanded_tokens = {k: v.repeat(current_batch_size, 1) for k, v in calib_tokens.items()}

                def batched_intervention_hook(module, input, output):
                    # output[0] is the attention context layer
                    context_layer = output[0]
                    # Reshape to expose heads and dims: [batch*n_dims, seq_len, n_heads, d_head]
                    bs_x_nd, sl, hidden_size = context_layer.shape
                    reshaped_context = context_layer.view(bs_x_nd, sl, n_heads, d_head)

                    for i in range(current_batch_size):
                        start_row, end_row = i * CALIBRATION_SAMPLES, (i + 1) * CALIBRATION_SAMPLES
                        dim_to_zero = dim_batch_start + i
                        reshaped_context[start_row:end_row, :, head_idx, dim_to_zero] = 0.
                    
                    return (reshaped_context.view(bs_x_nd, sl, hidden_size),) + output[1:]

                hook = module_to_hook.register_forward_hook(batched_intervention_hook)
                with torch.no_grad():
                    intervened_outputs = model(**expanded_tokens)
                hook.remove()

                intervened_embedding = get_sentence_embedding(intervened_outputs, expanded_tokens['attention_mask'])
                intervened_embedding = intervened_embedding.view(current_batch_size, CALIBRATION_SAMPLES, -1)
                
                diff = torch.mean((baseline_embedding.unsqueeze(0) - intervened_embedding) ** 2, dim=(1, 2))
                all_attn_neuron_scores[layer_idx, head_idx, dim_batch_start:dim_batch_end] = diff

    print("✅ Causal importance calculation complete.")
    
    flat_scores = all_attn_neuron_scores.flatten()
    top_scores, top_indices_flat = torch.topk(flat_scores, TOP_K_NEURONS_GLOBAL)
    top_neuron_coords = np.array(np.unravel_index(top_indices_flat.cpu().numpy(), all_attn_neuron_scores.shape)).T

    # ========================================================================
    # Phase 2: Feature Attribution for Top Neurons
    # ========================================================================
    print(f"\n--- Phase 2: Finding Causal Words for Top {TOP_K_NEURONS_GLOBAL} Neurons ---")
    
    attribution_results = {}
    
    for i, (layer_idx, head_idx, dim_idx) in enumerate(tqdm(top_neuron_coords, desc="Calculating Attributions")):
        neuron_id = f"L{layer_idx}_H{head_idx}_D{dim_idx}"
        
        # We need to re-run with hooks to get intermediate activations.
        captured_activations = {}
        def capture_hook(name):
            def hook(module, input, output):
                captured_activations[name] = output
            return hook

        target_name = f"bert.encoder.layer.{layer_idx}.attention.self"
        embedding_name = "bert.embeddings.word_embeddings"
        
        hooks = [
            model.get_submodule(target_name).register_forward_hook(capture_hook(target_name)),
            model.get_submodule(embedding_name).register_forward_hook(capture_hook(embedding_name))
        ]
        
        # Run forward pass to build graph and capture activations
        outputs = model(**calib_tokens)
        
        # Detach hooks immediately after use
        for h in hooks: h.remove()
        
        # Objective: the specific neuron's activation sum.
        attn_output = captured_activations[target_name][0]
        bs, sl, hs = attn_output.shape
        attn_output_reshaped = attn_output.view(bs, sl, n_heads, d_head)
        objective = attn_output_reshaped[:, :, head_idx, dim_idx].sum()
        
        # Calculate gradients manually
        grads = torch.autograd.grad(outputs=objective, inputs=captured_activations[embedding_name])[0]
        
        token_attributions = torch.einsum("bsd,bsd->bs", grads, captured_activations[embedding_name]).flatten()
        
        # Aggregate results
        flat_tokens = tokenizer.convert_ids_to_tokens(calib_tokens.flatten())
        df = pd.DataFrame({"token": flat_tokens, "attribution": token_attributions.cpu().numpy()})
        df = df[df.token.isin(['[CLS]', '[SEP]', '[PAD]']) == False]
        agg_df = df.groupby('token')['attribution'].agg(['mean', 'count'])
        agg_df = agg_df[agg_df['count'] >= 2].sort_values(by='mean', ascending=False).head(TOP_K_WORDS)
        
        attribution_results[neuron_id] = {
            "causal_score": top_scores[i].item(),
            "top_words": agg_df
        }
        
    # ========================================================================
    # Phase 3 & 4: UMAP, Save CSV, and Plot
    # ========================================================================
    print("\n--- Phase 3 & 4: UMAP Projection, Saving and Plotting ---")
    
    # Create the Neuron-x-Vocabulary matrix
    full_vocab = set()
    for data in attribution_results.values():
        full_vocab.update(data['top_words'].index.tolist())
    vocab_list = sorted(list(full_vocab))
    vocab_map = {word: i for i, word in enumerate(vocab_list)}
    
    neuron_vectors = np.zeros((len(attribution_results), len(vocab_list)))
    
    summary_data = []
    for i, (neuron_id, data) in enumerate(attribution_results.items()):
        profile = data['top_words']
        for word, row in profile.iterrows():
            if word in vocab_map:
                neuron_vectors[i, vocab_map[word]] = row['mean']
        
        l, h, d = [int(s[1:]) for s in neuron_id.split('_')]
        summary_data.append({
            "neuron_id": neuron_id, "layer": l, "head": h, "dim": d,
            "causal_score": data['causal_score'],
            "top_words": ", ".join(profile.head(TOP_K_WORDS_FOR_LABEL).index.tolist())
        })

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = reducer.fit_transform(neuron_vectors)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['umap_x'] = embedding_2d[:, 0]
    summary_df['umap_y'] = embedding_2d[:, 1]
    
    csv_path = Path(RESULTS_DIR) / "attention_neuron_summary_native.csv"
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full analysis data saved to: {csv_path}")

    fig, ax = plt.subplots(figsize=(20, 20))
    scatter = ax.scatter(summary_df['umap_x'], summary_df['umap_y'], c=summary_df['layer'], cmap='viridis', s=50, alpha=0.7)
    
    texts = [ax.text(row['umap_x'], row['umap_y'], row['top_words'], fontsize=8, alpha=0.9) for _, row in summary_df.iterrows()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    ax.set_title('UMAP Projection of Attention Neurons by Functional Similarity (Native PyTorch)', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Layer Index')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = Path(RESULTS_DIR) / "attention_neuron_umap_native.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Functional clustering plot saved to: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    run_neuron_cluster_analysis()

