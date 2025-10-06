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
CALIBRATION_SAMPLES = 1400      # Number of sentences for analysis, as requested.
ATTN_NEURON_BATCH_SIZE = 16     # How many attention dimensions to test in parallel. Adjust based on VRAM.
FFN_NEURON_BATCH_SIZE = 16      # Batch size for FFN neurons can often be larger.
TOP_K_NEURONS_GLOBAL = 200      # Analyse the top 200 most important neurons globally (from both attn and ffn).
TOP_K_WORDS_FOR_LABEL = 3       # Number of words to use for labeling points on the plot.
TOP_K_WORDS = 30 
BATCH_SIZE = 32                 # General batch size for forward passes.
RESULTS_DIR = "/app/results"

print(f"Using device: {DEVICE}")

# ============================================================================
# 2. Helper Functions
# ============================================================================
def get_sentence_embedding(outputs, attention_mask):
    """Calculates the mean-pooled sentence embedding from the final hidden state."""
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
    
    # Get model configuration details
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    d_ffn = model.config.intermediate_size
    sane_model_name = MODEL_NAME.replace('/', '_')

    # ========================================================================
    # Phase 1: Batched Causal Importance Calculation for Attention Neurons
    # ========================================================================
    causal_scores_path = Path(RESULTS_DIR) / f"attn_causal_scores_{sane_model_name}.pt"

    if causal_scores_path.exists():
        print(f"\n--- Phase 1: Loading Cached Attention Causal Scores from {causal_scores_path} ---")
        all_attn_neuron_scores = torch.load(causal_scores_path, map_location=DEVICE)
        print("✅ Attention causal scores loaded successfully.")
    else:
        print(f"\n--- Phase 1: Calculating Causal Importance for Attention Neurons ---")
        
        with torch.no_grad():
            baseline_outputs = model(**calib_tokens)
            baseline_embedding = get_sentence_embedding(baseline_outputs, calib_tokens['attention_mask'])

        all_attn_neuron_scores = torch.zeros(n_layers, n_heads, d_head).to(DEVICE)
        attn_modules = {name: module for name, module in model.named_modules() if "attention.self" in name}

        for layer_idx in tqdm(range(n_layers), desc="Analysing Attention Layers"):
            target_name = f"encoder.layer.{layer_idx}.attention.self"
            module_to_hook = attn_modules[target_name]

            for head_idx in tqdm(range(n_heads), desc=f"  L{layer_idx} Heads", leave=False):
                for dim_batch_start in range(0, d_head, ATTN_NEURON_BATCH_SIZE):
                    dim_batch_end = min(dim_batch_start + ATTN_NEURON_BATCH_SIZE, d_head)
                    current_batch_size = dim_batch_end - dim_batch_start
                    
                    expanded_tokens = {k: v.repeat(current_batch_size, 1) for k, v in calib_tokens.items()}

                    def batched_intervention_hook_attn(module, input, output):
                        context_layer = output[0]
                        bs_x_nd, sl, hs = context_layer.shape
                        reshaped_context = context_layer.view(bs_x_nd, sl, n_heads, d_head)

                        for i in range(current_batch_size):
                            start_row, end_row = i * CALIBRATION_SAMPLES, (i + 1) * CALIBRATION_SAMPLES
                            dim_to_zero = dim_batch_start + i
                            reshaped_context[start_row:end_row, :, head_idx, dim_to_zero] = 0.
                        
                        return (reshaped_context.view(bs_x_nd, sl, hs),) + output[1:]

                    hook = module_to_hook.register_forward_hook(batched_intervention_hook_attn)
                    with torch.no_grad():
                        intervened_outputs = model(**expanded_tokens)
                    hook.remove()

                    intervened_embedding = get_sentence_embedding(intervened_outputs, expanded_tokens['attention_mask'])
                    intervened_embedding = intervened_embedding.view(current_batch_size, CALIBRATION_SAMPLES, -1)
                    
                    diff = torch.mean((baseline_embedding.unsqueeze(0) - intervened_embedding) ** 2, dim=(1, 2))
                    all_attn_neuron_scores[layer_idx, head_idx, dim_batch_start:dim_batch_end] = diff

        print("✅ Attention causal importance calculation complete.")
        torch.save(all_attn_neuron_scores, causal_scores_path)
        print(f"✅ Attention causal scores saved to {causal_scores_path}")

    # ========================================================================
    # Phase 1.5: Batched Causal Importance Calculation for FFN Neurons
    # ========================================================================
    ffn_scores_path = Path(RESULTS_DIR) / f"ffn_causal_scores_{sane_model_name}.pt"

    if ffn_scores_path.exists():
        print(f"\n--- Phase 1.5: Loading Cached FFN Causal Scores from {ffn_scores_path} ---")
        all_ffn_neuron_scores = torch.load(ffn_scores_path, map_location=DEVICE)
        print("✅ FFN causal scores loaded successfully.")
    else:
        print(f"\n--- Phase 1.5: Calculating Causal Importance for FFN Neurons ---")
        
        # Baseline might already be calculated, but we recalculate here for modularity.
        with torch.no_grad():
            baseline_outputs = model(**calib_tokens)
            baseline_embedding = get_sentence_embedding(baseline_outputs, calib_tokens['attention_mask'])

        all_ffn_neuron_scores = torch.zeros(n_layers, d_ffn).to(DEVICE)
        ffn_intermediate_modules = {name: module for name, module in model.named_modules() if "intermediate.dense" in name}

        for layer_idx in tqdm(range(n_layers), desc="Analysing FFN Layers"):
            target_name = f"encoder.layer.{layer_idx}.intermediate.dense"
            module_to_hook = ffn_intermediate_modules[target_name]

            for neuron_batch_start in range(0, d_ffn, FFN_NEURON_BATCH_SIZE):
                neuron_batch_end = min(neuron_batch_start + FFN_NEURON_BATCH_SIZE, d_ffn)
                current_batch_size = neuron_batch_end - neuron_batch_start
                
                expanded_tokens = {k: v.repeat(current_batch_size, 1) for k, v in calib_tokens.items()}

                def ffn_intervention_hook(module, input, output):
                    for i in range(current_batch_size):
                        start_row, end_row = i * CALIBRATION_SAMPLES, (i + 1) * CALIBRATION_SAMPLES
                        neuron_to_zero = neuron_batch_start + i
                        output[start_row:end_row, :, neuron_to_zero] = 0.
                    return output

                hook = module_to_hook.register_forward_hook(ffn_intervention_hook)
                with torch.no_grad():
                    intervened_outputs = model(**expanded_tokens)
                hook.remove()

                intervened_embedding = get_sentence_embedding(intervened_outputs, expanded_tokens['attention_mask'])
                intervened_embedding = intervened_embedding.view(current_batch_size, CALIBRATION_SAMPLES, -1)
                
                diff = torch.mean((baseline_embedding.unsqueeze(0) - intervened_embedding) ** 2, dim=(1, 2))
                all_ffn_neuron_scores[layer_idx, neuron_batch_start:neuron_batch_end] = diff

        print("✅ FFN causal importance calculation complete.")
        torch.save(all_ffn_neuron_scores, ffn_scores_path)
        print(f"✅ FFN causal scores saved to {ffn_scores_path}")

    # ========================================================================
    # Phase 2: Feature Attribution for Top Neurons (Combined)
    # ========================================================================
    neuron_coords_list = []
    for l, h, d in np.ndindex(all_attn_neuron_scores.shape):
        neuron_coords_list.append({'type': 'attn', 'layer': l, 'head': h, 'dim': d, 'score': all_attn_neuron_scores[l,h,d].item()})
    for l, d in np.ndindex(all_ffn_neuron_scores.shape):
        neuron_coords_list.append({'type': 'ffn', 'layer': l, 'head': -1, 'dim': d, 'score': all_ffn_neuron_scores[l,d].item()})

    top_neurons = sorted(neuron_coords_list, key=lambda x: x['score'], reverse=True)[:TOP_K_NEURONS_GLOBAL]
    
    attribution_path = Path(RESULTS_DIR) / f"attribution_{sane_model_name}.pt"

    if attribution_path.exists():
        print(f"\n--- Phase 2: Loading Cached Attribution Results from {attribution_path} ---")
        attribution_results = torch.load(attribution_path)
        print("✅ Attribution results loaded successfully.")
    else:
        print(f"\n--- Phase 2: Finding Causal Words for Top {TOP_K_NEURONS_GLOBAL} Neurons ---")
        attribution_results = {}
        
        for neuron_info in tqdm(top_neurons, desc="Calculating Attributions"):
            l, h, d, n_type = neuron_info['layer'], neuron_info['head'], neuron_info['dim'], neuron_info['type']
            
            if n_type == 'attn':
                neuron_id = f"L{l}_H{h}_D{d}"
                target_name = f"encoder.layer.{l}.attention.self"
            else: # FFN
                neuron_id = f"L{l}_FFN_D{d}"
                target_name = f"encoder.layer.{l}.intermediate.dense"

            captured_activations = {}
            def capture_hook(name):
                def hook(module, input, output): captured_activations[name] = output
                return hook

            embedding_name = "embeddings.word_embeddings"
            
            hooks = [
                model.get_submodule(target_name).register_forward_hook(capture_hook(target_name)),
                model.get_submodule(embedding_name).register_forward_hook(capture_hook(embedding_name))
            ]
            
            model(**calib_tokens)
            for handle in hooks: 
                handle.remove()
            
            target_output = captured_activations[target_name]
            if isinstance(target_output, tuple): target_output = target_output[0]
            
            if n_type == 'attn':
                target_output = target_output.view(target_output.shape[0], target_output.shape[1], n_heads, d_head)
                objective = target_output[:, :, h, d].sum()
            else:
                objective = target_output[:, :, d].sum()
            
            grads = torch.autograd.grad(outputs=objective, inputs=captured_activations[embedding_name])[0]
            token_attributions = torch.einsum("bsd,bsd->bs", grads, captured_activations[embedding_name]).flatten()
            
            flat_tokens = tokenizer.convert_ids_to_tokens(calib_tokens['input_ids'].flatten())
    
            df = pd.DataFrame({"token": flat_tokens, "attribution": token_attributions.detach().cpu().numpy()})
            df = df[df.token.isin(['[CLS]', '[SEP]', '[PAD]']) == False]
            agg_df = df.groupby('token')['attribution'].agg(['mean', 'count'])
            agg_df = agg_df[agg_df['count'] >= 2].sort_values(by='mean', ascending=False).head(TOP_K_WORDS)
            
            attribution_results[neuron_id] = {"causal_score": neuron_info['score'], "top_words": agg_df}
            model.zero_grad()
        
        torch.save(attribution_results, attribution_path)
        print(f"✅ Attribution results saved to {attribution_path}")

    # ========================================================================
    # Phase 3 & 4: UMAP, Save CSV, and Plot
    # ========================================================================
    print("\n--- Phase 3 & 4: UMAP Projection, Saving and Plotting ---")
    
    full_vocab = set()
    for data in attribution_results.values():
        full_vocab.update(data['top_words'].index.tolist())
    vocab_list = sorted(list(full_vocab))
    vocab_map = {word: i for i, word in enumerate(vocab_list)}
    
    neuron_vectors = np.zeros((len(attribution_results), len(vocab_list)))
    neuron_ids = list(attribution_results.keys())
    
    for i, neuron_id in enumerate(neuron_ids):
        profile = attribution_results[neuron_id]['top_words']
        for word, row in profile.iterrows():
            if word in vocab_map:
                neuron_vectors[i, vocab_map[word]] = row['mean']
    
    summary_data = []
    for i, neuron_id in enumerate(neuron_ids):
        data = attribution_results[neuron_id]
        parts = neuron_id.split('_')
        l, n_type = int(parts[0][1:]), 'FFN' if 'FFN' in parts[1] else 'ATTN'
        
        summary_data.append({
            "neuron_id": neuron_id, "type": n_type, "layer": l,
            "causal_score": data['causal_score'],
            "top_words": ", ".join(data['top_words'].head(TOP_K_WORDS_FOR_LABEL).index.tolist())
        })

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = reducer.fit_transform(neuron_vectors)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['umap_x'] = embedding_2d[:, 0]
    summary_df['umap_y'] = embedding_2d[:, 1]
    
    csv_path = Path(RESULTS_DIR) / "comprehensive_neuron_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full analysis data saved to: {csv_path}")

    fig, ax = plt.subplots(figsize=(20, 20))
    attn_df = summary_df[summary_df['type'] == 'ATTN']
    ffn_df = summary_df[summary_df['type'] == 'FFN']
    
    ax.scatter(attn_df['umap_x'], attn_df['umap_y'], c=attn_df['layer'], cmap='viridis', s=50, alpha=0.7, marker='o', label='Attention')
    ax.scatter(ffn_df['umap_x'], ffn_df['umap_y'], c=ffn_df['layer'], cmap='plasma', s=80, alpha=0.9, marker='X', label='FFN')
    
    texts = [ax.text(row['umap_x'], row['umap_y'], row['top_words'], fontsize=8, alpha=0.9) for _, row in summary_df.iterrows()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    ax.set_title('UMAP Projection of Neurons by Functional Similarity (Native PyTorch)', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    cbar = fig.colorbar(ax.collections[0], ax=ax, label='Layer Index (Attention)') # Use first collection for colorbar
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = Path(RESULTS_DIR) / "comprehensive_neuron_umap.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Functional clustering plot saved to: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    run_neuron_cluster_analysis()