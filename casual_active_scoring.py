#
# Script 1: activation_patching_analysis.py
# Purpose: To calculate the causal importance of neurons based on the paper's
#          Activation Patching protocol.
#
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# ============================================================================
# 1. Configuration
# ============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DATASET_PATH = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION_SAMPLES = 500  # Adjust according to memory and time constraints.
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

# ============================================================================
# 3. Activation Patching Analysis
# ============================================================================
def run_activation_patching_analysis():
    """
    Calculates causal scores for all neurons based on the paper's Activation Patching protocol.
    I_causal(c_j) = E[ M(f(x_r | do(a=a_c))) - M(f(x_r)) ]
    """
    print("--- 🔬 Phase 1: Causal Circuit Discovery (via Activation Patching) ---")
    
    # --- Setup ---
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Prepare (x_clean, x_corrupt) pairs as per the paper ---
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train", trust_remote_code=True)
    dataset = dataset.select(range(min(CALIBRATION_SAMPLES, len(dataset))))
    
    # We treat sentence1 as clean and sentence2 as corrupt.
    clean_texts = [ex['sentence1'] for ex in dataset]
    corrupt_texts = [ex['sentence2'] for ex in dataset]

    # ▼▼▼ FIX ▼▼▼
    # Change padding=True to padding='max_length' to ensure consistent sequence length.
    clean_tokens = tokenizer(clean_texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    corrupt_tokens = tokenizer(corrupt_texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    # ▲▲▲ END FIX ▲▲▲

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    d_ffn = model.config.intermediate_size
    sane_model_name = MODEL_NAME.replace('/', '_')

    # --- Baseline Calculations ---
    with torch.no_grad():
        # 1. Original high-quality embedding for the clean input (M(f(x_c))).
        clean_outputs = model(**clean_tokens)
        clean_embedding_orig = get_sentence_embedding(clean_outputs, clean_tokens['attention_mask'])
        
        # 2. Embedding for the corrupt input without any intervention (M(f(x_r))).
        corrupt_outputs = model(**corrupt_tokens)
        corrupt_embedding_orig = get_sentence_embedding(corrupt_outputs, corrupt_tokens['attention_mask'])

    # --- Cache Clean Activations ---
    print("Caching clean activations...")
    clean_activations = {}
    hooks = []
    
    # Register hooks to save activations from Attention and FFN layers.
    for layer_idx in range(n_layers):
        # Attention
        attn_module = model.encoder.layer[layer_idx].attention.self
        def save_attn_hook(module, input, output, name):
            clean_activations[name] = output[0].detach()
        hooks.append(attn_module.register_forward_hook(lambda m, i, o, name=f"attn_{layer_idx}": save_attn_hook(m, i, o, name)))
        
        # FFN
        ffn_module = model.encoder.layer[layer_idx].intermediate.dense
        def save_ffn_hook(module, input, output, name):
            clean_activations[name] = output.detach()
        hooks.append(ffn_module.register_forward_hook(lambda m, i, o, name=f"ffn_{layer_idx}": save_ffn_hook(m, i, o, name)))

    with torch.no_grad():
        model(**clean_tokens) # Run a clean forward pass to cache the activations.
    for hook in hooks:
        hook.remove()

    # --- Perform Activation Patching for Each Neuron ---
    all_neuron_scores = []

    def compute_causal_effect(patched_embedding):
        """Calculates the change in the performance metric M_B."""
        # Here, we define the performance metric M_B as the degree of recovery
        # towards the original clean embedding, measured by cosine similarity.
        M_B = lambda emb: torch.nn.functional.cosine_similarity(emb, clean_embedding_orig, dim=-1).mean()
        
        patched_score = M_B(patched_embedding)
        corrupt_score = M_B(corrupt_embedding_orig)
        
        return patched_score - corrupt_score

    # 1. Calculate causal scores for Attention neurons.
    print("Calculating causal scores for Attention neurons...")
    for layer_idx in tqdm(range(n_layers), desc="ATTN Layers"):
        module_to_hook = model.encoder.layer[layer_idx].attention.self
        clean_act = clean_activations[f"attn_{layer_idx}"]
        for head_idx in range(n_heads):
            for dim_idx in range(d_head):
                
                def patch_hook_attn(module, input, output):
                    # Clone the original corrupted activation.
                    patched_act = output[0].clone()
                    # Patch in the value from the specific clean neuron.
                    neuron_val_to_patch = clean_act.view(-1, clean_act.shape[-2], n_heads, d_head)[:, :, head_idx, dim_idx]
                    patched_act_view = patched_act.view(-1, patched_act.shape[-2], n_heads, d_head)
                    patched_act_view[:, :, head_idx, dim_idx] = neuron_val_to_patch
                    return (patched_act,) + output[1:]

                hook = module_to_hook.register_forward_hook(patch_hook_attn)
                with torch.no_grad():
                    patched_outputs = model(**corrupt_tokens)
                hook.remove()
                
                patched_embedding = get_sentence_embedding(patched_outputs, corrupt_tokens['attention_mask'])
                score = compute_causal_effect(patched_embedding).item()
                
                neuron_id = f"L{layer_idx}_H{head_idx}_D{dim_idx}"
                all_neuron_scores.append({"neuron_id": neuron_id, "type": "ATTN", "layer": layer_idx, "head": head_idx, "dim": dim_idx, "causal_score": score})

    # 2. Calculate causal scores for FFN neurons.
    print("Calculating causal scores for FFN neurons...")
    for layer_idx in tqdm(range(n_layers), desc="FFN Layers"):
        module_to_hook = model.encoder.layer[layer_idx].intermediate.dense
        clean_act = clean_activations[f"ffn_{layer_idx}"]
        for dim_idx in range(d_ffn):
            
            def patch_hook_ffn(module, input, output):
                # Clone the original corrupted activation.
                patched_act = output.clone()
                # Patch in the value from the specific clean neuron.
                neuron_val_to_patch = clean_act[..., dim_idx]
                patched_act[..., dim_idx] = neuron_val_to_patch
                return patched_act
            
            hook = module_to_hook.register_forward_hook(patch_hook_ffn)
            with torch.no_grad():
                patched_outputs = model(**corrupt_tokens)
            hook.remove()

            patched_embedding = get_sentence_embedding(patched_outputs, corrupt_tokens['attention_mask'])
            score = compute_causal_effect(patched_embedding).item()

            neuron_id = f"L{layer_idx}_FFN_D{dim_idx}"
            all_neuron_scores.append({"neuron_id": neuron_id, "type": "FFN", "layer": layer_idx, "head": -1, "dim": dim_idx, "causal_score": score})


    # --- Save Results ---
    summary_df = pd.DataFrame(all_neuron_scores)
    csv_path = Path(RESULTS_DIR) / f"causal_scores_{sane_model_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✅ Causal scores saved to: {csv_path}")


if __name__ == "__main__":
    run_activation_patching_analysis()