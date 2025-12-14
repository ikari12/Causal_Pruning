# British English comments.
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
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
FFN_NEURON_BATCH_SIZE = 16
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
# 3. Main Analysis Function
# ============================================================================
def run_full_neuron_analysis():
    """Calculates causal scores for ALL neurons and saves them to a CSV."""
    
    # --- Setup ---
    print("Setting up model and data...")
    results_path = Path(RESULTS_DIR)
    if results_path.exists():
        print(f"🧹 Cleaning up existing results in {RESULTS_DIR}...")
        try:
            shutil.rmtree(results_path)
        except OSError:
            for item in results_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    results_path.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train", trust_remote_code=True)
    calib_dataset = dataset.select(range(min(CALIBRATION_SAMPLES, len(dataset))))
    calib_texts = [ex['sentence1'] for ex in calib_dataset]
    calib_tokens = tokenizer(calib_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    d_ffn = model.config.intermediate_size
    sane_model_name = MODEL_NAME.replace('/', '_')

    # ========================================================================
    # Phase 1: Batched Causal Importance Calculation for ALL Attention Neurons
    # ========================================================================
    causal_scores_path = Path(RESULTS_DIR) / f"attn_causal_scores_all_{sane_model_name}.pt"

    if causal_scores_path.exists():
        print(f"\n--- Loading Cached Attention Causal Scores from {causal_scores_path} ---")
        all_attn_neuron_scores = torch.load(causal_scores_path, map_location=DEVICE)
    else:
        print(f"\n--- Calculating Causal Importance for ALL Attention Neurons ---")
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
                        context_layer = output[0].view(current_batch_size, CALIBRATION_SAMPLES, -1, n_heads, d_head)
                        for i in range(current_batch_size):
                            context_layer[i, :, :, head_idx, dim_batch_start + i] = 0.
                        return (context_layer.view(output[0].shape),) + output[1:]
                    hook = module_to_hook.register_forward_hook(batched_intervention_hook_attn)
                    with torch.no_grad():
                        intervened_outputs = model(**expanded_tokens)
                    hook.remove()
                    intervened_embedding = get_sentence_embedding(intervened_outputs, expanded_tokens['attention_mask']).view(current_batch_size, CALIBRATION_SAMPLES, -1)
                    diff = torch.mean((baseline_embedding.unsqueeze(0) - intervened_embedding) ** 2, dim=(1, 2))
                    all_attn_neuron_scores[layer_idx, head_idx, dim_batch_start:dim_batch_end] = diff
        torch.save(all_attn_neuron_scores, causal_scores_path)
    print("✅ Attention causal scores calculation/loading complete.")

    # ========================================================================
    # Phase 1.5: Batched Causal Importance Calculation for ALL FFN Neurons
    # ========================================================================
    ffn_scores_path = Path(RESULTS_DIR) / f"ffn_causal_scores_all_{sane_model_name}.pt"
    if ffn_scores_path.exists():
        print(f"\n--- Loading Cached FFN Causal Scores from {ffn_scores_path} ---")
        all_ffn_neuron_scores = torch.load(ffn_scores_path, map_location=DEVICE)
    else:
        print(f"\n--- Calculating Causal Importance for ALL FFN Neurons ---")
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
                    reshaped_output = output.view(current_batch_size, CALIBRATION_SAMPLES, -1, d_ffn)
                    for i in range(current_batch_size):
                        reshaped_output[i, :, :, neuron_batch_start + i] = 0.
                    return reshaped_output.view(output.shape)
                hook = module_to_hook.register_forward_hook(ffn_intervention_hook)
                with torch.no_grad():
                    intervened_outputs = model(**expanded_tokens)
                hook.remove()
                intervened_embedding = get_sentence_embedding(intervened_outputs, expanded_tokens['attention_mask']).view(current_batch_size, CALIBRATION_SAMPLES, -1)
                diff = torch.mean((baseline_embedding.unsqueeze(0) - intervened_embedding) ** 2, dim=(1, 2))
                all_ffn_neuron_scores[layer_idx, neuron_batch_start:neuron_batch_end] = diff
        torch.save(all_ffn_neuron_scores, ffn_scores_path)
    print("✅ FFN causal scores calculation/loading complete.")

    # ========================================================================
    # Phase 2: Save ALL Neuron Scores to CSV
    # ========================================================================
    print("\n--- Creating comprehensive summary of ALL neurons ---")
    
    neuron_data_list = []
    # Process ATTN neurons
    all_attn_cpu = all_attn_neuron_scores.cpu().numpy()
    for l, h, d in tqdm(np.ndindex(all_attn_cpu.shape), desc="Packing ATTN scores"):
        neuron_id = f"L{l}_H{h}_D{d}"
        score = all_attn_cpu[l, h, d]
        neuron_data_list.append({"neuron_id": neuron_id, "type": "ATTN", "layer": l, "causal_score": score})
        
    # Process FFN neurons
    all_ffn_cpu = all_ffn_neuron_scores.cpu().numpy()
    for l, d in tqdm(np.ndindex(all_ffn_cpu.shape), desc="Packing FFN scores"):
        neuron_id = f"L{l}_FFN_D{d}"
        score = all_ffn_cpu[l, d]
        neuron_data_list.append({"neuron_id": neuron_id, "type": "FFN", "layer": l, "causal_score": score})

    summary_df_all = pd.DataFrame(neuron_data_list)
    
    csv_path = Path(RESULTS_DIR) / "comprehensive_neuron_summary_all.csv"
    summary_df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full analysis for ALL neurons saved to: {csv_path}")

if __name__ == "__main__":
    run_full_neuron_analysis()