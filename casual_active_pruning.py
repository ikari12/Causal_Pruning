#
# Script: hybrid_pruning.py (Theoretically Aligned and Corrected Version)
# Purpose: To prune the model using the Hybrid Pruning strategy (CC-Prune), ensuring complete protection of the Causal Circuit.
#
import torch
import torch.nn as nn
import copy
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from pathlib import Path
from mteb import MTEB
import time
import numpy as np
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message=".*model_card_data.*")

# ============================================================================
# 1. Configuration
# ============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/app/results"
sane_model_name = MODEL_NAME.replace('/', '_')
SUMMARY_CSV_PATH = Path(RESULTS_DIR) / f"causal_scores_{sane_model_name}.csv"

# ============================================================================
# 2. Helper Function
# ============================================================================
def report_sparsity(model: nn.Module, description: str):
    """Calculates and reports the parameter sparsity of the model."""
    total_params = 0
    nonzero_params = 0
    # Include all parameters (weights and biases) that are typically trainable
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
    
    if total_params == 0:
        sparsity = 0.0
    else:
        sparsity = (1 - (nonzero_params / total_params)) * 100
        
    print(f"--- Sparsity Report for: {description} ---")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Non-zero parameters:  {nonzero_params:,}")
    print(f"  Model sparsity:       {sparsity:.2f}%")
    print("-" * 40)
    return sparsity

# ============================================================================
# 3. Core Hybrid Pruning Function
# ============================================================================
def hybrid_prune_model(model: nn.Module, summary_df: pd.DataFrame, target_sparsity: float, circuit_retention_ratio: float):
    """
    Implements the Hybrid Pruning methodology.
    Phase 1: Protect the "causal circuit".
    Phase 2: Prune unprotected parameters by magnitude.
    """
    if not 0 <= target_sparsity <= 1:
        raise ValueError(f"Target sparsity must be between 0 and 1, but got {target_sparsity}")
    
    print(f"✂️ Starting Hybrid Pruning for {target_sparsity*100:.1f}% target sparsity...")
    print(f"   (Protecting top {circuit_retention_ratio*100:.1f}% of neurons as causal circuit)")

    # Phase 1: Causal Circuit Discovery
    score_threshold = summary_df['causal_score'].quantile(1.0 - circuit_retention_ratio)
    circuit_neurons = summary_df[summary_df['causal_score'] >= score_threshold]
    print(f"Identified {len(circuit_neurons)} neurons for the causal circuit (score >= {score_threshold:.6f}).")

    protection_mask = {}
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    
    # Initialize protection mask (True means prunable)
    for name, param in model.named_parameters():
        if param.requires_grad:
            protection_mask[name] = torch.ones_like(param, dtype=torch.bool)
            
    # Update protection mask based on causal circuit (False means protected)
    for _, neuron in circuit_neurons.iterrows():
        layer, n_type = neuron['layer'], neuron['type']
        
        if n_type == 'ATTN':
            head, dim = neuron['head'], neuron['dim']
            
            # Indices for the specific neuron (V and O)
            neuron_idx = head * d_head + dim
            # Indices for the entire head (Q and K)
            head_start_idx = head * d_head
            head_end_idx = (head + 1) * d_head

            # Rationale: To preserve the function of a neuron (H, D), we must protect:
            # 1. Output path (W_O at dimension level).
            # 2. Input Value (W_V, B_V at dimension level).
            # 3. Attention Pattern (W_Q, B_Q, W_K, B_K at head level).

            # ▼▼▼ FIX: Comprehensive Causal Circuit Protection for Attention ▼▼▼
            
            # 1. Protect W_O (Output projection) - Column
            w_o_name = f"encoder.layer.{layer}.attention.output.dense.weight"
            if w_o_name in protection_mask and neuron_idx < protection_mask[w_o_name].shape[1]:
                protection_mask[w_o_name][:, neuron_idx] = False
            # Note: B_O (output bias) is typically protected entirely or pruned based on overall magnitude, 
            # as it affects the entire layer output, not just this specific head/neuron. 
            # We rely on magnitude pruning to handle B_O if it's not crucial overall.

            # 2. Protect W_V and B_V (Value projection) - Row/Element (Neuron level)
            w_v_name = f"encoder.layer.{layer}.attention.self.value.weight"
            b_v_name = f"encoder.layer.{layer}.attention.self.value.bias"
            if w_v_name in protection_mask and neuron_idx < protection_mask[w_v_name].shape[0]:
                 protection_mask[w_v_name][neuron_idx, :] = False
            # Protect Bias
            if b_v_name in protection_mask and neuron_idx < protection_mask[b_v_name].shape[0]:
                 protection_mask[b_v_name][neuron_idx] = False

            # 3. Protect W_Q/B_Q and W_K/B_K (Query/Key projection) - Rows/Elements (Head level)
            for proj_type in ['query', 'key']:
                w_proj_name = f"encoder.layer.{layer}.attention.self.{proj_type}.weight"
                b_proj_name = f"encoder.layer.{layer}.attention.self.{proj_type}.bias"
                
                if w_proj_name in protection_mask and head_end_idx <= protection_mask[w_proj_name].shape[0]:
                    protection_mask[w_proj_name][head_start_idx:head_end_idx, :] = False
                # Protect Bias
                if b_proj_name in protection_mask and head_end_idx <= protection_mask[b_proj_name].shape[0]:
                    protection_mask[b_proj_name][head_start_idx:head_end_idx] = False
            # ▲▲▲ END FIX ▲▲▲
        
        elif n_type == 'FFN':
            dim = neuron['dim'] # Index corresponding to the neuron

            # ▼▼▼ FIX: Protect FFN Weights and Biases ▼▼▼
            # 1. Protect W_in (intermediate.dense) and B_in
            w_in_name = f"encoder.layer.{layer}.intermediate.dense.weight"
            b_in_name = f"encoder.layer.{layer}.intermediate.dense.bias"
            # Protect the row corresponding to the neuron input
            if w_in_name in protection_mask and dim < protection_mask[w_in_name].shape[0]:
                protection_mask[w_in_name][dim, :] = False
            # Protect Bias
            if b_in_name in protection_mask and dim < protection_mask[b_in_name].shape[0]:
                protection_mask[b_in_name][dim] = False

            # 2. Protect W_out (output.dense)
            w_out_name = f"encoder.layer.{layer}.output.dense.weight"
            # Protect the column corresponding to the neuron output
            if w_out_name in protection_mask and dim < protection_mask[w_out_name].shape[1]:
                protection_mask[w_out_name][:, dim] = False
            # ▲▲▲ END FIX ▲▲▲

    # Phase 2: Hybrid Pruning
    prunable_params = []
    total_prunable_count = 0
    for name, param in model.named_parameters():
        if name in protection_mask:
            prunable_indices = protection_mask[name]
            if prunable_indices.any():
                prunable_params.append(param.data[prunable_indices])
            total_prunable_count += prunable_indices.sum().item()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_protected_params = total_params - total_prunable_count
    
    # Calculate protection ratio
    protection_ratio = (num_protected_params / total_params) * 100 if total_params > 0 else 0
    
    print(f"Total parameters: {total_params:,}")
    print(f"Protected (circuit) parameters: {num_protected_params:,} ({protection_ratio:.2f}%)")
    print(f"Prunable (non-circuit) parameters: {total_prunable_count:,}")

    # Calculate the number of parameters to prune from the non-circuit part
    num_params_to_prune_total = int(total_params * target_sparsity)
    num_params_to_prune_from_non_circuit = num_params_to_prune_total
    
    if num_params_to_prune_from_non_circuit <= 0:
        print("✅ No parameters need to be removed to reach target sparsity.")
        return model

    if total_prunable_count == 0:
        if num_params_to_prune_from_non_circuit > 0:
             print("⚠️ Warning: No prunable parameters available, but target sparsity > 0. Cannot prune further.")
        return model

    if num_params_to_prune_from_non_circuit > total_prunable_count:
        print(f"⚠️ Warning: Target sparsity requires removing more parameters than available ({num_params_to_prune_from_non_circuit} > {total_prunable_count}). Pruning all non-circuit parameters.")
        num_params_to_prune_from_non_circuit = total_prunable_count

    # Determine the magnitude threshold for pruning
    all_prunable_values = torch.cat([p.view(-1).abs() for p in prunable_params])
    
    # Find the k-th smallest value (k = number of elements to prune). This is the threshold.
    # Ensure k is at least 1 if we need to prune anything, and capped at the total number of elements.
    k = min(max(1, num_params_to_prune_from_non_circuit), all_prunable_values.numel())
    threshold = torch.kthvalue(all_prunable_values, k=k).values

    # Apply pruning (magnitude-based) only to non-circuit parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in protection_mask:
                # mask (protection_mask[name]) defines which elements are prunable (True)
                mask = protection_mask[name]
                
                if mask.any():
                    # Apply magnitude threshold only to the prunable parts
                    prunable_data = param.data[mask]
                    # Create a mask for elements to keep (abs > threshold). 
                    # Using > ensures we prune approximately k elements.
                    keep_mask_prunable = prunable_data.abs() > threshold
                    # Apply the mask: set pruned elements to 0 by multiplying with the keep mask (cast to float)
                    param.data[mask] = prunable_data * keep_mask_prunable.float()


    print("✅ Hybrid Pruning complete.")
    return model

# ============================================================================
# 4. Evaluation Function (No major changes needed here, assuming it works reliably)
# ============================================================================
def evaluate_model(model_to_eval: nn.Module, tokenizer_to_use):
    """
    Evaluates the pruned model on the JSTS benchmark using a simple, robust MTEB wrapper.
    """
    print(f"\n🚀 Starting evaluation on JSTS benchmark...")
    start_time = time.time()
    
    class MTEBWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def _mean_pooling(self, model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        @torch.no_grad()
        def encode(self, sentences, batch_size=32, **kwargs):
            self.model.eval()
            all_embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt", max_length=512
                ).to(self.model.device)
                
                model_output = self.model(**inputs)
                pooled_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu())

            return torch.cat(all_embeddings, dim=0)

    mteb_model = MTEBWrapper(model=model_to_eval, tokenizer=tokenizer_to_use)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    
    output_folder_path = Path(RESULTS_DIR) / f"jsts_results_ccprune_{int(time.time())}"
    
    results = evaluation.run(mteb_model, output_folder=str(output_folder_path), verbosity=1, eval_splits=["validation"])
    
    end_time = time.time()
    print(f"✅ Evaluation finished in {end_time - start_time:.2f} seconds.")
    
    try:
        pearson_score = results[0].scores["validation"][0]["pearson"]
        return pearson_score
    except (KeyError, TypeError, IndexError) as e:
        print(f"⚠️ Could not extract Pearson score. Error: {e}")
        print("Full results:", results)
        return 0.0

# ============================================================================
# 5. Main Execution Block
# ============================================================================
if __name__ == "__main__":
    
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(f"Causal scores not found at '{SUMMARY_CSV_PATH}'. Run activation_patching_analysis.py first.")
    
    print("Loading causal score data...")
    neuron_summary_df = pd.read_csv(SUMMARY_CSV_PATH)
    print("✅ Causal score data loaded.")

    CIRCUIT_RETENTION_RATIO = 0.3
    TARGET_SPARSITY_LEVELS = [i / 100.0 for i in range(101)]
    performance_results = []
    
    print("Loading original model and tokenizer...")
    # trust_remote_code=True might be needed depending on the model source/version
    try:
        original_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load model with trust_remote_code=True, trying without. Error: {e}")
        original_model = AutoModel.from_pretrained(MODEL_NAME)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Model and tokenizer loaded.")

    for sparsity in TARGET_SPARSITY_LEVELS:
        print("\n" + "="*60)
        print(f"PROCESSING TARGET SPARSITY: {sparsity*100:.0f}%")
        print("="*60)

        # Deepcopy the model to ensure isolation between pruning runs
        model_to_prune = copy.deepcopy(original_model).to(DEVICE)
        pruned_model = hybrid_prune_model(model_to_prune, neuron_summary_df.copy(), sparsity, CIRCUIT_RETENTION_RATIO)
        
        actual_sparsity = report_sparsity(pruned_model, f"Pruned Model ({sparsity*100:.0f}%)")
        score = evaluate_model(pruned_model, tokenizer)
        performance_results.append({
            "target_sparsity": f"{sparsity*100:.0f}%",
            "circuit_retention": f"{CIRCUIT_RETENTION_RATIO*100:.0f}%",
            "actual_sparsity": f"{actual_sparsity:.2f}%",
            "jsts_pearson_score": f"{score:.4f}",
        })
        
        # Clean up memory
        del model_to_prune, pruned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\n" + "="*60)
    print("📊 FINAL PERFORMANCE REPORT")
    print("="*60)
    
    results_df = pd.DataFrame(performance_results)
    print(results_df.to_string(index=False))
    results_df.to_csv(Path(RESULTS_DIR) / f"hybrid_pruning_results_{sane_model_name}.csv", index=False)