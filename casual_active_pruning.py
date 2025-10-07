#
# Script: hybrid_pruning.py (Corrected Version)
# Purpose: To prune the model using the Hybrid Pruning strategy and evaluate it reliably.
#
import torch
import torch.nn as nn
import copy
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from mteb import MTEB
import time
import numpy as np
import torch.nn.functional as F


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
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
    
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
    if not 0 <= target_sparsity < 1:
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
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            protection_mask[name] = torch.ones_like(param, dtype=torch.bool)
            
    for _, neuron in circuit_neurons.iterrows():
        layer, n_type = neuron['layer'], neuron['type']
        
        if n_type == 'ATTN':
            head, dim = neuron['head'], neuron['dim']
            w_o_name = f"encoder.layer.{layer}.attention.output.dense.weight"
            start_col = head * d_head + dim
            if start_col < protection_mask[w_o_name].shape[1]:
                protection_mask[w_o_name][:, start_col] = False
        
        elif n_type == 'FFN':
            dim = neuron['dim']
            w_in_name = f"encoder.layer.{layer}.intermediate.dense.weight"
            if dim < protection_mask[w_in_name].shape[0]:
                protection_mask[w_in_name][dim, :] = False

            w_out_name = f"encoder.layer.{layer}.output.dense.weight"
            if dim < protection_mask[w_out_name].shape[1]:
                protection_mask[w_out_name][:, dim] = False

    # Phase 2: Hybrid Pruning
    prunable_params = []
    total_prunable_count = 0
    for name, param in model.named_parameters():
        if name in protection_mask:
            prunable_indices = protection_mask[name]
            prunable_params.append(param.data[prunable_indices])
            total_prunable_count += prunable_indices.sum().item()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_protected_params = total_params - total_prunable_count
    print(f"Total parameters: {total_params:,}")
    print(f"Protected (circuit) parameters: {num_protected_params:,}")
    print(f"Prunable (non-circuit) parameters: {total_prunable_count:,}")

    num_params_to_prune_total = int(total_params * target_sparsity)
    num_params_to_prune_from_non_circuit = num_params_to_prune_total
    
    if num_params_to_prune_from_non_circuit <= 0:
        print("✅ No prunable parameters need to be removed to reach target sparsity.")
        return model
    if num_params_to_prune_from_non_circuit > total_prunable_count:
        print(f"⚠️ Warning: Target sparsity requires removing more parameters than available. Pruning all non-circuit parameters.")
        num_params_to_prune_from_non_circuit = total_prunable_count

    all_prunable_values = torch.cat([p.abs() for p in prunable_params])
    threshold = torch.kthvalue(all_prunable_values, k=num_params_to_prune_from_non_circuit).values

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in protection_mask:
                mask = protection_mask[name]
                param.data[mask] = param.data[mask] * (param.data[mask].abs() > threshold)

    print("✅ Hybrid Pruning complete.")
    return model

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
    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    
    results = evaluation.run(mteb_model, output_folder=f"results/jsts_results", verbosity=1)
    
    end_time = time.time()
    print(f"✅ Evaluation finished in {end_time - start_time:.2f} seconds.")
    
    try:
        pearson_score = results["JSTS"]["test"]["cos_sim"]["pearson"]
        return pearson_score
    except (KeyError, TypeError):
        print("⚠️ Could not extract Pearson score from evaluation results.")
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
    TARGET_SPARSITY_LEVELS = [i / 10.0 for i in range(1, 10)] # 10% to 90%
    performance_results = []
    
    print("Loading original model and tokenizer...")
    original_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Model and tokenizer loaded.")

    for sparsity in TARGET_SPARSITY_LEVELS:
        print("\n" + "="*60)
        print(f"PROCESSING TARGET SPARSITY: {sparsity*100:.0f}%")
        print("="*60)

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

    print("\n\n" + "="*60)
    print("📊 FINAL PERFORMANCE REPORT")
    print("="*60)
    
    results_df = pd.DataFrame(performance_results)
    print(results_df.to_string(index=False))