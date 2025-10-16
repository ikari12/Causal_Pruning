#
# Script: hybrid_pruning_sae.py
# Purpose: To prune the model using the Hybrid Pruning strategy, where the
#          "causal circuit" is defined by high-scoring neurons based on
#          their alignment with SAE features.
#
import torch
import torch.nn as nn
import copy
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from pathlib import Path
from mteb import MTEB
import time
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*model_card_data.*")

# ============================================================================
# 1. Configuration
# ============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/app/results")
sane_model_name = MODEL_NAME.replace('/', '_')

# --- MODIFIED: Point to the new SAE score CSV ---
SUMMARY_CSV_PATH = RESULTS_DIR / f"sae_scores_{sane_model_name}.csv"
# --- MODIFIED: Define the score column to be used ---
SCORE_COLUMN_NAME = "sae_score"

# ============================================================================
# 2. Helper Function (No changes)
# ============================================================================
def report_sparsity(model: nn.Module, description: str):
    """Calculates and reports the parameter sparsity of the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
    sparsity = (1 - (nonzero_params / total_params)) * 100 if total_params > 0 else 0
    print(f"--- Sparsity Report for: {description} ---")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Non-zero parameters:  {nonzero_params:,}")
    print(f"  Model sparsity:       {sparsity:.2f}%")
    print("-" * 40)
    return sparsity

# ============================================================================
# 3. Core Hybrid Pruning Function (Modified for SAE score)
# ============================================================================
def hybrid_prune_model(model: nn.Module, summary_df: pd.DataFrame, target_sparsity: float, circuit_retention_ratio: float):
    """
    Implements the Hybrid Pruning methodology using SAE-based scores.
    """
    print(f"✂️ Starting Hybrid Pruning for {target_sparsity*100:.1f}% target sparsity...")
    print(f"   (Protecting top {circuit_retention_ratio*100:.1f}% of neurons based on '{SCORE_COLUMN_NAME}')")

    # --- MODIFIED: Use SAE score column for circuit discovery ---
    if SCORE_COLUMN_NAME not in summary_df.columns:
        raise ValueError(f"Score column '{SCORE_COLUMN_NAME}' not found in the provided CSV file.")
    
    score_threshold = summary_df[SCORE_COLUMN_NAME].quantile(1.0 - circuit_retention_ratio)
    circuit_neurons = summary_df[summary_df[SCORE_COLUMN_NAME] >= score_threshold]
    print(f"Identified {len(circuit_neurons)} neurons for the circuit (score >= {score_threshold:.6f}).")

    # The rest of the protection and pruning logic remains the same.
    # It correctly protects all weights associated with the identified neurons.
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
            neuron_idx = head * d_head + dim
            head_start_idx, head_end_idx = head * d_head, (head + 1) * d_head
            protection_mask[f"encoder.layer.{layer}.attention.output.dense.weight"][:, neuron_idx] = False
            protection_mask[f"encoder.layer.{layer}.attention.self.value.weight"][neuron_idx, :] = False
            protection_mask[f"encoder.layer.{layer}.attention.self.value.bias"][neuron_idx] = False
            for proj in ['query', 'key']:
                protection_mask[f"encoder.layer.{layer}.attention.self.{proj}.weight"][head_start_idx:head_end_idx, :] = False
                protection_mask[f"encoder.layer.{layer}.attention.self.{proj}.bias"][head_start_idx:head_end_idx] = False
        elif n_type == 'FFN':
            dim = neuron['dim']
            protection_mask[f"encoder.layer.{layer}.intermediate.dense.weight"][dim, :] = False
            protection_mask[f"encoder.layer.{layer}.intermediate.dense.bias"][dim] = False
            protection_mask[f"encoder.layer.{layer}.output.dense.weight"][:, dim] = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_prunable_count = sum(protection_mask[name].sum().item() for name in protection_mask)
    num_protected_params = total_params - total_prunable_count
    
    num_params_to_prune = int(total_params * target_sparsity)
    if num_params_to_prune <= 0: return model
    if num_params_to_prune > total_prunable_count:
        num_params_to_prune = total_prunable_count

    prunable_params = [p.data[protection_mask[name]] for name, p in model.named_parameters() if name in protection_mask and protection_mask[name].any()]
    all_prunable_values = torch.cat([p.view(-1).abs() for p in prunable_params])
    
    k = min(max(1, num_params_to_prune), all_prunable_values.numel())
    threshold = torch.kthvalue(all_prunable_values, k=k).values

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in protection_mask and protection_mask[name].any():
                mask = protection_mask[name]
                prunable_data = param.data[mask]
                keep_mask = (prunable_data.abs() > threshold).float()
                param.data[mask] = prunable_data * keep_mask

    print("✅ Hybrid Pruning complete.")
    return model

# ============================================================================
# 4. Evaluation Function (Corrected Version with Score Logging)
# ============================================================================
def evaluate_model(model_to_eval: nn.Module, tokenizer_to_use):
    """
    Evaluates the pruned model on the JSTS benchmark using a simple,
    robust MTEB wrapper. This version uses a corrected result parsing logic
    and explicitly logs the extracted score.
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
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.model.device)
                model_output = self.model(**inputs)
                pooled = self._mean_pooling(model_output, inputs['attention_mask'])
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu())
            return torch.cat(all_embeddings, dim=0)

    mteb_model = MTEBWrapper(model=model_to_eval, tokenizer=tokenizer_to_use)
    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    
    results = evaluation.run(mteb_model, output_folder=str(RESULTS_DIR / f"temp_mteb_{int(time.time())}"), verbosity=1, eval_splits=["validation"])

    try:
        pearson_score = results[0].scores["validation"][0]["pearson"]
        print(f"📊 Extracted JSTS Pearson Score: {pearson_score:.4f}")
        
        return pearson_score
    except (AttributeError, KeyError, TypeError, IndexError) as e:
        print(f"⚠️ Could not extract Pearson score. This is likely an MTEB version issue.")
        print(f"Error details: {e}")
        print("Full results object from MTEB:", results)
        return 0.0
    
# ============================================================================
# 5. Main Execution Block
# ============================================================================
if __name__ == "__main__":
    
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(f"SAE scores not found at '{SUMMARY_CSV_PATH}'. Run calculate_sae_scores.py first.")
    
    print(f"Loading SAE score data from {SUMMARY_CSV_PATH}...")
    neuron_summary_df = pd.read_csv(SUMMARY_CSV_PATH)
    print("✅ SAE score data loaded.")

    CIRCUIT_RETENTION_RATIO = 0.3
    TARGET_SPARSITY_LEVELS = [i / 100.0 for i in range(101)]
    performance_results = []
    
    print("\nLoading original model and tokenizer...")
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
        
        del model_to_prune, pruned_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n\n" + "="*60)
    print("📊 FINAL PERFORMANCE REPORT (SAE-based Pruning)")
    print("="*60)
    
    results_df = pd.DataFrame(performance_results)
    print(results_df.to_string(index=False))
    # --- MODIFIED: Save to a new results file ---
    results_df.to_csv(RESULTS_DIR / f"hybrid_pruning_sae_results_{sane_model_name}.csv", index=False)


