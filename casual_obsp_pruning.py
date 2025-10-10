# FILE: complete_obsp_pruning_experiment.py
# PURPOSE: To run pruning experiments using the Optimal Brain Surgeon (OBS) method
#          across multiple sparsity levels and generate a summary table.

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from mteb import MTEB
import torch.nn.functional as F
from sparseml.transformers.pruning import OBSPruner, PruningConfig
import warnings
import copy
import pandas as pd
import re

# Suppress all UserWarnings from the script.
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_ID = "cl-nagoya/ruri-base-v2"
DATASET_ID = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
RESULTS_DIR = Path("./pruning_results")
NUM_CALIBRATION_SAMPLES = 128
# Define the range of target sparsity levels for the experiment.
TARGET_SPARSITY_LEVELS = [i / 100.0 for i in range(101)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- End of Configuration ---


class MTEBWrapper:
    """A wrapper class for MTEB evaluation."""
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
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.model.device)
            model_output = self.model(**inputs)
            pooled = self._mean_pooling(model_output, inputs['attention_mask'])
            normalised_embeddings = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalised_embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


def evaluate_model_on_jsts(model, tokenizer, description):
    """Evaluates a given model on the JSTS benchmark with anti-caching."""
    print(f"\n🚀 Evaluating {description} on JSTS...")
    mteb_model = MTEBWrapper(model=model, tokenizer=tokenizer)
    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    
    # Create a unique output folder for each evaluation run to prevent caching.
    safe_desc = re.sub(r'[^a-zA-Z0-9_-]', '_', description)
    output_path = RESULTS_DIR / f"jsts_eval_obsp_{safe_desc}"
    print(f"Using unique output folder to prevent caching: {output_path}")
    
    results = evaluation.run(mteb_model, output_folder=output_path, verbosity=0, eval_splits=["validation"])
    
    # Correctly parse the results dictionary.
    pearson_score = results[0].scores["validation"][0]["pearson"]
    print(f"✅ JSTS Pearson Score for {description}: {pearson_score:.4f}")
    return pearson_score


def report_sparsity(model):
    """Calculates and reports the sparsity of the model, returning the ratio."""
    total_params, zero_params = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
    sparsity = (zero_params / total_params) if total_params > 0 else 0
    print(f"Model Sparsity: {sparsity*100:.2f}%")
    return sparsity


def print_results_table(results_data, baseline_score):
    """Formats and prints the final results in a pandas DataFrame and a LaTeX table."""
    df = pd.DataFrame(results_data)
    df['retention'] = (df['score'] / baseline_score) * 100
    
    print("\n\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY (OBS PRUNING)")
    print("="*80)
    print(df.to_string(index=False, formatters={
        'target_sparsity': '{:.0%}'.format,
        'actual_sparsity': '{:.2%}'.format,
        'score': '{:.4f}'.format,
        'retention': '{:.2f}%'.format
    }))
    print("\n\n" + "="*80)
    print("📋 LATEX TABLE FOR PUBLICATION (OBS PRUNING)")
    print("="*80)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance of OBS Pruning across various target sparsity levels.}")
    print("\\label{tab:obs_results}")
    print("\\begin{tabular}{@{}cccc@{}}")
    print("\\toprule")
    print("Target Sparsity ($s$) & Actual Sparsity ($s_{\\text{actual}}$) & JSTS Pearson Score ($\\MB$) & Performance Retention \\\\")
    print("\\midrule")
    
    print(f"0\\% (Baseline) & 0.00\\% & {baseline_score:.4f} & 100.00\\% \\\\")
    
    for res in results_data:
        target_s = res['target_sparsity'] * 100
        actual_s = res['actual_sparsity'] * 100
        score = res['score']
        retention = (score / baseline_score) * 100
        print(f"{target_s:.0f}\\% & {actual_s:.2f}\\% & {score:.4f} & {retention:.2f}\\% \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*80)


def main():
    """
    Main function to execute the OBS pruning experiment.
    """
    print("--- SPARSEGPT-EQUIVALENT (OBS) PRUNING EXPERIMENT SCRIPT ---")
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Load original model, tokenizer, and calibration data.
    print(f"Loading original model: {MODEL_ID}")
    original_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"\nLoading calibration data from {DATASET_ID} ({DATASET_SUBSET})...")
    dataset = load_dataset(DATASET_ID, DATASET_SUBSET, split="train", trust_remote_code=True)
    texts = [ex['sentence1'] for ex in dataset] + [ex['sentence2'] for ex in dataset]
    calibration_texts = texts[:NUM_CALIBRATION_SAMPLES]
    calibration_data = [
        tokenizer(text, return_tensors="pt") for text in calibration_texts if text.strip() != ""
    ]

    # 2. Evaluate the baseline model to get the reference score.
    baseline_score = evaluate_model_on_jsts(original_model, tokenizer, "Baseline Model")
    
    experiment_results = []
    
    # 3. Loop through all target sparsity levels.
    for target_sparsity in TARGET_SPARSITY_LEVELS:
        print("\n" + "-"*60)
        print(f"Processing Target Sparsity: {target_sparsity*100:.0f}%")
        print("-"*60)
        
        # Always start from a fresh copy of the original model.
        model_to_prune = copy.deepcopy(original_model).to(DEVICE)
        
        # Configure and run the OBS pruner for the current sparsity level.
        pruning_config = PruningConfig.optimal_brain_surgeon(sparsity=target_sparsity)
        pruner = OBSPruner(config=pruning_config)

        print("Scoring weights with OBS method...")
        pruner.score_weights(model=model_to_prune, dataloader=calibration_data, device=DEVICE)

        print("Pruning model and updating remaining weights...")
        pruner.prune()
        
        print("✅ OBS pruning complete.")
        
        # Evaluate and report actual sparsity.
        actual_sparsity = report_sparsity(model_to_prune)
        score = evaluate_model_on_jsts(model_to_prune, tokenizer, f"Pruned Model ({target_sparsity*100:.0f}%)")
        
        # Store results.
        experiment_results.append({
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'score': score,
        })
        
        # Clean up memory.
        del model_to_prune, pruner
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 4. Print the final summary table.
    print_results_table(experiment_results, baseline_score)


if __name__ == "__main__":
    main()