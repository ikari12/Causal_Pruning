# FILE: causal_wanda_pruning_experiment_fixed.py
# PURPOSE: To run pruning experiments with corrected evaluation caching.

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from mteb import MTEB
import torch.nn.functional as F
import warnings
import copy
import pandas as pd
import re

# Suppress all UserWarnings from the script.
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MODEL_ID = "cl-nagoya/ruri-base-v2"
RESULTS_DIR = Path("/app/results/")
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
    """Evaluates a given model on the JSTS benchmark."""
    print(f"\n🚀 Evaluating {description} on JSTS...")
    mteb_model = MTEBWrapper(model=model, tokenizer=tokenizer)
    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    
    safe_desc = re.sub(r'[^a-zA-Z0-9_-]', '_', description)
    output_path = RESULTS_DIR / f"jsts_eval_{safe_desc}"
    print(f"Using unique output folder to prevent caching: {output_path}")
    
    results = evaluation.run(mteb_model, output_folder=output_path, verbosity=0, eval_splits=["validation"])
    
    pearson_score = results[0].scores["validation"][0]["pearson"]
    print(f"✅ JSTS Pearson Score for {description}: {pearson_score:.4f}")
    return pearson_score


def report_sparsity(model):
    """Calculates and reports the sparsity of the model."""
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
    # Sort the results by target sparsity for a clean table
    results_data.sort(key=lambda x: x['target_sparsity'])
    
    df = pd.DataFrame(results_data)
    df['retention'] = (df['score'] / baseline_score) * 100
    
    print("\n\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string(index=False, formatters={
        'target_sparsity': '{:.0%}'.format,
        'actual_sparsity': '{:.2%}'.format,
        'score': '{:.4f}'.format,
        'retention': '{:.2f}%'.format
    }))
    print("\n\n" + "="*80)
    print("📋 LATEX TABLE FOR PUBLICATION")
    print("="*80)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance of Wanda Pruning across various target sparsity levels.}")
    print("\\label{tab:wanda_results}")
    print("\\begin{tabular}{@{}cccc@{}}")
    print("\\toprule")
    print("Target Sparsity ($s$) & Actual Sparsity ($s_{\\text{actual}}$) & JSTS Pearson Score ($\\MB$) & Performance Retention \\\\")
    print("\\midrule")
    
    # The baseline row is now the 0% case from our loop
    # print(f"0\\% (Baseline) & 0.00\\% & {baseline_score:.4f} & 100.00\\% \\\\")
    
    for res in results_data:
        target_s = res['target_sparsity'] * 100
        actual_s = res['actual_sparsity'] * 100
        score = res['score']
        retention = (score / baseline_score) * 100
        
        row_label = f"{target_s:.0f}\\%"
        if target_s == 0:
            row_label += " (Baseline)"
            
        print(f"{row_label} & {actual_s:.2f}\\% & {score:.4f} & {retention:.2f}\\% \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*80)


def main():
    """Main function to execute the Wanda pruning experiment."""
    print("--- WANDA PRUNING EXPERIMENT SCRIPT ---")
    sane_model_name = MODEL_ID.replace('/', '_')
    scores_file_path = RESULTS_DIR / f"{sane_model_name}_wanda_scores.pt"

    if not scores_file_path.exists():
        raise FileNotFoundError(f"Scores file not found at {scores_file_path}. Please run scoring script first.")

    print(f"Loading original model: {MODEL_ID}")
    original_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Loading Wanda scores from: {scores_file_path}")
    wanda_scores = torch.load(scores_file_path, map_location=DEVICE)

    baseline_score = evaluate_model_on_jsts(original_model, tokenizer, "Baseline Model")
    
    experiment_results = []
    
    for target_sparsity in TARGET_SPARSITY_LEVELS:
        print("\n" + "-"*60)
        print(f"Processing Target Sparsity: {target_sparsity*100:.0f}%")
        print("-"*60)
        
        # ▼▼▼ BUG FIX: Handle the 0% sparsity edge case ▼▼▼
        if target_sparsity == 0.0:
            print("Target sparsity is 0%, skipping pruning and using baseline results.")
            # We already have the baseline score. The actual sparsity is 0.
            actual_sparsity = 0.0
            score = baseline_score
            
            experiment_results.append({
                'target_sparsity': target_sparsity,
                'actual_sparsity': actual_sparsity,
                'score': score,
            })
            continue # Move to the next iteration
        # ▲▲▲ END FIX ▲▲▲

        model_to_prune = copy.deepcopy(original_model).to(DEVICE)
        
        all_scores = torch.cat([scores.view(-1) for scores in wanda_scores.values()])
        k = int(len(all_scores) * target_sparsity)
        threshold = torch.kthvalue(all_scores, k).values

        with torch.no_grad():
            for name, module in model_to_prune.named_modules():
                if isinstance(module, nn.Linear) and name in wanda_scores:
                    mask = wanda_scores[name] > threshold
                    module.weight.data *= mask.float()
        
        print("✅ Pruning complete.")
        
        actual_sparsity = report_sparsity(model_to_prune)
        score = evaluate_model_on_jsts(model_to_prune, tokenizer, f"Pruned Model ({target_sparsity*100:.0f}%)")
        
        experiment_results.append({
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'score': score,
        })
        
        del model_to_prune
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print_results_table(experiment_results, baseline_score)
    pd.DataFrame(experiment_results).to_csv(RESULTS_DIR / f"wanda_pruning_results_{sane_model_name}.csv", index=False)


if __name__ == "__main__":
    main()