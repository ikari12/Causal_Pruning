# FILE: causal_wanda_pruning.py
# PURPOSE: To prune a model using pre-calculated Wanda scores and evaluate it.

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from mteb import MTEB
import torch.nn.functional as F

# --- Configuration ---
MODEL_ID = "cl-nagoya/ruri-base-v2"
RESULTS_DIR = Path("./pruning_results")
TARGET_SPARSITY = 0.5  # Target 50% sparsity.
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
    results = evaluation.run(mteb_model, output_folder=RESULTS_DIR / "jsts_eval", verbosity=1, eval_splits=["test"])
    pearson_score = results[0].scores["test"]["cos_sim"]["pearson"]
    print(f"✅ JSTS Pearson Score for {description}: {pearson_score:.4f}")
    return pearson_score


def report_sparsity(model):
    """Calculates and reports the sparsity of the model."""
    total_params, zero_params = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
    sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0
    print(f"Model Sparsity: {sparsity:.2f}% ({zero_params:,} / {total_params:,} zero parameters)")
    return sparsity


def main():
    """
    Main function to execute the Wanda pruning and evaluation process.
    """
    print("--- WANDA PRUNING SCRIPT ---")
    sane_model_name = MODEL_ID.replace('/', '_')
    scores_file_path = RESULTS_DIR / f"{sane_model_name}_wanda_scores.pt"

    if not scores_file_path.exists():
        raise FileNotFoundError(f"Scores file not found at {scores_file_path}. Please run causal_wanda_scoring.py first.")

    # 1. Load the model, tokenizer, and pre-calculated scores.
    print(f"Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Loading Wanda scores from: {scores_file_path}")
    wanda_scores = torch.load(scores_file_path, map_location=DEVICE)

    # 2. Evaluate the baseline (unpruned) model.
    evaluate_model_on_jsts(model, tokenizer, "Baseline Model")
    report_sparsity(model)

    # 3. Determine the global pruning threshold.
    print(f"\nPruning model to {TARGET_SPARSITY*100:.1f}% target sparsity...")
    all_scores = torch.cat([scores.view(-1) for scores in wanda_scores.values()])
    k = int(len(all_scores) * TARGET_SPARSITY)
    threshold = torch.kthvalue(all_scores, k).values

    # 4. Apply the pruning mask based on the threshold.
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in wanda_scores:
                mask = wanda_scores[name] > threshold
                module.weight.data *= mask.float()
    
    print("✅ Pruning complete.")

    # 5. Evaluate the pruned model.
    evaluate_model_on_jsts(model, tokenizer, "Wanda Pruned Model")
    report_sparsity(model)

    # Save the pruned model (optional).
    pruned_model_path = RESULTS_DIR / f"{sane_model_name}-wanda-pruned"
    model.save_pretrained(pruned_model_path)
    tokenizer.save_pretrained(pruned_model_path)
    print(f"\nPruned model saved to: {pruned_model_path}")


if __name__ == "__main__":
    main()
