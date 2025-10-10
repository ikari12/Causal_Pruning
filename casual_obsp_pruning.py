# FILE: complete_obsp_pruning.py
# PURPOSE: To prune a model using the Optimal Brain Surgeon (OBS) method,
# which is the foundational algorithm for SparseGPT, and evaluate its performance.

import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from mteb import MTEB
import torch.nn.functional as F
from sparseml.transformers.pruning import OBSPruner, PruningConfig

# --- Configuration ---
MODEL_ID = "cl-nagoya/ruri-base-v2"
DATASET_ID = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
RESULTS_DIR = Path("./pruning_results")
NUM_CALIBRATION_SAMPLES = 128
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
    results = evaluation.run(mteb_model, output_folder=RESULTS_DIR / "jsts_eval_obsp", verbosity=1, eval_splits=["test"])
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
    Main function to execute the OBS pruning and evaluation process.
    """
    print("--- SPARSEGPT-EQUIVALENT (OBS) PRUNING SCRIPT ---")
    RESULTS_DIR.mkdir(exist_ok=True)
    sane_model_name = MODEL_ID.replace('/', '_')

    # 1. Load the model and tokenizer.
    print(f"Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. Evaluate the baseline model.
    evaluate_model_on_jsts(model, tokenizer, "Baseline Model")
    report_sparsity(model)

    # 3. Prepare calibration data.
    print(f"\nLoading calibration data from {DATASET_ID} ({DATASET_SUBSET})...")
    dataset = load_dataset(DATASET_ID, DATASET_SUBSET, split="train", trust_remote_code=True)
    texts = [ex['sentence1'] for ex in dataset] + [ex['sentence2'] for ex in dataset]
    calibration_texts = texts[:NUM_CALIBRATION_SAMPLES]
    # Create a simple dataloader.
    calibration_data = [
        tokenizer(text, return_tensors="pt") for text in calibration_texts if text.strip() != ""
    ]

    # 4. Configure and run the OBS pruner.
    print(f"Configuring OBS pruner for {TARGET_SPARSITY*100:.1f}% target sparsity...")
    pruning_config = PruningConfig.optimal_brain_surgeon(
        sparsity=TARGET_SPARSITY,
    )
    pruner = OBSPruner(config=pruning_config)

    # Score weights (calculates importance and Hessian information).
    print("Scoring weights with OBS method...")
    pruner.score_weights(
        model=model,
        dataloader=calibration_data,
        device=DEVICE,
    )

    # Prune the model (this also performs the weight updates).
    print("Pruning model and updating remaining weights...")
    pruner.prune()
    
    print("✅ OBS pruning complete.")

    # 5. Evaluate the pruned model.
    evaluate_model_on_jsts(model, tokenizer, "OBS Pruned Model")
    report_sparsity(model)

    # Save the pruned model (optional).
    pruned_model_path = RESULTS_DIR / f"{sane_model_name}-obsp-pruned"
    model.save_pretrained(pruned_model_path)
    tokenizer.save_pretrained(pruned_model_path)
    print(f"\nPruned model saved to: {pruned_model_path}")


if __name__ == "__main__":
    main()
