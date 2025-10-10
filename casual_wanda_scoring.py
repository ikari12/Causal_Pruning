# FILE: causal_wanda_scoring.py
# PURPOSE: To calculate and save Wanda importance scores for a given model.
# NOTE: Wanda is a magnitude-based pruning method, not a causal one.
# It uses the product of weight magnitude and activation magnitude as an importance heuristic.

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "cl-nagoya/ruri-base-v2"
DATASET_ID = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
RESULTS_DIR = Path("/app/results")
NUM_CALIBRATION_SAMPLES = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- End of Configuration ---


def main():
    """
    Main function to execute the Wanda scoring process.
    """
    print("---  WANDA SCORING SCRIPT ---")
    print(f"Using device: {DEVICE}")

    # Ensure the results directory exists.
    RESULTS_DIR.mkdir(exist_ok=True)
    sane_model_name = MODEL_ID.replace('/', '_')
    scores_file_path = RESULTS_DIR / f"{sane_model_name}_wanda_scores.pt"

    # 1. Load the model and tokenizer from Hugging Face.
    print(f"Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 2. Prepare the calibration dataset (JSTS).
    print(f"Loading calibration data from {DATASET_ID} ({DATASET_SUBSET})...")
    dataset = load_dataset(DATASET_ID, DATASET_SUBSET, split="train", trust_remote_code=True)
    # Using both sentence1 and sentence2 as calibration data.
    texts = [ex['sentence1'] for ex in dataset] + [ex['sentence2'] for ex in dataset]
    calibration_texts = texts[:NUM_CALIBRATION_SAMPLES]

    # 3. Instrument the model with forward hooks to capture activations.
    print("Instrumenting model with hooks to capture activation norms...")
    activation_data = {}

    def create_hook(name):
        # This hook function captures the input to a linear layer.
        def hook(module, inp, out):
            # We only need the input tensor.
            input_tensor = inp[0].detach()
            # To compute the L2 norm, we accumulate the sum of squares.
            # This is more memory-efficient than storing all activations.
            if name not in activation_data:
                activation_data[name] = torch.zeros(input_tensor.shape[-1], device=DEVICE)
            
            # Reduce over the batch and sequence length dimensions.
            activation_data[name] += torch.sum(input_tensor.pow(2), dim=[0, 1])
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Register a hook for each linear layer.
            hooks.append(module.register_forward_hook(create_hook(name)))

    # 4. Run the calibration data through the model.
    print(f"Running {len(calibration_texts)} calibration samples through the model...")
    total_tokens = 0
    with torch.no_grad():
        for text in tqdm(calibration_texts, desc="Calibration"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            model(**inputs)
            # Keep track of the total number of tokens for normalisation.
            total_tokens += inputs['input_ids'].numel()

    # Remove the hooks now that calibration is complete.
    for hook in hooks:
        hook.remove()

    # 5. Calculate and save the Wanda importance scores.
    print("Calculating and saving Wanda importance scores...")
    wanda_scores = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name not in activation_data:
                    print(f"Warning: No activation data found for layer {name}. Skipping.")
                    continue
                
                # Finalise the L2 norm: sqrt(sum_of_squares / num_tokens)
                activation_norm = torch.sqrt(activation_data[name] / total_tokens)
                
                # Reshape for broadcasting: [1, in_features]
                activation_norm = activation_norm.unsqueeze(0) 
                
                # Wanda Importance Score = |Weight| * ||Activation_Norm||
                wanda_scores[name] = module.weight.abs() * activation_norm

    torch.save(wanda_scores, scores_file_path)
    print(f"✅ Wanda scores successfully saved to: {scores_file_path}")


if __name__ == "__main__":
    main()
