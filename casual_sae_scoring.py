#
# Script: calculate_sae_scores.py
# Purpose: To train a Sparse Autoencoder (SAE) on a model's activations
#          and then calculate an importance score for each neuron based on
#          its alignment with the learned SAE features (basis vectors).
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. Configuration
# ============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/app/results")

# SAE Training Config
DATASET_PATH = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
ACTIVATION_EXTRACTION_BATCH_SIZE = 32
SAE_TRAINING_BATCH_SIZE = 128
SAE_EPOCHS = 100  # Adjust if needed for convergence

print(f"Using device: {DEVICE}")

# ============================================================================
# 2. Sparse Autoencoder (SAE) Class and Training Functions
# ============================================================================
class SparseAutoEncoder(nn.Module):
    """Represents a standard Sparse Autoencoder."""
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=0.01):
        super().__init__()
        self.input_dim = input_dim      # This line might also be missing, good to have
        self.hidden_dim = hidden_dim    # This line might also be missing, good to have
        self.sparsity_lambda = sparsity_lambda  # <<< ADD THIS LINE
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self._normalize_decoder_weights()

    def _normalize_decoder_weights(self):
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norm + 1e-8)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def get_dictionary(self):
        return self.decoder.weight.T

def train_sae(model, tokenizer):
    """Trains an SAE on the activations of the provided model."""
    print("--- 🔬 Phase 1: Training Sparse Autoencoder ---")
    
    # --- 1a. Extract Activations ---
    print("Extracting model activations...")
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train", trust_remote_code=True)
    
    all_sentence_representations = []
    model.eval()
    layer_idx = model.config.num_hidden_layers // 2
    target_layer_name = f"encoder.layer.{layer_idx}"
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), ACTIVATION_EXTRACTION_BATCH_SIZE), desc="Extracting Activations"):
            batch = dataset[i: i + ACTIVATION_EXTRACTION_BATCH_SIZE]
            inputs = tokenizer(
                batch['sentence1'], padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(DEVICE)
            
            # Get hidden states from the target layer
            outputs = model(**inputs, output_hidden_states=True)
            layer_output = outputs.hidden_states[layer_idx + 1] # +1 because hidden_states includes embeddings

            mask = inputs['attention_mask'].unsqueeze(-1)
            masked_output = layer_output * mask
            sentence_reps = masked_output.sum(dim=1) / mask.sum(dim=1)
            all_sentence_representations.append(sentence_reps.cpu())

    all_sentence_representations = torch.cat(all_sentence_representations)
    print(f"✅ Activation extraction complete. Shape: {all_sentence_representations.shape}")

    # --- 1b. Train SAE ---
    print("\nTraining SAE...")
    input_dim = all_sentence_representations.shape[1]
    hidden_dim = input_dim * 4  # 4x overcomplete dictionary
    sae = SparseAutoEncoder(input_dim, hidden_dim, sparsity_lambda=0.01).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    
    train_loader = DataLoader(
        TensorDataset(all_sentence_representations),
        batch_size=SAE_TRAINING_BATCH_SIZE,
        shuffle=True
    )
    
    sae.train()
    for epoch in tqdm(range(SAE_EPOCHS), desc="SAE Training Epochs"):
        for batch in train_loader:
            data_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_hat, z = sae(data_batch)
            reconstruction_loss = F.mse_loss(x_hat, data_batch)
            sparsity_loss = sae.sparsity_lambda * torch.mean(torch.abs(z))
            total_loss = reconstruction_loss + sparsity_loss
            total_loss.backward()
            optimizer.step()
            sae._normalize_decoder_weights()
            
    print("✅ SAE training complete.")
    return sae.eval()

# ============================================================================
# 3. SAE-based Neuron Scoring
# ============================================================================
def calculate_sae_scores(model, sae_dictionary):
    """
    Calculates importance scores for all neurons based on their alignment
    with the learned SAE dictionary features.
    """
    print("\n--- 🔬 Phase 2: Calculating SAE-based Neuron Scores ---")
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    d_ffn = model.config.intermediate_size
    
    all_neuron_scores = []
    
    # Normalize the SAE dictionary for cosine similarity calculation
    # Note: sae_dictionary has shape (num_features, feature_dim) -> (3072, 768)
    sae_dictionary_norm = F.normalize(sae_dictionary, p=2, dim=1).to(DEVICE)

    # 1. Calculate scores for FFN neurons
    print("Calculating scores for FFN neurons...")
    for layer_idx in tqdm(range(n_layers), desc="FFN Layers"):
        # An FFN neuron's output direction is a column in the output weight matrix
        # Shape: (hidden_dim, intermediate_dim) -> (768, 3072)
        ffn_out_weights = model.encoder.layer[layer_idx].output.dense.weight.data
        ffn_out_weights_norm = F.normalize(ffn_out_weights, p=2, dim=0)

        # FIX: Removed the unnecessary transpose (.T) from sae_dictionary_norm
        # Correct multiplication: (3072, 768) @ (768, 3072)
        similarities = torch.matmul(sae_dictionary_norm, ffn_out_weights_norm)
        
        # The score for each neuron is its maximum similarity to any SAE feature
        max_sims, _ = torch.max(torch.abs(similarities), dim=0)
        
        for dim_idx in range(d_ffn):
            score = max_sims[dim_idx].item()
            neuron_id = f"L{layer_idx}_FFN_D{dim_idx}"
            all_neuron_scores.append({
                "neuron_id": neuron_id, "type": "FFN", "layer": layer_idx,
                "head": -1, "dim": dim_idx, "sae_score": score
            })

    # 2. Calculate scores for Attention neurons
    print("Calculating scores for Attention neurons...")
    for layer_idx in tqdm(range(n_layers), desc="ATTN Layers"):
        # An Attention neuron's output direction is a column in the output weight matrix
        # Shape: (hidden_dim, hidden_dim) -> (768, 768)
        attn_out_weights = model.encoder.layer[layer_idx].attention.output.dense.weight.data
        attn_out_weights_norm = F.normalize(attn_out_weights, p=2, dim=0)

        # FIX: Removed the unnecessary transpose (.T) from sae_dictionary_norm
        # Correct multiplication: (3072, 768) @ (768, 768)
        similarities = torch.matmul(sae_dictionary_norm, attn_out_weights_norm)
        max_sims, _ = torch.max(torch.abs(similarities), dim=0)
        
        for head_idx in range(n_heads):
            for dim_idx in range(d_head):
                neuron_col_idx = head_idx * d_head + dim_idx
                score = max_sims[neuron_col_idx].item()
                neuron_id = f"L{layer_idx}_H{head_idx}_D{dim_idx}"
                all_neuron_scores.append({
                    "neuron_id": neuron_id, "type": "ATTN", "layer": layer_idx,
                    "head": head_idx, "dim": dim_idx, "sae_score": score
                })

    return pd.DataFrame(all_neuron_scores)

# ============================================================================
# 4. Main Execution Block
# ============================================================================
if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load base model
    print("Loading base model and tokenizer...")
    base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Model and tokenizer loaded.")
    
    # Train SAE to get the dictionary
    sae_model = train_sae(base_model, tokenizer)
    sae_dictionary = sae_model.get_dictionary()
    
    # Calculate neuron scores based on the dictionary
    summary_df = calculate_sae_scores(base_model, sae_dictionary)
    
    # Save results to CSV
    sane_model_name = MODEL_NAME.replace('/', '_')
    csv_path = RESULTS_DIR / f"sae_scores_{sane_model_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✅ SAE-based importance scores saved to: {csv_path}")


