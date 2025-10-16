"""
A complete implementation for a Sparse Autoencoder (SAE) and an analysis
of the basis vectors learned from the internal representations of the
Ruri-V2 model. This script performs dictionary learning using a standard
SAE as a method for analysing the internal workings of a Transformer,
offered as an alternative to the concept of DiscoGP.
"""

import time
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from mteb import MTEB
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Script Configuration and Global Parameters
# =============================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DATASET_PATH = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/app/results/causal_sae_analysis/")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Define the batch sizes for different stages of the process to manage memory.
ACTIVATION_EXTRACTION_BATCH_SIZE = 32
SAE_TRAINING_BATCH_SIZE = 128
SAE_EPOCHS = 100

print(f"Using compute device: {DEVICE}")


# =============================================================================
# 2. Definition of the Sparse Autoencoder (SAE) Class
# =============================================================================
class SparseAutoEncoder(nn.Module):
    """
    Represents a standard Sparse Autoencoder.

    The objective is to find a sparse representation 'z' for an input 'x'
    by minimising the reconstruction error plus an L1 penalty on the
    representation, according to the mathematical formulation:
    min ||x - Dz||² + λ||z||₁
    """

    def __init__(self, input_dim, hidden_dim, sparsity_lambda=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda

        # The encoder projects the input data onto the learned dictionary.
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        # The decoder reconstructs the original data from the sparse code.
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self._normalize_decoder_weights()

    def _normalize_decoder_weights(self):
        """
        Normalises the columns of the decoder's weight matrix (the
        dictionary atoms) to have a unit L2 norm. This is a crucial step
        in dictionary learning to prevent the arbitrary scaling of dictionary
        atoms and their corresponding codes.
        """
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norm + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the encoding step, transforming an input tensor into its
        sparse representation. The Rectified Linear Unit (ReLU) activation
        is applied to enforce non-negativity in the sparse codes.
        """
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the decoding step, reconstructing the original input
        from its sparse representation.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Defines the forward pass for the model."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x, x_hat, z):
        """
        Calculates the total loss for the Sparse Autoencoder.

        This loss is a composite of the mean squared error (reconstruction
        loss) and an L1 penalty on the activations (sparsity loss),
        weighted by the lambda hyperparameter.
        """
        reconstruction_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        return (reconstruction_loss + sparsity_loss,
                reconstruction_loss,
                sparsity_loss)

    def get_dictionary(self) -> torch.Tensor:
        """
        Returns the learned dictionary. The dictionary atoms are the
        transposed weights of the decoder.
        """
        return self.decoder.weight.T

    def compute_sparsity_metrics(self, z: torch.Tensor) -> dict:
        """
        Computes several metrics to evaluate the sparsity of the learned
        representations.
        """
        # The L0 norm approximates the count of non-zero feature activations.
        l0_norm = (z > 1e-6).float().sum(dim=-1).mean()
        # The proportion of feature activations that are non-zero.
        activation_rate = (z > 1e-6).float().mean()

        return {
            'l0_norm': l0_norm.item(),
            'activation_rate': activation_rate.item()
        }


# =============================================================================
# 3. Class for Extracting Activations from a Transformer Model
# =============================================================================
class ActivationExtractor:
    """
    Provides a mechanism for extracting the internal activation values
    from specified layers of a Transformer model using forward hooks.
    """

    def __init__(self, model: nn.Module, layer_names: list[str] = None):
        self.model = model
        self.activations = {}
        self.hooks = []

        if layer_names is None:
            # Default to registering hooks on all transformer encoder layers.
            num_layers = model.config.num_hidden_layers
            layer_names = [f"encoder.layer.{i}.output" for i in
                           range(num_layers)]

        self.register_hooks(layer_names)

    def register_hooks(self, layer_names: list[str]):
        """
        Iterates through a list of layer names and registers a forward
        hook on each corresponding module within the model.
        """
        for name in layer_names:
            module = self._get_module_by_name(name)
            if module is not None:
                callback = lambda m, i, o, n=name: self._save_activation(n, o)
                hook = module.register_forward_hook(callback)
                self.hooks.append(hook)

    def _get_module_by_name(self, name: str) -> nn.Module | None:
        """
        Retrieves a specific submodule from the main model using its fully
        qualified name (e.g., 'encoder.layer.0.output').
        """
        module = self.model
        for part in name.split('.'):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _save_activation(self, name: str, output: torch.Tensor | tuple):
        """
        The callback function for the forward hook. It saves the output
        tensor of a hooked layer into a dictionary.
        """
        if isinstance(output, tuple):
            output = output[0]
        self.activations[name] = output.detach()

    def extract(self, inputs: dict) -> dict:
        """
        Performs a forward pass with the provided inputs to trigger the
        hooks and capture the activations from the specified layers.
        """
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(**inputs)
        return self.activations.copy()

    def cleanup(self):
        """
        Removes all registered forward hooks to prevent memory leaks and
        unintended behaviour.
        """
        for hook in self.hooks:
            hook.remove()


# =============================================================================
# 4. Class for Analysing the Learned Dictionary Basis
# =============================================================================
class BasisAnalyzer:
    """
    A utility class containing static methods for the mathematical analysis
    and visualisation of the learned dictionary (basis vectors).
    """

    @staticmethod
    def analyze_dictionary_properties(dictionary: torch.Tensor) -> dict:
        """
        Performs a comprehensive mathematical analysis of the dictionary,
        calculating key properties such as coherence and condition number.
        """
        n_features, n_atoms = dictionary.shape

        # Coherence measures the maximum similarity between distinct atoms.
        normalized_dict = F.normalize(dictionary, dim=0)
        coherence_matrix = torch.abs(normalized_dict.T @ normalized_dict)
        coherence_matrix.fill_diagonal_(0)
        max_coherence = torch.max(coherence_matrix).item()

        # The condition number of the matrix indicates its stability.
        _, s_values, _ = torch.svd(dictionary)
        condition_number = ((s_values.max() / s_values.min()).item() if
                            s_values.min() > 1e-10 else float('inf'))

        return {
            'n_features': n_features,
            'n_atoms': n_atoms,
            'overcompleteness_ratio': n_atoms / n_features,
            'max_coherence': max_coherence,
            'condition_number': condition_number,
        }

    @staticmethod
    def visualize_dictionary(dictionary: torch.Tensor, save_path=None):
        """
        Generates plots to visualise the properties of the learned dictionary.
        This version replaces the original heatmap with a more informative clustermap
        and presents the other plots in a separate figure for clarity.
        """
        # --- Figure 1: Clustermap for Dictionary Atoms ---
        print("Generating Figure 1: Clustermap of Dictionary Atoms...")

        # Prepare data for the clustermap (first 50 atoms for readability)
        dict_subset = dictionary.detach().cpu().numpy()
        dict_subset = dict_subset[:, :min(50, dict_subset.shape[1])]

        # Create the clustermap.
        # row_cluster=False disables clustering for the Y-axis (Feature Dimension).
        # col_cluster=True enables clustering for the X-axis (Atom Index).
        g = sns.clustermap(
            dict_subset,
            row_cluster=False,
            col_cluster=True,
            figsize=(12, 8),
            cmap='RdBu_r',
            dendrogram_ratio=(0.1, 0.2), # Allocate space for dendrograms
            cbar_pos=(0.02, 0.8, 0.05, 0.18) # Position the colorbar
        )
        g.fig.suptitle('Clustermap of Dictionary Atoms (Basis Vectors)', fontsize=16)
        g.ax_heatmap.set_xlabel('Atom Index (Clustered by Similarity)')
        g.ax_heatmap.set_ylabel('Feature Dimension')

        # Save the clustermap figure if a path is provided
        if save_path:
            clustermap_save_path = save_path.with_name(save_path.stem + '_clustermap' + save_path.suffix)
            plt.savefig(clustermap_save_path, dpi=300, bbox_inches='tight')
            print(f"Clustermap saved to: {clustermap_save_path}")
        plt.show()


        # --- Figure 2: Other Analysis Plots ---
        print("\nGenerating Figure 2: Additional Dictionary Analyses...")
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # Plot 2.1: Semi-log plot of the singular values
        _, s_values, _ = torch.svd(dictionary)
        axes[0].semilogy(s_values.detach().cpu().numpy(), 'b-o', markersize=3)
        axes[0].set_title('Singular Value Distribution')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Singular Value (Logarithmic Scale)')
        axes[0].grid(True)

        # Plot 2.2: Mutual Coherence Distribution
        normalized_dict = F.normalize(dictionary, dim=0)
        coherence_matrix = torch.abs(normalized_dict.T @ normalized_dict)
        coherence_matrix.fill_diagonal_(0)
        indices = torch.triu_indices(
            coherence_matrix.size(0), coherence_matrix.size(1), offset=1
        )
        coherence_values = coherence_matrix[indices[0], indices[1]]
        axes[1].hist(coherence_values.detach().cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title('Mutual Coherence Distribution')
        axes[1].set_xlabel('Coherence Value')
        axes[1].set_ylabel('Frequency')

        # Plot 2.3: Cumulative Explained Variance
        eigenvals = s_values ** 2
        cumulative_ratio = torch.cumsum(eigenvals, dim=0) / eigenvals.sum()
        axes[2].plot(cumulative_ratio.detach().cpu().numpy(), 'g-', linewidth=2)
        axes[2].axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
        axes[2].axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
        axes[2].set_title('Cumulative Explained Variance')
        axes[2].set_xlabel('Number of Principal Components')
        axes[2].set_ylabel('Cumulative Ratio of Variance')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            other_plots_save_path = save_path.with_name(save_path.stem + '_metrics' + save_path.suffix)
            plt.savefig(other_plots_save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plots saved to: {other_plots_save_path}")
        plt.show()

# =============================================================================
# 5. MTEB Evaluation Function
# =============================================================================
def evaluate_model(model_to_eval: nn.Module, tokenizer_to_use) -> float:
    """Evaluates the given model on the JSTS benchmark using MTEB."""
    print("\n🚀 Starting baseline evaluation on JSTS benchmark...")
    start_time = time.time()

    class MTEBWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def _mean_pooling(self, model_output, attention_mask):
            token_embeds = model_output.last_hidden_state
            expanded_mask = attention_mask.unsqueeze(-1).expand(
                token_embeds.size()).float()
            sum_embeds = torch.sum(token_embeds * expanded_mask, 1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            return sum_embeds / sum_mask

        @torch.no_grad()
        def encode(self, sentences, batch_size=32, **kwargs):
            self.model.eval()
            all_embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i: i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True,
                    return_tensors="pt", max_length=512
                ).to(self.model.device)

                model_output = self.model(**inputs)
                pooled = self._mean_pooling(model_output,
                                            inputs['attention_mask'])
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu())
            return torch.cat(all_embeddings, dim=0)

    mteb_model = MTEBWrapper(model=model_to_eval, tokenizer=tokenizer_to_use)

    evaluation = MTEB(tasks=["JSTS"], task_langs=["ja"])
    folder_name = f"jsts_results_baseline_{int(time.time())}"
    output_folder_path = RESULTS_DIR / folder_name

    results = evaluation.run(
        mteb_model,
        output_folder=str(output_folder_path),
        verbosity=1,
        eval_splits=["validation"]
    )

    end_time = time.time()
    print(f"✅ Evaluation finished in {end_time - start_time:.2f} seconds.")

    try:
        # This structure is robust for MTEB v1.1.0 and later.
        pearson_score = results["JSTS"]["validation"]["main_score"]
        return pearson_score
    except (KeyError, TypeError, IndexError) as e:
        print(f"⚠️ Could not extract Pearson score. Error: {e}")
        print("Full results:", results)
        return 0.0


# =============================================================================
# 6. Main Execution Function
# =============================================================================
def main():
    """
    This function orchestrates the entire process, from loading the model
    and data to extracting activations, training the SAE, and analysing
    the results.
    """
    print("=== Basis Analysis of Ruri-V2 using a Sparse Autoencoder ===\n")

    # Step 1: Load the pre-trained model and its tokenizer.
    print("1. Loading the model and tokenizer...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Model loaded successfully. Hidden size: "
          f"{model.config.hidden_size} dimensions.")

    # Step 1.5: Evaluate the baseline performance of the original model.
    model.eval()
    baseline_score = evaluate_model(model, tokenizer)
    print(f"\n📊 Baseline JSTS Pearson Score for '{MODEL_NAME}': "
          f"{baseline_score:.4f}\n")

    # Step 2: Load the dataset for extracting activations.
    print("2. Preparing the dataset...")
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train", trust_remote_code=True)
    print(f"✅ Data preparation complete. Total samples: {len(dataset)}.")

    # Step 3: Extract internal activations in batches to conserve memory.
    print("\n3. Extracting internal activations from the Transformer...")
    layer_idx = model.config.num_hidden_layers // 2
    target_layer = f"encoder.layer.{layer_idx}.output"
    extractor = ActivationExtractor(model, [target_layer])

    all_sentence_representations = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset),
                            ACTIVATION_EXTRACTION_BATCH_SIZE),
                      desc="Extracting Activations"):
            batch = dataset[i: i + ACTIVATION_EXTRACTION_BATCH_SIZE]
            inputs = tokenizer(
                batch['sentence1'], padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            activations = extractor.extract(inputs)
            layer_output = activations[target_layer].to(DEVICE)

            mask = inputs['attention_mask'].unsqueeze(-1)
            masked_output = layer_output * mask
            sentence_reps = masked_output.sum(dim=1) / mask.sum(dim=1)
            all_sentence_representations.append(sentence_reps.cpu())

    all_sentence_representations = torch.cat(all_sentence_representations)
    print("✅ Activation extraction complete. Final tensor shape: "
          f"{all_sentence_representations.shape}.")

    # Step 4: Train the Sparse Autoencoder on the extracted activations.
    print("\n4. Training the Sparse Autoencoder in batches...")
    input_dim = all_sentence_representations.shape[1]
    hidden_dim = input_dim * 4
    sae = SparseAutoEncoder(input_dim, hidden_dim, sparsity_lambda=0.01)
    sae.to(DEVICE)
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
            total_loss, _, _ = sae.compute_loss(data_batch, x_hat, z)
            total_loss.backward()
            optimizer.step()
            sae._normalize_decoder_weights()
    print("✅ SAE training complete.")

    # Step 5: Analyse the learned dictionary.
    print("\n5. Analysing the learned dictionary (bases)...")
    sae.eval()
    dictionary = sae.get_dictionary()
    dict_properties = BasisAnalyzer.analyze_dictionary_properties(dictionary)
    print("\n=== Mathematical Properties of the Dictionary ===")
    print(f"Input dimensions: {dict_properties['n_features']}")
    print(f"Number of dictionary atoms: {dict_properties['n_atoms']}")
    print(f"Overcompleteness ratio: "
          f"{dict_properties['overcompleteness_ratio']:.2f}")
    print(f"Maximum coherence: {dict_properties['max_coherence']:.4f}")
    print(f"Condition number: {dict_properties['condition_number']:.2f}")

    # Step 6: Visualise the results.
    print("\n6. Visualising the results...")
    BasisAnalyzer.visualize_dictionary(
        dictionary,
        save_path=RESULTS_DIR / "dictionary_analysis.png"
    )

    print(f"\n✅ Analysis complete! All results have been saved to the "
          f"directory: {RESULTS_DIR}")
    extractor.cleanup()


if __name__ == "__main__":
    main()