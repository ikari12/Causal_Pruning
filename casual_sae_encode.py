"""
A complete implementation for a Sparse Autoencoder (SAE) and an analysis of the basis vectors learned
from the internal representations of the Ruri-V2 model. This script performs dictionary learning using a
standard SAE as a method for analysing the internal workings of a Transformer, offered as an alternative
to the concept of DiscoGP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =================================================================================================
# 1. Script Configuration and Global Parameters
# =================================================================================================
MODEL_NAME = "cl-nagoya/ruri-base-v2"
DATASET_PATH = "sbintuitions/JMTEB"
DATASET_SUBSET = "jsts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/home/sandbox/sae_results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Define the batch sizes for different stages of the process to manage memory usage.
ACTIVATION_EXTRACTION_BATCH_SIZE = 32  # The batch size used when extracting activations from the Transformer.
SAE_TRAINING_BATCH_SIZE = 128          # The mini-batch size used during the training of the Sparse Autoencoder.
SAE_EPOCHS = 100                       # The total number of epochs for training the Sparse Autoencoder.

print(f"Using compute device: {DEVICE}")

# =================================================================================================
# 2. Definition of the Sparse Autoencoder (SAE) Class
# =================================================================================================
class SparseAutoEncoder(nn.Module):
    """
    Represents a standard Sparse Autoencoder. The objective is to find a sparse representation 'z' for
    an input 'x' by minimising the reconstruction error plus an L1 penalty on the representation,
    according to the mathematical formulation: min ||x - Dz||² + λ||z||₁
    """
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        
        # The encoder component, which projects the input data onto the learned dictionary.
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        # The decoder component, which reconstructs the original data from the sparse representation.
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # This ensures that the dictionary's basis vectors (atoms) are consistently normalised.
        self._normalize_decoder_weights()
        
    def _normalize_decoder_weights(self):
        """
        This method normalises the columns of the decoder's weight matrix (the dictionary atoms)
        to have a unit L2 norm. This is a crucial step in dictionary learning to prevent the
        arbitrary scaling of dictionary atoms and their corresponding codes.
        """
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
    
    def encode(self, x):
        """Performs the encoding step, transforming an input tensor into its sparse representation."""
        # The Rectified Linear Unit (ReLU) activation function is applied to enforce non-negativity
        # in the sparse codes, a common practice in SAEs.
        return F.relu(self.encoder(x))
    
    def decode(self, z):
        """Performs the decoding step, reconstructing the original input from its sparse representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Defines the forward pass for the model."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def compute_loss(self, x, x_hat, z):
        """
        Calculates the total loss for the Sparse Autoencoder. This loss is a composite of the mean
        squared error (reconstruction loss) and an L1 penalty on the activations (sparsity loss),
        weighted by the lambda hyperparameter.
        """
        reconstruction_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        return reconstruction_loss + sparsity_loss, reconstruction_loss, sparsity_loss
    
    def get_dictionary(self):
        """Returns the learned dictionary. The dictionary atoms are the transposed weights of the decoder."""
        return self.decoder.weight.T
    
    def compute_sparsity_metrics(self, z):
        """Computes several metrics to evaluate the sparsity of the learned representations."""
        # The L0 norm, which approximates the count of non-zero elements in the feature activations.
        l0_norm = (z > 1e-6).float().sum(dim=-1).mean()
        # The L1 norm, which is the sum of the absolute values of the feature activations.
        l1_norm = torch.abs(z).sum(dim=-1).mean()
        # The proportion of feature activations that are non-zero.
        activation_rate = (z > 1e-6).float().mean()
        
        return {
            'l0_norm': l0_norm.item(),
            'l1_norm': l1_norm.item(), 
            'activation_rate': activation_rate.item()
        }

# =================================================================================================
# 3. Class for Extracting Activations from a Transformer Model
# =================================================================================================
class ActivationExtractor:
    """
    Provides a mechanism for extracting the internal activation values from specified layers of a
    Transformer model using forward hooks.
    """
    def __init__(self, model, layer_names=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        # If no specific layer names are provided, this class will default to registering hooks
        # on all transformer encoder layers.
        if layer_names is None:
            layer_names = [f"encoder.layer.{i}.output" for i in range(model.config.num_hidden_layers)]
        
        self.register_hooks(layer_names)
    
    def register_hooks(self, layer_names):
        """
        Iterates through a list of layer names and registers a forward hook on each corresponding
        module within the model.
        """
        for name in layer_names:
            module = self._get_module_by_name(name)
            if module is not None:
                hook = module.register_forward_hook(lambda m, i, o, n=name: self._save_activation(n, o))
                self.hooks.append(hook)
    
    def _get_module_by_name(self, name):
        """
        Retrieves a specific submodule from the main model using its fully qualified name
        (e.g., 'encoder.layer.0.output').
        """
        module = self.model
        for part in name.split('.'):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _save_activation(self, name, output):
        """
        This is the callback function for the forward hook. It saves the output tensor of a hooked
        layer into a dictionary.
        """
        if isinstance(output, tuple):
            output = output[0]
        self.activations[name] = output.detach()
    
    def extract(self, inputs):
        """
        Performs a forward pass with the provided inputs to trigger the hooks and capture the
        activations from the specified layers.
        """
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(**inputs)
        return self.activations.copy()
    
    def cleanup(self):
        """Removes all registered forward hooks to prevent memory leaks and unintended behaviour."""
        for hook in self.hooks:
            hook.remove()

# =================================================================================================
# 4. Class for Analysing the Learned Dictionary Basis
# =================================================================================================
class BasisAnalyzer:
    """
    A utility class containing static methods for the mathematical analysis and visualisation of the
    learned dictionary (basis vectors).
    """
    @staticmethod
    def analyze_dictionary_properties(dictionary):
        """
        Performs a comprehensive mathematical analysis of the dictionary, calculating key properties
        such as coherence, condition number, and spectral concentration.
        """
        # The dictionary is expected to be a tensor with dimensions corresponding to the input
        # feature size and the number of learned atoms.
        n_features, n_atoms = dictionary.shape
        
        # Coherence measures the maximum similarity between distinct dictionary atoms. A lower coherence is desirable.
        normalized_dict = F.normalize(dictionary, dim=0)
        coherence_matrix = torch.abs(normalized_dict.T @ normalized_dict)
        coherence_matrix.fill_diagonal_(0)  # The diagonal elements of the Gram matrix are always 1, so they are excluded.
        max_coherence = torch.max(coherence_matrix).item()
        mean_coherence = torch.mean(coherence_matrix).item()
        
        # The condition number of the dictionary matrix, indicating its stability. A high number suggests it is ill-conditioned.
        U, S, V = torch.svd(dictionary)
        condition_number = (S.max() / S.min()).item() if S.min() > 1e-10 else float('inf')
        
        # Analysis of the singular values (spectrum) of the dictionary to understand its energy distribution.
        eigenvals = S ** 2
        # This calculates the proportion of variance explained by the top 10 singular values.
        spectral_ratio = (eigenvals[:10].sum() / eigenvals.sum()).item()
        
        return {
            'n_features': n_features, 'n_atoms': n_atoms, 'overcompleteness_ratio': n_atoms / n_features,
            'max_coherence': max_coherence, 'mean_coherence': mean_coherence, 'condition_number': condition_number,
            'spectral_concentration': spectral_ratio, 'singular_values': S.cpu().numpy()
        }
    
    @staticmethod
    def visualize_dictionary(dictionary, save_path=None):
        """Generates a set of plots to visualise the properties of the learned dictionary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: A heatmap showing the values of the first 50 dictionary atoms.
        dict_np = dictionary.cpu().numpy()
        im1 = axes[0,0].imshow(dict_np[:, :min(50, dict_np.shape[1])], aspect='auto', cmap='RdBu_r')
        axes[0,0].set_title('Dictionary Atoms (Basis Vectors)')
        axes[0,0].set_xlabel('Atom Index')
        axes[0,0].set_ylabel('Feature Dimension')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Plot 2: A semi-log plot of the singular values of the dictionary matrix.
        U, S, V = torch.svd(dictionary)
        axes[0,1].semilogy(S.cpu().numpy(), 'b-o', markersize=3)
        axes[0,1].set_title('Singular Value Distribution')
        axes[0,1].set_xlabel('Index')
        axes[0,1].set_ylabel('Singular Value (Logarithmic Scale)')
        axes[0,1].grid(True)
        
        # Plot 3: A histogram showing the distribution of the mutual coherence values.
        normalized_dict = F.normalize(dictionary, dim=0)
        coherence_matrix = torch.abs(normalized_dict.T @ normalized_dict)
        coherence_matrix.fill_diagonal_(0)
        coherence_values = coherence_matrix[torch.triu_indices(coherence_matrix.size(0), coherence_matrix.size(1), offset=1)]
        axes[1,0].hist(coherence_values.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Mutual Coherence Distribution')
        axes[1,0].set_xlabel('Coherence Value')
        axes[1,0].set_ylabel('Frequency')
        
        # Plot 4: A plot of the cumulative explained variance as a function of the number of singular values.
        eigenvals = S ** 2
        cumulative_ratio = torch.cumsum(eigenvals, dim=0) / eigenvals.sum()
        axes[1,1].plot(cumulative_ratio.cpu().numpy(), 'g-', linewidth=2)
        axes[1,1].axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
        axes[1,1].axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
        axes[1,1].set_title('Cumulative Explained Variance')
        axes[1,1].set_xlabel('Number of Principal Components')
        axes[1,1].set_ylabel('Cumulative Ratio of Variance')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# =================================================================================================
# 5. Main Execution Function
# =================================================================================================
def main():
    """
    This function orchestrates the entire process, from loading the model and data to extracting
    activations, training the SAE, and analysing the results.
    """
    print("=== Basis Analysis of Ruri-V2 using a Sparse Autoencoder ===\n")
    
    # Step 1: Load the pre-trained Transformer model and its corresponding tokenizer.
    print("1. Loading the model and tokenizer...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"✅ Model loaded successfully. Hidden size: {model.config.hidden_size} dimensions.")
    
    # Step 2: Load the specified dataset for extracting activations.
    print("\n2. Preparing the dataset...")
    # The entire 'train' split of the JSTS dataset will be used.
    dataset = load_dataset(DATASET_PATH, DATASET_SUBSET, split="train", trust_remote_code=True)
    print(f"✅ Data preparation complete. Total samples: {len(dataset)}.")
    
    # Step 3: Extract internal activations. This is performed in batches to avoid overwhelming GPU memory.
    print("\n3. Extracting internal activations from the Transformer in batches...")
    target_layer = f"encoder.layer.{model.config.num_hidden_layers//2}"
    extractor = ActivationExtractor(model, [target_layer])
    
    all_sentence_representations = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), ACTIVATION_EXTRACTION_BATCH_SIZE), desc="Extracting Activations"):
            batch_texts = dataset[i:i+ACTIVATION_EXTRACTION_BATCH_SIZE]['sentence1']
            
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            # All input tensors are explicitly moved to the designated compute device (e.g., the GPU).
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            activations = extractor.extract(inputs)
            layer_output = activations[target_layer].to(DEVICE)
            
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_output = layer_output * attention_mask
            sentence_representations = masked_output.sum(dim=1) / attention_mask.sum(dim=1)
            
            # To conserve VRAM, the extracted activation tensors are moved back to the CPU for aggregation.
            all_sentence_representations.append(sentence_representations.cpu())

    # After processing all batches, the list of activation tensors is concatenated into a single large tensor.
    all_sentence_representations = torch.cat(all_sentence_representations, dim=0)
    print(f"✅ Activation extraction complete. Final tensor shape: {all_sentence_representations.shape}.")

    # Step 4: Train the Sparse Autoencoder on the extracted activations.
    print("\n4. Training the Sparse Autoencoder in batches...")
    input_dim = all_sentence_representations.shape[1]
    hidden_dim = input_dim * 4
    
    sae = SparseAutoEncoder(input_dim, hidden_dim, sparsity_lambda=0.01).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # The full dataset of activations is wrapped in a PyTorch DataLoader to facilitate efficient mini-batch training.
    train_dataset = TensorDataset(all_sentence_representations)
    train_loader = DataLoader(train_dataset, batch_size=SAE_TRAINING_BATCH_SIZE, shuffle=True)
    
    sae.train()
    losses = []
    sparsity_metrics_log = []
    
    pbar = tqdm(range(SAE_EPOCHS), desc="SAE Training Epochs")
    for epoch in pbar:
        epoch_losses = []
        for batch in train_loader:
            # Each mini-batch of data is moved to the GPU before being passed to the SAE model.
            data_batch = batch[0].to(DEVICE)
            
            optimizer.zero_grad()
            
            x_hat, z = sae(data_batch)
            total_loss, recon_loss, sparse_loss = sae.compute_loss(data_batch, x_hat, z)
            
            total_loss.backward()
            optimizer.step()
            sae._normalize_decoder_weights()
            
            epoch_losses.append({
                'total_loss': total_loss.item(),
                'reconstruction_loss': recon_loss.item(),
                'sparsity_loss': sparse_loss.item()
            })

        # The average loss for the entire epoch is calculated and recorded for later visualisation.
        avg_loss = pd.DataFrame(epoch_losses).mean().to_dict()
        losses.append({'epoch': epoch, **avg_loss})
        pbar.set_postfix(loss=avg_loss['total_loss'])

        # Periodically, and at the end of training, evaluate and print the sparsity metrics on the full dataset.
        if epoch % 20 == 0 or epoch == SAE_EPOCHS - 1:
            with torch.no_grad():
                # For evaluation, pass the entire dataset through the model to get representative metrics.
                _, z_full = sae(all_sentence_representations.to(DEVICE))
                metrics = sae.compute_sparsity_metrics(z_full)
                sparsity_metrics_log.append({'epoch': epoch, **metrics})
                print(f"\nEpoch {epoch}: Average Loss={avg_loss['total_loss']:.4f}, L0 Norm={metrics['l0_norm']:.2f}, Activation Rate={metrics['activation_rate']:.3f}")

    print("✅ SAE training complete.")
    
    # Step 5: Analyse the properties of the dictionary learned by the SAE.
    print("\n5. Analysing the learned dictionary (bases)...")
    sae.eval()
    dictionary = sae.get_dictionary()
    
    # Perform the quantitative analysis of the dictionary's mathematical characteristics.
    dict_properties = BasisAnalyzer.analyze_dictionary_properties(dictionary)
    
    print("\n=== Mathematical Properties of the Dictionary ===")
    print(f"Input dimensions: {dict_properties['n_features']}")
    print(f"Number of dictionary atoms: {dict_properties['n_atoms']}")
    print(f"Overcompleteness ratio: {dict_properties['overcompleteness_ratio']:.2f}")
    print(f"Maximum coherence: {dict_properties['max_coherence']:.4f}")
    print(f"Mean coherence: {dict_properties['mean_coherence']:.4f}")
    print(f"Condition number: {dict_properties['condition_number']:.2f}")
    print(f"Spectral concentration (top 10): {dict_properties['spectral_concentration']:.3f}")
    
    # Step 6: Generate and display visualisations of the training process and the final learned dictionary.
    print("\n6. Visualising the results...")
    
    # Visualise the dictionary properties.
    BasisAnalyzer.visualize_dictionary(
        dictionary, save_path=Path(RESULTS_DIR) / "dictionary_analysis.png"
    )
    
    # Create plots showing the progression of the different loss components and sparsity metrics.
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    loss_df = pd.DataFrame(losses)
    plt.plot(loss_df['epoch'], loss_df['total_loss'], label='Total Loss')
    plt.plot(loss_df['epoch'], loss_df['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(loss_df['epoch'], loss_df['sparsity_loss'], label='Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SAE Training Loss Progression')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if sparsity_metrics_log:
        sparse_df = pd.DataFrame(sparsity_metrics_log)
        plt.plot(sparse_df['epoch'], sparse_df['l0_norm'], 'o-', label='L0 Norm')
        ax2 = plt.gca().twinx() # Create a second y-axis for the activation rate.
        ax2.plot(sparse_df['epoch'], sparse_df['activation_rate'], 's-', color='seagreen', label='Activation Rate')
        plt.gca().set_xlabel('Epoch')
        plt.gca().set_ylabel('L0 Norm')
        ax2.set_ylabel('Activation Rate', color='seagreen')
        plt.title('Sparsity Metrics Progression')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "training_progress.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Step 7: Final cleanup and saving of artifacts.
    print(f"\n✅ Analysis complete! All results and plots have been saved to the directory: {RESULTS_DIR}")
    
    # Remove the forward hooks to free up resources.
    extractor.cleanup()
    return {}

if __name__ == "__main__":
    main()
