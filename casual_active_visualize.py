import torch
import copy
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModel, AutoTokenizer

# --- Configuration: Please modify these settings to match your environment ---
MODEL_NAME = "cl-nagoya/ruri-base-v2"
RESULTS_DIR = "/app/results"  # The directory configured in hybrid_pruning.py
TARGET_SPARSITY = 0.9         # The target sparsity for the pruned model to be visualised (e.g., 90%)
CIRCUIT_RETENTION_RATIO = 0.3 # The ratio configured in hybrid_pruning.py
# --- End of configuration ---

# The specific layer and head to visualise.
LAYER_TO_VISUALISE = 3
HEAD_TO_VISUALISE = 5

# Set up file paths.
sane_model_name = MODEL_NAME.replace('/', '_')
SUMMARY_CSV_PATH = Path(RESULTS_DIR) / f"causal_scores_{sane_model_name}.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hybrid_prune_model(model: torch.nn.Module, summary_df: pd.DataFrame, target_sparsity: float, circuit_retention_ratio: float):
    """
    Re-implements the core logic from hybrid_pruning.py to return a pruned model.
    """
    # This function is identical to 'hybrid_prune_model' from the previous script.
    # It is included here so that this script can generate the pruned model independently.
    score_threshold = summary_df['causal_score'].quantile(1.0 - circuit_retention_ratio)
    circuit_neurons = summary_df[summary_df['causal_score'] >= score_threshold]
    
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

            w_o_name = f"encoder.layer.{layer}.attention.output.dense.weight"
            if w_o_name in protection_mask:
                protection_mask[w_o_name][:, neuron_idx] = False
            
            w_v_name = f"encoder.layer.{layer}.attention.self.value.weight"
            b_v_name = f"encoder.layer.{layer}.attention.self.value.bias"
            if w_v_name in protection_mask:
                protection_mask[w_v_name][neuron_idx, :] = False
            if b_v_name in protection_mask:
                protection_mask[b_v_name][neuron_idx] = False

            for proj_type in ['query', 'key']:
                w_proj_name = f"encoder.layer.{layer}.attention.self.{proj_type}.weight"
                b_proj_name = f"encoder.layer.{layer}.attention.self.{proj_type}.bias"
                if w_proj_name in protection_mask:
                    protection_mask[w_proj_name][head_start_idx:head_end_idx, :] = False
                if b_proj_name in protection_mask:
                    protection_mask[b_proj_name][head_start_idx:head_end_idx] = False
        
        elif n_type == 'FFN':
            dim = neuron['dim']
            w_in_name = f"encoder.layer.{layer}.intermediate.dense.weight"
            b_in_name = f"encoder.layer.{layer}.intermediate.dense.bias"
            if w_in_name in protection_mask:
                protection_mask[w_in_name][dim, :] = False
            if b_in_name in protection_mask:
                protection_mask[b_in_name][dim] = False
                
            w_out_name = f"encoder.layer.{layer}.output.dense.weight"
            if w_out_name in protection_mask:
                protection_mask[w_out_name][:, dim] = False

    prunable_params = []
    total_prunable_count = 0
    for name, param in model.named_parameters():
        if name in protection_mask:
            prunable_indices = protection_mask[name]
            if prunable_indices.any():
                prunable_params.append(param.data[prunable_indices])
            total_prunable_count += prunable_indices.sum().item()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_to_prune_total = int(total_params * target_sparsity)
    num_params_to_prune_from_non_circuit = min(num_params_to_prune_total, total_prunable_count)

    if num_params_to_prune_from_non_circuit <= 0:
        return model

    all_prunable_values = torch.cat([p.view(-1).abs() for p in prunable_params])
    k = min(max(1, num_params_to_prune_from_non_circuit), all_prunable_values.numel())
    threshold = torch.kthvalue(all_prunable_values, k=k).values

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in protection_mask:
                mask = protection_mask[name]
                if mask.any():
                    prunable_data = param.data[mask]
                    keep_mask_prunable = prunable_data.abs() > threshold
                    param.data[mask] = prunable_data * keep_mask_prunable.float()
    return model


def visualise_weight_changes(original_model, pruned_model, layer_idx):
    """
    Function 1: Visualise the overall weight changes in the model.
    """
    print(f"\n--- 🎨 Function 1: Visualising overall weight changes for Layer {layer_idx} ---")
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    fig.suptitle(f'Weight Matrix Visualisation for Layer {layer_idx}\n(Left: Original, Right: Pruned)', fontsize=16)

    matrices_to_plot = {
        "Q (Query)": "attention.self.query.weight",
        "K (Key)": "attention.self.key.weight",
        "V (Value)": "attention.self.value.weight",
        "O (Output)": "attention.output.dense.weight",
    }
    
    with torch.no_grad():
        for i, (name, path) in enumerate(matrices_to_plot.items()):
            full_path = f"encoder.layer.{layer_idx}.{path}"
            
            orig_weights = original_model.state_dict()[full_path].cpu().numpy()
            pruned_weights = pruned_model.state_dict()[full_path].cpu().numpy()

            # Heatmap of the original weights.
            sns.heatmap(orig_weights, ax=axes[i, 0], cmap="viridis", cbar=False)
            axes[i, 0].set_title(f"Original {name}")
            axes[i, 0].set_xlabel("Input Dim")
            axes[i, 0].set_ylabel("Output Dim")
            
            # Heatmap of the pruned weights.
            # Use a different colour map to clearly show where values are zero.
            cmap = sns.color_palette("rocket", as_cmap=True)
            sns.heatmap(pruned_weights, ax=axes[i, 1], cmap=cmap, cbar=False)
            axes[i, 1].set_title(f"Pruned {name} (Sparsity: {TARGET_SPARSITY*100:.0f}%)")
            axes[i, 1].set_xlabel("Input Dim")
            axes[i, 1].set_ylabel("")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(RESULTS_DIR) / f"weight_changes_L{layer_idx}.png")
    print(f"✅ Visualisation saved to {RESULTS_DIR}/weight_changes_L{layer_idx}.png")
    plt.show()


def visualise_protection_for_input(pruned_model, causal_scores_df, layer_idx, head_idx):
    """
    Function 2: Visualise the protection mechanism for a specific input.
    """
    print(f"\n--- 🧠 Function 2: Visualising the protection mechanism for Layer {layer_idx} / Head {head_idx} ---")

    # 1. Extract and visualise the causal scores for the specified head.
    head_scores = causal_scores_df[
        (causal_scores_df['type'] == 'ATTN') &
        (causal_scores_df['layer'] == layer_idx) &
        (causal_scores_df['head'] == head_idx)
    ].sort_values(by='dim')

    if head_scores.empty:
        print(f"⚠️ Causal scores for Layer {layer_idx}, Head {head_idx} could not be found.")
        return

    # 2. Create a bar chart of the causal scores.
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :]) # Top panel (spans both columns).
    ax2 = fig.add_subplot(gs[1, 0]) # Bottom-left panel.
    ax3 = fig.add_subplot(gs[1, 1]) # Bottom-right panel.

    colours = plt.cm.plasma(head_scores['causal_score'] / head_scores['causal_score'].max())
    bars = ax1.bar(head_scores['dim'], head_scores['causal_score'], color=colours)
    ax1.set_title(f'Causal Importance Scores for Neurons in Layer {layer_idx}, Head {head_idx}', fontsize=14)
    ax1.set_xlabel('Neuron Dimension inside Head')
    ax1.set_ylabel('Causal Score')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the most important neuron.
    top_neuron_dim = head_scores.nlargest(1, 'causal_score')['dim'].iloc[0]
    ax1.axvline(top_neuron_dim, color='red', linestyle='--', linewidth=2, label=f'Top Neuron (Dim {top_neuron_dim})')
    ax1.legend()

    # 3. Visualise the protected weights.
    n_heads = pruned_model.config.num_attention_heads
    d_head = pruned_model.config.hidden_size // n_heads
    
    with torch.no_grad():
        # The portion of the W_V (Value) matrix corresponding to the head.
        w_v_path = f"encoder.layer.{layer_idx}.attention.self.value.weight"
        w_v_head = pruned_model.state_dict()[w_v_path][head_idx*d_head : (head_idx+1)*d_head, :].cpu().numpy()
        
        sns.heatmap(w_v_head, ax=ax2, cmap="rocket", cbar=True)
        ax2.set_title(f'Pruned W_V Matrix for Head {head_idx}')
        ax2.set_xlabel('Input Dimension (hidden_size)')
        ax2.set_ylabel('Output Dimension (d_head)')
        # Add a border to the row corresponding to the most important neuron.
        rect_v = plt.Rectangle((0, top_neuron_dim), w_v_head.shape[1], 1, fill=False, edgecolor='lime', lw=3)
        ax2.add_patch(rect_v)
        
        # The portion of the W_O (Output) matrix corresponding to the head.
        w_o_path = f"encoder.layer.{layer_idx}.attention.output.dense.weight"
        w_o_head = pruned_model.state_dict()[w_o_path][:, head_idx*d_head : (head_idx+1)*d_head].cpu().numpy()
        
        sns.heatmap(w_o_head.T, ax=ax3, cmap="rocket", cbar=True)
        ax3.set_title(f'Pruned W_O Matrix for Head {head_idx} (Transposed)')
        ax3.set_xlabel('Output Dimension (hidden_size)')
        ax3.set_ylabel('Input Dimension (d_head)')
        # Add a border to the column (which becomes a row after transposing) corresponding to the most important neuron.
        rect_o = plt.Rectangle((0, top_neuron_dim), w_o_head.shape[0], 1, fill=False, edgecolor='lime', lw=3)
        ax3.add_patch(rect_o)


    fig.suptitle(f'Visualising Protection Mechanism for a Causal Neuron\nInput: "A man is running." vs "A dog is running."', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / f"protection_L{layer_idx}_H{head_idx}.png")
    print(f"✅ Visualisation saved to {RESULTS_DIR}/protection_L{layer_idx}_H{head_idx}.png")
    plt.show()


if __name__ == "__main__":
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(f"Causal score file not found at: '{SUMMARY_CSV_PATH}'")
    
    print("1. Loading the model and causal scores...")
    original_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    causal_scores_df = pd.read_csv(SUMMARY_CSV_PATH)
    
    print(f"2. Pruning the model to a target sparsity of {TARGET_SPARSITY*100:.0f}%...")
    pruned_model = copy.deepcopy(original_model)
    pruned_model = hybrid_prune_model(pruned_model, causal_scores_df, TARGET_SPARSITY, CIRCUIT_RETENTION_RATIO)
    print("✅ Pruning complete.")

    # --- Calling the visualisation functions ---
    # Function 1: Visualise the overall weight changes in the model.
    visualise_weight_changes(original_model, pruned_model, layer_idx=LAYER_TO_VISUALISE)
    
    # Function 2: Visualise the protection mechanism for a specific input.
    # As an example, we will check whether neurons related to the ability to
    # capture the semantic difference between 'man' and 'dog' have been protected.
    visualise_protection_for_input(pruned_model, causal_scores_df, layer_idx=LAYER_TO_VISUALISE, head_idx=HEAD_TO_VISUALISE)
