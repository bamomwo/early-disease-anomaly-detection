import numpy as np
import matplotlib.pyplot as plt
import os

# Function to compute reconstruction error for group of features.
def compute_group_errors(inputs, outputs, feature_groups):
    """
    Computes MSE reconstruction error per feature group.
    Returns a dict with group-wise errors.
    """
    group_errors = {}

    for group_name, indices in feature_groups.items():
        input_slice = inputs[:, :, indices]   # shape: (N, seq_len, num_features)
        output_slice = outputs[:, :, indices] # same

        mse = ((input_slice - output_slice) ** 2).mean()  # scalar MSE
        group_errors[group_name] = mse

    return group_errors

def compute_sequence_losses(inputs, outputs):
    """
    Compute per-sequence losses for individual features.
    
    Args:
        inputs: numpy array of shape (num_sequences, seq_len, num_features)
        outputs: numpy array of shape (num_sequences, seq_len, num_features)
    
    Returns:
        dict: {feature_idx: [loss_seq1, loss_seq2, ...]}
    """
    num_sequences, seq_len, num_features = inputs.shape
    feature_losses = {}
    
    for feature_idx in range(num_features):
        feature_losses[feature_idx] = []
        
        for seq_idx in range(num_sequences):
            # Compute MSE for this feature in this sequence
            input_feature = inputs[seq_idx, :, feature_idx]
            output_feature = outputs[seq_idx, :, feature_idx]
            mse = np.mean((input_feature - output_feature) ** 2)
            feature_losses[feature_idx].append(mse)
    
    return feature_losses

def compute_group_losses_per_sequence(inputs, outputs, feature_groups):
    """
    Compute group-wise losses per sequence.
    
    Args:
        inputs: numpy array of shape (num_sequences, seq_len, num_features)
        outputs: numpy array of shape (num_sequences, seq_len, num_features)
        feature_groups: dict mapping group names to feature indices
    
    Returns:
        dict: {group_name: [loss_seq1, loss_seq2, ...]}
    """
    num_sequences = inputs.shape[0]
    group_losses = {}
    
    for group_name, indices in feature_groups.items():
        group_losses[group_name] = []
        
        for seq_idx in range(num_sequences):
            # Compute MSE for this group in this sequence
            input_group = inputs[seq_idx, :, indices]
            output_group = outputs[seq_idx, :, indices]
            mse = np.mean((input_group - output_group) ** 2)
            group_losses[group_name].append(mse)
    
    return group_losses

def aggregate_loss_analysis(inputs, outputs, feature_groups):
    """
    Aggregate all loss analysis data into a comprehensive structure.
    
    Args:
        inputs: numpy array of shape (num_sequences, seq_len, num_features)
        outputs: numpy array of shape (num_sequences, seq_len, num_features)
        feature_groups: dict mapping group names to feature indices
    
    Returns:
        dict: Comprehensive loss analysis data
    """
    # Compute all types of losses
    feature_losses = compute_sequence_losses(inputs, outputs)
    group_losses = compute_group_losses_per_sequence(inputs, outputs, feature_groups)
    
    # Compute overall sequence losses (all features combined)
    num_sequences = inputs.shape[0]
    overall_sequence_losses = []
    
    for seq_idx in range(num_sequences):
        input_seq = inputs[seq_idx]
        output_seq = outputs[seq_idx]
        mse = np.mean((input_seq - output_seq) ** 2)
        overall_sequence_losses.append(mse)
    
    # Compute statistics
    analysis = {
        'feature_losses': feature_losses,
        'group_losses': group_losses,
        'overall_sequence_losses': overall_sequence_losses,
        'num_sequences': num_sequences,
        'num_features': inputs.shape[2],
        'feature_groups': feature_groups,
        'statistics': {
            'overall_mean_loss': np.mean(overall_sequence_losses),
            'group_mean_losses': {group: np.mean(losses) for group, losses in group_losses.items()}
        }
    }
    
    return analysis

def plot_loss_analysis(analysis, save_dir="results/transformer_ae/general/checkpoints"):
    """
    Create comprehensive plots for loss analysis across sequences.
    
    Args:
        analysis: Dictionary containing loss analysis data from aggregate_loss_analysis()
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Overall sequence losses
    plt.figure(figsize=(12, 6))
    plt.plot(analysis['overall_sequence_losses'], 'b-', alpha=0.7, linewidth=1)
    plt.axhline(y=analysis['statistics']['overall_mean_loss'], color='r', linestyle='--', 
                label=f'Mean: {analysis["statistics"]["overall_mean_loss"]:.4f}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Overall Reconstruction Loss Across Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_sequence_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Group-wise losses across sequences
    num_groups = len(analysis['group_losses'])
    fig, axes = plt.subplots(num_groups, 1, figsize=(12, 4*num_groups))
    if num_groups == 1:
        axes = [axes]
    
    for i, (group_name, losses) in enumerate(analysis['group_losses'].items()):
        mean_loss = analysis['statistics']['group_mean_losses'][group_name]
        
        axes[i].plot(losses, alpha=0.7, linewidth=1, label=f'{group_name} Loss')
        axes[i].axhline(y=mean_loss, color='r', linestyle='--', 
                       label=f'Mean: {mean_loss:.4f}')
        axes[i].set_xlabel('Sequence Index')
        axes[i].set_ylabel('Reconstruction Loss (MSE)')
        axes[i].set_title(f'{group_name} Group Loss Across Sequences')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'group_losses_across_sequences.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot comparing group losses
    plt.figure(figsize=(10, 6))
    group_data = [losses for losses in analysis['group_losses'].values()]
    group_names = list(analysis['group_losses'].keys())
    
    bp = plt.boxplot(group_data, labels=group_names, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Distribution of Group Losses Across Sequences')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'group_losses_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of feature losses across sequences (sample first 20 sequences for clarity)
    max_sequences_to_plot = min(20, analysis['num_sequences'])
    feature_loss_matrix = np.zeros((max_sequences_to_plot, analysis['num_features']))
    
    for seq_idx in range(max_sequences_to_plot):
        for feature_idx in range(analysis['num_features']):
            feature_loss_matrix[seq_idx, feature_idx] = analysis['feature_losses'][feature_idx][seq_idx]
    
    plt.figure(figsize=(15, 8))
    im = plt.imshow(feature_loss_matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Reconstruction Loss (MSE)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Feature Index')
    plt.title(f'Feature-wise Loss Heatmap (First {max_sequences_to_plot} Sequences)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_loss_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation matrix between group losses
    group_loss_matrix = np.array([losses for losses in analysis['group_losses'].values()])
    correlation_matrix = np.corrcoef(group_loss_matrix)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xticks(range(len(group_names)), group_names, rotation=45)
    plt.yticks(range(len(group_names)), group_names)
    plt.title('Correlation Between Group Losses Across Sequences')
    
    # Add correlation values as text
    for i in range(len(group_names)):
        for j in range(len(group_names)):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'group_losses_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
