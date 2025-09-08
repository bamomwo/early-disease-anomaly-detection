import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    confusion_matrix
)

# Training and Validation Loss plots
def plot_loss_curves(train_losses, val_losses, hidden_size, num_layers, figs_dir, window=5):
    """
    Plot and save high-quality training and validation loss curves with moving average smoothing and early stopping annotation.

    Args:
        train_losses (list or np.ndarray): Training loss values per epoch.
        val_losses (list or np.ndarray): Validation loss values per epoch.
        hidden_size (int): Hidden size of the model (for title).
        num_layers (int): Number of layers in the model (for title).
        figs_dir (str): Directory to save the plot.
        window (int, optional): Window size for moving average. Default is 5.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(figs_dir, exist_ok=True)

    # 1) Compute best epoch & early-stop epoch
    best_epoch = int(np.argmin(val_losses))
    early_stop_epoch = len(val_losses)

    # 2) Smooth losses via moving average
    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    train_ma = moving_average(train_losses, window)
    val_ma = moving_average(val_losses, window)

    # 3) Prepare epoch axes
    epochs = np.arange(1, len(train_losses) + 1)
    ma_epochs = np.arange(window, len(train_losses) + 1)

    # 4) Plot
    plt.figure(figsize=(8,5))

    # raw scatter
    plt.scatter(epochs, train_losses, alpha=0.3, s=15, label='Train Loss')
    plt.scatter(epochs, val_losses,   alpha=0.3, s=15, label='Val Loss')

    # smoothed lines
    plt.plot(ma_epochs, train_ma, linewidth=2, label=f'Train MA (w={window})')
    plt.plot(ma_epochs, val_ma,   linewidth=2, label=f'Val MA   (w={window})')

    # highlight best epoch
    plt.axvline(best_epoch+1, color='k', linestyle='--', label='Best Epoch')

    # shade early-stop window
    plt.axvspan(best_epoch+1, early_stop_epoch, color='gray', alpha=0.2,
                label='Early-stop window')

    # annotate final values
    txt = (
        f"Final Train: {train_losses[-1]:.3f}\n"
        f"Final Val:   {val_losses[-1]:.3f}"
    )
    plt.gca().text(
        0.65, 0.75, txt,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss (hs={hidden_size}, nl={num_layers})')
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()

    # 5) Save high-res for publication
    plt.savefig(os.path.join(figs_dir, 'loss_curve_pub.png'), dpi=300)
    plt.close()

# Get Sequence Labels for evaluation
def get_sequence_labels(test_loader, participant_id, split='test'):
    """
    For each sequence, assign a binary label:
    0 = stress-free (all time steps in sequence are stress-free)
    1 = stressed (any time step in sequence is stressed)
    
    Args:
        test_loader: DataLoader containing test sequences
        participant_id: Single participant ID (str) or list of participant IDs
        split: Data split ('train', 'val', 'test')
    
    Returns:
        np.ndarray: Binary labels for each sequence
    """
    # Handle both single participant and multiple participants
    if isinstance(participant_id, str):
        participant_ids = [participant_id]
    else:
        participant_ids = participant_id
    
    all_sequence_labels = []
    
    # Try to derive base data path from the provided loader (supports new structure)
    base_data_path = None
    try:
        base_data_path = getattr(getattr(test_loader, 'dataset', None), 'data_path', None)
    except Exception:
        base_data_path = None

    for pid in participant_ids:
        # Prefer the loader's data_path (e.g., data/normalized_stratified)
        norm_path = None
        if base_data_path is not None:
            # 1) New structure: use filled data which contains stress_level
            filled_candidate = os.path.join(str(base_data_path), split, 'filled', f'{pid}_{split}_filled.csv')
            if os.path.exists(filled_candidate):
                norm_path = filled_candidate
            else:
                # 2) Try new structure but with norm subdir if present
                norm_candidate = os.path.join(str(base_data_path), split, 'norm', f'{pid}_{split}_norm.csv')
                if os.path.exists(norm_candidate):
                    norm_path = norm_candidate
        # 3) Legacy default location
        if norm_path is None:
            legacy_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
            if os.path.exists(legacy_path):
                norm_path = legacy_path

        if norm_path is None or not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find label source for participant {pid}. Tried: "
                                    f"{os.path.join(str(base_data_path), split, 'filled', f'{pid}_{split}_filled.csv') if base_data_path else 'N/A'} and "
                                    f"{os.path.join(str(base_data_path), split, 'norm', f'{pid}_{split}_norm.csv') if base_data_path else 'N/A'} and "
                                    f"data/normalized/{split}/norm/{pid}_{split}_norm.csv")

        df_pd = pd.read_csv(norm_path)
        if 'stress_level' not in df_pd.columns:
            raise ValueError("stress_level column not found in test data.")
        stress_labels = df_pd['stress_level'].values
        # Get sequence length and step size from loader
        seq_len = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n_windows = len(stress_labels)
        sequence_labels = []
        # For each sequence, assign label 1 if any time step is stressed, else 0
        for start_idx in range(0, n_windows - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            seq_labels = stress_labels[start_idx:end_idx]
            label = 1 if np.any(seq_labels > 0) else 0
            sequence_labels.append(label)
        all_sequence_labels.extend(sequence_labels)
    
    return np.array(all_sequence_labels)

# Reconstruction error distribution during validation. 
# def plot_recon_error_distribution(labels, errors, out_dir, bins=50):
#     """
#     Plot and save the reconstruction‐error distributions for normal vs. stress windows.

#     Args:
#         errors (np.ndarray): 1D array of reconstruction errors (one per window).
#         labels (np.ndarray): 1D binary array of same length (0=normal, 1=stress).
#         out_dir (str): Directory where figures will be saved.
#         bins (int): Number of histogram bins.
#     """
#     # Split errors
#     normal_err = errors[labels == 0]
#     stress_err = errors[labels == 1]

#     # Summary stats
#     mu_n, sigma_n = normal_err.mean(), normal_err.std()
#     mu_s, sigma_s = stress_err.mean(), stress_err.std()

#     # 95th percentile threshold (normal)
#     thresh = np.percentile(normal_err, 95)

#     # Create plot
#     plt.figure(figsize=(8, 5))
#     plt.hist(normal_err, bins=bins, density=True, alpha=0.5,
#              label=f'Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})')
#     plt.hist(stress_err, bins=bins, density=True, alpha=0.5,
#              label=f'Anomalous (μ={mu_s:.2f}, σ={sigma_s:.2f})')
#     plt.axvline(thresh, color='k', linestyle='--',
#                 label=f'95th %ile normal = {thresh:.2f}')

#     plt.xlabel('Reconstruction Error')
#     plt.ylabel('Density')
#     plt.title('Reconstruction Error Distribution')
#     plt.legend(fontsize='small')
#     plt.tight_layout()

#     # Save high‐quality figures
#     os.makedirs(out_dir, exist_ok=True)
#     pdf_path = os.path.join(out_dir, 'recon_error_dist.pdf')
#     png_path = os.path.join(out_dir, 'recon_error_dist.png')
#     plt.savefig(pdf_path)         # vector for publication
#     plt.savefig(png_path, dpi=300)  # high-res raster
#     plt.close()


def plot_recon_error_distribution(labels, errors, out_dir, bins=50):
    """
    Plot and save the reconstruction‐error distributions for normal vs. stress windows.

    Args:
        errors (np.ndarray): 1D array of reconstruction errors (one per window).
        labels (np.ndarray): 1D binary array of same length (0=normal, 1=stress).
        out_dir (str): Directory where figures will be saved.
        bins (int): Number of histogram bins.
    """
    # Ensure finite values
    finite_mask = np.isfinite(errors) & np.isfinite(labels)
    errors = errors[finite_mask]
    labels = labels[finite_mask]

    # Split and clean each class separately
    normal_err = errors[labels == 0]
    stress_err = errors[labels == 1]

    normal_err = normal_err[np.isfinite(normal_err)]
    stress_err = stress_err[np.isfinite(stress_err)]

    if len(normal_err) == 0 or len(stress_err) == 0:
        print("WARNING: One of the classes has no valid error values to plot.")
        return

    # Summary stats
    mu_n, sigma_n = normal_err.mean(), normal_err.std()
    mu_s, sigma_s = stress_err.mean(), stress_err.std()

    # 95th percentile threshold (normal)
    thresh = np.percentile(normal_err, 95)

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.hist(normal_err, bins=bins, density=True, alpha=0.5,
             label=f'Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})')
    plt.hist(stress_err, bins=bins, density=True, alpha=0.5,
             label=f'Anomalous (μ={mu_s:.2f}, σ={sigma_s:.2f})')
    plt.axvline(thresh, color='k', linestyle='--',
                label=f'95th %ile normal = {thresh:.2f}')

    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend(fontsize='small')
    plt.tight_layout()

    # Save figures
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'recon_error_dist.pdf'))         # vector
    plt.savefig(os.path.join(out_dir, 'recon_error_dist.png'), dpi=300)  # raster
    plt.close()


# ROC and PR curves with AOC value
def plot_roc_pr_curves(labels, errors, out_dir):
    """
    Plot and save ROC and Precision-Recall curves.

    Args:
        labels (np.ndarray): 1D binary array of true labels (0 or 1).
        errors (np.ndarray): 1D array of reconstruction errors.
        out_dir (str): Directory where figures will be saved.
    """
    # ROC curve
    auc_roc = roc_auc_score(labels, errors)
    fpr, tpr, _ = roc_curve(labels, errors)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'roc_curve.pdf'))
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, errors)
    auc_pr = auc(recall, precision)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PR (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.pdf'))
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=300)
    plt.close()

def get_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Find the optimal threshold for anomaly detection.
    
    If positive samples are present, it maximizes the F1-score.
    If only normal samples are present, it uses the 99th percentile of errors.
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_score: Reconstruction errors or anomaly scores
        
    Returns:
        The optimal threshold.
    """
    # Check if there are any positive samples
    if np.sum(y_true) == 0:
        # No positive samples, use a percentile of the reconstruction errors
        best_threshold = np.percentile(y_score, 99)
        print(f"Warning: No positive samples in validation set. Using 99th percentile threshold: {best_threshold:.4f}")
        return best_threshold

    # If positive samples exist, find the threshold that maximizes F1-score
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Adjust for length mismatch between precision/recall and thresholds
    if len(precision) > len(thresholds):
        precision = precision[:-1]
        recall = recall[:-1]
        
    # Calculate F1-score and handle division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find the threshold that maximizes F1 score
    if len(f1_scores) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]
    else:
        # Fallback if no valid F1 scores can be calculated
        best_threshold = np.percentile(y_score, 99)
        print(f"Warning: Could not calculate F1 scores. Using 99th percentile threshold: {best_threshold:.4f}")

    return best_threshold

# Confuction Matrix with F1-Score
def plot_confusion_matrix(labels, errors, out_dir, threshold=None):
    """
    Compute best-threshold via F1, plot and save confusion matrix at that threshold.

    Args:
        labels (np.ndarray): 1D binary array of true labels (0 or 1).
        errors (np.ndarray): 1D array of reconstruction errors.
        out_dir (str): Directory where figures will be saved.
        threshold (float, optional): If provided, use this threshold. 
                                     Otherwise, calculate the best threshold.
    """
    if threshold is None:
        threshold = get_optimal_threshold(labels, errors)

    # Compute confusion matrix at best threshold
    preds = (errors > threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Normal', 'Anomalous'],
                yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (thr={threshold:.2f}, F1={f1:.2f})')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'))
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

# Latent Space visualisation with PCA
def plot_latent_space_viz(latents, labels, out_dir):
    """
    Plot a 2D PCA of latent vectors, colored by label.

    Args:
        latents (np.ndarray): shape (num_seq, latent_dim)
        labels (np.ndarray): 1D array of sequence labels
        out_dir (str): directory to save plots
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(latents)

    plt.figure(figsize=(6, 6))
    for cls, name in [(0, 'Normal'), (1, 'Stress')]:
        idx = labels == cls
        plt.scatter(coords[idx, 0], coords[idx, 1], alpha=0.6, label=name)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Latent Space PCA')
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'latent_space.pdf'))
    plt.savefig(os.path.join(out_dir, 'latent_space.png'), dpi=300)
    plt.close()

# Reconstruction Examples
def plot_time_series_reconstructions(inputs, outputs, labels, out_dir, indices=None):
    """
    Plot example input vs. reconstructed time-series windows.

    Args:
        inputs (np.ndarray): shape (num_seq, seq_len, features)
        outputs (np.ndarray): same shape as inputs
        labels (np.ndarray): 1D array of sequence labels
        indices (list[int] or None): which sequences to plot; defaults to first 3
        out_dir (str): directory to save plots
    """
    num_seq = inputs.shape[0]
    if indices is None:
        indices = list(range(min(3, num_seq)))
    plt.figure(figsize=(10, 3 * len(indices)))
    for i, idx in enumerate(indices, 1):
        inp = inputs[idx]
        outp = outputs[idx]
        label = labels[idx]
        label_text = 'Normal' if label == 0 else 'Stress'

        inp_avg = inp.mean(axis=1)
        outp_avg = outp.mean(axis=1)

        ax = plt.subplot(len(indices), 1, i)
        ax.plot(inp_avg, label='Original')
        ax.plot(outp_avg, label='Reconstruction')
        ax.set_title(f'Example {i} (label={label_text})')
        if i == len(indices):
            ax.set_xlabel('Time Step')
        ax.set_ylabel('Avg Signal')
        if i == 1:
            ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'time_series_recon.pdf'))
    plt.savefig(os.path.join(out_dir, 'time_series_recon.png'), dpi=300)
    plt.close()


