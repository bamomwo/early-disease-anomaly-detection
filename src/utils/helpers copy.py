import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from typing import Tuple
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

# Method 1 ---- Get Sequence Labels for evaluation (A single anomalous row makes the entire sequence anomalous)
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
    
    for pid in participant_ids:
        # Path to the original normalized data (with stress_level column)
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
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


# Method Two ---- Get sequence labels for evaluation (Using ration of anomalous stress rows)
def get_sequence_labels_ratio(test_loader, participant_id, split='test', alpha=0.5, stress_col='stress_level'):
    """
    Label a sequence as positive if the fraction of stressed rows in the window >= alpha.
    Example: seq_len=10, alpha=0.3 -> at least 3 stressed rows to label the window as 1.

    Args:
        test_loader: DataLoader with dataset.sequence_length and dataset.step_size
        participant_id: str or list[str]
        split: 'train' | 'val' | 'test'
        alpha: float in [0,1], required stressed-row ratio
        stress_col: column name containing stress labels

    Returns:
        np.ndarray of 0/1 labels (one per sequence in order)
    """
    participant_ids = [participant_id] if isinstance(participant_id, str) else participant_id
    all_sequence_labels = []

    for pid in participant_ids:
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
        df = pd.read_csv(norm_path)
        if stress_col not in df.columns:
            raise ValueError(f"{stress_col} column not found in data: {norm_path}")

        stress = df[stress_col].values
        seq_len = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            seq_labels = stress[start_idx:end_idx]
            # binarize stress > 0 as stressed
            ratio = np.mean(seq_labels > 0)
            label = 1 if ratio >= alpha else 0
            all_sequence_labels.append(label)

    return np.array(all_sequence_labels)

# Method Three ---- Get sequence labels for evaluation (Using longest consecutive run in sequence)
def _longest_run_ones(binary_array: np.ndarray) -> int:
    """Return length of the longest consecutive 1s run in a 1D binary array."""
    best = run = 0
    for v in binary_array:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def get_sequence_labels_minrun(test_loader, participant_id, split='test', min_run=3, stress_col='stress_level'):
    """
    Label a sequence as positive if it contains at least `min_run` consecutive stressed rows.
    Example: min_run=3 -> need a streak of 3 stressed rows somewhere in the window.

    Args:
        test_loader: DataLoader with dataset.sequence_length and dataset.step_size
        participant_id: str or list[str]
        split: 'train' | 'val' | 'test'
        min_run: int >= 1, required consecutive stressed rows
        stress_col: column name containing stress labels

    Returns:
        np.ndarray of 0/1 labels (one per sequence in order)
    """
    participant_ids = [participant_id] if isinstance(participant_id, str) else participant_id
    all_sequence_labels = []

    for pid in participant_ids:
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
        df = pd.read_csv(norm_path)
        if stress_col not in df.columns:
            raise ValueError(f"{stress_col} column not found in data: {norm_path}")

        stress = df[stress_col].values
        seq_len = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            seq_labels = (stress[start_idx:end_idx] > 0).astype(int)
            label = 1 if _longest_run_ones(seq_labels) >= min_run else 0
            all_sequence_labels.append(label)

    return np.array(all_sequence_labels)


# Method Three ---- Get sequence labels for evaluation (Using longest consecutive run in sequence)

def get_sequence_labels_combined(
    test_loader,
    participant_id,
    split='test',
    # alpha: minimum consecutive stressed rows
    min_run=5,                      # <-- alpha
    # beta: set one of these (ratio or count). If both set, ratio is used.
    min_ratio=0.3,                 # e.g., 0.5 means ≥50% of rows stressed
    min_count=None,                 # e.g., 3 means ≥3 rows stressed
    stress_col='stress_level'
):
    """
    Combined labeling rule per window:
      Positive (1) if:
        - longest run of stressed rows >= min_run, OR
        - (ratio of stressed rows >= min_ratio) OR (count of stressed rows >= min_count)
      Otherwise Negative (0).

    Args:
        test_loader: DataLoader with dataset.sequence_length and dataset.step_size
        participant_id: str or list[str]
        split: 'train' | 'val' | 'test'
        min_run: int, alpha (run-length threshold)
        min_ratio: float in [0,1], beta as ratio (optional)
        min_count: int >= 0, beta as count (optional)
        stress_col: column containing stress labels (>0 means stressed)

    Returns:
        np.ndarray of 0/1 labels (one per sequence in order)
    """
    # Default beta if neither provided
    if min_ratio is None and min_count is None:
        min_ratio = 0.3  # sensible default for seq_len=10 (≈3 rows)

    participant_ids = [participant_id] if isinstance(participant_id, str) else participant_id
    all_sequence_labels = []

    for pid in participant_ids:
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
        df = pd.read_csv(norm_path)
        if stress_col not in df.columns:
            raise ValueError(f"{stress_col} column not found in data: {norm_path}")

        stress = df[stress_col].values
        seq_len = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            window = (stress[start_idx:end_idx] > 0).astype(int)

            # 1) Run-length check (alpha)
            longest_run = _longest_run_ones(window)
            if longest_run >= min_run:
                all_sequence_labels.append(1)
                continue

            # 2) Beta check (ratio OR count)
            count = int(window.sum())
            if min_ratio is not None:
                ratio = count / seq_len
                label = 1 if ratio >= min_ratio else 0
            else:
                label = 1 if count >= (min_count or 0) else 0

            all_sequence_labels.append(label)

    return np.array(all_sequence_labels)



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

# Confuction Matrix with F1-Score
def plot_confusion_matrix(labels, errors, out_dir):
    """
    Compute best-threshold via F1, plot and save confusion matrix at that threshold.

    Args:
        labels (np.ndarray): 1D binary array of true labels (0 or 1).
        errors (np.ndarray): 1D array of reconstruction errors.
        out_dir (str): Directory where figures will be saved.
    """
    # Determine best threshold by maximizing F1
    thresholds = np.unique(errors)
    best_f1 = 0
    best_thresh = thresholds[0]
    best_prec = best_rec = 0
    for thr in thresholds:
        preds = (errors > thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thr
            best_prec = prec
            best_rec = rec

    # Compute confusion matrix at best threshold
    preds = (errors > best_thresh).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Normal', 'Anomalous'],
                yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (thr={best_thresh:.2f}, F1={best_f1:.2f})')
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


def compute_row_errors_masked(inputs, outputs, masks):
    """
    inputs, outputs, masks: shape (N_seq, T, F)
    Returns per-row masked MSE: shape (N_seq, T)
    """
    diff = (inputs - outputs) ** 2 * masks
    num  = diff.sum(axis=2)                      # (N, T)
    den  = np.clip(masks.sum(axis=2), 1, None)   # (N, T)
    row_err = num / den
    return row_err

def smooth_row_errors_median3(row_err):
    """
    3-point median filter over time per sequence to suppress single-sample spikes.
    row_err: (N, T) -> (N, T)
    """
    N, T = row_err.shape
    if T < 3: 
        return row_err
    y = row_err.copy()
    y[:, 1:-1] = np.median(
        np.stack([row_err[:, :-2], row_err[:, 1:-1], row_err[:, 2:]], axis=0), axis=0
    )
    return y

def score_sequences_topk(row_err, k=10):
    """
    Top-k mean over time per sequence. row_err: (N, T) -> scores: (N,)
    Choose k ~ ceil(alpha * T). For T=32 and alpha=0.30, k≈10.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    # argsort ascending, take last k columns
    idx  = np.argsort(row_err, axis=1)[:, -k:]
    topk = np.take_along_axis(row_err, idx, axis=1)
    return topk.mean(axis=1)

def score_sequences_percentile(row_err, q=95):
    """
    High-percentile aggregator (q in [0,100]). row_err: (N, T) -> (N,)
    """
    return np.percentile(row_err, q, axis=1)

def build_sequence_scores(inputs, outputs, masks, mode='topk', k=10, q=95, smooth=True):
    """
    Convenience wrapper: returns sequence-level scores ready for thresholding.
    """
    row_err = compute_row_errors_masked(inputs, outputs, masks)
    if smooth:
        row_err = smooth_row_errors_median3(row_err)
    if mode == 'topk':
        return score_sequences_topk(row_err, k=k)
    elif mode == 'percentile':
        return score_sequences_percentile(row_err, q=q)
    elif mode == 'mean':
        return row_err.mean(axis=1)
    elif mode == 'max':
        return row_err.max(axis=1)
    else:
        raise ValueError("mode must be one of {'topk','percentile','mean','max'}")




def pick_threshold_precision(labels, scores, target_prec=0.90) -> Tuple[float, float, float]:
    """
    Choose threshold achieving at least target_prec with maximum recall (tie-breaker).
    Uses tie-robust midpoints + sentinels.
    Returns (thr, precision, recall).
    """
    s = np.sort(np.unique(scores))
    thrs = np.concatenate(([-np.inf], (s[:-1] + s[1:]) / 2.0, [np.inf]))
    best = (-1.0, thrs[0], 0.0, 0.0)  # (recall, thr, prec, rec)
    for t in thrs:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        if prec >= target_prec and rec > best[0]:
            best = (rec, t, prec, rec)
    # If nothing meets target_prec, fall back to max-F1
    if best[0] < 0:
        t, p, r = pick_threshold_fbeta(labels, scores, beta=1.0)
        return t, p, r
    _, t, p, r = best
    return t, p, r

def pick_threshold_fbeta(labels, scores, beta=1.0) -> Tuple[float, float, float]:
    """
    Threshold maximizing F_beta (beta<1: precision-leaning; >1: recall-leaning).
    Returns (thr, precision, recall).
    """
    s = np.sort(np.unique(scores))
    thrs = np.concatenate(([-np.inf], (s[:-1] + s[1:]) / 2.0, [np.inf]))
    best = ( -1.0, thrs[0], 0.0, 0.0)  # (Fbeta, thr, prec, rec)
    for t in thrs:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        fb   = (1+beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
        if fb > best[0]:
            best = (fb, t, prec, rec)
    _, t, p, r = best
    return t, p, r

def apply_prediction_persistence(preds, consecutive=2):
    """
    Require 'consecutive' positive windows to keep a positive.
    preds: (N,) 0/1 array in temporal order.
    Returns filtered preds of same shape.
    """
    if consecutive <= 1:
        return preds
    out = preds.copy()
    for _ in range(consecutive - 1):
        out = out & np.roll(preds, -1)  # AND with next window(s)
    # Optionally expand back to mark the whole short run as positive:
    # here we keep only the persistent ones; adjust if you want expansion.
    return out.astype(int)

def pick_threshold_by_fpr(labels, scores, target_fpr=0.05):
    """
    Calibrate threshold from negatives only: pick t so that ~target_fpr of normals would be flagged.
    Works even when there are 0 positives in validation.
    """
    neg = scores[labels == 0]
    if neg.size == 0:
        return np.inf  # no negatives either → predict none positive
    q = 1.0 - float(target_fpr)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(neg, q))