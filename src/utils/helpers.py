# helpers.py
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# ──────────────────────────────────────────────────────────────────────────────
# Training & Validation loss plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, hidden_size, num_layers, figs_dir, window: int = 5):
    """
    Plot and save training & validation loss curves with moving-average smoothing.

    Args:
        train_losses (list/np.ndarray): Training loss per epoch.
        val_losses   (list/np.ndarray): Validation loss per epoch.
        hidden_size  (int): model hidden size, used in title.
        num_layers   (int): number of layers, used in title.
        figs_dir     (str): directory where the figure is saved.
        window       (int): moving-average window for smoothing.
    """
    os.makedirs(figs_dir, exist_ok=True)

    best_epoch = int(np.argmin(val_losses))
    early_stop_epoch = len(val_losses)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    train_ma = moving_average(train_losses, window)
    val_ma   = moving_average(val_losses,   window)

    epochs    = np.arange(1, len(train_losses) + 1)
    ma_epochs = np.arange(window, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    # raw
    plt.scatter(epochs, train_losses, alpha=0.3, s=15, label="Train Loss")
    plt.scatter(epochs, val_losses,   alpha=0.3, s=15, label="Val Loss")
    # smoothed
    plt.plot(ma_epochs, train_ma, linewidth=2, label=f"Train MA (w={window})")
    plt.plot(ma_epochs, val_ma,   linewidth=2, label=f"Val MA (w={window})")

    plt.axvline(best_epoch + 1, color="k", linestyle="--", label="Best Epoch")
    plt.axvspan(best_epoch + 1, early_stop_epoch, color="gray", alpha=0.2, label="Early-stop window")

    txt = f"Final Train: {train_losses[-1]:.3f}\nFinal Val:   {val_losses[-1]:.3f}"
    plt.gca().text(
        0.65, 0.75, txt, transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss (hs={hidden_size}, nl={num_layers})")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_curve_pub.png"), dpi=300)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Window labelers (sequence-level ground truth)
# ──────────────────────────────────────────────────────────────────────────────

def get_sequence_labels(test_loader, participant_id, split: str = "test"):
    """
    Label a sequence as 1 if ANY time step in the window is stressed; else 0.
    """
    participant_ids = [participant_id] if isinstance(participant_id, str) else participant_id
    all_sequence_labels = []

    for pid in participant_ids:
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
        df_pd = pd.read_csv(norm_path)
        if "stress_level" not in df_pd.columns:
            raise ValueError("stress_level column not found in data.")
        stress_labels = df_pd["stress_level"].values

        seq_len   = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n         = len(stress_labels)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx    = start_idx + seq_len
            seq_labels = stress_labels[start_idx:end_idx]
            label      = 1 if np.any(seq_labels > 0) else 0
            all_sequence_labels.append(label)

    return np.array(all_sequence_labels, dtype=int)


def get_sequence_labels_ratio(test_loader, participant_id, split: str = "test",
                              alpha: float = 0.5, stress_col: str = "stress_level"):
    """
    Label a sequence as 1 if the FRACTION of stressed rows >= alpha; else 0.
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

        stress    = df[stress_col].values
        seq_len   = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n         = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx    = start_idx + seq_len
            seq_labels = stress[start_idx:end_idx]
            ratio      = np.mean(seq_labels > 0)
            label      = 1 if ratio >= alpha else 0
            all_sequence_labels.append(label)

    return np.array(all_sequence_labels, dtype=int)


def _longest_run_ones(binary_array: np.ndarray) -> int:
    """Return length of the longest consecutive 1s in a 1D binary array."""
    best = run = 0
    for v in binary_array:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def get_sequence_labels_minrun(test_loader, participant_id, split: str = "test",
                               min_run: int = 3, stress_col: str = "stress_level"):
    """
    Label a sequence as 1 if it contains at least `min_run` consecutive stressed rows; else 0.
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

        stress    = df[stress_col].values
        seq_len   = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n         = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            window  = (stress[start_idx:end_idx] > 0).astype(int)
            label   = 1 if _longest_run_ones(window) >= min_run else 0
            all_sequence_labels.append(label)

    return np.array(all_sequence_labels, dtype=int)


def get_sequence_labels_combined(
    test_loader,
    participant_id,
    split: str = "test",
    min_run: int = 5,             # alpha: min consecutive stressed rows
    min_ratio: Optional[float] = 0.3,  # beta as ratio (if None, min_count used)
    min_count: Optional[int] = None,   # beta as absolute count (used if min_ratio is None)
    stress_col: str = "stress_level",
):
    """
    Combined rule:
      window = 1 if (longest_run >= min_run) OR (ratio >= min_ratio) OR (count >= min_count).
      Otherwise 0.
    """
    if min_ratio is None and min_count is None:
        min_ratio = 0.3  # sensible default

    participant_ids = [participant_id] if isinstance(participant_id, str) else participant_id
    all_sequence_labels = []

    for pid in participant_ids:
        norm_path = f"data/normalized/{split}/norm/{pid}_{split}_norm.csv"
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Could not find: {norm_path}")
        df = pd.read_csv(norm_path)
        if stress_col not in df.columns:
            raise ValueError(f"{stress_col} column not found in data: {norm_path}")

        stress    = df[stress_col].values
        seq_len   = test_loader.dataset.sequence_length
        step_size = test_loader.dataset.step_size
        n         = len(stress)

        for start_idx in range(0, n - seq_len + 1, step_size):
            end_idx = start_idx + seq_len
            window  = (stress[start_idx:end_idx] > 0).astype(int)

            # 1) run-length check
            if _longest_run_ones(window) >= min_run:
                all_sequence_labels.append(1)
                continue

            # 2) ratio / count check
            count = int(window.sum())
            if min_ratio is not None:
                label = 1 if (count / seq_len) >= min_ratio else 0
            else:
                label = 1 if count >= (min_count or 0) else 0

            all_sequence_labels.append(label)

    return np.array(all_sequence_labels, dtype=int)

# ──────────────────────────────────────────────────────────────────────────────
# Sequence scoring (from per-row reconstruction errors)
# ──────────────────────────────────────────────────────────────────────────────

def compute_row_errors_masked(inputs, outputs, masks):
    """
    Compute per-row masked MSE across features.
    inputs/outputs/masks: (N_seq, T, F) → returns (N_seq, T)
    """
    diff = (inputs - outputs) ** 2 * masks
    num  = diff.sum(axis=2)                     # (N, T)
    den  = np.clip(masks.sum(axis=2), 1, None)  # (N, T)
    return num / den


def smooth_row_errors_median3(row_err: np.ndarray) -> np.ndarray:
    """
    3-point median filter over time to suppress one-sample spikes.
    row_err: (N, T) → (N, T)
    """
    N, T = row_err.shape
    if T < 3:
        return row_err
    y = row_err.copy()
    y[:, 1:-1] = np.median(
        np.stack([row_err[:, :-2], row_err[:, 1:-1], row_err[:, 2:]], axis=0), axis=0
    )
    return y


def score_sequences_topk(row_err: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Top-k mean over time per sequence. Caps k to T if needed.
    row_err: (N, T) → (N,)
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    T = row_err.shape[1]
    k = min(k, T)
    idx  = np.argsort(row_err, axis=1)[:, -k:]      # indices of the largest k
    topk = np.take_along_axis(row_err, idx, axis=1) # values
    return topk.mean(axis=1)


def score_sequences_percentile(row_err: np.ndarray, q: int = 95) -> np.ndarray:
    """
    High percentile across time per sequence.
    row_err: (N, T) → (N,)
    """
    return np.percentile(row_err, q, axis=1)

def score_sequences_runmean_max(row_err: np.ndarray, w: int = 5) -> np.ndarray:
    """
    Rolling-mean-over-w, then take the max inside each sequence.
    Highlights sustained runs of high error (aligned with min_run-style labels).

    Args:
        row_err: (N, T) per-row errors
        w:       run length for the rolling mean (use your min_run, e.g., 5)

    Returns:
        (N,) sequence scores
    """
    N, T = row_err.shape
    # Degenerate cases: fall back to max over time
    if w <= 1 or w > T:
        return row_err.max(axis=1)

    # cumulative-sum trick for fast rolling mean
    cs = np.cumsum(np.pad(row_err, ((0, 0), (1, 0)), mode="constant"), axis=1)  # (N, T+1)
    roll_sum  = cs[:, w:] - cs[:, :-w]          # (N, T-w+1)
    roll_mean = roll_sum / float(w)             # (N, T-w+1)
    return roll_mean.max(axis=1)                # (N,)


def build_sequence_scores(
    inputs,
    outputs,
    masks,
    mode: str = "topk",
    k: int = 10,
    q: int = 95,
    smooth: bool = True
) -> np.ndarray:
    """
    Convert reconstructions to a sequence-level anomaly score.

    Modes:
      - 'topk'       → mean of top-k row errors (time)
                        (set k ≈ expected stressed run length)
      - 'percentile' → q-th percentile of row errors (time)
      - 'mean'       → mean over time
      - 'max'        → max over time
      - 'runmax'     → rolling-mean over window k, then take max (aligns with min_run labels)

    Args:
      inputs/outputs/masks: arrays of shape (N_seq, T, F)
      mode: one of {'topk','percentile','mean','max','runmax'}
      k:    for 'topk' (top-k) and 'runmax' (rolling window length)
      q:    percentile for 'percentile'
      smooth: apply a 3-point median filter across time before aggregation

    Returns:
      (N_seq,) sequence scores
    """
    row_err = compute_row_errors_masked(inputs, outputs, masks)  # (N, T)
    if smooth:
        row_err = smooth_row_errors_median3(row_err)

    N, T = row_err.shape

    if mode == "topk":
        kk = int(max(1, min(k, T)))          # clamp k to [1, T]
        return score_sequences_topk(row_err, k=kk)

    if mode == "percentile":
        qq = int(max(0, min(q, 100)))        # clamp q to [0, 100]
        return score_sequences_percentile(row_err, q=qq)

    if mode == "mean":
        return row_err.mean(axis=1)

    if mode == "max":
        return row_err.max(axis=1)

    if mode == "runmax":
        ww = int(max(1, min(k, T)))          # reuse k as run length
        return score_sequences_runmean_max(row_err, w=ww)

    raise ValueError("mode must be one of {'topk','percentile','mean','max','runmax'}")

# ──────────────────────────────────────────────────────────────────────────────
# Threshold selection (validation calibration)
# ──────────────────────────────────────────────────────────────────────────────

def _midpoint_thresholds(scores: np.ndarray) -> np.ndarray:
    """Sorted unique midpoints with sentinels −inf/+inf (tie-robust)."""
    s = np.sort(np.unique(scores))
    if s.size == 0:
        return np.array([np.inf])
    return np.concatenate(([-np.inf], (s[:-1] + s[1:]) / 2.0, [np.inf]))


def pick_threshold_by_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """
    Calibrate threshold from NEGATIVES only:
      pick t so that ~target_fpr of normals (labels==0) would be flagged.
    Works even when there are 0 positives in validation.
    """
    neg = scores[labels == 0]
    if neg.size == 0:
        return np.inf  # no negatives → predict none positive
    q = 1.0 - float(target_fpr)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(neg, q))


def pick_threshold_fbeta(labels: np.ndarray, scores: np.ndarray, beta: float = 1.0) -> Tuple[float, float, float]:
    """
    Threshold maximizing F_beta (beta<1 precision-leaning; >1 recall-leaning).
    Conservative tie-break (prefers higher t).
    Returns (thr, precision, recall) evaluated on the given labels/scores.
    """
    thrs = _midpoint_thresholds(scores)
    best_fb, best_thr = -1.0, thrs[-1]
    best_prec = best_rec = 0.0
    for t in thrs:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        fb   = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
        if fb > best_fb or (fb == best_fb and t > best_thr):
            best_fb, best_thr, best_prec, best_rec = fb, t, prec, rec
    return float(best_thr), float(best_prec), float(best_rec)


def pick_threshold_precision(labels: np.ndarray, scores: np.ndarray,
                             target_prec: float = 0.90,
                             target_fpr_fallback: float = 0.05) -> Tuple[float, float, float]:
    """
    1) Try to meet a precision target on validation.
    2) If there are NO positives or the precision target is unreachable,
       fall back to an FPR cap on negatives (quantile rule).
    3) If still degenerate (e.g., all scores equal), fall back to max-F1.

    Returns: (thr, precision, recall) measured on validation labels/scores.
    """
    P = int(np.sum(labels == 1))

    # Case: no positives → precision/recall for positives undefined. Use FPR cap.
    if P == 0:
        thr = pick_threshold_by_fpr(labels, scores, target_fpr=target_fpr_fallback)
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        return float(thr), float(prec), float(rec)

    thrs = _midpoint_thresholds(scores)
    found = False
    best_rec, best_thr, best_prec = -1.0, thrs[-1], 0.0
    for t in thrs:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        if prec >= target_prec and rec > best_rec:
            found, best_rec, best_thr, best_prec = True, rec, t, prec

    if found:
        return float(best_thr), float(best_prec), float(best_rec)

    # Couldn’t meet target precision → try F1 (or other F_beta) with conservative tie-break.
    thr, prec, rec = pick_threshold_fbeta(labels, scores, beta=1.0)

    # Still degenerate? (e.g., all scores tied) → FPR cap fallback.
    if not np.isfinite(thr) or (prec == 0.0 and rec == 0.0):
        thr = pick_threshold_by_fpr(labels, scores, target_fpr=target_fpr_fallback)
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
    return float(thr), float(prec), float(rec)

# ──────────────────────────────────────────────────────────────────────────────
# Simple post-processing to reduce isolated false alarms
# ──────────────────────────────────────────────────────────────────────────────

def apply_prediction_persistence(preds: np.ndarray, consecutive: int = 2) -> np.ndarray:
    """
    Keep only windows that are the START of a run of >= 'consecutive' positives.
    No wrap-around. Returns a 0/1 array with the same length as preds.

    If you want to mark the WHOLE run as positive instead, ask and we’ll provide a variant.
    """
    preds = np.asarray(preds).astype(int)
    if consecutive <= 1 or preds.size == 0:
        return preds

    keep = np.ones_like(preds, dtype=bool)
    for d in range(consecutive):
        shifted = np.roll(preds, -d).astype(bool)
        if d > 0:
            shifted[-d:] = False  # prevent wrap-around at the end
        keep &= shifted
    out = np.zeros_like(preds)
    out[keep] = 1
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Plots: distributions, ROC/PR, confusion matrices, latents, recon examples
# ──────────────────────────────────────────────────────────────────────────────

def plot_recon_error_distribution(labels: np.ndarray, errors: np.ndarray, out_dir: str, bins: int = 50):
    """
    Plot reconstruction-error distributions for normal vs. stress windows.
    """
    os.makedirs(out_dir, exist_ok=True)
    finite_mask = np.isfinite(errors) & np.isfinite(labels)
    errors = errors[finite_mask]
    labels = labels[finite_mask]

    normal_err = errors[labels == 0]
    stress_err = errors[labels == 1]

    normal_err = normal_err[np.isfinite(normal_err)]
    stress_err = stress_err[np.isfinite(stress_err)]

    if len(normal_err) == 0 or len(stress_err) == 0:
        print("WARNING: One of the classes has no valid error values to plot. Skipping.")
        return

    mu_n, sigma_n = normal_err.mean(), normal_err.std()
    mu_s, sigma_s = stress_err.mean(), stress_err.std()

    thresh = np.percentile(normal_err, 95)

    plt.figure(figsize=(8, 5))
    plt.hist(normal_err, bins=bins, density=True, alpha=0.5,
             label=f'Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})')
    plt.hist(stress_err, bins=bins, density=True, alpha=0.5,
             label=f'Anomalous (μ={mu_s:.2f}, σ={sigma_s:.2f})')
    plt.axvline(thresh, color='k', linestyle='--', label=f'95th %ile normal = {thresh:.2f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'recon_error_dist.pdf'))
    plt.savefig(os.path.join(out_dir, 'recon_error_dist.png'), dpi=300)
    plt.close()


def plot_roc_pr_curves(labels: np.ndarray, errors: np.ndarray, out_dir: str):
    """
    Plot ROC and PR curves. ROC is skipped if only one class present.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ROC (guard one-class)
    try:
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
        plt.savefig(os.path.join(out_dir, 'roc_curve.pdf'))
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    except ValueError:
        print("WARNING: ROC not defined (only one class present). Skipping ROC plot.")

    # PR (always defined)
    precision, recall, _ = precision_recall_curve(labels, errors)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.pdf'))
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(labels: np.ndarray, errors: np.ndarray, out_dir: str):
    """
    Compute best-threshold via F1 on the PROVIDED data, then plot CM.
    (Note: for honest reporting, prefer using a FIXED threshold chosen on validation.)
    """
    os.makedirs(out_dir, exist_ok=True)

    thresholds = np.unique(errors)
    best_f1 = 0.0
    best_thr = thresholds[0] if thresholds.size else np.inf
    best_prec = best_rec = 0.0

    for thr in thresholds:
        preds = (errors >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr, best_prec, best_rec = f1, thr, prec, rec

    preds = (errors >= best_thr).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix (thr={best_thr:.4f}, F1={best_f1:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'))
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_confusion_matrix_fixed_thr(labels: np.ndarray, scores: np.ndarray, thr: float,
                                    out_path_png: str, out_path_pdf: Optional[str] = None):
    """
    Plot confusion matrix at a FIXED threshold (e.g., chosen on validation).
    """
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Normal','Anomalous'], yticklabels=['Normal','Anomalous'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix (thr={thr:.4f} | P={prec:.2f}, R={rec:.2f}, F1={f1:.2f})')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path_png) or ".", exist_ok=True)
    plt.savefig(out_path_png, dpi=300)
    if out_path_pdf:
        plt.savefig(out_path_pdf)
    plt.close()


def plot_latent_space_viz(latents: np.ndarray, labels: np.ndarray, out_dir: str):
    """
    2D PCA of latent vectors, colored by label.
    """
    os.makedirs(out_dir, exist_ok=True)
    pca   = PCA(n_components=2)
    coords = pca.fit_transform(latents)

    plt.figure(figsize=(6, 6))
    for cls, name in [(0, 'Normal'), (1, 'Stress')]:
        idx = labels == cls
        if np.sum(idx) > 0:
            plt.scatter(coords[idx, 0], coords[idx, 1], alpha=0.6, label=name)
    plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.title('Latent Space PCA')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latent_space.pdf'))
    plt.savefig(os.path.join(out_dir, 'latent_space.png'), dpi=300)
    plt.close()


def plot_time_series_reconstructions(inputs: np.ndarray, outputs: np.ndarray,
                                     labels: np.ndarray, out_dir: str,
                                     indices: Optional[list] = None):
    """
    Plot example input vs reconstructed windows (averaged across features).
    """
    os.makedirs(out_dir, exist_ok=True)
    num_seq = inputs.shape[0]
    if indices is None:
        indices = list(range(min(3, num_seq)))

    plt.figure(figsize=(10, 3 * len(indices)))
    for i, idx in enumerate(indices, 1):
        inp   = inputs[idx]
        outp  = outputs[idx]
        label = labels[idx]
        label_text = 'Normal' if label == 0 else 'Stress'

        inp_avg  = inp.mean(axis=1)
        outp_avg = outp.mean(axis=1)

        ax = plt.subplot(len(indices), 1, i)
        ax.plot(inp_avg,  label='Original')
        ax.plot(outp_avg, label='Reconstruction')
        ax.set_title(f'Example {i} (label={label_text})')
        ax.set_ylabel('Avg Signal')
        if i == 1:
            ax.legend()
        if i == len(indices):
            ax.set_xlabel('Time Step')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'time_series_recon.pdf'))
    plt.savefig(os.path.join(out_dir, 'time_series_recon.png'), dpi=300)
    plt.close()
