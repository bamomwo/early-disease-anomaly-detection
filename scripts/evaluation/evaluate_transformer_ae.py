#!/usr/bin/env python
import sys, os, json, argparse, numpy as np, torch

# ── Project setup ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.transformer_ae import TransformerAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import evaluate, extract_latents
from src.data.physiological_loader import PhysiologicalDataLoader

# ⬇️ you said you dropped these into helpers.py
from src.utils.helpers import (
    get_sequence_labels_combined,
    build_sequence_scores,              # NEW
    pick_threshold_precision,           # NEW
    pick_threshold_by_fpr,               # NEW
    apply_prediction_persistence,       # NEW
    plot_latent_space_viz,
    plot_recon_error_distribution,
    plot_roc_pr_curves,
)

# (Tiny utility) fixed-threshold confusion matrix plotter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt, seaborn as sns

def plot_confusion_matrix_fixed_thr(labels, scores, thr, out_path_png, out_path_pdf=None):
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Normal','Anomalous'], yticklabels=['Normal','Anomalous'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix (thr={thr:.4f} | P={prec:.2f}, R={rec:.2f}, F1={f1:.2f})')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png, dpi=300)
    if out_path_pdf:
        plt.savefig(out_path_pdf)
    plt.close()

SELECTED_FEATURES_PATH = "config/selected_features.json"

def get_input_size_from_selected_features():
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    return len(selected_features['features'])

def concat_batches(all_inputs, all_outputs, all_masks):
    X  = np.concatenate(all_inputs,  axis=0)
    Y  = np.concatenate(all_outputs, axis=0)
    M  = np.concatenate(all_masks,   axis=0)
    return X, Y, M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["pure", "general", "personalized"],
                        default="pure")
    parser.add_argument("--participant", default=None)
    parser.add_argument("--participants", nargs='+', default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--data-path", default="data/normalized")
    parser.add_argument("--figs-dir", default=None)
    parser.add_argument("--input-size", type=int, default=None)

    # Threshold/aggregation knobs
    parser.add_argument("--agg", choices=["topk","percentile","mean","max", "runmax"], default="topk")
    parser.add_argument("--alpha", type=float, default=0.30, help="label ratio for intuition (used to set k)")
    parser.add_argument("--k", type=int, default=None, help="top-k rows (if not set, k=ceil(alpha*T))")
    parser.add_argument("--q", type=int, default=95, help="percentile (if agg=percentile)")
    parser.add_argument("--smooth", action="store_true", help="median(3) smooth row errors before aggregation")
    parser.add_argument("--target-precision", type=float, default=0.90, help="precision constraint on validation")
    parser.add_argument("--target-fpr-fallback", type=float, default=0.05,
                        help="If val has no positives or precision target is unreachable, cap val FPR at this value")
    parser.add_argument("--persist", type=int, default=0, help="require N consecutive positive windows (0=off)")

    # Combined labeling rule knobs (keep same as you used for training/eval)
    parser.add_argument("--min-run", type=int, default=5)
    parser.add_argument("--min-ratio", type=float, default=0.30)

    args = parser.parse_args()

    # Validate args
    if args.model_type in ["pure", "personalized"]:
        if args.participant is None:
            parser.error("--participant is required for pure/personalized models")
        if args.participants is not None:
            print("Warning: --participants ignored for pure/personalized models")
    else:
        if args.participants is None:
            parser.error("--participants is required for general models")
        if args.participant is not None:
            print("Warning: --participant ignored for general models")

    # Paths
    if args.model_path is None:
        if args.model_type == "pure":
            args.model_path = f"results/transformer_ae/pure/{args.participant}/final_model_{args.participant}.pth"
        elif args.model_type == "personalized":
            args.model_path = f"results/transformer_ae/personalized/{args.participant}/final_model_{args.participant}.pth"
        else:
            if args.model_dir is None:
                args.model_dir = "results/transformer_ae/general"
            args.model_path = f"{args.model_dir}/final_model.pth"

    if args.figs_dir is None:
        if args.model_type == "pure":
            args.figs_dir = f"results/transformer_ae/pure/{args.participant}/figs"
        elif args.model_type == "personalized":
            args.figs_dir = f"results/transformer_ae/personalized/{args.participant}/figs"
        else:
            if args.model_dir is None:
                args.model_dir = "results/transformer_ae/general"
            args.figs_dir = f"{args.model_dir}/figs"

    os.makedirs(args.figs_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    config_path = "config/transformer_config.json"
    try:
        with open(config_path, 'r') as f:
            model_config = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found; using best_config.json")
        with open("config/best_config.json") as f:
            model_config = json.load(f)

    # Model
    input_size = args.input_size or get_input_size_from_selected_features()
    model = TransformerAutoencoder(
        input_size=input_size,
        model_dim=model_config["model_dim"],
        num_layers=model_config["num_layers"],
        nhead=model_config["nhead"],
        dropout=model_config["dropout"]
    )
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Data
    loader_factory = PhysiologicalDataLoader(args.data_path)
    if args.model_type in ["pure", "personalized"]:
        train_loader, val_loader, test_loader = loader_factory.create_personalized_loaders(args.participant)
        participants_to_evaluate = [args.participant]
    else:
        train_loader, val_loader, test_loader = loader_factory.create_general_loaders(args.participants)
        participants_to_evaluate = args.participants

    # Sequence length (for auto-k)
    T = test_loader.dataset.sequence_length
    k_auto = int(np.ceil(args.alpha * T))
    k_use = args.k if args.k is not None else max(1, k_auto)

    # ── Evaluate & score: VALIDATION ──
    loss_fn = MaskedMSELoss()

    # Run model on validation
    avg_loss_val, in_val, out_val, m_val = evaluate(model, val_loader, device, loss_fn)
    Xv, Yv, Mv = concat_batches(in_val, out_val, m_val)

    errors_val = build_sequence_scores(
        Xv, Yv, Mv,
        mode=args.agg,
        k=k_use,
        q=args.q,
        smooth=args.smooth
    )

    if args.model_type in ["pure", "personalized"]:
        labels_val = get_sequence_labels_combined(val_loader, args.participant, split="val",
                                                  min_run=args.min_run, min_ratio=args.min_ratio)
    else:
        labels_val = get_sequence_labels_combined(val_loader, participants_to_evaluate, split="val",
                                                  min_run=args.min_run, min_ratio=args.min_ratio)

    valid_val = np.isfinite(errors_val)
    errors_val = errors_val[valid_val]
    labels_val = labels_val[valid_val]


    # After you compute errors_val and labels_val
    val_P = int(labels_val.sum())
    val_N = int((labels_val == 0).sum())
    print(f"[VAL] positives={val_P}, negatives={val_N}")

    if val_P == 0:
        # stress-free validation → calibrate with an FPR cap
        t_star = pick_threshold_by_fpr(labels_val, errors_val, target_fpr=0.05)  # try 1–10%
        prec_v = np.nan; rec_v = np.nan  # not defined with P=0
        print(f"[VAL] no positives; using FPR-cap threshold. t*={t_star:.6f} (target FPR≈5%)")
    else:
        # If you sometimes have positives, you can still use your precision/F1 picker here
        from src.utils.helpers import pick_threshold_precision
        t_star, prec_v, rec_v = pick_threshold_precision(
            labels_val, errors_val,
            target_prec=args.target_precision,
            target_fpr_fallback=args.target_fpr_fallback
        )
        print(f"[VAL] chosen t*={t_star:.6f} | P={prec_v:.3f}, R={rec_v:.3f}")

    # ── Evaluate & score: TEST ──
    avg_loss_test, in_test, out_test, m_test = evaluate(model, test_loader, device, loss_fn)
    Xt, Yt, Mt = concat_batches(in_test, out_test, m_test)

    errors_test = build_sequence_scores(
        Xt, Yt, Mt,
        mode=args.agg,
        k=k_use,
        q=args.q,
        smooth=args.smooth
    )

    if args.model_type in ["pure", "personalized"]:
        labels_test = get_sequence_labels_combined(test_loader, args.participant, split="test",
                                                   min_run=args.min_run, min_ratio=args.min_ratio)
    else:
        labels_test = get_sequence_labels_combined(test_loader, participants_to_evaluate, split="test",
                                                   min_run=args.min_run, min_ratio=args.min_ratio)

    valid_test = np.isfinite(errors_test)
    errors_test = errors_test[valid_test]
    labels_test = labels_test[valid_test]

    # Apply fixed threshold to test
    preds_test = (errors_test >= t_star).astype(int)

    # Optional: persistence to reduce isolated FPs
    if args.persist and args.persist > 1:
        preds_test = apply_prediction_persistence(preds_test, consecutive=args.persist)

    # ── Figures ──
    # Threshold-free curves (AUC-PR, ROC) still computed from scores (unchanged by threshold)
    plot_recon_error_distribution(labels_test, errors_test, out_dir=args.figs_dir)
    plot_roc_pr_curves(labels_test, errors_test, out_dir=args.figs_dir)

    # Fixed-threshold confusion matrix at t*
    cm_png = os.path.join(args.figs_dir, "confusion_matrix_fixed_thr.png")
    cm_pdf = os.path.join(args.figs_dir, "confusion_matrix_fixed_thr.pdf")
    plot_confusion_matrix_fixed_thr(labels_test, errors_test, t_star, cm_png, cm_pdf)

    # Latent space (optional; filtered by valid indices)
    latents = extract_latents(model, test_loader, device)
    latents = latents[valid_test]
    plot_latent_space_viz(latents, labels_test, out_dir=args.figs_dir)

    # Save chosen threshold + knobs for reproducibility
    with open(os.path.join(args.figs_dir, "chosen_threshold.json"), "w") as f:
        json.dump({
            "threshold": float(t_star),
            "val_precision": float(prec_v),
            "val_recall": float(rec_v),
            "agg": args.agg,
            "k_used": int(k_use),
            "percentile_q": int(args.q),
            "smooth": bool(args.smooth),
            "persist": int(args.persist),
            "min_run": int(args.min_run),
            "min_ratio": float(args.min_ratio),
            "seq_len": int(T)
        }, f, indent=2)

    print(f"All evaluation figures saved to {args.figs_dir}")
    print("Note: AUC-PR is threshold-free; improvements there come from better scores (agg/smooth), not from t*.")

if __name__ == "__main__":
    main()
