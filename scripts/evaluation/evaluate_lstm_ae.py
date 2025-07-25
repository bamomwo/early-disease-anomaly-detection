#!/usr/bin/env python
import sys
import os
import json
import argparse
import numpy as np
import torch

# ── Project setup ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.lstm_ae import MaskedLSTMAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import evaluate
from src.data.physiological_loader import PhysiologicalDataLoader
from src.utils.helpers import (
    get_sequence_labels,
    plot_recon_error_distribution,
    plot_roc_pr_curves,
    plot_confusion_matrix
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant",    default=None)
    parser.add_argument("--model-path",     default=None,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--data-path",      default="data/normalized")
    parser.add_argument("--figs-dir",       default=None,
                        help="Directory to save evaluation figures")
    args = parser.parse_args()

    # Resolve defaults based on participant
    if args.model_path is None:
        args.model_path = f"results/lstm_ae/pure/{args.participant}/final_model_{args.participant}.pth"
    if args.figs_dir is None:
        args.figs_dir = f"results/lstm_ae/pure/{args.participant}/figs"
    os.makedirs(args.figs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load best hyperparameters from config ──
    with open("results/lstm_ae/best_config.json") as f:
        best_config = json.load(f)

    # ── 1. Load trained model ──
    model = MaskedLSTMAutoencoder(input_size=43, hidden_size=best_config["hidden_size"], num_layers=best_config["num_layers"])
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── 2. Prepare test data ──
    loader_factory = PhysiologicalDataLoader(args.data_path)
    _, _, test_loader = loader_factory.create_personalized_loaders(args.participant)

    # ── 3. Run evaluation to get reconstructions ──
    loss_fn = MaskedMSELoss()
    avg_loss, all_inputs, all_outputs = evaluate(model, test_loader, device, loss_fn)

    # ── 4. Compute per-sequence errors & labels ──
    inputs  = np.concatenate(all_inputs,  axis=0)  # (num_seq, seq_len, features)
    outputs = np.concatenate(all_outputs, axis=0)
    errors  = np.mean((inputs - outputs) ** 2, axis=(1,2))
    labels  = get_sequence_labels(test_loader, args.participant, split="test")

    # ── 5. Generate evaluation figures ──

    # 5.1 Reconstruction Error Distribution
    plot_recon_error_distribution(
        labels, errors, out_dir=args.figs_dir
    )

    # 5.2 ROC & Precision-Recall Curves
    plot_roc_pr_curves(
        labels, errors, out_dir=args.figs_dir
    )

    # 5.3 Confusion Matrix
    plot_confusion_matrix(
        labels, errors, out_dir=args.figs_dir
    )


    # # 5.3 Example Time-Series Reconstructions
    # plot_time_series_reconstructions(
    #     model, test_loader, device, out_dir=args.figs_dir
    # )

    # # 5.4 Latent-Space Visualization
    # plot_latent_space_viz(
    #     model, test_loader, device, out_dir=args.figs_dir
    # )

    print(f"All evaluation figures saved to {args.figs_dir}")

if __name__ == "__main__":
    main()
