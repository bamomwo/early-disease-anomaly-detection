#!/usr/bin/env python
import sys
import os
import json
import argparse
import numpy as np
import torch
from typing import List, Optional


# ── Project setup ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.lstm_ae import MaskedLSTMAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import evaluate, extract_latents
from src.data.physiological_loader import PhysiologicalDataLoader
from src.utils.helpers import (
    get_sequence_labels,
    plot_latent_space_viz,
    plot_recon_error_distribution,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    plot_time_series_reconstructions
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["personalized", "general"], 
                        default="personalized", help="Type of model to evaluate")
    parser.add_argument("--participant", default=None,
                        help="Single participant ID (required for personalized models)")
    parser.add_argument("--participants", nargs='+', default=None,
                        help="List of participant IDs (required for general models)")
    parser.add_argument("--model-path", default=None,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing the model (for general models)")
    parser.add_argument("--data-path", default="data/normalized")
    parser.add_argument("--figs-dir", default=None,
                        help="Directory to save evaluation figures")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Input size for the model (auto-detected if not provided)")
    args = parser.parse_args()

    # Validate arguments based on model type
    if args.model_type == "personalized":
        if args.participant is None:
            parser.error("--participant is required for personalized models")
        if args.participants is not None:
            print("Warning: --participants ignored for personalized models")
    else:  # general
        if args.participants is None:
            parser.error("--participants is required for general models")
        if args.participant is not None:
            print("Warning: --participant ignored for general models")

    # Resolve defaults based on model type and participant(s)
    if args.model_path is None:
        if args.model_type == "personalized":
            args.model_path = f"results/lstm_ae/pure/{args.participant}/final_model_{args.participant}.pth"
        else:  # general
            if args.model_dir is None:
                args.model_dir = "results/lstm_ae/general"
            args.model_path = f"{args.model_dir}/final_model.pth"
    
    if args.figs_dir is None:
        if args.model_type == "personalized":
            args.figs_dir = f"results/lstm_ae/pure/{args.participant}/figs"
        else:  # general
            if args.model_dir is None:
                args.model_dir = "results/lstm_ae/general"
            args.figs_dir = f"{args.model_dir}/figs"
    
    os.makedirs(args.figs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model configuration ──
    config_path = "config/lstm_config.json"
    
    try:
        with open(config_path, 'r') as f:
            model_config = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using best_config.json")
        with open("config/best_config.json") as f:
            model_config = json.load(f)

    # ── 1. Load trained model ──
    # Determine input size
    if args.input_size is None:
        # Try to get input size from config or determine from data
        input_size = model_config.get("input_size", 43)  # fallback to 43
    else:
        input_size = args.input_size
    
    model = MaskedLSTMAutoencoder(
        input_size=input_size, 
        hidden_size=model_config["hidden_size"], 
        num_layers=model_config["num_layers"]
    )
    
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── 2. Prepare test data ──
    loader_factory = PhysiologicalDataLoader(args.data_path)
    
    if args.model_type == "personalized":
        # Personalized model evaluation
        _, _, test_loader = loader_factory.create_personalized_loaders(args.participant)
        participants_to_evaluate = [args.participant]
    else:
        # General model evaluation
        _, _, test_loader = loader_factory.create_general_loaders(args.participants)
        participants_to_evaluate = args.participants

    # ── 3. Run evaluation to get reconstructions ──
    loss_fn = MaskedMSELoss()
    avg_loss, all_inputs, all_outputs, all_masks = evaluate(model, test_loader, device, loss_fn)

    # 4. Concatenate everything once
    inputs  = np.concatenate(all_inputs,  axis=0)
    outputs = np.concatenate(all_outputs, axis=0)
    masks   = np.concatenate(all_masks,   axis=0)

    # masked‐MSE per sequence
    diff   = (inputs - outputs)**2 * masks
    errors = diff.sum(axis=(1,2)) / masks.sum(axis=(1,2))
    
    if args.model_type == "personalized":
        labels = get_sequence_labels(test_loader, args.participant, split="test")
    else:
        # For general models, get labels for all participants
        labels = get_sequence_labels(test_loader, participants_to_evaluate, split="test")

    # drop any inf or nan just in case
    valid   = np.isfinite(errors)
    errors  = errors[valid]
    labels  = labels[valid]  # make sure labels was defined just above

    # 5. Latent Space Visualization
    latents = extract_latents(model, test_loader, device)

    # now filter latents based on valid indices
    latents = latents[valid]

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

    # 5.4 Example Time-Series Reconstructions
    plot_time_series_reconstructions(
        inputs, outputs, labels, out_dir=args.figs_dir
    )

    #5.5 Latent-Space Visualization
    plot_latent_space_viz(
        latents, labels, out_dir=args.figs_dir
    )

    print(f"All evaluation figures saved to {args.figs_dir}")
 

if __name__ == "__main__":
    main()
