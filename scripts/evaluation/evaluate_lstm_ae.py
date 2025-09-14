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
    get_optimal_threshold,
    plot_latent_space_viz,
    plot_recon_error_distribution,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    plot_time_series_reconstructions
)

# ── Constants ──
SELECTED_FEATURES_PATH = "config/selected_features.json"

def get_input_size_from_selected_features():
    """Load selected features file and return the number of features."""
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    return len(selected_features['features'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["pure", "personalized", "general"], 
                        default="pure", help="Type of model to evaluate")
    parser.add_argument("--participant", default=None,
                        help="Single participant ID (required for pure and personalized models)")
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
    if args.model_type in ["pure", "personalized"]:
        if args.participant is None:
            parser.error(f"--participant is required for {args.model_type} models")
        if args.participants is not None:
            print(f"Warning: --participants ignored for {args.model_type} models")
    else:  # general
        if args.participants is None:
            parser.error("--participants is required for general models")
        if args.participant is not None:
            print("Warning: --participant ignored for general models")

    # Resolve defaults based on model type and participant(s)
    if args.model_path is None:
        if args.model_type == "pure":
            args.model_path = f"results/lstm_ae/pure/{args.participant}/final_model_{args.participant}.pth"
        elif args.model_type == "personalized":
            args.model_path = f"results/lstm_ae/personalized/{args.participant}/best_model.pth"
        else:  # general
            if args.model_dir is None:
                args.model_dir = "results/lstm_ae/general"
            args.model_path = f"{args.model_dir}/final_model.pth"
    
    if args.figs_dir is None:
        if args.model_type == "pure":
            args.figs_dir = f"results/lstm_ae/pure/{args.participant}/figs"
        elif args.model_type == "personalized":
            args.figs_dir = f"results/lstm_ae/personalized/{args.participant}/figs"
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
        # Get input size from selected features file
        input_size = get_input_size_from_selected_features()
    else:
        input_size = args.input_size
    
    # Get data_params from config
    data_params = model_config.get("data_params")
    if data_params is None:
        raise ValueError("data_params not found in config file. Please ensure lstm_config.json contains data_params.")
    
    # Get sequence length from data_params
    sequence_length = data_params['train']['sequence_length']
    
    model = MaskedLSTMAutoencoder(
        input_size=input_size, 
        hidden_size=model_config["hidden_size"], 
        num_layers=model_config["num_layers"],
        sequence_length=sequence_length
    )
    
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── 2. Prepare loaders ──
    loader_factory = PhysiologicalDataLoader(args.data_path)
    
    if args.model_type in ["pure", "personalized"]:
        _, val_loader, test_loader = loader_factory.create_personalized_loaders(
            args.participant,
            data_params=data_params,
            filter_stress_val=False, # mixed data for thresholding
            filter_stress_test=False # mixed data for testing
        )
        participants_to_evaluate = [args.participant]
    else:
        _, val_loader, test_loader = loader_factory.create_general_loaders(
            args.participants,
            data_params=data_params,
            filter_stress_val=False, # mixed data for thresholding
            filter_stress_test=False # mixed data for testing
        )
        participants_to_evaluate = args.participants

    loss_fn = MaskedMSELoss()

    # ── 3. Evaluate on Validation Set to find threshold ──
    print("Evaluating on validation set to find optimal threshold...")
    _, val_inputs, val_outputs, val_masks = evaluate(model, val_loader, device, loss_fn)
    val_inputs  = np.concatenate(val_inputs,  axis=0)
    val_outputs = np.concatenate(val_outputs, axis=0)
    val_masks   = np.concatenate(val_masks,   axis=0)
    val_diff   = (val_inputs - val_outputs)**2 * val_masks
    val_errors = val_diff.sum(axis=(1,2)) / val_masks.sum(axis=(1,2))
    val_labels = get_sequence_labels(val_loader, participants_to_evaluate, split="val")
    
    valid_val = np.isfinite(val_errors)
    val_errors = val_errors[valid_val]
    val_labels = val_labels[valid_val]

    best_threshold = get_optimal_threshold(val_labels, val_errors)
    print(f"Optimal threshold found: {best_threshold:.4f}")

    # ── 4. Evaluate on Test Set ──
    print("Evaluating on test set...")
    _, test_inputs, test_outputs, test_masks = evaluate(model, test_loader, device, loss_fn)
    test_inputs  = np.concatenate(test_inputs,  axis=0)
    test_outputs = np.concatenate(test_outputs, axis=0)
    test_masks   = np.concatenate(test_masks,   axis=0)
    test_diff   = (test_inputs - test_outputs)**2 * test_masks
    test_errors = test_diff.sum(axis=(1,2)) / test_masks.sum(axis=(1,2))
    test_labels = get_sequence_labels(test_loader, participants_to_evaluate, split="test")

    valid_test = np.isfinite(test_errors)
    test_errors = test_errors[valid_test]
    test_labels = test_labels[valid_test]

    # ── 5. Latent Space Visualization on Test Set ──
    latents = extract_latents(model, test_loader, device)
    latents = latents[valid_test]

    # ── 6. Generate evaluation figures using the threshold from validation ──
    # 6.1 Reconstruction Error Distribution
    plot_recon_error_distribution(
        test_labels, test_errors, out_dir=args.figs_dir
    )

    # 6.2 ROC & Precision-Recall Curves
    plot_roc_pr_curves(
        test_labels, test_errors, out_dir=args.figs_dir
    )

    # 6.3 Confusion Matrix
    plot_confusion_matrix(
        test_labels, test_errors, out_dir=args.figs_dir, threshold=best_threshold
    )

    # 6.4 Example Time-Series Reconstructions
    plot_time_series_reconstructions(
        test_inputs, test_outputs, test_labels, out_dir=args.figs_dir
    )

    # 6.5 Latent-Space Visualization
    plot_latent_space_viz(
        latents, test_labels, out_dir=args.figs_dir
    )

    print(f"All evaluation figures saved to {args.figs_dir}")
 

if __name__ == "__main__":
    main()
