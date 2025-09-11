#!/usr/bin/env python
import sys
import os
import json
import argparse
import itertools

import torch
import matplotlib.pyplot as plt

# ── Project setup ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.helpers import plot_loss_curves
from src.models.tcn_ae import TCNAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import train_one_epoch, validate
from src.data.physiological_loader import PhysiologicalDataLoader

# ── Constants ──
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CONFIG_PATH   = f"config/tcn_config.json"
CHECKPOINT_DIR     = f"results/tcn_ae/general"
FIGS_DIR           = os.path.join(CHECKPOINT_DIR, "figs")

# ── Hyperparam search settings ──
HYPERPARAM_SPACE = {
    "latent_size": [32, 64, 128],
    "lr":          [1e-4, 5e-4, 1e-3],
    "num_levels":  [1, 2],
    "kernel_size": [3, 5],
    "sequence_length": [24, 48, 72],
    "overlap": [0.5, 0.8]
}
SEARCH_EPOCHS    = 50
FINAL_EPOCHS     = 600
PATIENCE         = 10

def train_and_evaluate(participants, data_path, latent_size, lr, num_levels, kernel_size, sequence_length, overlap, num_epochs=SEARCH_EPOCHS):
    """Train for up to num_epochs with early stopping; return best val loss."""
    data_params = {
        "train": {"sequence_length": sequence_length, "overlap": overlap},
        "val": {"sequence_length": sequence_length, "overlap": 0.2}, # Fixed low overlap for validation
        "test": {"sequence_length": sequence_length, "overlap": 0.2}, # Fixed low overlap for test
    }

    loader    = PhysiologicalDataLoader(data_path)
    train_loader, val_loader, _ = loader.create_general_loaders(
        participants,
        data_params=data_params,
        filter_stress_train=True,
        filter_stress_val=True
    )

    # Determine input size dynamically
    try:
        n_features = train_loader.dataset.get_feature_info()['n_features']
    except (IndexError, KeyError):
        print(f"Warning: Could not load data for participants. Skipping this combination.")
        return float('inf')

    model     = TCNAutoencoder(
        input_size=n_features,
        latent_size=latent_size,
        num_levels=num_levels,
        kernel_size=kernel_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    loss_fn   = MaskedMSELoss()
    model.to(DEVICE)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, loss_fn)
        val_loss   = validate(model, val_loader, DEVICE, loss_fn)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val

def do_grid_search(participants, data_path):
    """Loop over hyperparameter combinations; return the best-config dict."""
    best_score  = float('inf')
    best_config = None

    param_combinations = list(itertools.product(
        HYPERPARAM_SPACE["latent_size"],
        HYPERPARAM_SPACE["lr"],
        HYPERPARAM_SPACE["num_levels"],
        HYPERPARAM_SPACE["kernel_size"],
        HYPERPARAM_SPACE["sequence_length"],
        HYPERPARAM_SPACE["overlap"]
    ))

    for i, (ls, lr, nl, ks, sl, ov) in enumerate(param_combinations):
        print(f"[SEARCH {i+1}/{len(param_combinations)}] Running with config: ls={ls}, lr={lr:.0e}, nl={nl}, ks={ks}, sl={sl}, ov={ov}")
        val = train_and_evaluate(participants, data_path, ls, lr, nl, ks, sl, ov)
        print(f"→ val_loss={val:.4f}")
        if val < best_score:
            best_score  = val
            best_config = {
                "latent_size": ls, 
                "lr": lr, 
                "num_levels": nl, 
                "kernel_size": ks,
                "data_params": {
                    "train": {"sequence_length": sl, "overlap": ov},
                    "val": {"sequence_length": sl, "overlap": 0.2},
                    "test": {"sequence_length": sl, "overlap": 0.2}
                }
            }

    print(f"→ Best hyperparams: {best_config}, val_loss={best_score:.4f}")
    return best_config

def train_final(participants, data_path, latent_size, lr, num_levels, kernel_size, data_params):
    """Train a final model using the best hyperparams, with checkpoints & loss plotting."""
    loader    = PhysiologicalDataLoader(data_path)
    train_loader, val_loader, _ = loader.create_general_loaders(
        participants,
        data_params=data_params,
        filter_stress_train=True,
        filter_stress_val=True
    )

    # Determine input size dynamically
    try:
        n_features = train_loader.dataset.get_feature_info()['n_features']
    except (IndexError, KeyError):
        print(f"Error: Could not load data for participants. Aborting training.")
        return

    model     = TCNAutoencoder(
        input_size=n_features,
        latent_size=latent_size,
        num_levels=num_levels,
        kernel_size=kernel_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    loss_fn   = MaskedMSELoss()
    model.to(DEVICE)

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Attempt to resume
    resume_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    losses_path = os.path.join(CHECKPOINT_DIR, "losses.json")

    start_epoch      = 0
    train_losses     = []
    val_losses       = []
    best_val_loss    = float('inf')
    patience_counter = 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_loss    = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        start_epoch      = ckpt['epoch'] + 1

        if os.path.exists(losses_path):
            with open(losses_path) as f:
                log = json.load(f)
                train_losses = log.get("train_losses", [])
                val_losses   = log.get("val_losses", [])
        print(f"[TRAIN] Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, FINAL_EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, loss_fn)
        v_loss = validate(model, val_loader, DEVICE, loss_fn)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, resume_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[TRAIN] Early stopping at epoch {epoch+1}")
                break

        print(f"[TRAIN] Epoch {epoch+1}/{FINAL_EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}")

    # Save losses and final model
    with open(losses_path, "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Use the new helper function
    plot_loss_curves(train_losses, val_losses, latent_size, num_levels, FIGS_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["search", "train"],
        default="train",
        help="Mode 'search': run grid search & cache best params. 'train': load cached params & train final model."
    )
    parser.add_argument("--participants", nargs='+', required=True,
                        help="List of participant IDs to include in general training")
    parser.add_argument("--data-path", type=str, default="data/normalized_stratified",
                        help="Path to the normalized data root")
    args = parser.parse_args()

    if args.mode == "search":
        best_cfg = do_grid_search(args.participants, args.data_path)
        os.makedirs(os.path.dirname(BEST_CONFIG_PATH), exist_ok=True)
        with open(BEST_CONFIG_PATH, "w") as f:
            json.dump(best_cfg, f, indent=2)
        print(f"[SEARCH] Saved best config to {BEST_CONFIG_PATH}")

    elif args.mode == "train":
        if not os.path.exists(BEST_CONFIG_PATH):
            raise FileNotFoundError(
                f"No cached config found at {BEST_CONFIG_PATH}. "
                "Run with `--mode search` first."
            )
        with open(BEST_CONFIG_PATH) as f:
            cfg = json.load(f)
        
        # Unpack the nested data_params for the train_final function
        train_params = {
            "latent_size": cfg["latent_size"],
            "lr": cfg["lr"],
            "num_levels": cfg["num_levels"],
            "kernel_size": cfg["kernel_size"],
            "data_params": cfg["data_params"]
        }
        
        print(f"[TRAIN] Loaded hyperparams: {cfg}")
        train_final(args.participants, args.data_path, **train_params)

if __name__ == "__main__":
    main()






