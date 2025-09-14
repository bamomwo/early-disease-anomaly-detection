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
from src.models.lstm_ae import MaskedLSTMAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import train_one_epoch, validate
from src.data.physiological_loader import PhysiologicalDataLoader

# ── Constants ──
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CONFIG_PATH   = f"config/lstm_config.json"
SELECTED_FEATURES_PATH = "config/selected_features.json"

# ── Hyperparam search settings ──
HYPERPARAM_SPACE = {
    "hidden_size": [32, 64, 128],
    "lr":          [1e-4, 5e-4, 1e-3],
    "num_layers":  [1, 2],
    "sequence_length": [24, 48, 72],
    "overlap": [0.5, 0.8]
}
SEARCH_EPOCHS    = 50
FINAL_EPOCHS     = 200
PATIENCE         = 10

def get_input_size_from_selected_features():
    """Load selected features file and return the number of features."""
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    return len(selected_features['features'])

def train_and_evaluate(participant, data_path, hidden_size, lr, num_layers, sequence_length, overlap, num_epochs=SEARCH_EPOCHS):
    """Train for up to num_epochs with early stopping; return best val loss."""
    data_params = {
        "train": {"sequence_length": sequence_length, "overlap": overlap},
        "val": {"sequence_length": sequence_length, "overlap": overlap},
        "test": {"sequence_length": sequence_length, "overlap": overlap},
    }

    loader    = PhysiologicalDataLoader(data_path)
    train_loader, val_loader, _ = loader.create_personalized_loaders(
        participant,
        data_params=data_params,
        filter_stress_train=True,
        filter_stress_val=True
    )
    
    # Determine input size dynamically
    try:
        n_features = train_loader.dataset.get_feature_info()['n_features']
    except (IndexError, KeyError):
        print(f"Warning: Could not load data for participant {participant}. Skipping this combination.")
        return float('inf')

    model     = MaskedLSTMAutoencoder(input_size=n_features,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      sequence_length=sequence_length)
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

def do_grid_search(participant, data_path):
    """Loop over hyperparameter combinations; return the best-config dict."""
    best_score  = float('inf')
    best_config = None

    param_combinations = list(itertools.product(
        HYPERPARAM_SPACE["hidden_size"],
        HYPERPARAM_SPACE["lr"],
        HYPERPARAM_SPACE["num_layers"],
        HYPERPARAM_SPACE["sequence_length"],
        HYPERPARAM_SPACE["overlap"]
    ))

    for i, (hs, lr, nl, sl, ov) in enumerate(param_combinations):
        print(f"[SEARCH {i+1}/{len(param_combinations)}] Running with config: hs={hs}, lr={lr:.0e}, nl={nl}, sl={sl}, ov={ov}")
        val = train_and_evaluate(participant, data_path, hs, lr, nl, sl, ov)
        print(f"→ val_loss={val:.4f}")
        if val < best_score:
            best_score  = val
            best_config = {
                "hidden_size": hs, 
                "lr": lr, 
                "num_layers": nl,
                "data_params": {
                    "train": {"sequence_length": sl, "overlap": ov},
                    "val": {"sequence_length": sl, "overlap": ov},
                    "test": {"sequence_length": sl, "overlap": ov}
                }
            }

    print(f"→ Best hyperparams: {best_config}, val_loss={best_score:.4f}")
    return best_config

def train_final(participant, data_path, hidden_size, lr, num_layers, data_params):
    """Train a final model using the best hyperparams, with checkpoints & loss plotting."""
    loader    = PhysiologicalDataLoader(data_path)
    train_loader, val_loader, _ = loader.create_personalized_loaders(
        participant,
        data_params=data_params,
        filter_stress_train=True,
        filter_stress_val=True
    )

    # Determine input size dynamically
    try:
        n_features = train_loader.dataset.get_feature_info()['n_features']
    except (IndexError, KeyError):
        print(f"Error: Could not load data for participant {participant}. Aborting training.")
        return

    # Get sequence length from data_params
    sequence_length = data_params['train']['sequence_length']

    model     = MaskedLSTMAutoencoder(input_size=n_features,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    loss_fn   = MaskedMSELoss()
    model.to(DEVICE)

    checkpoint_dir = f"results/lstm_ae/pure/{participant}"
    figs_dir = os.path.join(checkpoint_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Attempt to resume
    resume_path = os.path.join(checkpoint_dir, "best_model.pth")
    losses_path = os.path.join(checkpoint_dir, f"losses_{participant}.json")

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

    final_model_path = os.path.join(checkpoint_dir, f"final_model_{participant}.pth")
    torch.save(model.state_dict(), final_model_path)

    # Use the helper function for plotting
    plot_loss_curves(train_losses, val_losses, hidden_size, num_layers, figs_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=str, required=True, help="Participant ID")
    parser.add_argument("--data-path", type=str, default="data/normalized_stratified", help="Path to the data directory")
    parser.add_argument(
        "--mode",
        choices=["search", "train"],
        default="train",
        help="Mode ‘search’: run grid search & cache best params. ‘train’: load cached params & train final model."
    )
    args = parser.parse_args()

    if args.mode == "search":
        best_cfg = do_grid_search(args.participant, args.data_path)
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
        
        train_params = {
            "hidden_size": cfg["hidden_size"],
            "lr": cfg["lr"],
            "num_layers": cfg["num_layers"],
            "data_params": cfg["data_params"]
        }
        
        print(f"[TRAIN] Loaded hyperparams: {cfg}")
        train_final(args.participant, args.data_path, **train_params)

if __name__ == "__main__":
    main()