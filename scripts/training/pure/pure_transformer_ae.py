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
from src.models.transformer_ae import TransformerAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import train_one_epoch, validate
from src.data.physiological_loader import PhysiologicalDataLoader

# ── Constants ──
DATA_PATH          = "data/normalized"
PARTICIPANT        = "94"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CONFIG_PATH   = f"config/transformer_config.json"
SELECTED_FEATURES_PATH = "config/selected_features.json"
CHECKPOINT_DIR     = f"results/transformer_ae/pure/{PARTICIPANT}"
FIGS_DIR           = os.path.join(CHECKPOINT_DIR, "figs")

# ── Hyperparam search settings ──
HYPERPARAM_SPACE = {
    "model_dim": [32, 64, 128],
    "lr":        [1e-4, 5e-4, 1e-3],
    "num_layers": [1, 2, 3],
    "nhead":     [2, 4],
    "dropout":   [0.1, 0.2],
}


SEARCH_EPOCHS    = 50
FINAL_EPOCHS     = 200
PATIENCE         = 10

def get_input_size_from_selected_features():
    """Load selected features file and return the number of features."""
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    return len(selected_features['features'])

def train_and_evaluate(model_dim, lr, num_layers, nhead, dropout, num_epochs=SEARCH_EPOCHS):
    # Get dynamic input size from selected features
    input_size = get_input_size_from_selected_features()
    
    model     = TransformerAutoencoder(
        input_size=input_size,
        model_dim=model_dim,
        num_layers=num_layers,
        nhead=nhead,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = MaskedMSELoss()

    loader    = PhysiologicalDataLoader(DATA_PATH, config={'num_workers':1})
    train_loader, val_loader, _ = loader.create_personalized_loaders(PARTICIPANT)
    model.to(DEVICE)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, loss_fn)
        val_loss   = validate(model, val_loader, DEVICE, loss_fn)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val

def do_grid_search():
    best_score  = float('inf')
    best_config = None

    for md, lr, nl, nh, do in itertools.product(
            HYPERPARAM_SPACE["model_dim"],
            HYPERPARAM_SPACE["lr"],
            HYPERPARAM_SPACE["num_layers"],
            HYPERPARAM_SPACE["nhead"],
            HYPERPARAM_SPACE["dropout"]):
        val = train_and_evaluate(md, lr, nl, nh, do)
        print(f"[SEARCH] model_dim={md}, lr={lr:.0e}, num_layers={nl}, nhead={nh}, dropout={do} → val_loss={val:.4f}")
        if val < best_score:
            best_score  = val
            best_config = {"model_dim": md, "lr": lr, "num_layers": nl, "nhead": nh, "dropout": do}

    print(f"→ Best hyperparams: {best_config}, val_loss={best_score:.4f}")
    return best_config

def train_final(model_dim, lr, num_layers, nhead, dropout):
    # Get dynamic input size from selected features
    input_size = get_input_size_from_selected_features()
    
    model     = TransformerAutoencoder(
        input_size=input_size,
        model_dim=model_dim,
        num_layers=num_layers,
        nhead=nhead,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = MaskedMSELoss()

    loader    = PhysiologicalDataLoader(DATA_PATH, config={'num_workers':1})
    train_loader, val_loader, _ = loader.create_personalized_loaders(PARTICIPANT)
    model.to(DEVICE)

    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    resume_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    losses_path = os.path.join(CHECKPOINT_DIR, f"losses_{PARTICIPANT}.json")

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

    with open(losses_path, "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    final_model_path = os.path.join(CHECKPOINT_DIR, f"final_model_{PARTICIPANT}.pth")
    torch.save(model.state_dict(), final_model_path)

    plot_loss_curves(train_losses, val_losses, model_dim, num_layers, FIGS_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["search", "train"],
        default="train",
        help="Mode ‘search’: run grid search & cache best params. ‘train’: load cached params & train final model."
    )
    args = parser.parse_args()

    if args.mode == "search":
        best_cfg = do_grid_search()
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
        print(f"[TRAIN] Loaded hyperparams: {cfg}")
        train_final(**cfg)

if __name__ == "__main__":
    main()






