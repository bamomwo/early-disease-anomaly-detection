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
DATA_PATH          = "data/normalized"
PARTICIPANT        = "BG"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CONFIG_PATH   = f"results/tcn_config.json"
CHECKPOINT_DIR     = f"results/tcn_ae/pure/{PARTICIPANT}"
FIGS_DIR           = os.path.join(CHECKPOINT_DIR, "figs")

# ── Hyperparam search settings ──
HYPERPARAM_SPACE = {
    "latent_size": [32, 64, 128],
    "lr":          [1e-4, 5e-4, 1e-3],
    "num_levels":  [1, 2],
    "kernel_size": [3, 5],
}
SEARCH_EPOCHS    = 50
FINAL_EPOCHS     = 200
PATIENCE         = 10

def train_and_evaluate(latent_size, lr, num_levels, kernel_size, num_epochs=SEARCH_EPOCHS):
    model     = TCNAutoencoder(input_size=43,
                              latent_size=latent_size,
                              num_levels=num_levels,
                              kernel_size=kernel_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn   = MaskedMSELoss()

    loader    = PhysiologicalDataLoader(DATA_PATH)
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

    for ls, lr, nl, ks in itertools.product(
            HYPERPARAM_SPACE["latent_size"],
            HYPERPARAM_SPACE["lr"],
            HYPERPARAM_SPACE["num_levels"],
            HYPERPARAM_SPACE["kernel_size"]):
        val = train_and_evaluate(ls, lr, nl, ks)
        print(f"[SEARCH] latent_size={ls}, lr={lr:.0e}, num_levels={nl}, kernel_size={ks} → val_loss={val:.4f}")
        if val < best_score:
            best_score  = val
            best_config = {"latent_size": ls, "lr": lr, "num_levels": nl, "kernel_size": ks}

    print(f"→ Best hyperparams: {best_config}, val_loss={best_score:.4f}")
    return best_config

def train_final(latent_size, lr, num_levels, kernel_size):
    model     = TCNAutoencoder(input_size=43,
                              latent_size=latent_size,
                              num_levels=num_levels,
                              kernel_size=kernel_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn   = MaskedMSELoss()

    loader    = PhysiologicalDataLoader(DATA_PATH)
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

    plot_loss_curves(train_losses, val_losses, latent_size, num_levels, FIGS_DIR)

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






