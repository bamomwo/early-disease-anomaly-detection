#!/usr/bin/env python
import sys
import os
import json
import argparse

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
DATA_PATH = "data/normalized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CONFIG_PATH = "config/transformer_config.json"
SELECTED_FEATURES_PATH = "config/selected_features.json"
GENERAL_PRETRAINED_PATH = "results/transformer_ae/general/best_model.pth"
PERSONALIZED_DIR = "results/transformer_ae/personalized"

DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 5


def get_input_size_from_selected_features():
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    return len(selected_features['features'])


def build_model(config):
    input_size = get_input_size_from_selected_features()
    return TransformerAutoencoder(
        input_size=input_size,
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        nhead=config["nhead"],
        dropout=config["dropout"],
    )


def fine_tune_participant(participant, lr=None, epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE, pretrained_path=GENERAL_PRETRAINED_PATH, freeze_encoder=False):
    # Load best hyperparameters
    if not os.path.exists(BEST_CONFIG_PATH):
        raise FileNotFoundError(
            f"No cached config found at {BEST_CONFIG_PATH}. Run general or pure search first."
        )
    with open(BEST_CONFIG_PATH) as f:
        cfg = json.load(f)

    # Learning rate override or use config
    effective_lr = lr if lr is not None else cfg.get("lr", 1e-4)

    # Directories
    save_dir = os.path.join(PERSONALIZED_DIR, participant)
    figs_dir = os.path.join(save_dir, "figs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Build model and load pretrained weights
    model = build_model(cfg)
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path)
        # Support either full checkpoint or state_dict
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
    else:
        print(f"[WARN] Pretrained checkpoint not found at {pretrained_path}. Training from scratch for this participant.")

    if freeze_encoder:
        for param in getattr(model, 'encoder', model).parameters():
            param.requires_grad = False

    model.to(DEVICE)

    # Data
    loader_factory = PhysiologicalDataLoader(DATA_PATH, config={'num_workers': 1})
    train_loader, val_loader, _ = loader_factory.create_personalized_loaders(participant)

    loss_fn = MaskedMSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=effective_lr)

    # Resume support
    resume_path = os.path.join(save_dir, "best_model.pth")
    losses_path = os.path.join(save_dir, f"losses_{participant}.json")

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
        patience_counter = ckpt.get('patience_counter', 0)
        start_epoch = ckpt.get('epoch', -1) + 1

        if os.path.exists(losses_path):
            with open(losses_path) as f:
                log = json.load(f)
                train_losses = log.get("train_losses", [])
                val_losses = log.get("val_losses", [])
        print(f"[TRAIN] Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        t_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, loss_fn)
        v_loss = validate(model, val_loader, DEVICE, loss_fn)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
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
            if patience_counter >= patience:
                print(f"[TRAIN] Early stopping at epoch {epoch+1}")
                break

        print(f"[TRAIN] Epoch {epoch+1}/{epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

    # Persist losses and final model
    with open(losses_path, "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    final_model_path = os.path.join(save_dir, f"final_model_{participant}.pth")
    torch.save(model.state_dict(), final_model_path)

    # Plot losses with shared helper
    plot_loss_curves(train_losses, val_losses, cfg["model_dim"], cfg["num_layers"], figs_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=str, default="BG", help="Participant ID to fine-tune on")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate override for fine-tuning")
    parser.add_argument("--pretrained", type=str, default=GENERAL_PRETRAINED_PATH, help="Path to pretrained general checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder layers during fine-tuning")
    args = parser.parse_args()

    fine_tune_participant(
        participant=args.participant,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        pretrained_path=args.pretrained,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()