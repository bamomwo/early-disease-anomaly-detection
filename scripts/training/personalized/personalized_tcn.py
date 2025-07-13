import torch
import os
import sys
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.tcn_ae import TCNAutoencoder
from src.utils.losses import MaskedMSELoss
from src.utils.train_utils import train_one_epoch, validate
from src.data.physiological_loader import PhysiologicalDataLoader


def main():
    # ------------------- CONFIG ------------------- #
    participant = "BG"  # replace with desired participant
    pretrained_path = "results/tcn_ae/general/best_model.pth"
    save_dir = f"results/tcn_ae/personalized/{participant}"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------- MODEL PARAMS ..............#
    input_size = 43
    latent_size = 64
    kernel_size = 3
    #dropout = 0.1

    learning_rate = 1e-4  # smaller LR for fine-tuning
    num_epochs = 30
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- LOAD MODEL ------------------- #
    model = TCNAutoencoder(
        input_size=input_size,
        latent_size=latent_size,
        kernel_size=kernel_size,
        #dropout=dropout
    )

    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Optionally freeze encoder
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # ------------------- PREPARE DATA ------------------- #
    loader_factory = PhysiologicalDataLoader("data")
    train_loader, val_loader, _ = loader_factory.create_personalized_loaders(participant)

    loss_fn = MaskedMSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=0.0001
    )

    # ------------------- FINE-TUNING LOOP ------------------- #
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)


        train_losses.append(train_loss)
        val_losses.append(val_loss)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"Epoch {epoch+1}: New best model saved. Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Val Loss: {val_loss:.4f}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

     # Saving the losses
    loss_log = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(save_dir, f"losses.json"), "w") as f:
        json.dump(loss_log, f)


    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig(f'results/tcn_ae/personalized/{participant}/losses.png')
    plt.close()

if __name__ == "__main__":
    main()