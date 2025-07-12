import sys
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def main():
    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    data_path = "data" 
    participant = "6B"

    from src.models.tcn_ae import TCNAutoencoder
    from src.utils.losses import MaskedMSELoss
    from src.utils.train_utils import train_one_epoch, validate
    from src.data.physiological_loader import PhysiologicalDataLoader

    # LSTM Autoencoder
    model = TCNAutoencoder(
        input_size=43,
        latent_size=64,
        num_levels=1,
        kernel_size=3
        # dropout=0.2,
    )

    # Optimizer and device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Data loader
    loader_factory = PhysiologicalDataLoader(data_path)
    train_loader, val_loader, _ = loader_factory.create_personalized_loaders(participant)

    # Loss function
    loss_fn = MaskedMSELoss()


    # Training loop 
    train_losses = []
    val_losses = []
    start_epoch = 0


    num_epochs = 200
    best_val_loss = float('inf')
    checkpoint_dir = f"results/tcn_ae/pure/checkpoints/{participant}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parameters for early stopping
    patience = 10
    patience_counter = 0

    # Move model to device
    model.to(device)

    # Resume training from checkpoint
    resume_path = os.path.join(checkpoint_dir, "best_model.pth")
    losses_path = os.path.join(checkpoint_dir, f"losses_{participant}.json")
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        start_epoch = checkpoint['epoch'] + 1


        if os.path.exists(losses_path):
            with open(losses_path, "r") as f:
                loss_log = json.load(f)
                train_losses = loss_log.get("train_losses", [])
                val_losses = loss_log.get("val_losses", [])
    else:
        print("No checkpoint found, starting from scratch")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # Saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    

    # Saving the losses
    loss_log = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }

    with open(os.path.join(checkpoint_dir, f"losses_{participant}.json"), "w") as f:
        json.dump(loss_log, f)

    final_path = os.path.join(checkpoint_dir, f"final_model_{participant}.pth")
    torch.save(model.state_dict(), final_path) 

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for Participant {participant} with 32 hidden units')
    plt.legend()
    plt.savefig(f'results/tcn_ae/pure/checkpoints/{participant}/losses.png')
    plt.close()

     
  



if __name__ == "__main__":
    main()






