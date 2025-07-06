import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def main():
    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    data_path = "data" 

    from src.models.lstm_ae import MaskedLSTMAutoencoder
    from src.utils.losses import MaskedMSELoss
    from src.utils.train_utils import train_one_epoch, validate
    from src.data.physiological_loader import PhysiologicalDataLoader

    # LSTM Autoencoder
    model = MaskedLSTMAutoencoder(
        input_size=43,
        hidden_size=64,
        num_layers=1
    )

    # Optimizer and device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Data loader
    loader_factory = PhysiologicalDataLoader(data_path)
    participants = ["5C", "6B"]
    # train_loader, test_loader = loader_factory.create_personalized_loaders("5C")
    train_loader, test_loader = loader_factory.create_general_loaders(participants)
    # Loss function
    loss_fn = MaskedMSELoss()


    # Training loop 
    train_losses = []
    val_losses = []

    num_epochs = 100
    best_val_loss = float('inf')
    checkpoint_dir = "results/lstm_ae/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parameters for early stopping
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate(model, test_loader, device, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # Saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('results/lstm_ae/losses.png')
    plt.close()

    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)  
  



if __name__ == "__main__":
    main()






