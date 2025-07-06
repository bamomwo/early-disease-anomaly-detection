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
        num_layers=1,
        # dropout=0.2,
    )

    # Optimizer and device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Data loader
    loader_factory = PhysiologicalDataLoader(data_path)
    train_loader, test_loader = loader_factory.create_personalized_loaders("94")

    # Loss function
    loss_fn = MaskedMSELoss()


    # Training loop 
    train_losses = []
    val_losses = []

    num_epochs = 200
    best_val_loss = float('inf')
    checkpoint_dir = "results/lstm_ae/pure/checkpoints/94"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parameters for early stopping
    patience = 10
    patience_counter = 0

    # Move model to device
    model.to(device)

    resume_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoint found, starting from scratch")
    
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
    

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Participant 94')
    plt.legend()
    plt.savefig('results/lstm_ae/pure/checkpoints/94/losses.png')
    plt.close()

    final_path = os.path.join(checkpoint_dir, "final_model_94.pth")
    torch.save(model.state_dict(), final_path)  
  



if __name__ == "__main__":
    main()






