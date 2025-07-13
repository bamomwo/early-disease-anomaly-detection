import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_ae import MaskedLSTMAutoencoder
from src.utils.losses import MaskedMSELoss
from src.data.physiological_loader import PhysiologicalDataLoader
from src.utils.train_utils import evaluate
from src.utils.helpers import compute_group_errors

def main():
    # 1. Set paths for model and device
    participant = "BG"
    data_path = "data" 
    checkpoint_path = f"results/lstm_ae/pure/checkpoints/{participant}/best_model.pth"
    
    # Device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model
    model = MaskedLSTMAutoencoder(input_size=43, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load test data
    loader_factory = PhysiologicalDataLoader(data_path, config={"num_workers":0})
    _, _, test_loader = loader_factory.create_personalized_loaders("5C")

    # 4. Define loss and evaluation loop
    loss_fn = MaskedMSELoss()

    # Use the new evaluate function
    avg_loss, all_inputs, all_outputs = evaluate(model, test_loader, device, loss_fn)

    # 5. Analyze and visualize
    print(f"Average test reconstruction loss: {avg_loss:.4f}")

    # Optional: Save outputs or visualize sample reconstructions
    # e.g., plot a few sequences and their reconstructions


    # Convert lists to numpy arrays
    inputs = np.concatenate(all_inputs, axis=0)    # shape: (num_seq, seq_len, input_size)
    outputs = np.concatenate(all_outputs, axis=0)  # shape: same

    # Select a sample sequence (e.g., index 0)
    seq_idx = 0
    seq_input = inputs[seq_idx]    # shape: (seq_len, input_size)
    seq_output = outputs[seq_idx]  # same

    # obtain reconstruction error for feature groups
    feature_groups = {
    "HR_related": [0, 1, 2, 3, 4],
    "TEMP_related": [5, 6, 7, 8, 9],
}


    group_errors = compute_group_errors(inputs, outputs, feature_groups)
    print(group_errors)



    # Plot a few features (e.g., first 3 features)
    # features_to_plot = [0, 1, 2]

    # plt.figure(figsize=(12, 6))
    # for i, f in enumerate(features_to_plot):
    #     plt.subplot(len(features_to_plot), 1, i+1)
    #     plt.plot(seq_input[:, f], label=f"Original Feature {f}")
    #     plt.plot(seq_output[:, f], label=f"Reconstructed Feature {f}", linestyle='--')
    #     plt.legend()
    #     plt.title(f"Feature {f} Reconstruction - Sequence {seq_idx}")
    #     plt.tight_layout()

    # plt.savefig("results/lstm_ae/pure/reconstruction_seq0.png")
    # plt.show()
    

if __name__ == "__main__":
    main()

