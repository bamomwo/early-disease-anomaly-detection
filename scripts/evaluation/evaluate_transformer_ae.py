import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer_ae import TransformerAutoencoder
from src.utils.losses import MaskedMSELoss
from src.data.physiological_loader import PhysiologicalDataLoader
from src.utils.train_utils import evaluate
from src.utils.helpers import compute_group_errors, aggregate_loss_analysis, plot_loss_analysis

def main():

    participant = "BG"
    #participants = ["5C", "6B", "6D", "7A", "7E", "8B", "94", "BG" ]

    # Path to data and model checkpoints
    data_path = "data" 
    checkpoint_path = "results/transformer_ae/general/checkpoints/best_model.pth"
    save_visualisations = "results/transformer_ae/general/checkpoints/visuals"
    # Device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model
    model = TransformerAutoencoder(input_size=43, model_dim=128, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load test data
    loader_factory = PhysiologicalDataLoader(data_path, config={"num_workers":1})
    _, _, test_loader = loader_factory.create_personalized_loaders(participant)
    #train_loader, _, _ = loader_factory.create_general_loaders(participants)

    # 4. Define loss and evaluation loop
    loss_fn = MaskedMSELoss()

    # Use the new evaluate function
    avg_loss, all_inputs, all_outputs = evaluate(model, test_loader, device, loss_fn)

    # 5. Analyze and visualize
    print(f"Average test reconstruction loss: {avg_loss:.4f}")

    # Convert lists to numpy arrays
    inputs = np.concatenate(all_inputs, axis=0)    # shape: (num_seq, seq_len, input_size)
    outputs = np.concatenate(all_outputs, axis=0)  # shape: same

    # Define feature groups
    feature_groups = {
        "HR_related": [0, 1, 2, 3, 4],
        "TEMP_related": [5, 6, 7, 8, 9],
        "another": [10]
    }

    # Perform comprehensive loss analysis
    print("Performing comprehensive loss analysis...")
    analysis = aggregate_loss_analysis(inputs, outputs, feature_groups)
    
    # Print summary statistics
    print(f"\n=== LOSS ANALYSIS SUMMARY ===")
    print(f"Number of sequences: {analysis['num_sequences']}")
    print(f"Number of features: {analysis['num_features']}")
    print(f"Overall mean loss: {analysis['statistics']['overall_mean_loss']:.4f}")
    
    print(f"\n=== GROUP-WISE STATISTICS ===")
    for group_name in analysis['group_losses'].keys():
        mean_loss = analysis['statistics']['group_mean_losses'][group_name]
        print(f"{group_name}: Mean={mean_loss:.4f}")
    
    # Create comprehensive visualizations
    print("\nCreating loss analysis visualizations...")
    
    plot_loss_analysis(analysis, save_dir=save_visualisations)
    print(f"Visualizations saved to {save_visualisations}/")

if __name__ == "__main__":
    main()

