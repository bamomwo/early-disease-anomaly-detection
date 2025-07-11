import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_ae import MaskedLSTMAutoencoder
from src.data.physiological_loader import PhysiologicalDataLoader


def main():
    # 1. Set paths for model and device
    participant = "BG"
    data_path = "data"
    checkpoint_path = f"results/lstm_ae/pure/checkpoints/{participant}/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model
    model = MaskedLSTMAutoencoder(input_size=43, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load test data
    loader_factory = PhysiologicalDataLoader(data_path)
    _, _, test_loader = loader_factory.create_personalized_loaders("5C")

    # 4. Extract latent vectors
    all_latents = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["data"].to(device)
            latents = model.get_latent_representation(x)
            all_latents.append(latents.cpu().numpy())
    all_latents = np.concatenate(all_latents, axis=0)

    # 5. PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)

 
    # Optionally, save latent vectors for further analysis
    np.save("results/lstm_ae/pure/latent_vectorss.npy", all_latents)

if __name__ == "__main__":
    main() 