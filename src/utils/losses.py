import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss that ignores NaN values based on mask
    """
    
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss
        
        Args:
            predictions: Predicted values
            targets: Target values
            mask: Mask tensor (True for valid data, False for NaN)
            
        Returns:
            Masked MSE loss
        """
        # Compute squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply mask - only consider valid data points
        masked_error = squared_error * mask.float()
        
        # Compute mean only over valid data points
        total_error = masked_error.sum()
        total_valid = mask.float().sum()
        
        # Avoid division by zero
        if total_valid > 0:
            return total_error / total_valid
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
