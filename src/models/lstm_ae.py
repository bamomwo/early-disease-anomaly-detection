"""
LSTM Autoencoder for Anomaly Detection
File: models/lstm_ae.py

This module contains the LSTM autoencoder model for temporal anomaly detection
in physiological data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import os
import json
from datetime import datetime


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked MSE loss that ignores NaN values
    
    Args:
        predictions: Predicted values
        targets: Target values  
        mask: Mask tensor (True for valid data, False for NaN)
        
    Returns:
        Masked MSE loss
    """
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask.float()
    
    total_error = masked_error.sum()
    total_valid = mask.float().sum()
    
    if total_valid > 0:
        return total_error / total_valid
    else:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)


class MaskedLSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder with masking support for anomaly detection.
    Handles temporal dependencies and NaN values through masking.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 sequence_length: int = 10):
        """
        Initialize LSTM Autoencoder
        
        Args:
            input_size: Number of features per timestep
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Length of input sequences
        """
        super(MaskedLSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layer to reconstruct input
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the autoencoder
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            mask: Mask tensor (optional, for compatibility)
        
        Returns:
            Reconstructed tensor of same shape as input
        """
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use the last hidden state as the compressed representation
        latent = encoded[:, -1, :]
        
        # Prepare decoder input - repeat latent vector for each timestep
        decoder_input = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input)
        
        # Apply dropout
        decoded = self.dropout(decoded)
        
        # Reconstruct to original input size
        reconstruction = self.output_layer(decoded)
        
        return reconstruction
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the latent representation of the input"""
        with torch.no_grad():
            encoded, _ = self.encoder(x)
            return encoded[:, -1, :]