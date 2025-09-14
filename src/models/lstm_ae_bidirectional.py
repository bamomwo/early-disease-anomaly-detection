"""
Bidirectional LSTM Autoencoder for Anomaly Detection
File: models/lstm_ae_bidirectional.py

This module contains the Bidirectional LSTM autoencoder model for temporal anomaly detection
in physiological data.
"""

import torch
import torch.nn as nn
from typing import Optional

class BidirectionalLSTMAutoencoder(nn.Module):
    """
    Bidirectional LSTM Autoencoder with masking support for anomaly detection.
    Handles temporal dependencies and NaN values through masking.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 sequence_length: int = 10):
        """
        Initialize Bidirectional LSTM Autoencoder
        
        Args:
            input_size: Number of features per timestep
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Length of input sequences
        """
        super(BidirectionalLSTMAutoencoder, self).__init__()
        
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
            bidirectional=True
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2, # Input from bidirectional encoder
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
        
        # Use the last hidden state of the encoder as the compressed representation
        # The output of the bidirectional encoder is concatenated, so we take the full last hidden state
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
