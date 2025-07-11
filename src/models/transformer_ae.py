import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size, model_dim=128, num_layers=2, nhead=4, dropout=0.1, max_len=500):
        super().__init__()
        self.input_size = input_size
        self.model_dim = model_dim

        # 1. Input projection
        self.input_proj = nn.Linear(input_size, model_dim)

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Output projection
        self.output_proj = nn.Linear(model_dim, input_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # 1. Project input to model_dim
        x_proj = self.input_proj(x)  # (batch, seq_len, model_dim)
        x_proj = x_proj.permute(1, 0, 2)  # → (seq_len, batch, model_dim)

        # 2. Add positional encoding
        x_encoded = self.pos_encoder(x_proj)

        # 3. Encoder output
        memory = self.encoder(x_encoded)

        # 4. Decoder input — here, we just use the same encoded input as a starting point
        # (In masked denoising, you'd zero or shift the input here)
        decoded = self.decoder(x_encoded, memory)  # (seq_len, batch, model_dim)

        # 5. Output projection
        out = self.output_proj(decoded)  # (seq_len, batch, input_size)

        return out.permute(1, 0, 2)  # → back to (batch, seq_len, input_size)


    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the latent representation (from encoder output) for each sequence.
        Returns a tensor of shape (batch_size, model_dim)
        """
        x_proj = self.input_proj(x)               # (batch, seq_len, model_dim)
        x_proj = x_proj.permute(1, 0, 2)          # → (seq_len, batch, model_dim)
        x_encoded = self.pos_encoder(x_proj)      # Add positional encoding
        memory = self.encoder(x_encoded)          # (seq_len, batch, model_dim)

        # Option 1: Use the last time step
        latent = memory[-1]                       # (batch, model_dim)

        # Option 2 (Alternative): Average over time steps
        # latent = memory.mean(dim=0)             # (batch, model_dim)

        return latent
