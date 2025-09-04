import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Removes padding on the right (to maintain causality if needed)."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        # Calculate causal padding: (kernel_size - 1) * dilation
        # This ensures we only look at past and present, not future
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size=64, num_levels=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        
        # Encoder
        self.encoder = TemporalConvNet(
            num_inputs=input_size,
            num_channels=[latent_size] * num_levels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Decoder: Use a symmetric TCN structure
        # Reverse the channel progression for the decoder
        decoder_channels = [latent_size] * num_levels
        decoder_channels.reverse()  # Reverse to go from latent_size back to input_size
        
        self.decoder = TemporalConvNet(
            num_inputs=latent_size,
            num_channels=decoder_channels + [input_size],
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        
        encoded = self.encoder(x)  # (batch, latent_size, encoded_seq_len)
        decoded = self.decoder(encoded)  # (batch, input_size, decoded_seq_len)
        
        # Ensure output has the same sequence length as input
        if decoded.shape[2] != seq_len:
            # If decoded sequence is longer, truncate to match input length
            if decoded.shape[2] > seq_len:
                decoded = decoded[:, :, :seq_len]
            # If decoded sequence is shorter, pad with zeros
            else:
                padding = seq_len - decoded.shape[2]
                decoded = torch.nn.functional.pad(decoded, (0, padding))
        
        return decoded.permute(0, 2, 1)  # (batch, seq_len, input_size)

    def get_latent_representation(self, x):
        with torch.no_grad():
            # x: (batch, seq_len, features)  → TCN expects (batch, features, seq_len)
            x = x.permute(0, 2, 1)
            encoded = self.encoder(x)         # (batch, latent_size, seq_len)
            last_step = encoded[:, :, -1]     # (batch, latent_size)
            return last_step 

