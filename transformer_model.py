import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):

    def __init__(
        self,
        num_features,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        output_size,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super(TransformerModel, self).__init__()
        self.num_layers = num_encoder_layers
        self.d_model = d_model

        self.input_linear = nn.Linear(num_features, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.decoder = nn.Linear(d_model, output_size)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        # Project input features to d_model
        src = self.input_linear(src)  # (batch_size, seq_length, d_model)

        # Apply positional encoding
        src = self.positional_encoding(src)  # (batch_size, seq_length, d_model)

        # Pass through Transformer Encoder
        output = self.transformer_encoder(
            src, mask=src_mask
        )  # (batch_size, seq_length, d_model)

        # Take the output from the last time step
        output = output[:, -1, :]  # (batch_size, d_model)

        # Pass through the fully connected layer
        output = self.decoder(output)  # (batch_size, output_size)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)
