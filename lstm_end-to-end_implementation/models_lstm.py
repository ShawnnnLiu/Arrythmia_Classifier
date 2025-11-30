# models_lstm_rhythm.py

"""
LSTM-based model for end-to-end rhythm classification.

Takes raw ECG segments (e.g., 5–30s @ 360Hz) and runs an LSTM over time,
optionally bidirectional, with an attention layer over time steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmAttention(nn.Module):
    """
    Simple additive attention over LSTM outputs.

    Input:  [batch, seq_len, hidden_dim]
    Output: context [batch, hidden_dim], attn_weights [batch, seq_len]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        scores = self.attn(lstm_out)           # [B, T, 1]
        weights = F.softmax(scores, dim=1)     # [B, T, 1]
        context = torch.sum(weights * lstm_out, dim=1)  # [B, H]
        return context, weights.squeeze(-1)    # [B, H], [B, T]


class RhythmLSTM(nn.Module):
    """
    Pure LSTM model for rhythm classification (end-to-end).

    Input:  ECG segment [batch, channels, time_steps]
    Output: class logits [batch, num_classes]
    """

    def __init__(self,
                 num_classes: int = 4,
                 input_channels: int = 1,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.5,
                 use_attention: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        lstm_input_dim = input_channels
        lstm_hidden_dim = hidden_dim
        num_directions = 2 if bidirectional else 1

        # LSTM over time (treat each time step as input_dim = #channels)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,              # [B, T, D]
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention over time steps (optional)
        if use_attention:
            self.attention = RhythmAttention(lstm_hidden_dim * num_directions)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_dim * num_directions, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_attention: bool = False):
        """
        x: [batch, channels, time_steps]
        """
        # Reorder to [batch, time_steps, features]
        # Features here are just the channels (1 or 2 leads)
        x = x.transpose(1, 2)  # [B, T, C]

        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, H*D]

        if self.use_attention:
            # Attention over all time steps
            context, attn_weights = self.attention(lstm_out)  # [B, H*D], [B, T]
        else:
            # Use last hidden state from LSTM
            if self.bidirectional:
                # h_n: [num_layers*2, B, H]
                h_forward = h_n[-2, :, :]   # [B, H]
                h_backward = h_n[-1, :, :]  # [B, H]
                context = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]
            else:
                context = h_n[-1, :, :]     # [B, H]
            attn_weights = None

        x = self.dropout(context)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_attention:
            return logits, attn_weights
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity test
    print("Testing RhythmLSTM...")
    model = RhythmLSTM(num_classes=4, input_channels=1)
    batch_size = 4
    segment_length = 3600   # e.g., 10s @ 360Hz
    dummy = torch.randn(batch_size, 1, segment_length)

    with torch.no_grad():
        out, attn = model(dummy, return_attention=True)

    print("Input shape: ", dummy.shape)
    print("Logits shape:", out.shape)
    print("Attention shape:", attn.shape if attn is not None else None)
    print("Trainable params:", model.count_parameters())