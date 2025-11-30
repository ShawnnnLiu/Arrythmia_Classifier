"""
Complex CNN with temporal modeling for rhythm classification

This module provides an advanced architecture combining:
- Residual convolutional blocks for feature extraction
- Bidirectional LSTM for temporal pattern recognition
- Attention mechanism for important time-step weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual Block for time-series data"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class AttentionLayer(nn.Module):
    """Attention mechanism for time-series"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output):
        """
        Parameters:
        -----------
        lstm_output : torch.Tensor
            LSTM output of shape [batch, seq_len, hidden_dim]
            
        Returns:
        --------
        context : torch.Tensor
            Attention-weighted context vector of shape [batch, hidden_dim]
        attention_weights : torch.Tensor
            Attention weights of shape [batch, seq_len]
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        
        # Compute context vector
        context = torch.sum(attention_weights * lstm_output, dim=1)  # [batch, hidden_dim]
        
        return context, attention_weights.squeeze(-1)


class ComplexRhythmCNN(nn.Module):
    """
    Complex CNN-LSTM with Attention for rhythm classification
    
    Architecture:
    1. Residual CNN blocks for feature extraction
    2. Bidirectional LSTM for temporal modeling
    3. Attention mechanism for important time-step weighting
    4. Fully connected classifier
    
    This architecture is designed to capture both:
    - Local morphological features (via CNN)
    - Long-term temporal patterns (via LSTM + Attention)
    """
    
    def __init__(self, 
                 num_classes: int = 4,
                 input_channels: int = 1,
                 dropout: float = 0.5,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2):
        """
        Initialize ComplexRhythmCNN
        
        Parameters:
        -----------
        num_classes : int
            Number of rhythm classes to classify
        input_channels : int
            Number of input channels (1 for single-lead ECG, 2 for dual-lead)
        dropout : float
            Dropout rate for regularization
        lstm_hidden_dim : int
            Hidden dimension for LSTM layers
        lstm_layers : int
            Number of LSTM layers
        """
        super(ComplexRhythmCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        # Block 1: 64 -> 64
        self.res_block1 = ResidualBlock1D(64, 64, kernel_size=3)
        
        # Block 2: 64 -> 128 with downsampling
        downsample2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128)
        )
        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=3, stride=2, downsample=downsample2)
        
        # Block 3: 128 -> 256 with downsampling
        downsample3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm1d(256)
        )
        self.res_block3 = ResidualBlock1D(128, 256, kernel_size=3, stride=2, downsample=downsample3)
        
        # Block 4: 256 -> 512 with downsampling
        downsample4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm1d(512)
        )
        self.res_block4 = ResidualBlock1D(256, 512, kernel_size=3, stride=2, downsample=downsample4)
        
        # Additional pooling to reduce sequence length for LSTM
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)  # Reduce to 100 time steps
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(lstm_hidden_dim * 2)  # *2 for bidirectional
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch, channels, time_steps]
        return_attention : bool
            If True, also returns attention weights
            
        Returns:
        --------
        logits : torch.Tensor
            Class logits of shape [batch, num_classes]
        attention_weights : torch.Tensor (optional)
            Attention weights if return_attention=True
        """
        # CNN feature extraction
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Reduce sequence length
        x = self.adaptive_pool(x)  # [batch, 512, 100]
        
        # Prepare for LSTM: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)  # [batch, 100, 512]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, 100, lstm_hidden_dim*2]
        
        # Attention
        context, attention_weights = self.attention(lstm_out)  # [batch, lstm_hidden_dim*2]
        
        # Fully connected classifier
        x = self.dropout(context)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComplexRhythmCNN_NoLSTM(nn.Module):
    """
    Complex CNN without LSTM (pure convolutional approach)
    
    Alternative architecture that uses only residual CNN blocks
    and global pooling. Faster than LSTM version but may miss
    some long-term temporal dependencies.
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1, dropout: float = 0.5):
        super(ComplexRhythmCNN_NoLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock1D(64, 64, kernel_size=3)
        
        downsample2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128)
        )
        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=3, stride=2, downsample=downsample2)
        
        downsample3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm1d(256)
        )
        self.res_block3 = ResidualBlock1D(128, 256, kernel_size=3, stride=2, downsample=downsample3)
        
        downsample4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm1d(512)
        )
        self.res_block4 = ResidualBlock1D(256, 512, kernel_size=3, stride=2, downsample=downsample4)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512 * 2, 256)  # *2 because we concat avg and max pooling
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Dual global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    print("Testing ComplexRhythmCNN models...")
    print("="*70)
    
    # Test LSTM version
    print("\n1. ComplexRhythmCNN (with LSTM + Attention)")
    print("-"*70)
    model_lstm = ComplexRhythmCNN(num_classes=4, input_channels=1)
    n_params = model_lstm.count_parameters()
    print(f"Parameters: {n_params:,}")
    
    # Test with dummy input (10 seconds @ 360Hz)
    batch_size = 4
    segment_length = 3600
    dummy_input = torch.randn(batch_size, 1, segment_length)
    
    print(f"Input shape: {dummy_input.shape}")
    
    model_lstm.eval()
    with torch.no_grad():
        output, attention = model_lstm(dummy_input, return_attention=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Output logits (first sample): {output[0]}")
    
    # Test no-LSTM version
    print("\n2. ComplexRhythmCNN_NoLSTM (pure CNN)")
    print("-"*70)
    model_cnn = ComplexRhythmCNN_NoLSTM(num_classes=4, input_channels=1)
    n_params = model_cnn.count_parameters()
    print(f"Parameters: {n_params:,}")
    
    model_cnn.eval()
    with torch.no_grad():
        output = model_cnn(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output[0]}")
    
    # Test with different segment lengths
    print("\nTesting with different segment lengths (LSTM version):")
    for duration_sec in [5, 10, 15, 30]:
        length = int(duration_sec * 360)
        dummy = torch.randn(2, 1, length)
        with torch.no_grad():
            out = model_lstm(dummy)
        print(f"  {duration_sec:2d}s ({length:5d} samples) -> output shape: {out.shape}")
    
    print("\n" + "="*70)
    print("ComplexRhythmCNN test passed!")
    print("="*70)

