"""
Simple 1D CNN model for rhythm classification

This module provides a lightweight baseline CNN architecture for classifying
ECG rhythm patterns from longer segments (10-30s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRhythmCNN(nn.Module):
    """
    Simple 1D CNN for rhythm classification
    
    Architecture:
    - 4 convolutional blocks (conv -> bn -> relu -> maxpool)
    - Global average pooling
    - Fully connected classifier
    
    Designed for longer ECG segments (10-30 seconds) compared to beat classification.
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1, dropout: float = 0.5):
        """
        Initialize SimpleRhythmCNN
        
        Parameters:
        -----------
        num_classes : int
            Number of rhythm classes to classify
        input_channels : int
            Number of input channels (1 for single-lead ECG, 2 for dual-lead)
        dropout : float
            Dropout rate for regularization
        """
        super(SimpleRhythmCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional blocks
        # Input shape: [batch, 1, ~3600] for 10s @ 360Hz
        
        # Block 1: [batch, 1, T] -> [batch, 32, T/4]
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        # Block 2: [batch, 32, T/4] -> [batch, 64, T/16]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        # Block 3: [batch, 64, T/16] -> [batch, 128, T/64]
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        
        # Block 4: [batch, 128, T/64] -> [batch, 256, T/256]
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        
        # Global pooling: [batch, 256, T/256] -> [batch, 256]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch, channels, time_steps]
            
        Returns:
        --------
        logits : torch.Tensor
            Class logits of shape [batch, num_classes]
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [batch, 256, 1] -> [batch, 256]
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("Testing SimpleRhythmCNN...")
    print("="*70)
    
    # Create model
    model = SimpleRhythmCNN(num_classes=4, input_channels=1)
    
    # Count parameters
    n_params = model.count_parameters()
    print(f"\nModel has {n_params:,} trainable parameters")
    
    # Test with dummy input (10 seconds @ 360Hz)
    batch_size = 8
    segment_length = 3600  # 10s * 360Hz
    dummy_input = torch.randn(batch_size, 1, segment_length)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output[0]}")
    
    # Test with different segment lengths
    print("\nTesting with different segment lengths:")
    for duration_sec in [5, 10, 15, 30]:
        length = int(duration_sec * 360)
        dummy = torch.randn(4, 1, length)
        with torch.no_grad():
            out = model(dummy)
        print(f"  {duration_sec:2d}s ({length:5d} samples) -> output shape: {out.shape}")
    
    print("\n" + "="*70)
    print("SimpleRhythmCNN test passed!")
    print("="*70)

