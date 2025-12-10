"""
Simple CNN model for per-beat ECG classification

A lightweight 1D CNN with a few Conv1d layers, suitable for quick prototyping
and establishing baseline performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBeatCNN(nn.Module):
    """
    Simple 1D CNN for beat classification
    
    Architecture:
    - 3 convolutional blocks (Conv1d -> ReLU -> MaxPool)
    - Global average pooling
    - Fully connected classifier
    """
    
    def __init__(self, num_classes: int = 6, input_channels: int = 1, dropout: float = 0.5):
        """
        Initialize SimpleBeatCNN
        
        Parameters:
        -----------
        num_classes : int
            Number of output classes
        input_channels : int
            Number of input channels (typically 1 for single-lead ECG)
        dropout : float
            Dropout probability for regularization
        """
        super(SimpleBeatCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Block 1
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, time_steps]
            
        Returns:
        --------
        out : torch.Tensor
            Output logits of shape [batch_size, num_classes]
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    # Create a simple model
    model = SimpleBeatCNN(num_classes=6)
    
    print("SimpleBeatCNN Architecture:")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Total trainable parameters: {model.get_num_params():,}")
    
    # Test with random input (batch_size=8, channels=1, time_steps=288)
    # Assuming 360 Hz sampling rate and 0.8s window -> 288 samples
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1, 288)
    
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output[0]}")

