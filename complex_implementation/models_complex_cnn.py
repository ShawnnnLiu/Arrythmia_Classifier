"""
Complex CNN model with residual connections for per-beat ECG classification

A deeper architecture with residual blocks to capture more complex ECG morphology
while maintaining trainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D Residual block for time-series data
    
    Uses two convolutional layers with a skip connection
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.3):
        """
        Initialize ResidualBlock1D
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size for convolutions
        stride : int
            Stride for the first convolution
        dropout : float
            Dropout probability
        """
        super(ResidualBlock1D, self).__init__()
        
        padding = kernel_size // 2
        
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (1x1 conv if dimensions change)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        """Forward pass with residual connection"""
        identity = self.skip(x)
        
        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = F.relu(out)
        
        return out


class ComplexBeatCNN(nn.Module):
    """
    Complex 1D CNN with residual blocks for beat classification
    
    Architecture:
    - Initial convolutional layer
    - 4 residual blocks with increasing channels
    - Global average pooling
    - Fully connected classifier with attention-like mechanism
    """
    
    def __init__(self, num_classes: int = 6, input_channels: int = 1, dropout: float = 0.3):
        """
        Initialize ComplexBeatCNN
        
        Parameters:
        -----------
        num_classes : int
            Number of output classes
        input_channels : int
            Number of input channels (typically 1 for single-lead ECG)
        dropout : float
            Dropout probability for regularization
        """
        super(ComplexBeatCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv_init = nn.Conv1d(input_channels, 32, kernel_size=15, padding=7)
        self.bn_init = nn.BatchNorm1d(32)
        
        # Residual blocks
        self.res_block1 = ResidualBlock1D(32, 64, kernel_size=11, stride=2, dropout=dropout)
        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=7, stride=2, dropout=dropout)
        self.res_block3 = ResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=dropout)
        self.res_block4 = ResidualBlock1D(256, 512, kernel_size=5, stride=2, dropout=dropout)
        
        # Additional conv layer for feature refinement
        self.conv_final = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm1d(256)
        
        # Global average and max pooling (concatenated)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 2, 128)  # *2 because we concat avg and max pooling
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        
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
        # Initial convolution
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Final convolution
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = F.relu(x)
        
        # Global pooling (both average and max)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers with batch norm
        x = self.dropout(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_num_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    # Create a complex model
    model = ComplexBeatCNN(num_classes=6)
    
    print("ComplexBeatCNN Architecture:")
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

