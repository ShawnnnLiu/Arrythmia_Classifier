"""
LSTM Autoencoder for ECG beat classification

This model combines reconstruction (via autoencoder) and classification,
inspired by Liu et al. "Arrhythmia classification of LSTM autoencoder based 
on time series anomaly detection" (Biomed Signal Process Control 2022).

Architecture:
- LSTM Encoder: Compresses ECG beat into latent representation
- LSTM Decoder: Reconstructs original beat from latent code
- Classifier Head: Predicts arrhythmia class from latent code

The model is trained with a combined loss:
    Total Loss = alpha * MSE(reconstruction) + beta * CrossEntropy(classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMAutoencoderClassifier(nn.Module):
    """
    LSTM-based autoencoder with classification head for ECG arrhythmia detection
    
    This model learns to:
    1. Encode ECG beats into a compact latent representation
    2. Reconstruct the original beat from the latent code
    3. Classify the beat into arrhythmia categories using the latent code
    
    The dual objective helps the model learn meaningful representations that
    are both reconstructive (preserving beat morphology) and discriminative
    (separating arrhythmia classes).
    """
    
    def __init__(self, 
                 seq_len: int = 288,
                 num_classes: int = 6,
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 latent_dim: int = 64,
                 dropout: float = 0.3):
        """
        Initialize LSTM Autoencoder Classifier
        
        Parameters:
        -----------
        seq_len : int
            Length of input sequence (number of time steps)
        num_classes : int
            Number of output classes for classification
        input_size : int
            Dimensionality of each time step (typically 1 for single-lead ECG)
        hidden_size : int
            Number of features in LSTM hidden state
        num_layers : int
            Number of stacked LSTM layers
        latent_dim : int
            Dimensionality of the bottleneck latent representation
        dropout : float
            Dropout probability for regularization
        """
        super(LSTMAutoencoderClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        
        # ===== ENCODER =====
        # LSTM encoder: processes the input sequence and outputs hidden states
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bottleneck: compress final hidden state to latent representation
        self.fc_enc = nn.Linear(hidden_size, latent_dim)
        
        # ===== DECODER =====
        # LSTM decoder: reconstructs the sequence from the latent code
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Time-distributed linear layer: maps decoder outputs back to original dimensionality
        self.fc_dec = nn.Linear(hidden_size, input_size)
        
        # ===== CLASSIFIER HEAD =====
        # Multi-layer perceptron for classification from latent code
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
        """
        # Pass through LSTM encoder
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        
        # Use the final hidden state from the top LSTM layer
        # h_n[-1] shape: (batch_size, hidden_size)
        final_hidden = h_n[-1]
        
        # Compress to latent dimension
        # latent shape: (batch_size, latent_dim)
        latent = self.fc_enc(final_hidden)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to original sequence
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
            
        Returns:
        --------
        recon : torch.Tensor
            Reconstructed sequence of shape (batch_size, seq_len, input_size)
        """
        batch_size = latent.size(0)
        
        # Repeat latent vector along time axis to match sequence length
        # This allows the decoder to "see" the latent code at each time step
        # latent_expanded shape: (batch_size, seq_len, latent_dim)
        latent_expanded = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Pass through LSTM decoder
        # decoder_out shape: (batch_size, seq_len, hidden_size)
        decoder_out, _ = self.decoder_lstm(latent_expanded)
        
        # Apply time-distributed linear layer to each time step
        # recon shape: (batch_size, seq_len, input_size)
        recon = self.fc_dec(decoder_out)
        
        return recon
    
    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Classify from latent representation
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
            
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        """
        logits = self.classifier(latent)
        return logits
    
    def forward(self, 
                x: torch.Tensor, 
                return_latent: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: encode, decode, and classify
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            or (batch_size, 1, seq_len) for CNN compatibility
        return_latent : bool
            If True, also return the latent representation
            
        Returns:
        --------
        recon : torch.Tensor
            Reconstructed sequence of shape (batch_size, seq_len, 1)
        logits : torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        latent : torch.Tensor (optional)
            Latent representation of shape (batch_size, latent_dim), 
            only returned if return_latent=True
        """
        # Handle different input shapes (for compatibility with CNN inputs)
        if x.dim() == 2:
            # Shape: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(1) == 1:
            # Shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
            # This is the CNN format, need to transpose
            x = x.transpose(1, 2)
        
        # Now x has shape (batch_size, seq_len, input_size)
        
        # Encode to latent representation
        latent = self.encode(x)
        
        # Decode to reconstruction
        recon = self.decode(latent)
        
        # Classify from latent
        logits = self.classify(latent)
        
        if return_latent:
            return recon, logits, latent
        else:
            return recon, logits
    
    def reconstruction_loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (Mean Squared Error)
        
        Parameters:
        -----------
        recon : torch.Tensor
            Reconstructed sequence of shape (batch_size, seq_len, input_size)
        x : torch.Tensor
            Original input of shape (batch_size, seq_len) or (batch_size, seq_len, input_size)
            or (batch_size, input_size, seq_len)
            
        Returns:
        --------
        loss : torch.Tensor
            Scalar MSE loss
        """
        # Ensure x has the same shape as recon
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2)
        
        # MSE loss
        loss = F.mse_loss(recon, x)
        return loss
    
    def classification_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss (Cross Entropy)
        
        Parameters:
        -----------
        logits : torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        targets : torch.Tensor
            Ground truth labels of shape (batch_size,)
            
        Returns:
        --------
        loss : torch.Tensor
            Scalar cross entropy loss
        """
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def get_num_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    # Create model
    seq_len = 288  # ~0.8s at 360 Hz
    num_classes = 6
    model = LSTMAutoencoderClassifier(seq_len=seq_len, num_classes=num_classes)
    
    print("LSTM Autoencoder Classifier Architecture:")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Total trainable parameters: {model.get_num_params():,}")
    
    # Test with different input shapes
    batch_size = 8
    
    # Test shape 1: (batch_size, seq_len)
    print("\n" + "=" * 70)
    print("Testing with different input shapes:")
    print("=" * 70)
    
    input_2d = torch.randn(batch_size, seq_len)
    print(f"\nInput shape (2D): {input_2d.shape}")
    recon, logits = model(input_2d)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Classification logits shape: {logits.shape}")
    
    # Test shape 2: (batch_size, seq_len, 1)
    input_3d = torch.randn(batch_size, seq_len, 1)
    print(f"\nInput shape (3D): {input_3d.shape}")
    recon, logits = model(input_3d)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Classification logits shape: {logits.shape}")
    
    # Test shape 3: (batch_size, 1, seq_len) - CNN format
    input_cnn = torch.randn(batch_size, 1, seq_len)
    print(f"\nInput shape (CNN format): {input_cnn.shape}")
    recon, logits, latent = model(input_cnn, return_latent=True)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Classification logits shape: {logits.shape}")
    print(f"Latent representation shape: {latent.shape}")
    
    # Test loss computation
    print("\n" + "=" * 70)
    print("Testing loss computation:")
    print("=" * 70)
    
    targets = torch.randint(0, num_classes, (batch_size,))
    recon_loss = model.reconstruction_loss(recon, input_cnn)
    class_loss = model.classification_loss(logits, targets)
    
    print(f"\nReconstruction loss: {recon_loss.item():.4f}")
    print(f"Classification loss: {class_loss.item():.4f}")
    
    # Example combined loss
    alpha, beta = 1.0, 1.0
    total_loss = alpha * recon_loss + beta * class_loss
    print(f"Total loss (alpha={alpha}, beta={beta}): {total_loss.item():.4f}")

