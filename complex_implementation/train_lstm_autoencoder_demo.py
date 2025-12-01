"""
Simple demonstration script for LSTM Autoencoder training

This is a minimal example showing how to train the LSTM autoencoder
on ECG beat data. For production training, use train.py instead.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models_lstm_autoencoder import LSTMAutoencoderClassifier


class DummyECGBeatsDataset(Dataset):
    """
    Dummy dataset for demonstration purposes
    
    In practice, use the BeatDataset from dataset.py with real MIT-BIH data.
    """
    
    def __init__(self, num_samples=1000, seq_len=288, num_classes=6):
        """
        Create synthetic ECG-like data
        
        Parameters:
        -----------
        num_samples : int
            Number of beat samples to generate
        seq_len : int
            Length of each beat sequence
        num_classes : int
            Number of arrhythmia classes
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Generate synthetic ECG-like beats
        # Real ECG beats would come from MIT-BIH database
        np.random.seed(42)
        
        self.beats = []
        self.labels = []
        
        for i in range(num_samples):
            # Create synthetic beat with QRS complex-like pattern
            t = np.linspace(0, 1, seq_len)
            
            # Simple synthetic ECG: baseline + QRS spike + some noise
            class_id = i % num_classes
            
            # Different classes have different morphologies
            if class_id == 0:  # Normal
                beat = 0.2 * np.sin(2 * np.pi * 3 * t) + \
                       np.exp(-((t - 0.5) ** 2) / 0.01)
            elif class_id == 1:  # Supraventricular
                beat = 0.3 * np.sin(2 * np.pi * 4 * t) + \
                       0.8 * np.exp(-((t - 0.5) ** 2) / 0.008)
            elif class_id == 2:  # Ventricular
                beat = 0.15 * np.sin(2 * np.pi * 2 * t) + \
                       1.2 * np.exp(-((t - 0.45) ** 2) / 0.015)
            else:  # Other classes
                beat = 0.1 * np.sin(2 * np.pi * 5 * t) + \
                       np.exp(-((t - 0.5) ** 2) / 0.012)
            
            # Add noise
            beat += 0.05 * np.random.randn(seq_len)
            
            # Normalize
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)
            
            self.beats.append(beat.astype(np.float32))
            self.labels.append(class_id)
        
        self.beats = np.array(self.beats)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Return a single beat and its label
        
        Returns:
        --------
        beat : torch.Tensor
            ECG beat of shape (seq_len,)
        label : int
            Class label
        """
        beat = torch.FloatTensor(self.beats[idx])
        label = self.labels[idx]
        return beat, label


def train_lstm_autoencoder_demo():
    """
    Demonstration of LSTM autoencoder training
    """
    print("=" * 70)
    print("LSTM Autoencoder - Training Demonstration")
    print("=" * 70)
    
    # Hyperparameters
    seq_len = 288          # ~0.8s at 360 Hz (MIT-BIH sampling rate)
    num_classes = 6        # Number of arrhythmia classes
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    
    # Loss weights
    alpha = 1.0  # Reconstruction loss weight
    beta = 1.0   # Classification loss weight
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    print("NOTE: This uses synthetic data for demonstration.")
    print("      For real training, use dataset.py with MIT-BIH data.")
    
    train_dataset = DummyECGBeatsDataset(num_samples=1000, seq_len=seq_len, num_classes=num_classes)
    val_dataset = DummyECGBeatsDataset(num_samples=200, seq_len=seq_len, num_classes=num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating LSTM Autoencoder model...")
    model = LSTMAutoencoderClassifier(
        seq_len=seq_len,
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2,
        latent_dim=64,
        dropout=0.3
    ).to(device)
    
    print(f"  Model parameters: {model.get_num_params():,}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Latent dimension: {model.latent_dim}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Loss weights: alpha={alpha} (reconstruction), beta={beta} (classification)")
    print("=" * 70)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_recon_loss_total = 0.0
        train_class_loss_total = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (beats, labels) in enumerate(train_loader):
            # Move to device
            beats = beats.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon, logits = model(beats)
            
            # Compute losses
            recon_loss = model.reconstruction_loss(recon, beats)
            class_loss = model.classification_loss(logits, labels)
            total_loss = alpha * recon_loss + beta * class_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss_total += total_loss.item()
            train_recon_loss_total += recon_loss.item()
            train_class_loss_total += class_loss.item()
            
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Compute average training metrics
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_recon = train_recon_loss_total / len(train_loader)
        avg_train_class = train_class_loss_total / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_recon_loss_total = 0.0
        val_class_loss_total = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for beats, labels in val_loader:
                beats = beats.to(device)
                labels = labels.to(device)
                
                # Forward pass
                recon, logits = model(beats)
                
                # Compute losses
                recon_loss = model.reconstruction_loss(recon, beats)
                class_loss = model.classification_loss(logits, labels)
                total_loss = alpha * recon_loss + beta * class_loss
                
                # Statistics
                val_loss_total += total_loss.item()
                val_recon_loss_total += recon_loss.item()
                val_class_loss_total += class_loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Compute average validation metrics
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_recon = val_recon_loss_total / len(val_loader)
        avg_val_class = val_class_loss_total / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        # Print epoch summary
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        print(f"  Train - Loss: {avg_train_loss:.4f} "
              f"(Recon: {avg_train_recon:.4f}, Class: {avg_train_class:.4f}) "
              f"| Acc: {train_accuracy:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f} "
              f"(Recon: {avg_val_recon:.4f}, Class: {avg_val_class:.4f}) "
              f"| Acc: {val_accuracy:.2f}%")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # Demonstrate latent representation extraction
    print("\nDemonstrating latent representation extraction...")
    model.eval()
    with torch.no_grad():
        sample_beats, sample_labels = next(iter(val_loader))
        sample_beats = sample_beats.to(device)
        
        # Get reconstruction, logits, and latent representation
        recon, logits, latent = model(sample_beats, return_latent=True)
        
        print(f"  Input shape: {sample_beats.shape}")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Latent representation shape: {latent.shape}")
        
        # Compute prediction accuracy for this batch
        _, predicted = logits.max(1)
        accuracy = 100. * predicted.eq(sample_labels.to(device)).sum().item() / len(sample_labels)
        print(f"  Batch accuracy: {accuracy:.2f}%")
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("\nFor real training on MIT-BIH data, use:")
    print("  python train.py --model lstm_autoencoder --epochs 50 --batch_size 64")
    print("=" * 70)


if __name__ == "__main__":
    train_lstm_autoencoder_demo()






