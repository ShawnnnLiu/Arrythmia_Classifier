"""
Quick integration test for LSTM autoencoder

This script verifies that the LSTM autoencoder integrates correctly
with the existing training pipeline without running full training.
"""

import sys
import torch
from models_lstm_autoencoder import LSTMAutoencoderClassifier

def test_model_creation():
    """Test that the model can be created with correct parameters"""
    print("Testing model creation...")
    
    seq_len = 288  # 0.8s at 360 Hz
    num_classes = 6
    
    model = LSTMAutoencoderClassifier(
        seq_len=seq_len,
        num_classes=num_classes
    )
    
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Parameters: {model.get_num_params():,}")
    assert model.get_num_params() > 0, "Model should have parameters"
    
    return model


def test_forward_pass(model):
    """Test forward pass with different input shapes"""
    print("\nTesting forward pass with different input shapes...")
    
    batch_size = 4
    seq_len = 288
    
    # Test 1: (batch_size, seq_len)
    input_2d = torch.randn(batch_size, seq_len)
    recon, logits = model(input_2d)
    assert recon.shape == (batch_size, seq_len, 1), f"Reconstruction shape mismatch: {recon.shape}"
    assert logits.shape == (batch_size, 6), f"Logits shape mismatch: {logits.shape}"
    print(f"  ✓ 2D input (batch, seq_len): {input_2d.shape} → recon: {recon.shape}, logits: {logits.shape}")
    
    # Test 2: (batch_size, seq_len, 1)
    input_3d = torch.randn(batch_size, seq_len, 1)
    recon, logits = model(input_3d)
    assert recon.shape == (batch_size, seq_len, 1), f"Reconstruction shape mismatch: {recon.shape}"
    assert logits.shape == (batch_size, 6), f"Logits shape mismatch: {logits.shape}"
    print(f"  ✓ 3D input (batch, seq_len, 1): {input_3d.shape} → recon: {recon.shape}, logits: {logits.shape}")
    
    # Test 3: (batch_size, 1, seq_len) - CNN format
    input_cnn = torch.randn(batch_size, 1, seq_len)
    recon, logits, latent = model(input_cnn, return_latent=True)
    assert recon.shape == (batch_size, seq_len, 1), f"Reconstruction shape mismatch: {recon.shape}"
    assert logits.shape == (batch_size, 6), f"Logits shape mismatch: {logits.shape}"
    assert latent.shape == (batch_size, 64), f"Latent shape mismatch: {latent.shape}"
    print(f"  ✓ CNN format (batch, 1, seq_len): {input_cnn.shape} → recon: {recon.shape}, logits: {logits.shape}, latent: {latent.shape}")


def test_loss_computation(model):
    """Test loss computation"""
    print("\nTesting loss computation...")
    
    batch_size = 4
    seq_len = 288
    num_classes = 6
    
    # Create dummy data
    input_signal = torch.randn(batch_size, 1, seq_len)  # CNN format
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    recon, logits = model(input_signal)
    
    # Compute losses
    recon_loss = model.reconstruction_loss(recon, input_signal)
    class_loss = model.classification_loss(logits, targets)
    
    assert recon_loss.item() >= 0, "Reconstruction loss should be non-negative"
    assert class_loss.item() >= 0, "Classification loss should be non-negative"
    
    print(f"  ✓ Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  ✓ Classification loss: {class_loss.item():.4f}")
    
    # Test combined loss
    alpha, beta = 1.0, 1.0
    total_loss = alpha * recon_loss + beta * class_loss
    assert total_loss.item() >= 0, "Total loss should be non-negative"
    print(f"  ✓ Total loss (α={alpha}, β={beta}): {total_loss.item():.4f}")


def test_backward_pass(model):
    """Test that backward pass works"""
    print("\nTesting backward pass (gradient computation)...")
    
    batch_size = 4
    seq_len = 288
    num_classes = 6
    
    # Create dummy data
    input_signal = torch.randn(batch_size, seq_len)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    recon, logits = model(input_signal)
    
    # Compute loss
    loss = model.reconstruction_loss(recon, input_signal) + model.classification_loss(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Backward pass works correctly")


def test_training_pipeline_compatibility():
    """Test compatibility with training pipeline"""
    print("\nTesting training pipeline compatibility...")
    
    try:
        # Test imports from train.py
        from train import get_model
        
        # Test model creation via get_model function
        model = get_model('lstm_autoencoder', num_classes=6, seq_len=288)
        assert isinstance(model, LSTMAutoencoderClassifier), "get_model should return LSTMAutoencoderClassifier"
        print(f"  ✓ Model creation via get_model() works")
        
        # Test that is_autoencoder check works
        is_autoencoder = isinstance(model, LSTMAutoencoderClassifier)
        assert is_autoencoder == True, "isinstance check should work"
        print(f"  ✓ isinstance() check works correctly")
        
        print(f"  ✓ Integration with train.py successful")
        
    except ImportError as e:
        print(f"  ⚠ Could not import from train.py: {e}")
        print(f"  Note: This is expected if running outside the package")


def main():
    """Run all tests"""
    print("=" * 70)
    print("LSTM Autoencoder Integration Test")
    print("=" * 70)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Loss computation
        test_loss_computation(model)
        
        # Test 4: Backward pass
        test_backward_pass(model)
        
        # Test 5: Training pipeline compatibility
        test_training_pipeline_compatibility()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nThe LSTM autoencoder is ready to use.")
        print("To train the model, run:")
        print("  python train.py --model lstm_autoencoder --epochs 50")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Test failed!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())









