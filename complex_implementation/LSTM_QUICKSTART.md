# LSTM Autoencoder Quick Start

## Installation Check

Make sure you have PyTorch and other dependencies installed:

```bash
pip install torch numpy matplotlib pandas scikit-learn wfdb
```

## Step 1: Verify Installation

Test that the LSTM model works:

```bash
cd complex_implementation
python test_lstm_integration.py
```

You should see: `✓ All tests passed!`

## Step 2: Run Demo (Optional)

Try the standalone demo with synthetic data:

```bash
python train_lstm_autoencoder_demo.py
```

This shows how the model trains without requiring real ECG data.

## Step 3: Train on Real Data

### Basic Training

```bash
python train.py --model lstm_autoencoder --epochs 50
```

### Recommended Configuration

Use curated patient-wise split with class imbalance handling:

```bash
python train.py \
    --model lstm_autoencoder \
    --curated_test 207 217 \
    --class_weights \
    --oversample \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --alpha 1.0 \
    --beta 1.0
```

## Step 4: Monitor Training

During training, you'll see:

```
Epoch [1/50]
----------------------------------------------------------------------
  Batch [50/150] Loss: 0.3245 (Recon: 0.0012, Class: 0.3233) Acc: 92.15%
  
  Summary:
    Train Loss: 0.2745 | Train Acc: 93.12%
    Val Loss:   0.2534 | Val Acc:   94.23%
```

Where:
- **Loss**: Total combined loss
- **Recon**: Reconstruction MSE
- **Class**: Classification cross-entropy
- **Acc**: Classification accuracy

## Step 5: Check Results

After training, find results in:

```
checkpoints/lstm_autoencoder_YYYYMMDD_HHMMSS/
├── best_model.pth           # Best model weights
├── training_history.json    # Loss/accuracy per epoch
├── training_curves.png      # Training plots
├── confusion_matrix.png     # Confusion matrix
└── SUMMARY.txt             # Human-readable summary
```

## Key Parameters

### Loss Weights

- **`--alpha`**: Reconstruction loss weight (default: 1.0)
- **`--beta`**: Classification loss weight (default: 1.0)

**Examples:**
```bash
# Emphasize reconstruction (for anomaly detection)
python train.py --model lstm_autoencoder --alpha 2.0 --beta 1.0

# Emphasize classification (for clinical deployment)
python train.py --model lstm_autoencoder --alpha 1.0 --beta 2.0

# Balanced (default)
python train.py --model lstm_autoencoder --alpha 1.0 --beta 1.0
```

### Common Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--model` | `simple_cnn` | Use `lstm_autoencoder` |
| `--alpha` | `1.0` | Reconstruction weight |
| `--beta` | `1.0` | Classification weight |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--curated_test` | `None` | Test patients (e.g., `207 217`) |
| `--class_weights` | `False` | Handle class imbalance |
| `--oversample` | `False` | Oversample minority classes |

## Troubleshooting

### Training too slow?
- Use GPU if available
- Reduce `--batch_size 32`
- Try CNN models for faster iteration

### Poor reconstruction?
- Increase `--alpha 2.0`
- Reduce `--lr 0.0005`
- Train longer `--epochs 100`

### Poor classification?
- Increase `--beta 2.0`
- Use `--class_weights --oversample`
- Check data split (use `--curated_test`)

### Out of memory?
- Reduce `--batch_size 32` or `16`
- Reduce `--num_workers 2`

## Next Steps

1. **Read full documentation**: `LSTM_AUTOENCODER_GUIDE.md`
2. **Compare with CNNs**: Train CNN models for comparison
3. **Tune hyperparameters**: Adjust α, β, learning rate
4. **Analyze results**: Check reconstruction quality and classification metrics

## Example Workflow

```bash
# 1. Test installation
python test_lstm_integration.py

# 2. Run quick demo
python train_lstm_autoencoder_demo.py

# 3. Train on real data (short test)
python train.py --model lstm_autoencoder --epochs 10 --batch_size 64

# 4. Full training with best settings
python train.py \
    --model lstm_autoencoder \
    --curated_test 207 217 \
    --class_weights \
    --oversample \
    --alpha 1.0 \
    --beta 1.0 \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 64

# 5. Compare with CNN
python train.py --model complex_cnn --curated_test 207 217 --epochs 50
```

## Getting Help

- **Detailed guide**: See `LSTM_AUTOENCODER_GUIDE.md`
- **Implementation details**: See `LSTM_IMPLEMENTATION_SUMMARY.md`
- **General info**: See `README.md`
- **Model code**: See `models_lstm_autoencoder.py`

## Success Indicators

Your model is training well if:

✓ Reconstruction loss decreases (ideally < 0.01)  
✓ Classification accuracy increases (> 90% on normal beats)  
✓ No NaN losses  
✓ Validation accuracy tracks training accuracy  
✓ Test accuracy is reasonable (> 85%)  

Happy training! 🚀






