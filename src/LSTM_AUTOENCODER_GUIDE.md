# LSTM Autoencoder for ECG Beat Classification

## Overview

This implementation provides an LSTM-based autoencoder for ECG arrhythmia classification, inspired by Liu et al. "Arrhythmia classification of LSTM autoencoder based on time series anomaly detection" (Biomed Signal Process Control 2022).

The model combines two objectives:
1. **Reconstruction**: Learn to reconstruct ECG beats using an autoencoder
2. **Classification**: Classify beats into arrhythmia categories using the learned latent representation

This dual-task approach helps the model learn more robust and meaningful representations of ECG morphology.

## Architecture

### Model Components

```
Input (ECG beat) → Encoder → Latent Representation → Decoder → Reconstruction
                                    ↓
                              Classifier → Class Predictions
```

**Encoder:**
- 2-layer bidirectional LSTM (hidden_size=128)
- Fully connected bottleneck layer (latent_dim=64)

**Decoder:**
- Latent vector repeated along time axis
- 2-layer LSTM (hidden_size=128)
- Time-distributed linear layer to reconstruct signal

**Classifier:**
- Fully connected layers: 64 → ReLU → Dropout → num_classes
- Uses same latent representation as decoder

### Loss Function

The model is trained with a combined loss:

```
Total Loss = α × MSE(reconstruction) + β × CrossEntropy(classification)
```

Where:
- `α` (alpha) = weight for reconstruction loss
- `β` (beta) = weight for classification loss

Default values: α = 1.0, β = 1.0

## Usage

### Basic Training Command

```bash
python train.py --model lstm_autoencoder --epochs 50 --batch_size 64
```

### With Curated Patient-Wise Split (Recommended)

```bash
python train.py \
    --model lstm_autoencoder \
    --curated_test 207 217 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001
```

### Adjusting Loss Weights

You can adjust the balance between reconstruction and classification:

```bash
# Emphasize reconstruction
python train.py --model lstm_autoencoder --alpha 2.0 --beta 1.0

# Emphasize classification
python train.py --model lstm_autoencoder --alpha 1.0 --beta 2.0

# Focus mainly on classification (like a regular classifier)
python train.py --model lstm_autoencoder --alpha 0.1 --beta 1.0
```

### With Class Imbalance Handling

```bash
python train.py \
    --model lstm_autoencoder \
    --curated_test 207 217 \
    --class_weights \
    --oversample \
    --alpha 1.0 \
    --beta 1.0 \
    --epochs 50
```

### Full Example with All Options

```bash
python train.py \
    --model lstm_autoencoder \
    --data_dir ../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0 \
    --curated_test 207 217 \
    --window_size 0.8 \
    --lead 0 \
    --class_weights \
    --oversample \
    --alpha 1.0 \
    --beta 1.0 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --seed 42 \
    --checkpoint_dir checkpoints \
    --num_workers 4
```

## Model Parameters

The LSTM autoencoder has approximately **~1.5M trainable parameters** (depending on num_classes and seq_len).

Key hyperparameters:
- `seq_len`: Automatically calculated from window_size × sampling_rate (360 Hz)
  - Default: 0.8s × 360 Hz = 288 samples
- `hidden_size`: 128 (LSTM hidden dimension)
- `num_layers`: 2 (stacked LSTM layers)
- `latent_dim`: 64 (bottleneck dimension)
- `dropout`: 0.3 (regularization)

## Input Format Compatibility

The model handles different input shapes automatically:
- `(batch_size, seq_len)` → converted to `(batch_size, seq_len, 1)`
- `(batch_size, seq_len, 1)` → used directly
- `(batch_size, 1, seq_len)` → transposed to `(batch_size, seq_len, 1)` (CNN format)

This ensures compatibility with the existing CNN-based pipeline.

## Training Output

During training, you'll see additional metrics for the autoencoder:

```
Batch [50/150] Loss: 0.3245 (Recon: 0.0012, Class: 0.3233) Acc: 92.15%
```

Where:
- **Loss**: Total combined loss (α × Recon + β × Class)
- **Recon**: Reconstruction MSE loss
- **Class**: Classification cross-entropy loss
- **Acc**: Classification accuracy

## Hyperparameter Tuning Recommendations

### Loss Weight Tuning

1. **Start with balanced weights** (α=1.0, β=1.0)
2. **Monitor both losses** during training
3. **Adjust based on goals:**
   - If classification is poor but reconstruction is good → increase β
   - If reconstruction is poor → increase α
   - If both are poor → try adjusting learning rate

### Learning Rate

The autoencoder may benefit from different learning rates than CNNs:
- Start with `lr=0.001` (default)
- If training is unstable → reduce to `lr=0.0005`
- If convergence is slow → increase to `lr=0.002`

### Batch Size

- Default: 64
- Larger batches (128-256) can help with LSTM training stability
- Smaller batches (32) may help with generalization

## Expected Performance

Based on similar architectures in literature:
- **Reconstruction MSE**: Should decrease to ~0.001-0.005
- **Classification Accuracy**: Comparable to CNN models (85-95% on normal/abnormal)
- **Training Time**: ~2-3x slower than CNN (due to sequential LSTM processing)

## Comparison with CNN Models

| Feature | Simple CNN | Complex CNN | LSTM Autoencoder |
|---------|------------|-------------|------------------|
| Parameters | ~50K | ~1.5M | ~1.5M |
| Training Speed | Fast | Medium | Slow |
| Reconstruction | No | No | Yes |
| Temporal Modeling | Local | Multi-scale | Sequential |
| Interpretability | Medium | Low | High (via reconstruction) |

## Advanced Usage

### Accessing Latent Representations

The model's latent representations can be extracted for visualization or downstream tasks:

```python
model.eval()
with torch.no_grad():
    recon, logits, latent = model(input_signal, return_latent=True)
    # latent has shape (batch_size, 64)
```

### Custom Loss Weights Per Epoch

You can implement curriculum learning by adjusting α and β during training:
- Start with high α (e.g., 2.0) to learn good reconstructions
- Gradually increase β (e.g., to 2.0) to focus on classification

(This requires modifying the training loop in train.py)

## Troubleshooting

### Issue: Very high reconstruction loss
- **Solution**: Increase α or reduce learning rate
- Check input normalization (should be enabled by default)

### Issue: Poor classification accuracy
- **Solution**: Increase β, try class weights or oversampling
- Ensure you're using patient-wise or curated hybrid split

### Issue: Training is very slow
- **Solution**: Reduce batch size, use fewer workers, or use GPU if available
- LSTM is inherently slower than CNN due to sequential processing

### Issue: NaN losses
- **Solution**: Reduce learning rate (try lr=0.0005 or lr=0.0001)
- Check for extreme values in input data

## References

1. Liu, W., et al. (2022). "Arrhythmia classification of LSTM autoencoder based on time series anomaly detection." Biomedical Signal Processing and Control.

2. MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/

## File Structure

```
complex_implementation/
├── models_lstm_autoencoder.py    # LSTM autoencoder model
├── train.py                       # Training script (supports LSTM)
├── dataset.py                     # Dataset loader
├── LSTM_AUTOENCODER_GUIDE.md     # This file
└── checkpoints/                   # Saved models
    └── lstm_autoencoder_YYYYMMDD_HHMMSS/
        ├── best_model.pth
        ├── training_curves.png
        ├── confusion_matrix.png
        └── SUMMARY.txt
```

## Next Steps

1. **Train baseline model**: Start with default hyperparameters
2. **Analyze results**: Check both reconstruction quality and classification metrics
3. **Tune hyperparameters**: Adjust α, β, learning rate based on results
4. **Compare with CNNs**: Use same data split to compare performance
5. **Visualize reconstructions**: Plot original vs reconstructed beats to verify model learning

