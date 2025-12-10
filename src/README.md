# Stage 1: Per-Beat ECG Classification

This directory contains the Stage 1 implementation for per-beat arrhythmia classification using deep learning on the MIT-BIH Arrhythmia Database.

## Overview

The goal of Stage 1 is to classify individual heartbeats into different arrhythmia categories. Given a fixed-length ECG window centered around an R-peak, the model predicts the beat type (e.g., Normal, Ventricular, Supraventricular, etc.).

## Files

- **`dataset.py`**: PyTorch Dataset for loading per-beat ECG windows
  - `BeatDataset`: Main dataset class that extracts beat windows around R-peaks
  - `create_patient_splits()`: Creates patient-wise train/val/test splits
  - `create_dataloaders()`: Helper to create PyTorch DataLoaders
  - Beat class mapping based on AAMI EC57 standard

- **`models_simple_cnn.py`**: Simple baseline CNN model
  - `SimpleBeatCNN`: Lightweight 1D CNN with 3 conv blocks
  - ~65K parameters, fast to train
  - Good for establishing baseline performance

- **`models_complex_cnn.py`**: Complex CNN with residual connections
  - `ComplexBeatCNN`: Deeper 1D CNN with 4 residual blocks
  - `ResidualBlock1D`: Custom residual block for time-series
  - ~3.3M parameters, better feature extraction
  - Uses both average and max global pooling

- **`models_lstm_autoencoder.py`**: LSTM autoencoder for beat classification
  - `LSTMAutoencoderClassifier`: Combines reconstruction and classification
  - ~1.5M parameters, learns temporal dependencies
  - Dual loss: reconstruction (MSE) + classification (CrossEntropy)
  - Inspired by Liu et al. 2022 (Biomed Signal Process Control)

- **`train.py`**: Training script with full pipeline
  - Supports multiple model architectures
  - Patient-wise data splitting
  - Comprehensive evaluation metrics (per-class precision, recall, F1)
  - Model checkpointing and training history logging
  - Learning rate scheduling

- **`__init__.py`**: Package initialization with exports

## Beat Classes

The implementation groups MIT-BIH beat annotations into 6 classes:

| Class ID | Class Name | Beat Types Included |
|----------|------------|-------------------|
| 0 | Normal | N, L, R, e, j |
| 1 | Supraventricular | A, a, J, S |
| 2 | Ventricular | V, E |
| 3 | Fusion | F |
| 4 | Paced | /, f |
| 5 | Unknown | Q, ? |

This grouping follows clinical conventions and helps with class balance.

## Quick Start

### 1. Test Individual Components

Test the simple CNN model:
```bash
python complex_implementation/models_simple_cnn.py
```

Test the complex CNN model:
```bash
python complex_implementation/models_complex_cnn.py
```

Test the LSTM autoencoder model:
```bash
python complex_implementation/models_lstm_autoencoder.py
```

Test the dataset:
```bash
python complex_implementation/dataset.py
```

Run LSTM autoencoder demo:
```bash
python complex_implementation/train_lstm_autoencoder_demo.py
```

### 2. Train a Model

**Train Simple CNN (baseline):**
```bash
python -m complex_implementation.train --model simple_cnn --epochs 50 --batch_size 64
```

**Train Complex CNN (better performance):**
```bash
python -m complex_implementation.train --model complex_cnn --epochs 50 --batch_size 32 --lr 0.0005
```

**Train LSTM Autoencoder (reconstruction + classification):**
```bash
python -m complex_implementation.train --model lstm_autoencoder --epochs 50 --batch_size 64 --alpha 1.0 --beta 1.0
```

**LSTM Autoencoder with curated patient split:**
```bash
python -m complex_implementation.train --model lstm_autoencoder --curated_test 207 217 --epochs 50 --class_weights
```

**With custom parameters:**
```bash
python -m complex_implementation.train \
    --model complex_cnn \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --window_size 0.8 \
    --lead 0 \
    --weight_decay 1e-5 \
    --seed 42
```

### 3. Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `simple_cnn` | Model architecture (`simple_cnn`, `complex_cnn`, or `lstm_autoencoder`) |
| `--alpha` | `1.0` | Reconstruction loss weight (LSTM autoencoder only) |
| `--beta` | `1.0` | Classification loss weight (LSTM autoencoder only) |
| `--data_dir` | (auto-detected) | Directory containing MIT-BIH data |
| `--window_size` | `0.8` | Window size around R-peak in seconds |
| `--lead` | `0` | Which ECG lead to use (0 or 1) |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `64` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay for regularization |
| `--seed` | `42` | Random seed for reproducibility |
| `--checkpoint_dir` | `complex_implementation/checkpoints` | Directory to save checkpoints |
| `--save_freq` | `10` | Save checkpoint every N epochs |
| `--num_workers` | `4` | Number of data loading workers |

## Output

Training produces the following output structure:

```
complex_implementation/checkpoints/<model>_<timestamp>/
├── config.json              # Training configuration
├── training_history.json    # Loss and accuracy per epoch
├── best_model.pth          # Best model checkpoint (highest val accuracy)
├── checkpoint_epoch_10.pth # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── test_results.json       # Final test set evaluation
```

### Training Output Example

```
Training Per-Beat ECG Classifier
======================================================================

Using device: cuda

Creating patient-wise train/val/test splits...
  Training:   33 records (70%)
  Validation: 7 records (15%)
  Test:       8 records (15%)

Loading data...
Dataset Statistics:
  Total beats: 75234
  Class distribution:
    Normal               (class 0):  67821 (90.15%)
    Supraventricular     (class 1):   2154 ( 2.86%)
    Ventricular          (class 2):   5012 ( 6.66%)
    ...

Creating model: simple_cnn
Model has 64,966 trainable parameters

Starting training...
======================================================================

Epoch [1/50]
----------------------------------------------------------------------
  Batch [50/1176] Loss: 0.4523 Acc: 89.23%
  ...

  Summary:
    Train Loss: 0.3245 | Train Acc: 91.23%
    Val Loss:   0.2891 | Val Acc:   92.45%
    
  Per-class metrics:
    Normal               Precision: 0.9623  Recall: 0.9854  F1: 0.9737
    Supraventricular     Precision: 0.7234  Recall: 0.6891  F1: 0.7058
    ...
```

## Model Architectures

### SimpleBeatCNN
- **Parameters**: ~65K
- **Architecture**: 3 conv blocks + fully connected
- **Training time**: ~5-10 minutes per epoch (CPU)
- **Use case**: Quick prototyping, baseline performance

### ComplexBeatCNN
- **Parameters**: ~3.3M
- **Architecture**: 4 residual blocks + dual pooling + fully connected
- **Training time**: ~15-30 minutes per epoch (CPU)
- **Use case**: Better performance, captures complex ECG morphology

### LSTMAutoencoderClassifier (NEW)
- **Parameters**: ~1.5M
- **Architecture**: LSTM encoder → latent representation → LSTM decoder + classifier
  - Encoder: 2-layer LSTM (hidden_size=128) + bottleneck (latent_dim=64)
  - Decoder: 2-layer LSTM to reconstruct signal
  - Classifier: MLP head for arrhythmia classification
- **Loss**: Dual objective (α × reconstruction MSE + β × classification CE)
- **Training time**: ~20-40 minutes per epoch (CPU, slower due to sequential LSTM)
- **Use case**: Learning interpretable representations, anomaly detection, temporal modeling
- **See**: `LSTM_AUTOENCODER_GUIDE.md` for detailed documentation

## Data Splits

The implementation uses **patient-wise splitting** to ensure:
- No patient appears in both training and test sets
- Better evaluation of model generalization
- More realistic clinical scenario

Default split ratios:
- Training: 70% of patients
- Validation: 15% of patients
- Test: 15% of patients

## Evaluation Metrics

For each epoch, the training script computes:
- **Loss**: Cross-entropy loss
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, Recall, F1-score for each beat class
- **Macro-average**: Average metrics across all classes

## Using Trained Models

To load and use a trained model:

```python
import torch
from complex_implementation.models_simple_cnn import SimpleBeatCNN

# Load model
model = SimpleBeatCNN(num_classes=6)
checkpoint = torch.load('path/to/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

## LSTM Autoencoder Model

The LSTM autoencoder combines reconstruction and classification in a single model. This approach:
- Learns meaningful latent representations of ECG morphology
- Can detect anomalies via reconstruction error
- Provides interpretability through signal reconstruction
- Models temporal dependencies in ECG signals

**Key files:**
- `models_lstm_autoencoder.py` - Model implementation
- `LSTM_AUTOENCODER_GUIDE.md` - Comprehensive guide and hyperparameter tuning
- `train_lstm_autoencoder_demo.py` - Standalone demo with synthetic data

**Quick start:**
```bash
# Basic training
python train.py --model lstm_autoencoder --epochs 50

# With class imbalance handling
python train.py --model lstm_autoencoder --curated_test 207 217 --class_weights --oversample

# Adjust loss weights
python train.py --model lstm_autoencoder --alpha 2.0 --beta 1.0  # Emphasize reconstruction
python train.py --model lstm_autoencoder --alpha 1.0 --beta 2.0  # Emphasize classification
```

## Next Steps

After Stage 1 (per-beat classification), possible extensions include:
- **Stage 2**: Rhythm-level classification (analyzing sequences of beats)
- **Multi-lead**: Using both ECG leads simultaneously
- **Attention mechanisms**: Adding attention layers for interpretability
- **Real-time inference**: Optimizing for low-latency prediction
- **Data augmentation**: Adding noise, shifts, and scaling
- **Transfer learning**: Pre-training on larger ECG datasets
- **Reconstruction analysis**: Visualize LSTM autoencoder reconstructions for anomaly detection

## Tips for Better Performance

1. **Class imbalance**: The Normal class dominates (~90% of beats)
   - Consider using weighted loss functions
   - Try oversampling minority classes
   - Use focal loss for hard examples

2. **Hyperparameter tuning**:
   - Start with simple_cnn to establish baseline
   - Use learning rate scheduling (already included)
   - Experiment with window sizes (0.6s - 1.0s)
   - Try different batch sizes based on available memory

3. **Data augmentation**:
   - Add random noise to simulate real-world conditions
   - Apply small time shifts
   - Normalize per-beat or per-record

4. **Model selection**:
   - Use SimpleBeatCNN for quick iterations
   - Use ComplexBeatCNN for final performance
   - Monitor overfitting with train/val loss curves

## Troubleshooting

**Out of memory errors:**
- Reduce batch size (`--batch_size 32` or `--batch_size 16`)
- Use simple_cnn instead of complex_cnn
- Reduce number of workers (`--num_workers 2`)

**Poor validation accuracy:**
- Train for more epochs
- Try different learning rates
- Check for data leakage (patient-wise split should prevent this)
- Verify data preprocessing

**Training too slow:**
- Increase batch size if memory allows
- Use GPU if available
- Reduce number of workers if CPU-bound
- Use simple_cnn for faster iterations

## License

This implementation is for educational purposes as part of the CS184A course project.


