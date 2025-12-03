# Quick Start Guide - Stage 1 Per-Beat Classification

This guide will get you training models in under 5 minutes.

## Step 1: Ensure Data is Downloaded

Make sure you have MIT-BIH data downloaded:

```bash
# If not already downloaded:
python download_data.py
```

Data will be saved to: `data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/`

## Step 2: Train Your First Model

### Option A: Simple CNN (Fast, Baseline)

```bash
python -m complex_implementation.train --model simple_cnn --epochs 30
```

**Expected results:**
- Training time: ~5-10 min per epoch (CPU)
- Validation accuracy: ~92-95%
- Total parameters: ~65K

### Option B: Complex CNN (Better Performance)

```bash
python -m complex_implementation.train --model complex_cnn --epochs 30 --batch_size 32
```

**Expected results:**
- Training time: ~15-30 min per epoch (CPU)
- Validation accuracy: ~94-97%
- Total parameters: ~3.3M

## Step 3: Monitor Training

Watch the terminal output for:
- Training/validation loss and accuracy
- Per-class precision, recall, and F1-scores
- Best model checkpointing

```
Epoch [5/30]
----------------------------------------------------------------------
  Summary:
    Train Loss: 0.2341 | Train Acc: 93.45%
    Val Loss:   0.2012 | Val Acc:   94.23%
    
  Per-class metrics:
    Normal               Precision: 0.9645  Recall: 0.9876  F1: 0.9759
    Supraventricular     Precision: 0.7823  Recall: 0.7345  F1: 0.7576
    Ventricular          Precision: 0.8912  Recall: 0.8634  F1: 0.8771
    ...
```

## Step 4: Check Results

After training completes, find your results in:
```
complex_implementation/checkpoints/<model>_<timestamp>/
├── best_model.pth          # Best checkpoint
├── config.json             # Training config
├── training_history.json   # Per-epoch metrics
└── test_results.json       # Final test performance
```

## Common Training Commands

**Quick test run (3 epochs):**
```bash
python -m complex_implementation.train --model simple_cnn --epochs 3 --batch_size 128
```

**Longer training with custom learning rate:**
```bash
python -m complex_implementation.train \
    --model complex_cnn \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0005 \
    --weight_decay 1e-4
```

**Train on second ECG lead:**
```bash
python -m complex_implementation.train --model simple_cnn --lead 1
```

**Different window size:**
```bash
python -m complex_implementation.train --model simple_cnn --window_size 1.0
```

## Testing Individual Components

**Test dataset loading:**
```bash
python complex_implementation/dataset.py
```

**Test simple CNN:**
```bash
python complex_implementation/models_simple_cnn.py
```

**Test complex CNN:**
```bash
python complex_implementation/models_complex_cnn.py
```

**Import in Python:**
```python
from complex_implementation import SimpleBeatCNN, ComplexBeatCNN
from complex_implementation import create_patient_splits, create_dataloaders

# Create splits
train_records, val_records, test_records = create_patient_splits()

# Create dataloaders
train_loader, val_loader, test_loader, num_classes = create_dataloaders(
    train_records, val_records, test_records, batch_size=64
)

# Create model
model = SimpleBeatCNN(num_classes=num_classes)
print(f"Model has {model.get_num_params():,} parameters")
```

## Troubleshooting

**Out of memory?**
- Reduce batch size: `--batch_size 16`
- Use simple model: `--model simple_cnn`

**Training too slow?**
- Use simple model for testing: `--model simple_cnn`
- Increase batch size: `--batch_size 128`
- Reduce epochs for testing: `--epochs 10`

**Want to see all options?**
```bash
python -m complex_implementation.train --help
```

## Next Steps

1. **Analyze results**: Look at the per-class metrics to see which beat types are hardest to classify
2. **Experiment**: Try different hyperparameters and model architectures
3. **Visualize**: Create plots from `training_history.json`
4. **Improve**: Add data augmentation, weighted loss, or try ensemble methods

## Expected Performance Benchmarks

Based on MIT-BIH dataset characteristics:

| Model | Parameters | Train Time/Epoch | Val Accuracy | F1-Score |
|-------|------------|------------------|--------------|----------|
| SimpleBeatCNN | 65K | 5-10 min | 92-95% | 0.85-0.90 |
| ComplexBeatCNN | 3.3M | 15-30 min | 94-97% | 0.88-0.93 |

*Note: Actual results may vary based on data splits and hyperparameters*

## Class Distribution (Typical)

- **Normal**: ~90% (majority class)
- **Ventricular**: ~5-7%
- **Supraventricular**: ~2-3%
- **Others**: <1% (fusion, paced, unknown)

The high class imbalance makes this a challenging but realistic medical classification problem!


