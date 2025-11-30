# Rhythm Classification - End-to-End ECG Rhythm Detection

This directory contains a complete end-to-end implementation for **rhythm classification** from ECG signals using the MIT-BIH Arrhythmia Database.

## 🎯 What is Rhythm Classification?

Unlike **beat classification** (which classifies individual heartbeats as Normal, PVC, etc.), **rhythm classification** analyzes longer ECG segments (10-30 seconds) to identify underlying **rhythm patterns** such as:

- **Normal Sinus Rhythm (NSR)**: Regular, healthy heart rhythm
- **Atrial Fibrillation (AFIB)**: Irregular atrial activity
- **Atrial Flutter (AFL)**: Rapid but regular atrial activity  
- **Ventricular Tachycardia (VT)**: Fast ventricular rhythm
- **Pre-excitation**: Abnormal conduction pathway

This is an **end-to-end solution** that directly maps ECG segments → Rhythm labels, without using beat classifications as an intermediate step.

## 📁 Files

| File | Description |
|------|-------------|
| `dataset.py` | PyTorch Dataset for rhythm segments with annotation parsing |
| `models_simple_cnn.py` | Lightweight CNN baseline (~200K parameters) |
| `models_complex_cnn.py` | Advanced CNN-LSTM with Attention (~3M parameters) |
| `train.py` | Complete training pipeline with evaluation metrics |
| `find_optimal_patient_split.py` | Utility to analyze rhythm diversity across patients |
| `__init__.py` | Package initialization |
| `checkpoints/` | Saved model weights and training results |

## 🚀 Quick Start

### 1. Train with Patient-Wise Split (Recommended)

**No data leakage** - each patient appears in only one split:

```bash
python -m rhythm_classification.train \
    --model simple_cnn \
    --split patient_wise \
    --epochs 50 \
    --batch_size 32 \
    --segment_length 10.0
```

### 2. Train with Segment-Wise Split (For Comparison)

**⚠️ Data leakage** - same patient's segments appear in train/val/test:

```bash
python -m rhythm_classification.train \
    --model simple_cnn \
    --split segment_wise \
    --epochs 50 \
    --batch_size 32
```

### 3. Train Complex Model with LSTM + Attention

For better temporal modeling:

```bash
python -m rhythm_classification.train \
    --model complex_cnn \
    --split patient_wise \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0005
```

## 🎛️ Command-Line Arguments

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/mitdb` | Directory containing MIT-BIH data |
| `--segment_length` | `10.0` | ECG segment length in seconds |
| `--segment_stride` | `5.0` | Stride between segments (creates overlap) |
| `--lead` | `0` | ECG lead to use (0 or 1) |

### Model Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--model` | `simple_cnn` | `simple_cnn`, `complex_cnn`, `complex_cnn_nolstm` | Model architecture |
| `--dropout` | `0.5` | - | Dropout rate for regularization |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size (reduce if out of memory) |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `1e-5` | L2 regularization |
| `--optimizer` | `adam` | Optimizer (`adam`, `sgd`, `adamw`) |
| `--loss` | `crossentropy` | Loss function (`crossentropy`, `focal`) |

### Split Strategy

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--split` | `patient_wise` | `patient_wise`, `segment_wise` | Data splitting strategy |
| `--train_ratio` | `0.75` | - | Fraction of data for training |
| `--val_ratio` | `0.125` | - | Fraction of data for validation |

## 🏗️ Architecture Details

### SimpleRhythmCNN

**Lightweight baseline model:**
- 4 convolutional blocks (conv → batch norm → relu → maxpool)
- Global average pooling
- 2 fully connected layers
- **~200K parameters**
- **Use for**: Quick experimentation, baseline performance

### ComplexRhythmCNN

**Advanced temporal modeling:**
- 4 residual convolutional blocks for feature extraction
- Bidirectional LSTM (2 layers) for temporal patterns
- Attention mechanism for important time-step weighting
- Fully connected classifier
- **~3M parameters**
- **Use for**: Best performance, capturing long-term dependencies

### ComplexRhythmCNN_NoLSTM

**Pure convolutional approach:**
- 4 residual blocks
- Dual global pooling (average + max)
- Faster than LSTM version
- **~1M parameters**
- **Use for**: Balance between speed and performance

## 📊 Rhythm Classes

The implementation uses a **simplified 4-class mapping** for better class balance:

| Class ID | Class Name | Rhythm Annotations Included |
|----------|------------|---------------------------|
| 0 | Normal | (N, (SBR, (NOD |
| 1 | Atrial Arrhythmia | (AFIB, (AFL, (AB, (SVTA |
| 2 | Ventricular Arrhythmia | (VT, (VFL, (B, (T |
| 3 | Pre-excitation | (PREX |

## 🔍 Data Splits: Patient-Wise vs Segment-Wise

### Patient-Wise Split ✅ (Recommended)

**What it does:**
- Divides **patients** into train/val/test
- Each patient appears in only ONE split
- No data leakage

**Pros:**
- ✅ Clinically valid evaluation
- ✅ Tests generalization to new patients
- ✅ Publishable results

**Cons:**
- ⚠️ May have class imbalance if rhythms cluster in specific patients
- ⚠️ Smaller effective dataset size

**Example:**
```python
train_records = ['100', '101', '102', ...]  # 36 patients
val_records = ['103', '104', ...]           # 6 patients  
test_records = ['105', '106', ...]          # 6 patients
```

### Segment-Wise Split ⚠️ (For Comparison Only)

**What it does:**
- Pools ALL segments from ALL patients
- Randomly divides **segments** into train/val/test
- Same patient's segments appear in multiple splits

**Pros:**
- ✅ Better class balance
- ✅ Larger effective dataset
- ✅ Establishes upper-bound performance

**Cons:**
- ❌ **Data leakage** - not clinically valid
- ❌ Overly optimistic performance estimates
- ❌ NOT suitable for publication

**Use only for:**
- Quick prototyping
- Comparing with patient-wise results
- Understanding upper-bound performance

## 📈 Output Structure

Training produces comprehensive outputs:

```
rhythm_classification/checkpoints/<model>_<timestamp>_<split>/
├── config.json                 # Training configuration
├── training_history.json       # Loss/accuracy per epoch
├── training_history.csv        # Same as JSON but in CSV format
├── best_model.pth             # Best model weights (highest val accuracy)
├── checkpoint_epoch_10.pth    # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── results_summary.json       # Final test set metrics
├── training_curves.png        # Loss and accuracy plots
├── confusion_matrix.png       # Confusion matrix visualization
└── SUMMARY.txt                # Human-readable summary
```

## 🔬 Example Training Output

```
Training ECG Rhythm Classifier
======================================================================

Using device: cuda

Creating patient_wise splits...
  Training:   36 records (75%)
  Validation: 6 records (12%)
  Test:       6 records (12%)

Loading data...
  Loaded  245 segments from record 100
  Loaded  312 segments from record 101
  ...

Rhythm Dataset Statistics:
  Total segments: 4,523
  Records: 36 unique
  Segment length: 10.0s
  Segment stride: 5.0s

  Class distribution:
    Normal                    (class 0):   3421 (75.63%)
    Atrial_Arrhythmia        (class 1):    654 (14.46%)
    Ventricular_Arrhythmia   (class 2):    398 ( 8.80%)
    Pre-excitation           (class 3):     50 ( 1.11%)

Creating model: simple_cnn
Model has 197,324 trainable parameters

Starting training...
======================================================================

Epoch [1/50]
----------------------------------------------------------------------
  Batch [50/142] Loss: 0.8234 Acc: 72.34%
  ...

  Summary:
    Train Loss: 0.7123 | Train Acc: 75.23%
    Val Loss:   0.6891 | Val Acc:   77.45%
    
  Per-class metrics (validation):
    Normal                     Precision: 0.8234  Recall: 0.9123  F1: 0.8656
    Atrial_Arrhythmia         Precision: 0.7456  Recall: 0.6789  F1: 0.7105
    ...
```

## 🛠️ Analyzing Patient Rhythm Diversity

Find patients with diverse rhythms for optimal test sets:

```bash
python rhythm_classification/find_optimal_patient_split.py
```

This analyzes which patients have which rhythms and suggests patients to hold out for testing.

## 📊 Evaluation Metrics

For each epoch and final test evaluation:

- **Loss**: Cross-entropy or Focal loss
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**:
  - Precision
  - Recall  
  - F1-score
  - Support (number of samples)
- **Macro averages**: Unweighted average across all classes
- **Confusion matrix**: Visual breakdown of predictions

## 💡 Tips for Better Performance

### 1. Handling Class Imbalance

Rhythm annotations are sparse and imbalanced:

```bash
# Use focal loss (down-weights easy examples)
python -m rhythm_classification.train --loss focal --focal_gamma 2.0

# Adjust segment stride for more overlap
python -m rhythm_classification.train --segment_stride 2.5  # More segments
```

### 2. Hyperparameter Tuning

```bash
# Try different segment lengths
python -m rhythm_classification.train --segment_length 15.0  # Longer context

# Adjust learning rate
python -m rhythm_classification.train --lr 0.0005  # Lower for complex models

# Increase dropout for regularization
python -m rhythm_classification.train --dropout 0.6
```

### 3. Model Selection

| Scenario | Recommended Model |
|----------|------------------|
| Quick baseline | `simple_cnn` |
| Best performance | `complex_cnn` (LSTM + Attention) |
| Limited memory | `complex_cnn_nolstm` |
| Real-time inference | `simple_cnn` |

## 🔧 Troubleshooting

**Out of memory errors:**
```bash
# Reduce batch size
python -m rhythm_classification.train --batch_size 16

# Use simpler model
python -m rhythm_classification.train --model simple_cnn

# Reduce segment length
python -m rhythm_classification.train --segment_length 5.0
```

**No rhythm annotations found:**
- Not all MIT-BIH records have rhythm annotations
- The dataset will automatically skip records without rhythm data
- Check `dataset.py` output for which records were loaded

**Poor validation accuracy:**
- Train for more epochs
- Try focal loss for class imbalance
- Increase segment overlap (reduce stride)
- Use complex model for better temporal modeling

## 🔬 Using Trained Models

### Load and Use a Trained Model

```python
import torch
from rhythm_classification.models_simple_cnn import SimpleRhythmCNN

# Load model
model = SimpleRhythmCNN(num_classes=4)
checkpoint = torch.load('path/to/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
    probabilities = torch.softmax(predictions, dim=1)
```

### Example: Classify a New ECG Segment

```python
import numpy as np
import torch
import wfdb

# Load ECG data
record = wfdb.rdrecord('data/mitdb/100')
signal = record.p_signal[:3600, 0]  # 10 seconds @ 360Hz, lead 0

# Normalize
signal = (signal - signal.mean()) / signal.std()

# Convert to tensor [1, 1, 3600]
signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(signal_tensor)
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

print(f"Predicted rhythm: {CLASS_NAMES[predicted_class]}")
print(f"Confidence: {probs[0, predicted_class].item():.2%}")
```

## 📚 Key Differences from Beat Classification

| Aspect | Beat Classification | Rhythm Classification |
|--------|-------------------|---------------------|
| **Input** | Single beat (~0.8s) | ECG segment (10-30s) |
| **Labels** | Beat annotations (N, V, etc.) | Rhythm annotations ((AFIB, (VT, etc.) |
| **Goal** | Classify individual beats | Identify rhythm pattern |
| **Model** | Simpler (few beats) | Temporal modeling (LSTM/Attention) |
| **Clinical Use** | Detect abnormal beats | Diagnose rhythm disorders |

## 🎓 Next Steps

After rhythm classification, consider:

- **Multi-lead**: Use both ECG leads simultaneously
- **Patient-specific models**: Fine-tune on individual patients
- **Ensemble methods**: Combine beat and rhythm predictions
- **Real-time detection**: Streaming rhythm classification
- **Transfer learning**: Pre-train on larger ECG datasets
- **Explainability**: Attention visualization, GradCAM

## 📄 License

This implementation is for educational purposes as part of the CS184A course project.

## 🙏 Acknowledgments

- **MIT-BIH Arrhythmia Database**: Moody & Mark (2001)
- **PhysioNet**: Goldberger et al. (2000)
- **WFDB Python package**: for ECG data access

---

**Questions or issues?** Check the documentation or review example training runs in the `checkpoints/` directory.

