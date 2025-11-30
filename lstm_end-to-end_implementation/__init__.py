"""
LSTM Rhythm Classification Package

This package contains an End-to-End LSTM implementation for ECG rhythm classification
using the MIT-BIH Arrhythmia Database.

Unlike beat-level classifiers, this model ingests raw ECG signal segments (e.g., 10 seconds)
and uses Recurrent Neural Networks (LSTM) with Attention to identify rhythm patterns
over time.

Modules:
--------
- dataset: PyTorch Dataset for loading raw ECG segments with rhythm labels.
- models_lstm_rhythm: Bidirectional LSTM with Attention mechanism.
- train: Training script with patient-wise splitting and comprehensive metrics.

Example Usage:
--------------
```python
from lstm_rhythm.dataset import create_patient_splits, create_dataloaders
from lstm_rhythm.models_lstm_rhythm import RhythmLSTM

# 1. Create Data Splits (Patient-wise to prevent leakage)
train_recs, val_recs, test_recs = create_patient_splits()

# 2. Create Loaders
train_loader, val_loader, test_loader, num_classes = create_dataloaders(
    train_recs, val_recs, test_recs,
    segment_length=10.0,
    batch_size=32
)

# 3. Initialize Model (128 units, Bidirectional)
model = RhythmLSTM(
    num_classes=num_classes,
    input_channels=1,
    hidden_dim=128,
    bidirectional=True,
    use_attention=True
)
```

To train from the command line:
```bash
python train.py --model lstm --hidden_dim 128 --epochs 50
```
"""

__version__ = '1.0.0'

# Import Dataset Utilities
from .dataset import (
    RhythmDataset,
    create_patient_splits,
    create_segment_wise_splits,
    create_dataloaders,
    CLASS_NAMES,
    RHYTHM_CLASS_MAPPING,
    RHYTHM_CLASS_MAPPING_SIMPLE
)

# Import LSTM Models
# Note: Ensure your file is named 'models_lstm_rhythm.py'
from .models_lstm import (
    RhythmLSTM,
    RhythmAttention
)

__all__ = [
    # Dataset tools
    'RhythmDataset',
    'create_patient_splits',
    'create_segment_wise_splits',
    'create_dataloaders',
    'CLASS_NAMES',
    'RHYTHM_CLASS_MAPPING',
    'RHYTHM_CLASS_MAPPING_SIMPLE',
    
    # Models
    'RhythmLSTM',
    'RhythmAttention',
]