"""
Complex Implementation Package for Arrhythmia Classification

This package contains the Stage 1 implementation for per-beat ECG classification
using deep learning models on the MIT-BIH Arrhythmia Database.

Modules:
--------
- dataset: PyTorch Dataset for beat-level ECG data
- models_simple_cnn: Simple CNN architecture for baseline performance
- models_complex_cnn: Complex CNN with residual blocks for improved performance
- train: Training script with evaluation and checkpointing

Example Usage:
--------------
```python
from complex_implementation.dataset import create_patient_splits, create_dataloaders
from complex_implementation.models_simple_cnn import SimpleBeatCNN
from complex_implementation.models_complex_cnn import ComplexBeatCNN

# Create data splits
train_records, val_records, test_records = create_patient_splits()

# Create dataloaders
train_loader, val_loader, test_loader, num_classes = create_dataloaders(
    train_records, val_records, test_records
)

# Create model
model = SimpleBeatCNN(num_classes=num_classes)
```

To train a model from the command line:
```bash
python -m complex_implementation.train --model simple_cnn --epochs 50 --batch_size 64
```
"""

__version__ = '1.0.0'

from .dataset import (
    BeatDataset,
    create_patient_splits,
    create_dataloaders,
    CLASS_NAMES,
    BEAT_CLASS_MAPPING
)

from .models_simple_cnn import SimpleBeatCNN
from .models_complex_cnn import ComplexBeatCNN, ResidualBlock1D

__all__ = [
    'BeatDataset',
    'create_patient_splits',
    'create_dataloaders',
    'CLASS_NAMES',
    'BEAT_CLASS_MAPPING',
    'SimpleBeatCNN',
    'ComplexBeatCNN',
    'ResidualBlock1D',
]

