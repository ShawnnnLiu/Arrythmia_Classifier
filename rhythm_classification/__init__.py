"""
Rhythm Classification Package for Arrhythmia Detection

This package contains an end-to-end implementation for ECG rhythm classification
using deep learning models on the MIT-BIH Arrhythmia Database.

This is distinct from beat-level classification - it analyzes longer ECG segments
(10-30 seconds) to identify rhythm patterns like atrial fibrillation, ventricular
tachycardia, etc.

Modules:
--------
- dataset: PyTorch Dataset for segment-level ECG data with rhythm labels
- models_simple_cnn: Simple CNN architecture for baseline rhythm classification
- models_complex_cnn: Complex CNN with LSTM/Attention for temporal modeling
- train: Training script with comprehensive evaluation and checkpointing

Example Usage:
--------------
```python
from rhythm_classification.dataset import create_patient_splits, create_dataloaders
from rhythm_classification.models_simple_cnn import SimpleRhythmCNN
from rhythm_classification.models_complex_cnn import ComplexRhythmCNN

# Create data splits (patient-wise, no leakage)
train_records, val_records, test_records = create_patient_splits()

# Create dataloaders
train_loader, val_loader, test_loader, num_classes = create_dataloaders(
    train_records, val_records, test_records,
    segment_length=10.0,
    segment_stride=5.0
)

# Create model
model = SimpleRhythmCNN(num_classes=num_classes)
```

To train a model from the command line:
```bash
# Patient-wise split (recommended, no data leakage)
python -m rhythm_classification.train --model simple_cnn --split patient_wise --epochs 50

# Segment-wise split (for comparison, has data leakage)
python -m rhythm_classification.train --model simple_cnn --split segment_wise --epochs 50

# Complex model with LSTM
python -m rhythm_classification.train --model complex_cnn --split patient_wise --epochs 50
```
"""

__version__ = '1.0.0'

from .dataset import (
    RhythmDataset,
    create_patient_splits,
    create_segment_wise_splits,
    create_dataloaders,
    CLASS_NAMES,
    RHYTHM_CLASS_MAPPING,
    RHYTHM_CLASS_MAPPING_SIMPLE
)

from .models_simple_cnn import SimpleRhythmCNN
from .models_complex_cnn import (
    ComplexRhythmCNN,
    ComplexRhythmCNN_NoLSTM,
    ResidualBlock1D,
    AttentionLayer
)

__all__ = [
    'RhythmDataset',
    'create_patient_splits',
    'create_segment_wise_splits',
    'create_dataloaders',
    'CLASS_NAMES',
    'RHYTHM_CLASS_MAPPING',
    'RHYTHM_CLASS_MAPPING_SIMPLE',
    'SimpleRhythmCNN',
    'ComplexRhythmCNN',
    'ComplexRhythmCNN_NoLSTM',
    'ResidualBlock1D',
    'AttentionLayer',
]

