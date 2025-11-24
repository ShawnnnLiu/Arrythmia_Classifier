# Arrhythmia Classifier

An AI-powered system for detecting and classifying cardiac arrhythmias from ECG (Electrocardiogram) data using the MIT-BIH Arrhythmia Database.

## Dataset

This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) from PhysioNet, which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Run the download script to fetch the MIT-BIH database:

```bash
python download_data.py
```

This will download all 48 ECG records (~90MB) to the `data/mitdb` directory. The download takes approximately 5-10 minutes.

## Usage

### Load and Visualize ECG Data

```python
from load_ecg_data import load_ecg_record, visualize_ecg, get_annotation_statistics

# Load a record
record, annotation = load_ecg_record('100')

# Visualize 10 seconds of ECG
visualize_ecg('100', start_sec=0, duration_sec=10)

# Get statistics about beat annotations
get_annotation_statistics('100')
```

### Run the Example Script

```bash
python load_ecg_data.py
```

This will display statistics and visualizations for record 100.

## Project Structure

```
arrhythmia_classifier/
├── data/
│   └── mitdb/          # MIT-BIH database files
├── download_data.py    # Script to download the dataset
├── load_ecg_data.py    # Helper functions for loading and visualizing ECG data
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Beat Annotation Types

The MIT-BIH database includes various beat annotations:
- **N**: Normal beat
- **L**: Left bundle branch block beat
- **R**: Right bundle branch block beat
- **A**: Atrial premature beat
- **V**: Premature ventricular contraction
- And more (see `load_ecg_data.py` for complete list)

## Next Steps

- [ ] Implement data preprocessing pipeline
- [ ] Build neural network model for arrhythmia classification
- [ ] Train and evaluate the model
- [ ] Create inference pipeline

