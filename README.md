# ECG Arrhythmia Classifier

An AI-powered system for detecting and classifying cardiac arrhythmias from ECG (Electrocardiogram) data using deep learning on the MIT-BIH Arrhythmia Database.

**Course:** CS184A - Introduction to Machine Learning  
**Dataset:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) from PhysioNet

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Demo: Training & Testing](#demo-training--testing)
5. [Project Structure](#project-structure)
6. [Models](#models)
7. [Data Split Strategies](#data-split-strategies)
8. [Results](#results)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Arrythmia_Classifier.git
cd Arrythmia_Classifier

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
python download_data.py

# 5. Run demo training (Simple CNN, 10 epochs)
python -m complex_implementation.train --model simple_cnn --epochs 10 --batch_size 64
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~500MB disk space for dataset
- (Optional) NVIDIA GPU with CUDA for faster training

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Arrythmia_Classifier.git
   cd Arrythmia_Classifier
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Requirements

All dependencies are listed in `requirements.txt`:
- **PyTorch** (>=2.0.0) - Deep learning framework
- **wfdb** (>=4.1.0) - PhysioNet data access
- **numpy**, **pandas** - Data processing
- **matplotlib**, **seaborn** - Visualization
- **scikit-learn** - Evaluation metrics
- **tqdm** - Progress bars

---

## Dataset Setup

### Option 1: Automatic Download (Recommended)

Run the download script to fetch the MIT-BIH database (~90MB):

```bash
python download_data.py
```

This downloads all 48 ECG records to `data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/`.

**Note:** Download takes approximately 5-10 minutes depending on internet speed.

### Option 2: Manual Download

1. Go to [PhysioNet MIT-BIH Database](https://physionet.org/content/mitdb/1.0.0/)
2. Download the ZIP file
3. Extract to: `data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/`

### Verify Dataset

After downloading, you should see files like:
```
data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/
├── 100.atr    # Annotations
├── 100.dat    # Signal data
├── 100.hea    # Header
├── 101.atr
├── 101.dat
...
```

---

## Demo: Training & Testing

### Demo 1: Quick Training Demo (5 minutes)

Train a simple CNN model for 10 epochs to verify everything works:

```bash
python -m complex_implementation.train --model simple_cnn --epochs 10 --batch_size 64
```

**Expected output:**
- Training progress with loss/accuracy per epoch
- Per-class precision, recall, F1-score
- Model saved to `complex_implementation/checkpoints/`

### Demo 2: Full Model Training

**Simple CNN (Baseline, ~65K parameters):**
```bash
python -m complex_implementation.train --model simple_cnn --epochs 30 --batch_size 64
```

**Complex CNN (Better Performance, ~3.3M parameters):**
```bash
python -m complex_implementation.train --model complex_cnn --epochs 30 --batch_size 32
```

**LSTM Autoencoder (Reconstruction + Classification, ~1.5M parameters):**
```bash
python -m complex_implementation.train --model lstm_autoencoder --epochs 30 --batch_size 64
```

### Demo 3: Explore the Data

Open the Jupyter notebook for data exploration:

```bash
jupyter notebook explore_data.ipynb
```

Or run the visualization script:

```bash
python load_ecg_data.py
```

### Demo 4: Compare Trained Models

After training multiple models, compare their performance:

```bash
python training_results/compare_models.py
```

This generates comparison plots and tables in `training_results/`.

---

## Project Structure

```
Arrythmia_Classifier/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── download_data.py             # Dataset download script
├── load_ecg_data.py             # ECG data loading utilities
├── explore_data.ipynb           # Data exploration notebook
│
├── complex_implementation/      # Beat classification models & training
│   ├── train.py                 # Main training script
│   ├── dataset.py               # PyTorch Dataset for beat windows
│   ├── models_simple_cnn.py     # Simple CNN architecture
│   ├── models_complex_cnn.py    # Complex CNN with residual blocks
│   ├── models_lstm_autoencoder.py  # LSTM autoencoder model
│   ├── checkpoints/             # Saved models and results
│   └── QUICKSTART.md            # Quick start guide
│
├── training_results/            # Model comparison tools
│   ├── compare_models.py        # Compare multiple models
│   └── quick_view.py            # Quick results viewer
│
└── data/                        # Dataset directory
    └── mit-bih-arrhythmia-database-1.0.0/
        └── mit-bih-arrhythmia-database-1.0.0/
            ├── 100.atr, 100.dat, 100.hea
            ├── 101.atr, 101.dat, 101.hea
            └── ... (48 records total)
```

---

## Models

### 1. Simple CNN (`simple_cnn`)
- **Parameters:** ~65,000
- **Architecture:** 3 convolutional blocks + fully connected layers
- **Training time:** ~5-10 min/epoch (CPU)
- **Best for:** Quick prototyping, baseline performance

### 2. Complex CNN (`complex_cnn`)
- **Parameters:** ~3.3M
- **Architecture:** 4 residual blocks + dual pooling + FC layers
- **Training time:** ~15-30 min/epoch (CPU)
- **Best for:** Better performance, production use

### 3. LSTM Autoencoder (`lstm_autoencoder`)
- **Parameters:** ~1.5M
- **Architecture:** LSTM encoder → latent space → decoder + classifier
- **Training time:** ~20-40 min/epoch (CPU)
- **Best for:** Temporal modeling, anomaly detection

### Beat Classes (AAMI Standard)

| Class | Name | Beat Types |
|-------|------|------------|
| 0 | Normal | N, L, R, e, j |
| 1 | Supraventricular | A, a, J, S |
| 2 | Ventricular | V, E |
| 3 | Fusion | F |
| 4 | Paced | /, f |
| 5 | Unknown | Q, ? |

---

## Data Split Strategies

Choosing the right data split is **critical** for valid evaluation. The MIT-BIH database contains 48 records from 47 patients. We provide four split strategies:

### 1. Beat-Wise Split (`--beat_wise`) ⚠️ DATA LEAKAGE

```bash
python -m complex_implementation.train --model simple_cnn --beat_wise
```

- **How it works:** Pools all beats from all patients, then randomly splits into train/val/test
- **Problem:** Same patient's beats appear in train AND test sets
- **Result:** Artificially high accuracy (~98%+) that won't generalize
- **Use case:** ❌ **NOT recommended** - only for debugging or quick sanity checks
- **Clinical validity:** ❌ Invalid - model memorizes patient-specific patterns

### 2. Patient-Wise Stratified (`--stratified`) ✅ Valid but Uneven

```bash
python -m complex_implementation.train --model simple_cnn --stratified
```

- **How it works:** Splits by patient while trying to balance class distribution
- **Advantage:** No data leakage, attempts to balance rare classes
- **Limitation:** May result in uneven split ratios (not exactly 75/12.5/12.5)
- **Use case:** ✅ Good for initial experiments
- **Clinical validity:** ✅ Valid - patients don't appear across splits

### 3. Curated Patient-Wise (`--curated_val` + `--curated_test`) ✅ Manual Control

```bash
python -m complex_implementation.train --model simple_cnn \
    --curated_val 203 208 213 233 104 223 \
    --curated_test 215 205 210 200 214 219
```

- **How it works:** You manually specify which patients go to val/test sets
- **Advantage:** Full control over split, can ensure specific classes are represented
- **Use case:** ✅ Good when you know which patients have rare arrhythmias
- **Clinical validity:** ✅ Valid - completely patient-wise

### 4. Optimal Patient-Wise (Recommended) ✅ Best Statistical Split

```bash
# First, find the optimal split:
python complex_implementation/find_optimal_patient_split.py

# Then use the recommended patients:
python train.py --model simple_cnn `
  --curated_val 116 118 200 201 214 219 `
  --curated_test 104 105 109 124 202 223 `
  --class_weights --epochs 20
```

- **How it works:** Algorithm finds patient combination that best matches overall class distribution
- **Advantage:** Statistically optimal representation while maintaining patient separation
- **Target split:** 75% train / 12.5% val / 12.5% test (by beat count)
- **Use case:** ✅ **Recommended for final results and publication**
- **Clinical validity:** ✅ Valid - rigorous patient-wise separation

### Split Strategy Comparison

| Strategy | Data Leakage | Class Balance | Split Ratio | Recommended |
|----------|--------------|---------------|-------------|-------------|
| `--beat_wise` | ⚠️ YES | ✅ Good | ✅ Exact | ❌ No |
| `--stratified` | ✅ No | ✅ Good | ⚠️ Variable | ✅ Yes |
| `--curated_val/test` | ✅ No | ⚠️ Manual | ⚠️ Manual | ✅ Yes |
| Optimal (curated) | ✅ No | ✅ Best | ✅ ~75/12.5/12.5 | ✅ **Best** |

### Why Patient-Wise Splitting Matters

In medical applications, **data leakage** leads to overoptimistic results:

- **With leakage (beat-wise):** Model sees Patient A's beats in training, then is tested on different beats from the same Patient A → learns patient-specific patterns → ~98% accuracy
- **Without leakage (patient-wise):** Model never sees Patient A during training, tested on Patient A → must learn generalizable arrhythmia patterns → ~94% accuracy (but clinically meaningful)

**Always use patient-wise splits for valid medical AI evaluation!**

---

## Results

### Training Output

After training, results are saved to:
```
complex_implementation/checkpoints/<model>_<timestamp>/
├── best_model.pth           # Best model weights
├── config.json              # Training configuration
├── training_history.json    # Per-epoch metrics
├── results_summary.json     # Final test results
├── confusion_matrix.png     # Confusion matrix plot
├── training_curves.png      # Loss/accuracy curves
├── per_class_metrics.csv    # Detailed metrics
└── SUMMARY.txt              # Human-readable summary
```

### Expected Performance

| Model | Test Accuracy |
|-------|--------------|
| Simple CNN | 90% |
| Complex CNN | 75% |
| LSTM Autoencoder | 91% |

---

## Command Reference

### Training Commands

```bash
# Basic training
python -m complex_implementation.train --model simple_cnn --epochs 30

# With custom hyperparameters
python -m complex_implementation.train \
    --model complex_cnn \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --weight_decay 1e-4

# LSTM with reconstruction emphasis
python -m complex_implementation.train \
    --model lstm_autoencoder \
    --epochs 50 \
    --alpha 2.0 \
    --beta 1.0

# See all options
python -m complex_implementation.train --help
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `simple_cnn` | Model: `simple_cnn`, `complex_cnn`, `lstm_autoencoder` |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--window_size` | `0.8` | Window around R-peak (seconds) |
| `--seed` | `42` | Random seed for reproducibility |

### Data Split Arguments

| Argument | Description |
|----------|-------------|
| `--beat_wise` | ⚠️ Beat-wise split (DATA LEAKAGE - not for publication) |
| `--stratified` | Patient-wise with class balancing (may have uneven ratios) |
| `--curated_val P1 P2...` | Manually specify validation patients |
| `--curated_test P1 P2...` | Manually specify test patients |

**Recommended:** Use `--curated_val` and `--curated_test` with optimal patients from `find_optimal_patient_split.py`

---

## Troubleshooting

### "Out of memory" error
```bash
# Reduce batch size
python -m complex_implementation.train --batch_size 16
```

### Training too slow
```bash
# Use simpler model
python -m complex_implementation.train --model simple_cnn

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Dataset not found
```bash
# Re-download dataset
python download_data.py
```

---

## Authors

CS184A Course Project Team

---

## License

This project is for educational purposes as part of the CS184A course at UC Irvine.
