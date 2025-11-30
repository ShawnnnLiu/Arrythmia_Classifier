# 🚀 Rhythm Classification - Quick Start Guide

Get started with ECG rhythm classification in 3 simple steps!

## Step 1: Verify Setup

Make sure you have the MIT-BIH data downloaded:

```bash
# From project root, check if data exists
ls data/mitdb/

# If not, download it
python download_data.py
```

**Note:** The rhythm classification scripts expect to be run from the project root OR from inside the `rhythm_classification/` folder. The default data path is set to `../data/mitdb` to work from inside the folder.

## Step 2: Run Your First Training

### Option A: Simple Baseline (Patient-Wise) ⭐ Recommended

```bash
# From project root:
python -m rhythm_classification.train \
    --model simple_cnn \
    --split patient_wise \
    --epochs 30 \
    --batch_size 32

# OR from inside rhythm_classification/:
python train.py \
    --model simple_cnn \
    --split patient_wise \
    --epochs 30 \
    --batch_size 32
```

**Expected output:**
- Training time: ~5-10 minutes per epoch (CPU) or ~1-2 min (GPU)
- Validation accuracy: ~70-85% (depending on rhythm distribution)
- Output saved to: `rhythm_classification/checkpoints/simple_cnn_<timestamp>_patient_wise/`

### Option B: Segment-Wise (For Comparison)

```bash
python -m rhythm_classification.train \
    --model simple_cnn \
    --split segment_wise \
    --epochs 30 \
    --batch_size 32
```

**Note:** This will likely show higher accuracy due to data leakage, but it's useful for comparison.

## Step 3: View Results

After training completes, check your results:

```bash
# Navigate to checkpoint directory
cd rhythm_classification/checkpoints/simple_cnn_<timestamp>_patient_wise/

# View summary
cat SUMMARY.txt

# View plots (on Windows)
start training_curves.png
start confusion_matrix.png

# View plots (on Mac/Linux)
open training_curves.png
open confusion_matrix.png
```

## Next Steps

### 1. Train the Complex Model

For better performance with temporal modeling:

```bash
python -m rhythm_classification.train \
    --model complex_cnn \
    --split patient_wise \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0005
```

**Note:** This model has LSTM + Attention, so it's slower but more accurate.

### 2. Experiment with Segment Length

```bash
# Longer segments (more context)
python -m rhythm_classification.train \
    --segment_length 15.0 \
    --segment_stride 7.5

# Shorter segments (faster training)
python -m rhythm_classification.train \
    --segment_length 5.0 \
    --segment_stride 2.5
```

### 3. Handle Class Imbalance

```bash
# Use focal loss
python -m rhythm_classification.train \
    --loss focal \
    --focal_gamma 2.0
```

### 4. Analyze Patient Diversity

Find which patients have which rhythms:

```bash
python rhythm_classification/find_optimal_patient_split.py
```

## Troubleshooting

### "Out of memory" error

```bash
# Reduce batch size
python -m rhythm_classification.train --batch_size 16

# Or use smaller segment length
python -m rhythm_classification.train --segment_length 5.0
```

### "No rhythm annotations found"

This is normal! Not all MIT-BIH records have rhythm annotations. The dataset will automatically skip those records and use only records with rhythm data.

### Training is too slow

```bash
# Use simpler model
python -m rhythm_classification.train --model simple_cnn

# Reduce epochs for quick test
python -m rhythm_classification.train --epochs 10

# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Understanding the Output

After training, you'll see:

```
rhythm_classification/checkpoints/simple_cnn_20251129_153045_patient_wise/
├── SUMMARY.txt                 ← Read this first!
├── training_curves.png         ← Loss/accuracy plots
├── confusion_matrix.png        ← Where did the model make mistakes?
├── best_model.pth             ← Trained model weights
├── config.json                ← All settings used
└── results_summary.json       ← Detailed metrics
```

## Quick Reference: Common Commands

```bash
# Basic training (recommended start)
python -m rhythm_classification.train --model simple_cnn --split patient_wise

# Full training with all options
python -m rhythm_classification.train \
    --model complex_cnn \
    --split patient_wise \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0005 \
    --segment_length 10.0 \
    --segment_stride 5.0 \
    --loss focal

# Quick test (fast iteration)
python -m rhythm_classification.train --epochs 5 --batch_size 64

# Best performance (long training)
python -m rhythm_classification.train \
    --model complex_cnn \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001
```

## What's Next?

1. ✅ Run basic training
2. ✅ Check results in SUMMARY.txt
3. ✅ Compare patient_wise vs segment_wise splits
4. ✅ Try complex_cnn model
5. ✅ Experiment with hyperparameters
6. ✅ Analyze patient diversity
7. 🎓 Read full documentation in README.md

---

**Need help?** Check the main README.md for detailed documentation!

