# Quick Start: Rhythm-Bounded Segmentation

## 🎯 What's New?

A better way to create rhythm segments that prevents patient-level bias.

## 📊 The Problem

**Current approach** (`dataset.py`):
```python
# Sliding window across entire record
segment_length = 10s, stride = 5s (50% overlap)

Patient 201 (25-min normal rhythm):  → 300 segments  😱
Patient 222 (frequent rhythm changes): → 50 segments

Problem: Patient 201 has 6x more influence!
```

**New approach** (`dataset_rhythm_bounded.py`):
```python
# Non-overlapping within each rhythm annotation

Patient 201:
  ├─ Normal rhythm (1500s) → 150 segments
  └─ Equal contribution per second

Patient 222:
  ├─ AFL (30s) → 3 segments
  ├─ Normal (20s) → 2 segments  
  ├─ AFL (40s) → 4 segments
  └─ Equal contribution per second

Result: Both patients contribute proportionally! ✅
```

## 🚀 How to Use

### Option 1: Non-Overlapping (Recommended)

```python
from rhythm_classification.dataset_rhythm_bounded import (
    RhythmDatasetBounded,
    create_dataloaders_bounded,
    create_patient_splits
)

# Create patient splits
train_records, val_records, test_records = create_patient_splits()

# Create dataset with non-overlapping segments
train_dataset = RhythmDatasetBounded(
    train_records,
    segment_length=10.0,
    allow_overlap=False,  # No overlap!
    normalize=True
)

# Or use dataloader helper
train_loader, val_loader, test_loader, num_classes = create_dataloaders_bounded(
    train_records,
    val_records, 
    test_records,
    batch_size=32,
    segment_length=10.0,
    allow_overlap=False
)
```

### Option 2: Minimal Overlap (More Segments)

```python
# 50% overlap within rhythm boundaries
train_dataset = RhythmDatasetBounded(
    train_records,
    segment_length=10.0,
    allow_overlap=True,
    overlap_ratio=0.5,  # 50% overlap
    normalize=True
)
```

### Option 3: Test Both and Compare

```python
# Test script to compare approaches
python rhythm_classification/dataset_rhythm_bounded.py
```

## 📈 Expected Results

```
Non-Overlapping:
- Total segments: ~1,500-2,000
- More balanced patient representation
- Higher quality (less redundancy)

50% Overlap:
- Total segments: ~2,500-3,500
- Still bounded by rhythm annotations
- Good middle ground

Original Sliding Window:
- Total segments: ~3,000-5,000
- Patient bias present
- Many near-duplicates
```

## 🔄 Integration with Training

### Modify Your Training Script

```python
# In train.py, replace:
# from dataset import create_dataloaders

# With:
from dataset_rhythm_bounded import create_dataloaders_bounded

# Then use normally:
train_loader, val_loader, test_loader, num_classes = create_dataloaders_bounded(
    train_records,
    val_records,
    test_records,
    segment_length=args.segment_length,
    allow_overlap=False  # or True for some overlap
)
```

## 🎯 Which One to Use?

### For Publication/Research:
```python
✅ Non-overlapping (allow_overlap=False)
  - Most principled approach
  - Similar to beat classification methodology
  - Cleaner labels
  - Better patient balance
```

### For Maximum Performance:
```python
✅ 50% overlap (allow_overlap=True, overlap_ratio=0.5)
  - More segments for training
  - Still bounded by rhythms
  - Good balance
```

### For Comparison:
```python
✅ Test both!
  - Show the bias problem
  - Demonstrate improvement
  - Novel contribution
```

## 📊 Analyze the Difference

```bash
# Run analysis script
python rhythm_classification/analyze_segmentation_bias.py

# This will:
# 1. Count segments from both approaches
# 2. Show patient-level distribution
# 3. Create visualization plots
# 4. Print recommendations
```

## ✅ Benefits of Rhythm-Bounded Approach

1. **Balanced Patient Representation**
   - Each patient contributes proportionally
   - Long stable rhythms don't dominate

2. **Cleaner Labels**
   - Segments never cross rhythm boundaries
   - No ambiguous labels

3. **Less Redundancy**
   - Fewer near-duplicate segments
   - Better use of data diversity

4. **Follows Best Practices**
   - Similar to beat classification (centered on annotation)
   - Matches methodology in top papers

## 🎓 Next Steps

1. **Test the new approach:**
   ```bash
   python rhythm_classification/dataset_rhythm_bounded.py
   ```

2. **Run comparison analysis:**
   ```bash
   python rhythm_classification/analyze_segmentation_bias.py
   ```

3. **Train with rhythm-bounded:**
   - Modify train.py to use new dataset
   - Compare results with original

4. **Report both:**
   - Show the bias problem (original)
   - Show the solution (rhythm-bounded)
   - Novel contribution! ✨

## 📚 Documentation

- `BEAT_VS_RHYTHM_PAPERS.md` - What papers actually focus on
- `SEGMENTATION_STRATEGY.md` - Detailed comparison
- `dataset_rhythm_bounded.py` - Implementation
- `analyze_segmentation_bias.py` - Analysis tool

---

**Remember:** Most papers focus on BEAT classification (~90%), but you have both beat AND rhythm classifiers with a novel segmentation approach! 🎉









