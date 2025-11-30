# Class Distribution in Train/Val/Test Splits

## ❌ Problem: Imbalanced Class Distribution

Your current split **does NOT have even class distribution** across train/val/test sets!

### Example from Recent Training

```
Test Set Class Distribution:
  Normal:           9,608 samples  ✅
  Supraventricular:   204 samples  ⚠️  Very small
  Ventricular:        631 samples  ⚠️  Small
  Fusion:               0 samples  ❌ MISSING!
  Paced:            2,077 samples  ✅
  Unknown:              0 samples  ❌ MISSING!
```

**Problems:**
- ❌ **Fusion** and **Unknown** classes have 0 samples in test set
- Can't evaluate model performance on these classes
- Reported metrics (precision, recall, F1) are 0.0000 for missing classes

## Why This Happens

### Simple Random Split (Current Default)

```python
# Shuffles ALL patients randomly
records = shuffle(['100', '101', '102', ..., '234'])

# Splits sequentially
train = records[0:36]    # First 75%
val   = records[36:42]   # Next 12.5%  
test  = records[42:48]   # Last 12.5%
```

**Problem:** Different patients have different arrhythmias!
- Patient 100 might have mostly Normal + Ventricular beats
- Patient 207 might have Fusion beats
- Patient 217 might have Unknown beats

If all patients with Fusion beats happen to fall in the training set, the test set will have **zero** Fusion beats!

## ✅ Solution: Stratified Split

The new stratified split ensures **rare classes are distributed across all splits**.

### How It Works

```python
# Step 1: Identify which patients have rare classes
records_with_fusion = ['207', '219', ...]
records_with_unknown = ['102', '112', ...]
records_without_rare = ['100', '101', ...]

# Step 2: Split EACH group proportionally
rare_train = 75% of records_with_rare
rare_val   = 12.5% of records_with_rare
rare_test  = 12.5% of records_with_rare

common_train = 75% of records_without_rare
common_val   = 12.5% of records_without_rare
common_test  = 12.5% of records_without_rare

# Step 3: Combine
train = rare_train + common_train
val   = rare_val + common_val
test  = rare_test + common_test
```

**Result:** All splits get some patients with rare classes!

## 🔍 Check Your Current Distribution

Run this command to see the distribution:

```bash
cd complex_implementation
python check_split_distribution.py
```

Output will show:
- Class counts per split
- Which classes are missing from which splits
- Percentage differences across splits

### Compare Stratified vs Non-Stratified

```bash
# Check random split (current default)
python check_split_distribution.py

# Check stratified split (improved)
python check_split_distribution.py --stratified
```

## 🚀 Using Stratified Split in Training

### Enable Stratified Split

Add the `--stratified` flag:

```bash
python train.py --stratified --model complex_cnn --epochs 20
```

This will:
1. Analyze all records to find which have rare classes
2. Distribute rare-class records across train/val/test
3. Ensure all classes appear in all splits (if possible)

### Full Example

```bash
cd complex_implementation
conda activate gpu5070

# Train with stratified split
python train.py \
    --model complex_cnn \
    --epochs 50 \
    --batch_size 128 \
    --num_workers 12 \
    --stratified
```

## 📊 Expected Improvements

### Before (Random Split)
```
Test Set:
  Fusion:   0 samples    → Precision: 0.0000, Recall: 0.0000
  Unknown:  0 samples    → Precision: 0.0000, Recall: 0.0000
```

### After (Stratified Split)
```
Test Set:
  Fusion:   15-30 samples  → Can evaluate performance!
  Unknown:  5-15 samples   → Can evaluate performance!
```

## ⚠️ Important Notes

### 1. Patient-Wise Split is Maintained

Both methods maintain **patient-wise splitting**:
- No patient appears in multiple splits
- Tests generalization to new patients
- Critical for medical ML

### 2. Perfect Balance Not Guaranteed

Stratification **helps** but doesn't guarantee perfect balance because:
- Limited number of patients (48 total)
- Some classes are very rare
- Can't split a patient across sets

### 3. When to Use Each

**Random Split (Default):**
- Quick prototyping
- When all classes are common
- When you want pure random patient selection

**Stratified Split (`--stratified`):**
- ✅ **Final model training**
- ✅ When rare classes exist (Fusion, Unknown)
- ✅ When fair evaluation across all classes is critical
- ✅ **For reporting/publication**

## 🎯 Recommendation

**Always use `--stratified` for final training runs!**

```bash
# Recommended for final results
python train.py --stratified --epochs 50
```

This ensures:
- ✅ All classes represented in test set
- ✅ Fair evaluation across all arrhythmia types
- ✅ Meaningful precision/recall/F1 scores
- ✅ Publishable results

## 📈 Verification Workflow

1. **Check distribution before training:**
   ```bash
   python check_split_distribution.py --stratified
   ```

2. **Train with stratified split:**
   ```bash
   python train.py --stratified --epochs 50
   ```

3. **Review test metrics:**
   - Check `per_class_metrics.csv`
   - Verify all classes have non-zero support
   - Confirm F1-scores are meaningful

## Common Questions

**Q: Will stratified split change my random seed results?**
A: Yes, the split will be different, but still reproducible with the same seed.

**Q: Can I compare models trained with different splits?**
A: No, you should use the same split method and seed for fair comparison.

**Q: What if a class only exists in 1-2 patients?**
A: Stratification will try to distribute them, but with only 6 test records, some imbalance is unavoidable. Consider combining very rare classes.

**Q: Does this affect training time?**
A: Minimal - only adds ~1-2 seconds at the start for distribution analysis.

## 🔬 Beat-Wise Split (For Prototyping Only!)

### What is Beat-Wise Split?

Instead of splitting patients, beat-wise split pools ALL beats from ALL patients, then randomly splits the beats:

```python
# Beat-wise (NOT recommended for production!)
All 109,494 beats → shuffle → 75/12.5/12.5 split

# Result: Same patient's beats in multiple splits
Patient 100: some beats in train, some in val, some in test
```

### ⚠️ Critical Warnings

**DO NOT USE for:**
- ❌ Final model evaluation
- ❌ Clinical validation
- ❌ Publications / papers
- ❌ FDA submissions
- ❌ Real-world deployment

**ONLY USE for:**
- ✓ Quick prototyping
- ✓ Debugging models
- ✓ Establishing upper-bound performance
- ✓ Teaching / learning

### Why It Creates Data Leakage

```python
# Patient 100 has 2,000 beats with unique ECG morphology

Beat-wise split assigns:
  Train: Beats #1, #3, #5, ... (1,500 beats from Patient 100)
  Test:  Beats #2, #4, #6, ... (500 beats from Patient 100)

# Model learns Patient 100's unique patterns in training
# Then "predicts" on more beats from SAME patient
# This is artificially easy → inflated accuracy!
```

### Testing Beat-Wise Split

```bash
# Check distribution with beat-wise split
python check_split_distribution.py --beat_wise

# Compare: patient-wise vs beat-wise
python check_split_distribution.py --stratified
python check_split_distribution.py --beat_wise
```

### Training with Beat-Wise Split

```bash
# ⚠️ For prototyping ONLY!
python train.py --beat_wise --epochs 20

# Results will be MUCH better but INVALID for clinical use
```

### Expected Performance Difference

**Patient-Wise (Correct):**
- Test Accuracy: 82-85% ✅ Realistic
- Rare Class F1: 0.30-0.50 ✅ Honest

**Beat-Wise (Inflated):**
- Test Accuracy: 95-98% 🎈 Too good to be true!
- Rare Class F1: 0.80-0.90 🎈 Overly optimistic

### When to Use Each

| Use Case | Patient-Wise | Beat-Wise |
|----------|--------------|-----------|
| **Final results** | ✅ Always | ❌ Never |
| **Clinical deployment** | ✅ Required | ❌ Invalid |
| **Publications** | ✅ Standard | ❌ Rejected |
| **Quick prototyping** | ⚠️ Slow | ✅ Fast |
| **Model debugging** | ⚠️ Hard | ✅ Easy |
| **Upper-bound performance** | ❌ Not applicable | ✅ Shows ceiling |

### Proper Reporting

If you use beat-wise for comparison:

```markdown
## Results

### Patient-Wise Split (Primary - Clinically Valid)
- Test Accuracy: 83.5%
- Tests generalization to NEW patients
- ✅ Suitable for clinical deployment

### Beat-Wise Split (Supplementary - Data Leakage)
- Test Accuracy: 96.2%
- ⚠️ Includes data leakage (same patient in train/test)
- ❌ NOT suitable for clinical use
- Shows upper-bound performance if patient-specific tuning were possible
```

## Summary

✅ **Use `--stratified` flag for fair class distribution**
✅ **Check distribution with `check_split_distribution.py`**
✅ **Verify all classes appear in test set**
✅ **Report results using patient-wise split**
⚠️  **Use `--beat_wise` ONLY for prototyping, clearly mark as such**

This ensures your model evaluation is fair and all arrhythmia types are properly tested!

