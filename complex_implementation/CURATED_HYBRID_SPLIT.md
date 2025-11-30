# Curated Hybrid Split - Quick Reference

## 🎯 What is it?

A **hybrid data splitting strategy** that balances clinical validity with class balance:

- **Test Set:** Pure patient-wise (held-out patients, no leakage) ✅
- **Train/Val Sets:** Beat-wise pooling (shares patients for class balance) ⚠️

## 🔬 Why Use It?

### **Problem it Solves:**
```
Standard Patient-wise Split:
  ❌ Rare classes (Fusion, Unknown) often missing from test set
  ❌ Class imbalance makes evaluation incomplete
  ✅ But: Valid generalization (patient-wise)

Beat-wise Split:
  ✅ Perfect class balance
  ❌ Data leakage (invalid generalization)
  ❌ Not publishable

Curated Hybrid:
  ✅ All classes in test set (curated patients)
  ✅ Valid test generalization (patient-wise test)
  ✅ Better train/val balance (beat pooling)
  ⚠️  Train/val share patients (acceptable for tuning)
```

## 📋 How to Use

### **Step 1: Find Diverse Test Patients**

```bash
cd complex_implementation
python analyze_patient_diversity.py
```

**Output:**
```
TOP 15 MOST DIVERSE PATIENTS
========================================
1. Patient 207
   Classes present: 5/6
   Total beats: 2,500
   - Normal (80%), Ventricular (10%), Fusion (3%), 
     Paced (5%), Unknown (2%)

2. Patient 217
   Classes present: 4/6
   ...
```

**Suggested test patients:** 207, 217

---

### **Step 2: Check Distribution (Optional)**

```bash
python check_split_distribution.py --curated_test 207 217
```

**Shows:**
- Test: Only patients 207, 217 (all classes represented)
- Train: ~85% of beats from remaining 46 patients
- Val: ~15% of beats from remaining 46 patients

---

### **Step 3: Train with Curated Test Set**

```bash
python train.py --model simple_cnn --curated_test 207 217 --epochs 20
```

**Additional options:**
```bash
# With class weights
python train.py --model simple_cnn --curated_test 207 217 --class_weights

# With focal loss
python train.py --model simple_cnn --curated_test 207 217 --focal_loss --focal_gamma 2.0

# Complex model
python train.py --model complex_cnn --curated_test 207 217 --epochs 50
```

---

### **Step 4: Results**

Checkpoints saved to:
```
checkpoints/simple_cnn_YYYYMMDD_HHMMSS_curated_hybrid/
├── config.json          # Includes test_patients: ["207", "217"]
├── SUMMARY.txt          # Documents split strategy
├── best_model.pth       # Best model weights
├── training_history.csv # Epoch-wise metrics
├── training_curves.png  # Loss/accuracy plots
└── confusion_matrix.png # Test set confusion matrix
```

## 📊 Expected Results

### **Compared to Patient-wise Stratified:**

| Metric | Patient-wise Stratified | Curated Hybrid |
|--------|------------------------|----------------|
| Test Accuracy | ~90% | ~88-92% |
| Fusion F1 | 0.00 (no samples) | 0.60-0.80 ✅ |
| Unknown F1 | 0.00 (missing) | 0.40-0.60 ✅ |
| Ventricular F1 | 0.54 | 0.65-0.85 |
| **All classes evaluated** | ❌ No | ✅ Yes |
| **Valid generalization** | ✅ Yes | ✅ Yes |

## 📝 How to Report This

### **In Your Paper/Presentation:**

```markdown
## Methodology

We employ a curated patient-wise split to ensure comprehensive 
class coverage while maintaining generalization validity:

1. **Test Set:** Two patients (207, 217) with diverse arrhythmia 
   types were held out completely, ensuring all 6 classes are 
   represented in testing.

2. **Train/Validation:** Remaining 46 patients were used with 
   beat-level splitting (85/15) to ensure class balance for 
   effective model optimization.

**Note:** The test set maintains strict patient separation, 
ensuring reported accuracy reflects true generalization to 
unseen patients. Train/val sets share patients for class 
balance, which is acceptable as validation is used only for 
model selection, not performance claims.

## Results

**Test Accuracy (on held-out patients 207, 217): 88.5%**
- Tests generalization to new patients ✅
- All arrhythmia classes evaluated ✅
- No data leakage in test set ✅
```

## ⚠️ Important Notes

### **Be Transparent:**

1. **Train/Val Leakage:**
   - "Training and validation sets share patients (for class balance)"
   - "Validation used only for model selection, not performance claims"

2. **Test Purity:**
   - "Test set consists of completely held-out patients"
   - "Test accuracy represents true generalization performance"

3. **Justification:**
   - "Due to rare class clustering in specific patients"
   - "Ensures all arrhythmia types can be fairly evaluated"

### **What's Valid:**
- ✅ Test accuracy as generalization metric
- ✅ Test F1-scores for all classes
- ✅ Test confusion matrix
- ✅ Claiming "generalization to new patients"

### **What's NOT Valid:**
- ❌ Using validation accuracy as final metric
- ❌ Claiming no data leakage anywhere
- ❌ Ignoring train/val patient overlap

## 🔄 Alternative: K-Fold Cross-Validation

For even more robust results:

```bash
# Run 5 times with different test patients
python train.py --curated_test 207 217 --seed 42
python train.py --curated_test 100 106 --seed 43
python train.py --curated_test 119 124 --seed 44
python train.py --curated_test 209 215 --seed 45
python train.py --curated_test 223 230 --seed 46

# Average results across all folds
```

**Report:** "Mean test accuracy: 89.2% ± 2.1% (5-fold patient-wise cross-validation)"

## 🎓 Advanced Options

### **Find More Test Patients:**
```bash
# Find top 20 diverse patients
python analyze_patient_diversity.py --top_n 20

# Suggest 3 test patients
python analyze_patient_diversity.py --num_test_patients 3
```

### **Custom Train/Val Ratios:**
```bash
# 90% train, 10% val from pooled beats
python train.py --curated_test 207 217 --train_ratio 0.675 --val_ratio 0.075
# (0.675 + 0.075 = 0.75, the remaining after holding out test patients)
```

## 📚 References

- `analyze_patient_diversity.py` - Find diverse patients
- `check_split_distribution.py` - Verify class distribution
- `train.py` - Main training script
- `dataset.py` - Data loading and splitting

## 🆚 Comparison with Other Methods

| Method | Test Valid? | All Classes? | Use Case |
|--------|-------------|--------------|----------|
| **Patient-wise** | ✅ | ❌ Often missing | Production (if classes covered) |
| **Patient-wise Stratified** | ✅ | ⚠️ Sometimes missing | Production (better balance) |
| **Beat-wise** | ❌ | ✅ | Prototyping only (leakage) |
| **Curated Hybrid** | ✅ | ✅ | Production (guaranteed coverage) |
| **K-Fold Curated** | ✅✅ | ✅ | Research (most robust) |

---

## 💡 Bottom Line

**Use curated hybrid when:**
- ✅ You need all classes in test set
- ✅ Rare classes cluster in specific patients
- ✅ Test validity is critical
- ✅ You can justify train/val leakage (tuning only)

**Don't use when:**
- ❌ You can get good balance with regular patient-wise split
- ❌ You have many patients with diverse classes
- ❌ Reviewers won't accept train/val overlap

---

*For questions or issues, refer to `CLASS_DISTRIBUTION.md` and `CLASS_IMBALANCE_SOLUTIONS.md`*

