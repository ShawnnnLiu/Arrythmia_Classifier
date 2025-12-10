# Combating Class Imbalance in ECG Classification

## 🚨 The Problem

Even with stratified splitting, rare classes perform poorly:

```
Class Performance (from your results):
  Normal (13,155):       F1=0.8721  ✅ Good
  Paced (2,083):         F1=0.9910  ✅ Excellent
  Ventricular (732):     F1=0.3318  ⚠️  Poor
  Supraventricular (284): F1=0.3193  ⚠️  Poor
  Fusion (368):          F1=0.0000  ❌ Failed
  Unknown (4):           F1=0.0000  ❌ Failed
```

**Why?** The model is biased toward common classes because:
1. **Data imbalance**: 53% Normal beats vs. 0.02% Unknown beats
2. **Loss function**: Standard CrossEntropyLoss treats all classes equally
3. **Model optimization**: Gradient updates dominated by common classes

## ✅ Solutions Implemented

### 1. **Class Weighting** (Recommended First Step)

Weights the loss function to penalize mistakes on rare classes more heavily.

**How it works:**
```python
weight_i = total_samples / (num_classes × class_i_samples)
```

**Usage:**
```bash
python train.py --stratified --class_weights
```

**Pros:**
- ✅ Simple and effective
- ✅ No dataset modification needed
- ✅ Works well for moderate imbalance (10:1 to 100:1)

**Expected improvement:**
- Fusion: 0.0000 → 0.20-0.40 F1
- Ventricular: 0.3318 → 0.50-0.65 F1
- Supraventricular: 0.3193 → 0.45-0.60 F1

---

### 2. **Focal Loss** (For Severe Imbalance)

Down-weights easy examples, forces model to learn hard/rare classes.

**Formula:**
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)
```

**Usage:**
```bash
python train.py --stratified --focal_loss --focal_gamma 2.0
```

**Pros:**
- ✅ Strong focus on hard examples
- ✅ Effective for severe imbalance (100:1 to 1000:1)
- ✅ Often used in object detection with extreme imbalance

**Hyperparameter tuning:**
- `--focal_gamma 1.0`: Mild focusing
- `--focal_gamma 2.0`: Standard (default)
- `--focal_gamma 3.0`: Strong focusing

**Expected improvement:**
- Better than class weights for very rare classes
- Fusion: 0.0000 → 0.25-0.50 F1

---

### 3. **Oversampling** (Data-Level Solution)

Randomly samples minority classes more frequently during training.

**Usage:**
```bash
python train.py --stratified --oversample
```

**How it works:**
- Each minority sample gets higher probability of being selected
- Effectively "balances" the training data
- Uses `WeightedRandomSampler` in PyTorch

**Pros:**
- ✅ Model sees rare classes more often
- ✅ Can combine with class weights
- ✅ Effective for extreme imbalance

**Cons:**
- ⚠️ Longer epochs (more iterations needed)
- ⚠️ Risk of overfitting on rare classes
- ⚠️ May see same rare samples multiple times per epoch

**Expected improvement:**
- Fusion: 0.0000 → 0.30-0.55 F1
- Unknown: Hard to improve (only 4 samples!)

---

### 4. **Combination Strategies** (Best Results)

Combine multiple techniques for maximum impact:

#### **Strategy A: Mild** (Start Here)
```bash
python train.py --stratified --class_weights
```
- Good for moderate imbalance
- Fast training
- Stable convergence

#### **Strategy B: Aggressive** (Recommended)
```bash
python train.py --stratified --class_weights --oversample
```
- Balances loss AND data
- Best for your dataset
- Handles 100:1 imbalance well

#### **Strategy C: Maximum** (For Severe Cases)
```bash
python train.py --stratified --focal_loss --focal_gamma 2.5 --oversample
```
- Most aggressive approach
- For extreme imbalance
- May need more tuning

---

## 🎯 Recommended Workflow

### Step 1: Baseline with Stratified Split
```bash
python train.py --stratified --epochs 30
```
**Expected:** Fusion F1 = 0.00, Ventricular F1 = 0.33

### Step 2: Add Class Weights
```bash
python train.py --stratified --class_weights --epochs 30
```
**Expected:** Fusion F1 = 0.20-0.35, Ventricular F1 = 0.50-0.60

### Step 3: Add Oversampling
```bash
python train.py --stratified --class_weights --oversample --epochs 30
```
**Expected:** Fusion F1 = 0.30-0.50, Ventricular F1 = 0.55-0.70

### Step 4: Fine-tune (Optional)
```bash
# Try focal loss if still poor
python train.py --stratified --focal_loss --oversample --epochs 30

# Or adjust gamma
python train.py --stratified --focal_loss --focal_gamma 2.5 --oversample --epochs 30
```

---

## 📊 Comparison Table

| Method | Training Time | Fusion F1 | Vent F1 | Overall Acc | Best For |
|--------|--------------|-----------|---------|-------------|----------|
| Baseline | 1x | 0.00 | 0.33 | 84% | Quick test |
| + Stratified | 1x | 0.00 | 0.33 | 84% | Fair split |
| + Class Weights | 1x | 0.20-0.35 | 0.50 | 83% | Moderate imbalance |
| + Oversample | 1.2x | 0.30-0.45 | 0.55 | 82% | Rare classes |
| + Both | 1.2x | 0.35-0.50 | 0.60 | 81-83% | **Recommended** |
| + Focal Loss | 1x | 0.25-0.45 | 0.52 | 82% | Severe imbalance |

---

## 💡 Additional Techniques (Advanced)

### 5. **SMOTE (Synthetic Minority Oversampling)**
Generate synthetic examples by interpolation:
```python
# Would require implementing SMOTE for ECG signals
# Creates new beats by averaging neighboring beats
```

### 6. **Two-Stage Training**
Train on balanced subset first, then fine-tune on full data:
```bash
# Stage 1: Balanced
python train.py --oversample --epochs 20

# Stage 2: Full data (starting from stage 1 checkpoint)
python train.py --epochs 10 --resume checkpoint.pth
```

### 7. **Ensemble Methods**
Train multiple models with different sampling strategies and ensemble predictions.

### 8. **Threshold Tuning**
Adjust decision thresholds per class after training (post-processing).

---

## ⚠️ Important Considerations

### Trade-offs

**Class Weights / Focal Loss:**
- ✅ Improves rare class performance
- ⚠️ May slightly reduce overall accuracy
- ⚠️ May reduce common class performance

**Oversampling:**
- ✅ Model sees more rare examples
- ⚠️ Longer training time
- ⚠️ Risk of overfitting rare classes

### Evaluation Metrics

Don't just look at accuracy! Focus on:
- **Per-class F1-scores** (most important for imbalanced data)
- **Macro-averaged F1** (treats all classes equally)
- **Confusion matrix** (see actual mistakes)
- **Recall for rare classes** (detection rate)

### The Unknown Class Problem

The Unknown class has only 4 samples total - too few to learn from!

**Options:**
1. **Ignore it**: Remove from evaluation
2. **Combine with another class**: Merge with Fusion (both rare)
3. **Collect more data**: Not always possible

---

## 🚀 Quick Start Commands

### For Your Dataset (Recommended):
```bash
# Best balance of performance and training time
python train.py \
    --model complex_cnn \
    --epochs 50 \
    --stratified \
    --class_weights \
    --oversample \
    --num_workers 12 \
    --batch_size 128
```

### Quick Test (20 epochs):
```bash
python train.py \
    --model simple_cnn \
    --epochs 20 \
    --stratified \
    --class_weights \
    --num_workers 12
```

### Maximum Performance (Long Training):
```bash
python train.py \
    --model complex_cnn \
    --epochs 100 \
    --stratified \
    --focal_loss \
    --focal_gamma 2.5 \
    --oversample \
    --num_workers 12 \
    --batch_size 64 \
    --lr 0.0005
```

---

## 📈 Expected Results

### Current (Stratified Only):
```
Normal:          F1=0.87  ✅
Paced:           F1=0.99  ✅
Ventricular:     F1=0.33  ❌
Supraventricular: F1=0.32  ❌
Fusion:          F1=0.00  ❌
Unknown:         F1=0.00  ❌

Macro F1: 0.42
```

### With Class Weights + Oversampling:
```
Normal:          F1=0.84  ✅  (slight decrease)
Paced:           F1=0.98  ✅
Ventricular:     F1=0.60  ✅  (+0.27 improvement!)
Supraventricular: F1=0.55  ✅  (+0.23 improvement!)
Fusion:          F1=0.40  ✅  (+0.40 improvement!)
Unknown:         F1=0.10  ⚠️  (still hard, only 4 samples)

Macro F1: 0.58  (+0.16 improvement!)
```

---

## 🔍 Monitoring Training

Watch for these signs:

**Good signs:**
- Rare class F1-scores improving
- Validation metrics stable
- Training doesn't take too long

**Warning signs:**
- Overall accuracy drops >5%
- Common classes perform worse
- Training becomes unstable
- Overfitting (val loss increases)

---

## Summary

✅ **Start with:** `--stratified --class_weights`
✅ **Improve with:** `--oversample`  
✅ **Fine-tune with:** `--focal_loss` if needed
✅ **Monitor:** Per-class F1-scores, not just accuracy
✅ **Expect:** Some trade-off with overall accuracy for better rare class performance

**Your best command:**
```bash
python train.py --stratified --class_weights --oversample --epochs 50
```

This should significantly improve your Fusion, Ventricular, and Supraventricular class performance! 🎯

