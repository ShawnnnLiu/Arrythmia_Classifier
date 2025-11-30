# Comparison: Beat Classification vs Rhythm Classification

This document explains the key differences between the two implementations in this project.

## 📁 Two Separate Implementations

```
Arrythmia_Classifier/
├── complex_implementation/      # Beat-level classification
│   └── Classifies individual heartbeats (0.8s windows)
│
└── rhythm_classification/       # Rhythm-level classification
    └── Identifies rhythm patterns (10-30s segments)
```

## 🎯 Different Goals

### Beat Classification (`complex_implementation/`)
**Question:** "What type of heartbeat is this?"

- **Input**: Single beat window (~0.8 seconds)
- **Output**: Beat type (Normal, PVC, Supraventricular, etc.)
- **Use case**: Detect individual abnormal beats
- **Clinical**: "Patient has premature ventricular contractions"

### Rhythm Classification (`rhythm_classification/`)
**Question:** "What rhythm pattern is the heart in?"

- **Input**: Long ECG segment (10-30 seconds)
- **Output**: Rhythm type (Normal Sinus, AFIB, VT, etc.)
- **Use case**: Diagnose sustained rhythm disorders
- **Clinical**: "Patient is in atrial fibrillation"

## 📊 Technical Comparison

| Aspect | Beat Classifier | Rhythm Classifier |
|--------|----------------|------------------|
| **Input Length** | ~0.8 seconds (288 samples @ 360Hz) | 10-30 seconds (3600-10800 samples) |
| **Annotation Source** | `symbol` field | `aux_note` field |
| **Labels** | N, V, A, F, /, Q | (N, (AFIB, (VT, etc. |
| **Number of Classes** | 6 beat types | 4 rhythm types |
| **Samples per Patient** | 100-3000 beats | 10-50 segments |
| **Total Dataset** | ~110,000 beats | ~1,000-5,000 segments |
| **Model Complexity** | Simple CNN sufficient | LSTM/Attention beneficial |
| **Training Time** | 5-10 min/epoch | 10-20 min/epoch |
| **Inference Speed** | Very fast | Moderate |

## 🧬 Data Annotation Details

### Beat Annotations (complex_implementation)

```python
# From annotation.symbol field
annotation.symbol = ['N', 'N', 'V', 'N', 'N', 'A', ...]

# Each symbol marks ONE beat
# Appears at R-peak location
# Very dense (~2 per second)
```

**All 48 MIT-BIH records have beat annotations** ✅

### Rhythm Annotations (rhythm_classification)

```python
# From annotation.aux_note field  
annotation.aux_note = ['', '', '(AFIB', '', '', '(N', ...]

# Each marks START of a rhythm segment
# Sparse (~5-10 per record)
# Only some records have this
```

**Only ~10-15 MIT-BIH records have rhythm annotations** ⚠️

## 🏗️ Architecture Differences

### Beat Classifier Architecture

```
Input [1, 288]
    ↓
Conv Block 1 → Conv Block 2 → Conv Block 3
    ↓
Global Pooling
    ↓
FC Layers → [6 classes]
```

**Why it works:**
- Beats are short, morphology-focused
- Spatial features sufficient
- No long-term dependencies needed

### Rhythm Classifier Architecture

```
Input [1, 3600]
    ↓
ResNet Blocks (feature extraction)
    ↓
Bidirectional LSTM (temporal patterns)
    ↓
Attention (important time steps)
    ↓
FC Layers → [4 classes]
```

**Why it's different:**
- Long sequences need temporal modeling
- LSTM captures rhythm regularity/irregularity
- Attention weights important segments

## 📈 Class Distributions

### Beat Classification Classes

```
Normal            : ~90% (very dominant)
Supraventricular : ~3%
Ventricular      : ~6%
Fusion           : <1%
Paced            : <1%
Unknown          : <1%
```

**Challenge:** Severe class imbalance

### Rhythm Classification Classes

```
Normal           : ~60-70%
Atrial Arrhythmia: ~15-25%
Ventricular Arr. : ~10-15%
Pre-excitation   : ~1-5%
```

**Challenge:** Sparse annotations, small dataset

## 🎓 Training Strategies

### Beat Classifier

```bash
# Patient-wise split works well
python -m complex_implementation.train \
    --model simple_cnn \
    --epochs 50 \
    --batch_size 64

# 110K beats → good training data
```

### Rhythm Classifier

```bash
# Patient-wise split is harder (fewer patients)
python -m rhythm_classification.train \
    --model complex_cnn \
    --split patient_wise \
    --epochs 50 \
    --batch_size 16

# Only ~1000 segments → more challenging
```

## 🔬 When to Use Each

### Use Beat Classification When:
- ✅ Detecting individual abnormal beats
- ✅ Counting PVCs, PACs, etc.
- ✅ Real-time beat-by-beat monitoring
- ✅ You have R-peak annotations
- ✅ Need fast inference

### Use Rhythm Classification When:
- ✅ Diagnosing rhythm disorders (AFIB, VT)
- ✅ Assessing overall heart rhythm
- ✅ You have rhythm annotations
- ✅ Need sustained pattern detection
- ✅ Clinical diagnosis focus

## 🔄 Complementary Approaches

In practice, both are useful:

```python
# Step 1: Beat classification
beats = classify_individual_beats(ecg_signal)
# → "Patient has frequent PVCs"

# Step 2: Rhythm classification  
rhythm = classify_rhythm_pattern(ecg_signal)
# → "Patient is in normal sinus rhythm with PVCs"

# Combined insight
# → "Normal sinus rhythm with frequent PVCs"
```

## 📊 Performance Expectations

### Beat Classifier
- **Patient-wise**: 85-95% accuracy
- **Segment-wise**: 95-98% accuracy
- **Main challenge**: Rare beat types (Fusion, Paced)
- **Strengths**: Large dataset, clear annotations

### Rhythm Classifier
- **Patient-wise**: 70-85% accuracy
- **Segment-wise**: 85-95% accuracy
- **Main challenge**: Limited rhythm annotations
- **Strengths**: Clinically meaningful, end-to-end

## 🎯 Key Takeaways

1. **Different Problems**
   - Beat = single heartbeat morphology
   - Rhythm = sustained pattern over time

2. **Different Data**
   - Beat annotations: dense, all records
   - Rhythm annotations: sparse, few records

3. **Different Models**
   - Beat: Simple CNN works
   - Rhythm: LSTM/Attention helps

4. **Different Evaluation**
   - Beat: More data, easier to validate
   - Rhythm: Less data, more clinical impact

5. **Both Are Valuable**
   - Beat: Detailed arrhythmia detection
   - Rhythm: Overall rhythm diagnosis

## 🚀 Running Both

```bash
# Train beat classifier
cd complex_implementation
python -m complex_implementation.train --model simple_cnn

# Train rhythm classifier
cd ..
python -m rhythm_classification.train --model simple_cnn

# Compare results!
```

## 📝 Bottom Line

- **Beat classification** = Fine-grained, per-beat analysis
- **Rhythm classification** = Big-picture, rhythm pattern diagnosis

**Both are important for comprehensive cardiac monitoring!** 💓

---

Choose based on your clinical question and available annotations.

