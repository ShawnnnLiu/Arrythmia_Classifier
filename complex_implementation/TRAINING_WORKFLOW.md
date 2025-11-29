# Training, Validation, and Testing Workflow

## Data Split (Default: 75/12.5/12.5)

Your data is split into **three separate sets**:

```
Total: 48 MIT-BIH Records

┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Training Set: 75% (36 records)                              │
│  ├─ Used for: Learning model parameters                      │
│  ├─ Model updates: Every batch                               │
│  └─ Purpose: Teach the model ECG patterns                    │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Validation Set: 12.5% (6 records)                           │
│  ├─ Used for: Monitoring training progress                   │
│  ├─ Evaluated: Every epoch (no gradient updates)             │
│  ├─ Purpose: Detect overfitting, select best model           │
│  └─ Saves: Best checkpoint based on validation accuracy      │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Test Set: 12.5% (6 records)                                 │
│  ├─ Used for: Final performance evaluation                   │
│  ├─ Evaluated: Once at the end (using best model)            │
│  ├─ Purpose: Unbiased estimate of real-world performance     │
│  └─ Reports: Final accuracy for publication/reporting        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Training Workflow

### Phase 1: Training Loop (Each Epoch)

```
For each epoch (1 to 50):
  
  1. TRAINING PHASE
     ├─ Iterate through all training batches
     ├─ Forward pass → compute loss → backward pass → update weights
     ├─ Track: training loss, training accuracy
     └─ Model learns from this data
  
  2. VALIDATION PHASE
     ├─ Evaluate on validation set (no weight updates)
     ├─ Track: validation loss, validation accuracy
     ├─ Check if this is the best model so far
     └─ If best: Save checkpoint as 'best_model.pth'
  
  3. LEARNING RATE ADJUSTMENT
     └─ Reduce LR if validation accuracy plateaus
```

### Phase 2: Final Test Evaluation

```
After all epochs complete:

  1. LOAD BEST MODEL
     ├─ Load the checkpoint with highest validation accuracy
     ├─ This might be from epoch 10, 20, etc. (not necessarily the last epoch)
     └─ Print: "Using best model from Epoch X, Val Acc: Y%"
  
  2. EVALUATE ON TEST SET
     ├─ Run the best model on the test set (unseen data)
     ├─ Compute: test accuracy, per-class metrics, confusion matrix
     └─ This is your FINAL PERFORMANCE metric
  
  3. SAVE RESULTS
     ├─ Generate confusion matrix plots
     ├─ Save per-class metrics
     ├─ Create comprehensive results summary
     └─ Generate visualizations
```

## Why This Workflow?

### Training Set (75%)
- **Largest portion** because the model needs many examples to learn
- Direct feedback loop: loss → gradients → weight updates
- The model "memorizes" patterns from this data

### Validation Set (12.5%)
- **Guards against overfitting**: If validation accuracy decreases while training accuracy increases, we're overfitting
- **Model selection**: Automatically saves the best model (might not be the last epoch!)
- **Hyperparameter tuning**: Learning rate scheduler uses validation accuracy
- Model indirectly "sees" this data through early stopping decisions

### Test Set (12.5%)
- **Completely unseen**: Never used during training or model selection
- **Unbiased evaluation**: True measure of generalization
- **Final metric**: This is what you report in papers/presentations
- If test accuracy >> validation accuracy → lucky split
- If test accuracy << validation accuracy → overfitting to validation set

## Example Training Run

```
Epoch 1/50:  Train Acc: 85.23%  Val Acc: 82.45%  [Save best model ✓]
Epoch 2/50:  Train Acc: 88.12%  Val Acc: 84.31%  [Save best model ✓]
Epoch 3/50:  Train Acc: 89.56%  Val Acc: 85.67%  [Save best model ✓]
...
Epoch 15/50: Train Acc: 95.23%  Val Acc: 89.12%  [Save best model ✓]
Epoch 16/50: Train Acc: 95.89%  Val Acc: 88.98%  [No improvement]
Epoch 17/50: Train Acc: 96.12%  Val Acc: 88.76%  [No improvement - overfitting!]
...
Epoch 50/50: Train Acc: 98.34%  Val Acc: 87.23%  [No improvement]

Best model: Epoch 15 with Val Acc: 89.12%

Loading best model (Epoch 15)...
Evaluating on test set...

FINAL TEST RESULTS:
  Test Accuracy: 88.95%  ← This is your reportable accuracy!
```

## Key Points

✅ **Best model is NOT always the last epoch**
   - Early stopping prevents overfitting
   - Validation accuracy guides model selection

✅ **Test set is evaluated ONCE at the end**
   - Using the best model from training
   - Completely unbiased evaluation

✅ **Validation ≠ Test**
   - Validation: Used during training for decisions
   - Test: Used once for final reporting

✅ **Patient-wise split**
   - No patient appears in multiple sets
   - Tests generalization to new patients
   - Critical for medical applications

## Changing the Split

Default is 75/12.5/12.5, but you can customize:

```bash
# Use default (75/12.5/12.5)
python train.py

# Custom split (e.g., 70/15/15)
python train.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# Larger training set (80/10/10)
python train.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

**Note:** Ratios must sum to 1.0!

## Best Practices

1. **Never train on validation or test data** ✓ (Enforced by patient-wise split)
2. **Always use the best model for final test** ✓ (Automatic in our code)
3. **Test only once at the end** ✓ (To avoid bias from multiple testing)
4. **Report test accuracy, not validation** ✓ (Test is unbiased)
5. **Keep validation for hyperparameter tuning** ✓ (Allows fair comparison)

## Summary

The training script ensures:
- ✅ Clean separation of train/val/test sets
- ✅ Automatic best model selection
- ✅ Final test evaluation using best model
- ✅ Comprehensive result logging
- ✅ No data leakage between sets

**Your final test accuracy is the gold standard for reporting model performance!**

