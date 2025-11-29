# Training Results Directory

This directory contains all training results for ECG arrhythmia classification models.

## Directory Structure

Each training run creates a timestamped subdirectory in `complex_implementation/checkpoints/` with the following structure:

```
checkpoints/
└── {model_name}_{timestamp}/
    ├── results_summary.json          # Complete results (JSON format)
    ├── training_history.json         # Epoch-by-epoch metrics (JSON)
    ├── training_history.csv          # Epoch-by-epoch metrics (CSV)
    ├── per_class_metrics.csv         # Per-class performance metrics
    ├── training_curves.png           # Training/validation curves (PNG, 300 DPI)
    ├── training_curves.pdf           # Training/validation curves (PDF)
    ├── loss_curve.png                # Loss curve only (high-res)
    ├── accuracy_curve.png            # Accuracy curve only (high-res)
    ├── confusion_matrix.png          # Confusion matrix heatmap (PNG, 300 DPI)
    ├── confusion_matrix.pdf          # Confusion matrix heatmap (PDF)
    ├── SUMMARY.txt                   # Human-readable summary report
    ├── config.json                   # Training configuration
    ├── best_model.pth                # Best model checkpoint
    └── checkpoint_epoch_*.pth        # Periodic checkpoints
```

## Files Explanation

### For Presentations and Slideshows

1. **High-Resolution Plots** (300 DPI, publication-ready):
   - `training_curves.png/pdf` - Combined loss and accuracy plots
   - `loss_curve.png` - Standalone loss curve
   - `accuracy_curve.png` - Standalone accuracy curve
   - `confusion_matrix.png/pdf` - Confusion matrix with percentages

2. **Summary Report**:
   - `SUMMARY.txt` - Quick overview with all key metrics

### For Data Analysis

1. **CSV Files** (Easy to import into Excel, Python, R):
   - `training_history.csv` - Loss, accuracy per epoch
   - `per_class_metrics.csv` - Precision, recall, F1-score per class

2. **JSON Files** (Complete structured data):
   - `results_summary.json` - All results in one file
   - `training_history.json` - Training metrics
   - `config.json` - Model configuration

### For Model Deployment

- `best_model.pth` - Best performing model (highest validation accuracy)
- `checkpoint_epoch_*.pth` - Periodic checkpoints

## Using Results for Presentations

### Quick Comparison Between Models

1. **Load CSV files** from different model runs:
```python
import pandas as pd

# Load training history
simple_cnn = pd.read_csv('checkpoints/simple_cnn_20251128_201700/training_history.csv')
complex_cnn = pd.read_csv('checkpoints/complex_cnn_20251128_203000/training_history.csv')

# Compare final accuracies
print(f"Simple CNN:  {simple_cnn['val_acc'].max():.2f}%")
print(f"Complex CNN: {complex_cnn['val_acc'].max():.2f}%")
```

2. **Create comparison plots**:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(simple_cnn['epoch'], simple_cnn['val_acc'], label='Simple CNN')
plt.plot(complex_cnn['epoch'], complex_cnn['val_acc'], label='Complex CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('model_comparison.png', dpi=300)
```

3. **Compare per-class performance**:
```python
simple_metrics = pd.read_csv('checkpoints/simple_cnn_20251128_201700/per_class_metrics.csv')
complex_metrics = pd.read_csv('checkpoints/complex_cnn_20251128_203000/per_class_metrics.csv')

comparison = pd.DataFrame({
    'Class': simple_metrics['class_name'],
    'Simple_F1': simple_metrics['f1_score'],
    'Complex_F1': complex_metrics['f1_score']
})
print(comparison)
```

### Generating Presentation Tables

**Load results summary**:
```python
import json

with open('checkpoints/simple_cnn_20251128_201700/results_summary.json', 'r') as f:
    results = json.load(f)

# Create summary table
print(f"Model: {results['model_name']}")
print(f"Parameters: {results['model_parameters']:,}")
print(f"Test Accuracy: {results['test']['test_accuracy']:.2f}%")
print(f"Training Time: {results['training_duration_formatted']}")
```

### Key Metrics for Slideshow

Include these in your presentation:

1. **Model Architecture**:
   - Model name
   - Number of parameters
   - Training time

2. **Performance**:
   - Test accuracy
   - Best validation accuracy
   - Confusion matrix (use the PNG/PDF files)

3. **Training Dynamics**:
   - Training curves (use the PNG/PDF files)
   - Learning rate schedule
   - Convergence behavior

4. **Per-Class Analysis**:
   - Precision, recall, F1-score per class
   - Support (number of samples) per class
   - Challenging classes (low F1-scores)

## Time Format

All timestamps use **ISO 8601 format** (industry standard):
- Example: `2025-11-28T20:17:00.123456`
- Includes: Year-Month-Day T Hour:Minute:Second.Microseconds
- Timezone-aware and internationally recognized

## Comparing Multiple Runs

To compare multiple training runs:

1. Check the `training_results/` folder for all timestamped runs
2. Use the CSV files for quantitative comparison
3. Use the PDF files for presentation slides
4. Reference the SUMMARY.txt for quick overview

## Best Practices

1. **Archive important runs**: Copy the checkpoint directory to `training_results/` for long-term storage
2. **Document experiments**: Add notes about what you changed in each run
3. **Use PDFs for presentations**: Higher quality than PNGs in slides
4. **Keep CSVs for analysis**: Easy to load into any tool

## Example: Creating a Comparison Table

```python
import pandas as pd
import json
from pathlib import Path

# Collect all results
results = []
for checkpoint_dir in Path('complex_implementation/checkpoints').glob('*/'):
    summary_file = checkpoint_dir / 'results_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            results.append({
                'Model': data['model_name'],
                'Parameters': data['model_parameters'],
                'Test Acc (%)': data['test']['test_accuracy'],
                'Val Acc (%)': data['training']['best_val_accuracy'],
                'Training Time': data['training_duration_formatted'],
                'Date': data['end_time'][:10]
            })

# Create comparison DataFrame
df = pd.DataFrame(results).sort_values('Test Acc (%)', ascending=False)
print(df.to_string(index=False))

# Save to CSV
df.to_csv('training_results/model_comparison.csv', index=False)
```

## Questions?

For more information about the training pipeline, see:
- `complex_implementation/README.md` - Implementation details
- `complex_implementation/train.py` - Training script
- `complex_implementation/dataset.py` - Data loading

