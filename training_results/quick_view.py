"""
Quick View - Rapidly inspect training results

This script provides a quick way to view results from the most recent
or a specific training run.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime


def find_latest_run(checkpoint_base_dir='complex_implementation/checkpoints'):
    """Find the most recent training run"""
    checkpoint_path = Path(checkpoint_base_dir)
    
    if not checkpoint_path.exists():
        return None
    
    runs = list(checkpoint_path.glob('*/results_summary.json'))
    if not runs:
        return None
    
    # Sort by modification time
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest.parent


def display_summary(run_dir):
    """Display a formatted summary of the training run"""
    run_path = Path(run_dir)
    summary_file = run_path / 'results_summary.json'
    
    if not summary_file.exists():
        print(f"No results found in {run_dir}")
        return
    
    with open(summary_file) as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print(f"TRAINING RUN SUMMARY: {run_path.name}")
    print("="*80)
    
    # Basic Info
    print(f"\n📊 MODEL INFORMATION")
    print(f"{'─'*80}")
    print(f"  Model Name:        {data['model_name'].upper()}")
    print(f"  Parameters:        {data['model_parameters']:,}")
    print(f"  Device:            {data['config']['device']}")
    
    # Timing
    print(f"\n⏱️  TIMING")
    print(f"{'─'*80}")
    start = datetime.fromisoformat(data['start_time'])
    end = datetime.fromisoformat(data['end_time'])
    print(f"  Started:           {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished:          {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:          {data['training_duration_formatted']}")
    
    # Configuration
    print(f"\n⚙️  CONFIGURATION")
    print(f"{'─'*80}")
    print(f"  Epochs:            {data['config']['epochs']}")
    print(f"  Batch Size:        {data['config']['batch_size']}")
    print(f"  Learning Rate:     {data['config']['learning_rate']}")
    print(f"  Window Size:       {data['config']['window_size']}s")
    print(f"  Random Seed:       {data['config']['seed']}")
    
    # Dataset
    print(f"\n📁 DATASET")
    print(f"{'─'*80}")
    print(f"  Classes:           {data['dataset']['num_classes']}")
    print(f"  Train Records:     {data['dataset']['num_train_records']}")
    print(f"  Val Records:       {data['dataset']['num_val_records']}")
    print(f"  Test Records:      {data['dataset']['num_test_records']}")
    
    # Training Results
    print(f"\n🎯 TRAINING RESULTS")
    print(f"{'─'*80}")
    print(f"  Best Epoch:        {data['training']['best_epoch']}")
    print(f"  Best Val Acc:      {data['training']['best_val_accuracy']:.2f}%")
    print(f"  Final Train Acc:   {data['training']['final_train_acc']:.2f}%")
    print(f"  Final Val Acc:     {data['training']['final_val_acc']:.2f}%")
    
    # Test Results
    print(f"\n✅ TEST RESULTS")
    print(f"{'─'*80}")
    print(f"  Test Accuracy:     {data['test']['test_accuracy']:.2f}%")
    print(f"  Test Loss:         {data['test']['test_loss']:.4f}")
    
    # Per-Class Performance
    print(f"\n📈 PER-CLASS PERFORMANCE (Test Set)")
    print(f"{'─'*80}")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    metrics = data['test']['per_class_metrics']
    for i, class_name in enumerate(data['dataset']['class_names']):
        p = metrics['per_class']['precision'][i]
        r = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1'][i]
        s = int(metrics['per_class']['support'][i])
        print(f"  {class_name:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {s:>10d}")
    
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Macro Average':<20} "
          f"{metrics['macro']['precision']:>10.4f} "
          f"{metrics['macro']['recall']:>10.4f} "
          f"{metrics['macro']['f1']:>10.4f}")
    
    # Files Generated
    print(f"\n📂 OUTPUT FILES")
    print(f"{'─'*80}")
    files = [
        'results_summary.json',
        'training_history.csv',
        'per_class_metrics.csv',
        'training_curves.png',
        'confusion_matrix.png',
        'SUMMARY.txt',
        'best_model.pth'
    ]
    for file in files:
        if (run_path / file).exists():
            size = (run_path / file).stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            print(f"  ✓ {file:<35} {size_str:>10}")
    
    print(f"\n{'='*80}")
    print(f"Location: {run_path}")
    print(f"{'='*80}\n")


def quick_plot(run_dir):
    """Display training curves in terminal using ASCII art"""
    run_path = Path(run_dir)
    history_file = run_path / 'training_history.csv'
    
    if not history_file.exists():
        print("Training history not found")
        return
    
    df = pd.read_csv(history_file)
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    
    print(f"\nEpoch {df['epoch'].max()}: "
          f"Train Acc={df['train_acc'].iloc[-1]:.2f}%, "
          f"Val Acc={df['val_acc'].iloc[-1]:.2f}%")
    
    print(f"Best Val Acc: {df['val_acc'].max():.2f}% (Epoch {df['val_acc'].idxmax() + 1})")
    
    # Simple ASCII plot of validation accuracy
    print("\nValidation Accuracy Progress:")
    max_val = df['val_acc'].max()
    min_val = df['val_acc'].min()
    
    for i, row in df.iterrows():
        epoch = int(row['epoch'])
        acc = row['val_acc']
        # Normalize to 0-50 range for display
        bar_len = int((acc - min_val) / (max_val - min_val + 0.01) * 50)
        bar = '█' * bar_len
        print(f"  Epoch {epoch:3d}: {bar} {acc:.2f}%")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Quick view of training results')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Specific run directory to view (default: latest)')
    parser.add_argument('--plot', action='store_true',
                       help='Show ASCII plot of training progress')
    parser.add_argument('--checkpoint_base', type=str,
                       default='complex_implementation/checkpoints',
                       help='Base checkpoint directory')
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(args.checkpoint_base)
        if run_dir is None:
            print(f"No training runs found in {args.checkpoint_base}")
            return
        print(f"\n🔍 Showing latest run: {run_dir.name}\n")
    
    # Display summary
    display_summary(run_dir)
    
    # Show plot if requested
    if args.plot:
        quick_plot(run_dir)


if __name__ == '__main__':
    main()

