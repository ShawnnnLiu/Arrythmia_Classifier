"""
Script to compare multiple trained models

This script helps you compare the performance of different models
for presentation and analysis purposes.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


def load_all_results(checkpoint_base_dir='complex_implementation/checkpoints'):
    """
    Load results from all training runs
    
    Returns:
    --------
    results : list of dict
        List of result summaries from all runs
    """
    results = []
    checkpoint_path = Path(checkpoint_base_dir)
    
    if not checkpoint_path.exists():
        print(f"Warning: {checkpoint_base_dir} does not exist")
        return results
    
    for run_dir in checkpoint_path.glob('*/'):
        summary_file = run_dir / 'results_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                data['run_dir'] = str(run_dir)
                results.append(data)
    
    print(f"Found {len(results)} training runs")
    return results


def create_comparison_table(results):
    """Create a comparison table of all models"""
    if not results:
        print("No results to compare")
        return None
    
    comparison = []
    for data in results:
        comparison.append({
            'Model': data['model_name'],
            'Run_Dir': Path(data['run_dir']).name,
            'Parameters': data['model_parameters'],
            'Test_Acc_%': round(data['test']['test_accuracy'], 2),
            'Val_Acc_%': round(data['training']['best_val_accuracy'], 2),
            'Best_Epoch': data['training']['best_epoch'],
            'Training_Time': data['training_duration_formatted'],
            'Finished': data['end_time'][:19].replace('T', ' '),
            'Batch_Size': data['config']['batch_size'],
            'Learning_Rate': data['config']['learning_rate']
        })
    
    df = pd.DataFrame(comparison).sort_values('Test_Acc_%', ascending=False)
    return df


def plot_model_comparison(results, save_path='training_results/model_comparison.png'):
    """Create comparison plots for all models"""
    if not results:
        print("No results to plot")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Test Accuracy Comparison
    model_names = [r['model_name'] + '\n' + Path(r['run_dir']).name[-6:] for r in results]
    test_accs = [r['test']['test_accuracy'] for r in results]
    
    axes[0, 0].bar(range(len(model_names)), test_accs, color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(test_accs):
        axes[0, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Parameters vs Accuracy
    params = [r['model_parameters'] / 1000 for r in results]  # Convert to thousands
    
    scatter = axes[0, 1].scatter(params, test_accs, s=200, alpha=0.6, c=test_accs, cmap='viridis')
    axes[0, 1].set_xlabel('Model Parameters (thousands)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Test Accuracy (%)')
    
    # Annotate points
    for i, (p, a, name) in enumerate(zip(params, test_accs, model_names)):
        axes[0, 1].annotate(name.split('\n')[0], (p, a), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Training Curves Comparison (if available)
    for i, result in enumerate(results):
        run_dir = Path(result['run_dir'])
        history_file = run_dir / 'training_history.csv'
        
        if history_file.exists():
            history = pd.read_csv(history_file)
            label = result['model_name'] + ' ' + run_dir.name[-6:]
            axes[1, 0].plot(history['epoch'], history['val_acc'], 
                           label=label, linewidth=2, alpha=0.7)
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Per-Class F1-Score Comparison (first two models)
    if len(results) >= 1:
        for i, result in enumerate(results[:2]):  # Compare first 2 models
            run_dir = Path(result['run_dir'])
            metrics_file = run_dir / 'per_class_metrics.csv'
            
            if metrics_file.exists():
                metrics = pd.read_csv(metrics_file)
                x_pos = np.arange(len(metrics)) + i * 0.35
                axes[1, 1].bar(x_pos, metrics['f1_score'], width=0.35, 
                             label=result['model_name'], alpha=0.7)
        
        if len(results) >= 1:
            # Use class names from first model
            run_dir = Path(results[0]['run_dir'])
            metrics_file = run_dir / 'per_class_metrics.csv'
            if metrics_file.exists():
                metrics = pd.read_csv(metrics_file)
                axes[1, 1].set_xticks(np.arange(len(metrics)) + 0.175)
                axes[1, 1].set_xticklabels(metrics['class_name'], rotation=45, ha='right', fontsize=10)
        
        axes[1, 1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nComparison plots saved to: {save_path}")
    plt.close()


def generate_latex_table(df, save_path='training_results/comparison_table.tex'):
    """Generate a LaTeX table for academic papers/presentations"""
    if df is None or df.empty:
        return
    
    # Select key columns
    table_df = df[['Model', 'Parameters', 'Test_Acc_%', 'Val_Acc_%', 'Training_Time']]
    
    latex_str = table_df.to_latex(index=False, 
                                   column_format='lrrrr',
                                   caption='Model Performance Comparison',
                                   label='tab:model_comparison')
    
    with open(save_path, 'w') as f:
        f.write(latex_str)
    
    print(f"LaTeX table saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare trained ECG models')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='complex_implementation/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str,
                       default='training_results',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ECG MODEL COMPARISON TOOL")
    print("="*70)
    
    # Load all results
    results = load_all_results(args.checkpoint_dir)
    
    if not results:
        print("\nNo training results found!")
        print(f"Please ensure you have trained models in: {args.checkpoint_dir}")
        return
    
    # Create comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    df = create_comparison_table(results)
    print("\n", df.to_string(index=False))
    
    # Save comparison table
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path / 'model_comparison.csv', index=False)
    print(f"\nComparison table saved to: {output_path / 'model_comparison.csv'}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_model_comparison(results, save_path=str(output_path / 'model_comparison.png'))
    
    # Generate LaTeX table (for academic use)
    generate_latex_table(df, save_path=str(output_path / 'comparison_table.tex'))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total models trained: {len(results)}")
    print(f"Best test accuracy: {df['Test_Acc_%'].max():.2f}% ({df.loc[df['Test_Acc_%'].idxmax(), 'Model']})")
    print(f"Average test accuracy: {df['Test_Acc_%'].mean():.2f}%")
    print(f"Smallest model: {df['Parameters'].min():,} params ({df.loc[df['Parameters'].idxmin(), 'Model']})")
    print(f"Largest model: {df['Parameters'].max():,} params ({df.loc[df['Parameters'].idxmax(), 'Model']})")
    print("="*70)


if __name__ == '__main__':
    main()

