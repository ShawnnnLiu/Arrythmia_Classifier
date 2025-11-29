"""
Training script for per-beat ECG classification

Supports multiple model architectures and provides comprehensive
logging and evaluation metrics.
"""

import os
import argparse
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Import models and dataset
from dataset import (
    create_patient_splits, 
    create_dataloaders,
    CLASS_NAMES
)
from models_simple_cnn import SimpleBeatCNN
from models_complex_cnn import ComplexBeatCNN


def get_model(model_name: str, num_classes: int):
    """
    Get model by name
    
    Parameters:
    -----------
    model_name : str
        Model name ('simple_cnn' or 'complex_cnn')
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    model : nn.Module
        PyTorch model
    """
    if model_name == 'simple_cnn':
        return SimpleBeatCNN(num_classes=num_classes)
    elif model_name == 'complex_cnn':
        return ComplexBeatCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'simple_cnn' or 'complex_cnn'")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Returns:
    --------
    avg_loss : float
        Average training loss
    accuracy : float
        Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (signals, labels) in enumerate(train_loader):
        # Move to device
        signals = signals.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, split_name='Validation'):
    """
    Evaluate model on a dataset
    
    Returns:
    --------
    avg_loss : float
        Average loss
    accuracy : float
        Accuracy
    all_preds : np.array
        All predictions
    all_labels : np.array
        All true labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in loader:
            # Move to device
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute detailed metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary containing precision, recall, f1 per class
    """
    # Compute per-class metrics for all classes (including those with 0 samples)
    num_classes = len(class_names)
    labels = list(range(num_classes))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    
    # Also compute macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=labels, zero_division=0
    )
    
    metrics = {
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro)
        }
    }
    
    return metrics


def print_metrics(metrics, class_names):
    """Print metrics in a readable format"""
    print("\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 70)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['per_class']['precision'][i]
        recall = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1'][i]
        support = metrics['per_class']['support'][i]
        
        print(f"  {class_name:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10.0f}")
    
    print("  " + "-" * 70)
    print(f"  {'Macro Average':<20} "
          f"{metrics['macro']['precision']:>10.4f} "
          f"{metrics['macro']['recall']:>10.4f} "
          f"{metrics['macro']['f1']:>10.4f}")


def save_checkpoint(model, optimizer, epoch, best_val_acc, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, save_path)
    print(f"  Checkpoint saved to {save_path}")


def save_training_curves(training_history, save_dir, model_name):
    """
    Save training curves as plots for presentation
    
    Parameters:
    -----------
    training_history : dict
        Dictionary containing training metrics per epoch
    save_dir : str
        Directory to save plots
    model_name : str
        Name of the model for plot titles
    """
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.close()
    
    # Also create individual high-res plots for presentations
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2.5)
    plt.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'{model_name} - Loss Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2.5)
    plt.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2.5)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'{model_name} - Accuracy Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Training curves saved to {save_dir}")


def plot_confusion_matrix(cm, class_names, save_path, model_name):
    """
    Plot confusion matrix for presentation
    
    Parameters:
    -----------
    cm : np.array
        Confusion matrix
    class_names : list
        List of class names
    save_path : str
        Path to save the plot
    model_name : str
        Name of the model
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.title(f'{model_name} - Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Confusion matrix plot saved to {save_path}")


def train(args):
    """
    Main training function
    """
    # Record start time (ISO 8601 format - industry standard)
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    start_time_iso = start_time.isoformat()
    
    print("\n" + "="*70)
    print(f"Training Per-Beat ECG Classifier")
    print(f"Started at: {start_time_str}")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create patient-wise splits
    print("\nCreating patient-wise train/val/test splits...")
    if args.stratified:
        print("Using stratified split (balances rare classes across splits)")
    train_records, val_records, test_records = create_patient_splits(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratified=args.stratified
    )
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        lead=args.lead
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, num_classes)
    model = model.to(device)
    
    print(f"Model has {model.get_num_params():,} trainable parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 
                                  f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Save training configuration
    config = vars(args)
    config['num_classes'] = num_classes
    config['num_train_records'] = len(train_records)
    config['num_val_records'] = len(val_records)
    config['num_test_records'] = len(test_records)
    
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    best_val_acc = 0.0
    best_epoch = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, split_name='Validation'
        )
        
        # Compute detailed metrics
        val_metrics = compute_metrics(val_labels, val_preds, CLASS_NAMES)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"    Learning Rate: {current_lr:.6f}")
        print(f"    Epoch Time: {epoch_time:.2f}s")
        
        # Print validation metrics
        print_metrics(val_metrics, CLASS_NAMES)
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, best_val_acc,
                os.path.join(checkpoint_dir, 'best_model.pth')
            )
        
        # Save latest model
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, best_val_acc,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Record end time
    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_iso = end_time.isoformat()
    training_duration = end_time - start_time
    duration_str = str(training_duration).split('.')[0]  # Remove microseconds
    
    # Save training history (JSON)
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training history (CSV for easy plotting)
    history_df = pd.DataFrame({
        'epoch': range(1, len(training_history['train_loss']) + 1),
        'train_loss': training_history['train_loss'],
        'train_acc': training_history['train_acc'],
        'val_loss': training_history['val_loss'],
        'val_acc': training_history['val_acc'],
        'learning_rate': training_history['learning_rate']
    })
    history_df.to_csv(os.path.join(checkpoint_dir, 'training_history.csv'), index=False)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Training duration: {duration_str}")
    print("="*70)
    
    # Generate training curves
    print("\nGenerating training curves...")
    save_training_curves(training_history, checkpoint_dir, args.model.upper().replace('_', ' '))
    
    # Load best model and evaluate on test set
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION (using best model from training)")
    print("="*70)
    print(f"Loading best model checkpoint (Epoch {best_epoch}, Val Acc: {best_val_acc:.2f}%)...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Evaluating on test set ({len(test_records)} records)...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, split_name='Test'
    )
    
    test_metrics = compute_metrics(test_labels, test_preds, CLASS_NAMES)
    
    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS (Best Model from Epoch {best_epoch})")
    print(f"{'='*70}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    print_metrics(test_metrics, CLASS_NAMES)
    
    # Generate confusion matrix plot
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(
        cm, CLASS_NAMES, 
        os.path.join(checkpoint_dir, 'confusion_matrix.png'),
        args.model.upper().replace('_', ' ')
    )
    
    # Print confusion matrix (text)
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)
    
    # Save comprehensive results summary
    results_summary = {
        # Model information
        'model_name': args.model,
        'model_parameters': model.get_num_params(),
        
        # Timing (ISO 8601 format - industry standard)
        'start_time': start_time_iso,
        'end_time': end_time_iso,
        'training_duration_seconds': training_duration.total_seconds(),
        'training_duration_formatted': duration_str,
        
        # Configuration
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'window_size': args.window_size,
            'lead': args.lead,
            'seed': args.seed,
            'device': str(device)
        },
        
        # Dataset information
        'dataset': {
            'num_classes': num_classes,
            'class_names': CLASS_NAMES,
            'num_train_records': len(train_records),
            'num_val_records': len(val_records),
            'num_test_records': len(test_records),
            'train_records': train_records,
            'val_records': val_records,
            'test_records': test_records
        },
        
        # Training results
        'training': {
            'best_epoch': int(best_epoch),
            'best_val_accuracy': float(best_val_acc),
            'final_train_loss': float(training_history['train_loss'][-1]),
            'final_train_acc': float(training_history['train_acc'][-1]),
            'final_val_loss': float(training_history['val_loss'][-1]),
            'final_val_acc': float(training_history['val_acc'][-1]),
            'final_learning_rate': float(training_history['learning_rate'][-1])
        },
        
        # Test results
        'test': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'per_class_metrics': test_metrics,
            'confusion_matrix': cm.tolist()
        }
    }
    
    # Save comprehensive summary
    with open(os.path.join(checkpoint_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save per-class metrics to CSV for easy comparison
    per_class_df = pd.DataFrame({
        'class_name': CLASS_NAMES,
        'class_id': list(range(len(CLASS_NAMES))),
        'precision': test_metrics['per_class']['precision'],
        'recall': test_metrics['per_class']['recall'],
        'f1_score': test_metrics['per_class']['f1'],
        'support': test_metrics['per_class']['support']
    })
    per_class_df.to_csv(os.path.join(checkpoint_dir, 'per_class_metrics.csv'), index=False)
    
    # Create a summary text report for quick reference
    with open(os.path.join(checkpoint_dir, 'SUMMARY.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"ECG ARRHYTHMIA CLASSIFICATION - TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Parameters: {model.get_num_params():,}\n")
        f.write(f"Started: {start_time_str}\n")
        f.write(f"Finished: {end_time_str}\n")
        f.write(f"Duration: {duration_str}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Window Size: {args.window_size}s\n")
        f.write(f"ECG Lead: {args.lead}\n")
        f.write(f"Random Seed: {args.seed}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("DATASET\n")
        f.write("-"*70 + "\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Train Records: {len(train_records)}\n")
        f.write(f"Val Records: {len(val_records)}\n")
        f.write(f"Test Records: {len(test_records)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PER-CLASS PERFORMANCE (Test Set)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-"*70 + "\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"{class_name:<20} "
                   f"{test_metrics['per_class']['precision'][i]:>10.4f} "
                   f"{test_metrics['per_class']['recall'][i]:>10.4f} "
                   f"{test_metrics['per_class']['f1'][i]:>10.4f} "
                   f"{int(test_metrics['per_class']['support'][i]):>10d}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Macro Average':<20} "
               f"{test_metrics['macro']['precision']:>10.4f} "
               f"{test_metrics['macro']['recall']:>10.4f} "
               f"{test_metrics['macro']['f1']:>10.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {checkpoint_dir}")
    print(f"{'='*70}")
    print(f"\nFiles generated:")
    print(f"  - results_summary.json     (Complete results in JSON)")
    print(f"  - training_history.json    (Epoch-by-epoch metrics)")
    print(f"  - training_history.csv     (CSV for plotting)")
    print(f"  - per_class_metrics.csv    (Per-class performance)")
    print(f"  - training_curves.png/pdf  (Loss & accuracy plots)")
    print(f"  - confusion_matrix.png/pdf (Confusion matrix heatmap)")
    print(f"  - SUMMARY.txt              (Human-readable summary)")
    print(f"  - best_model.pth           (Best model checkpoint)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train per-beat ECG classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'complex_cnn'],
                       help='Model architecture to use')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data/mitdb',
                       help='Directory containing MIT-BIH data')
    parser.add_argument('--window_size', type=float, default=0.8,
                       help='Window size around R-peak in seconds')
    parser.add_argument('--lead', type=int, default=0, choices=[0, 1],
                       help='Which ECG lead to use (0 or 1)')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                       help='Fraction of records for training (default: 0.75)')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                       help='Fraction of records for validation (default: 0.125)')
    parser.add_argument('--test_ratio', type=float, default=0.125,
                       help='Fraction of records for testing (default: 0.125)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified split to balance rare classes across train/val/test')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 12, recommended: 4-16 for multi-core CPUs)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        print(f"ERROR: Split ratios must sum to 1.0")
        print(f"Current: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
        print(f"Sum = {total_ratio}")
        return
    
    if args.val_ratio == 0:
        print("WARNING: No validation set! Model selection and overfitting detection will not be available.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            return
    
    # Train model
    train(args)


if __name__ == "__main__":
    main()

