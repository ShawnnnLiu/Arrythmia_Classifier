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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Import models and dataset
from complex_implementation.dataset import (
    create_patient_splits, 
    create_dataloaders,
    CLASS_NAMES
)
from complex_implementation.models_simple_cnn import SimpleBeatCNN
from complex_implementation.models_complex_cnn import ComplexBeatCNN


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
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Also compute macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
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


def train(args):
    """
    Main training function
    """
    print("\n" + "="*70)
    print(f"Training Per-Beat ECG Classifier")
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
    train_records, val_records, test_records = create_patient_splits(
        data_dir=args.data_dir,
        random_seed=args.seed
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
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
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
        start_time = time.time()
        
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
        epoch_time = time.time() - start_time
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
    
    # Save training history
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print("="*70)
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, split_name='Test'
    )
    
    test_metrics = compute_metrics(test_labels, test_preds, CLASS_NAMES)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    print_metrics(test_metrics, CLASS_NAMES)
    
    # Save test metrics
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_metrics': test_metrics
    }
    
    with open(os.path.join(checkpoint_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    print(f"\nAll results saved to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train per-beat ECG classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'complex_cnn'],
                       help='Model architecture to use')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/mitdb',
                       help='Directory containing MIT-BIH data')
    parser.add_argument('--window_size', type=float, default=0.8,
                       help='Window size around R-peak in seconds')
    parser.add_argument('--lead', type=int, default=0, choices=[0, 1],
                       help='Which ECG lead to use (0 or 1)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
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
    parser.add_argument('--checkpoint_dir', type=str, default='complex_implementation/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Train model
    train(args)


if __name__ == "__main__":
    main()

