##### Modify this file for the LSTM end-to-end implementation #####


"""
Training script for rhythm classification

Supports multiple model architectures with comprehensive
logging and evaluation metrics for ECG rhythm patterns.
"""

import os
import argparse
import time
import json
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Import models and dataset
from dataset import (
    create_patient_splits,
    create_segment_wise_splits,
    create_dataloaders,
    CLASS_NAMES
)

# import LSTM model


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        probs = torch.softmax(inputs, dim=1)
        probs_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - probs_target) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_model(model_name: str, num_classes: int, **model_kwargs): # <-- switch to LSTM model
    """Get model by name"""
    # if model_name == 'simple_cnn':
    #     return SimpleRhythmCNN(num_classes=num_classes, **model_kwargs)
    # elif model_name == 'complex_cnn':
    #     return ComplexRhythmCNN(num_classes=num_classes, **model_kwargs)
    # elif model_name == 'complex_cnn_nolstm':
    #     return ComplexRhythmCNN_NoLSTM(num_classes=num_classes, **model_kwargs)
    # else:
    #     raise ValueError(f"Unknown model: {model_name}")
    pass


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (signals, labels) in enumerate(train_loader):
        signals, labels = signals.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            acc = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device, class_names):
    """Evaluate model on a dataset"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(data_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * (all_predictions == all_labels).sum() / len(all_labels)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro average
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, class_names, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='True',
           title='Confusion Matrix - Rhythm Classification')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_text(config, history, test_results, save_dir):
    """Save a human-readable summary"""
    summary_path = os.path.join(save_dir, 'SUMMARY.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"ECG RHYTHM CLASSIFICATION - TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Split Strategy: {config['split_strategy']}\n")
        f.write(f"Segment Length: {config['segment_length']}s\n")
        f.write(f"Segment Stride: {config['segment_stride']}s\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Batch Size: {config['batch_size']}\n")
        f.write(f"Learning Rate: {config['lr']}\n")
        f.write(f"Optimizer: {config['optimizer']}\n")
        f.write(f"Loss Function: {config['criterion']}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-"*70 + "\n")
        best_epoch = np.argmax(history['val_acc']) + 1
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.2f}% (Epoch {best_epoch})\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%\n\n")
        
        f.write("TEST SET RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Test Loss: {test_results['loss']:.4f}\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")
        f.write(f"Macro Precision: {test_results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {test_results['macro_recall']:.4f}\n")
        f.write(f"Macro F1: {test_results['macro_f1']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS (Test Set)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-"*70 + "\n")
        
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(test_results['precision']):
                f.write(f"{class_name:<25} "
                       f"{test_results['precision'][i]:>10.4f} "
                       f"{test_results['recall'][i]:>10.4f} "
                       f"{test_results['f1'][i]:>10.4f} "
                       f"{test_results['support'][i]:>10d}\n")
        
        f.write("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train ECG Rhythm Classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                       help='Directory containing MIT-BIH data')
    parser.add_argument('--segment_length', type=float, default=10.0,
                       help='Length of ECG segments in seconds')
    parser.add_argument('--segment_stride', type=float, default=5.0,
                       help='Stride between segments in seconds')
    parser.add_argument('--lead', type=int, default=0, choices=[0, 1],
                       help='ECG lead to use (0 or 1)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'complex_cnn', 'complex_cnn_nolstm'],
                       help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--loss', type=str, default='crossentropy',
                       choices=['crossentropy', 'focal'],
                       help='Loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    
    # Split strategy
    parser.add_argument('--split', type=str, default='patient_wise',
                       choices=['patient_wise', 'segment_wise'],
                       help='Data split strategy')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                       help='Validation data ratio')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='rhythm_classification/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "="*70)
    print("Training ECG Rhythm Classifier")
    print("="*70)
    print(f"\nUsing device: {device}")
    
    # Create data splits
    print(f"\nCreating {args.split} splits...")
    if args.split == 'patient_wise':
        train_records, val_records, test_records = create_patient_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            stratified=True
        )
        segment_wise_split = False
    else:  # segment_wise
        train_records, val_records, test_records = create_segment_wise_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed
        )
        segment_wise_split = True
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        train_records, val_records, test_records,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_wise_split=segment_wise_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        segment_length=args.segment_length,
        segment_stride=args.segment_stride,
        lead=args.lead
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create loss function
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, f"{args.model}_{timestamp}_{args.split}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['num_classes'] = num_classes
    config['num_parameters'] = n_params
    config['split_strategy'] = args.split
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\nStarting training...")
    print("="*70 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        print("\n  Evaluating on validation set...")
        val_results = evaluate(model, val_loader, criterion, device, CLASS_NAMES)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_results['loss']:.4f} | Val Acc:   {val_results['accuracy']:.2f}%")
        
        # Print per-class metrics
        print(f"\n  Per-class metrics (validation):")
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(val_results['precision']):
                print(f"    {class_name:25s} "
                     f"Precision: {val_results['precision'][i]:.4f}  "
                     f"Recall: {val_results['recall'][i]:.4f}  "
                     f"F1: {val_results['f1'][i]:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_results['accuracy'])
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"\n  ✓ New best model saved! Val Acc: {best_val_acc:.2f}%")
        
        # Periodic checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_results['accuracy'],
                'config': config
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print("\n" + "="*70 + "\n")
    
    # Final evaluation on test set
    print("Final evaluation on test set...")
    print("-"*70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, device, CLASS_NAMES)
    
    print(f"\nTest Set Results:")
    print(f"  Test Loss: {test_results['loss']:.4f}")
    print(f"  Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"  Macro Precision: {test_results['macro_precision']:.4f}")
    print(f"  Macro Recall: {test_results['macro_recall']:.4f}")
    print(f"  Macro F1: {test_results['macro_f1']:.4f}")
    
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(test_results['precision']):
            print(f"  {class_name:25s} "
                 f"Precision: {test_results['precision'][i]:.4f}  "
                 f"Recall: {test_results['recall'][i]:.4f}  "
                 f"F1: {test_results['f1'][i]:.4f}  "
                 f"Support: {test_results['support'][i]:5d}")
    
    # Save results
    print("\nSaving results...")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test results
    test_results_serializable = {
        'loss': float(test_results['loss']),
        'accuracy': float(test_results['accuracy']),
        'macro_precision': float(test_results['macro_precision']),
        'macro_recall': float(test_results['macro_recall']),
        'macro_f1': float(test_results['macro_f1']),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': float(test_results['precision'][i]),
                'recall': float(test_results['recall'][i]),
                'f1': float(test_results['f1'][i]),
                'support': int(test_results['support'][i])
            }
            for i in range(len(CLASS_NAMES)) if i < len(test_results['precision'])
        }
    }
    
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['labels'], test_results['predictions'], CLASS_NAMES, save_dir)
    
    # Save summary text
    save_summary_text(config, history, test_results, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()