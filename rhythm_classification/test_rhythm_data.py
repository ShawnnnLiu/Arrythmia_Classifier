"""
Quick test to verify rhythm annotations are working
"""
import sys
sys.path.append('..')

from dataset import create_patient_splits, create_dataloaders, CLASS_NAMES

print("Testing rhythm annotation loading...")
print("="*70)

try:
    # Create splits
    print("\nCreating patient-wise splits...")
    train_records, val_records, test_records = create_patient_splits()
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        train_records, val_records, test_records,
        batch_size=8,
        num_workers=0
    )
    
    print(f"\n✅ SUCCESS! Rhythm data loaded successfully!")
    print(f"   Number of classes: {num_classes}")
    print(f"   Class names: {CLASS_NAMES}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Get a sample
    for signals, labels in train_loader:
        print(f"\n   Sample batch shape: {signals.shape}")
        print(f"   Sample labels: {labels[:5].tolist()}")
        print(f"   Sample rhythms: {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")
        break
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()









