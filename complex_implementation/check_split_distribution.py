"""
Script to check class distribution across train/val/test splits

This helps verify that all classes are represented in each split.
"""

import os
import argparse
from collections import Counter
import wfdb
from dataset import BEAT_CLASS_MAPPING, CLASS_NAMES, create_patient_splits


def analyze_split_distribution(train_records, val_records, test_records, data_dir='../data/mitdb'):
    """
    Analyze and print class distribution for each split
    """
    
    def count_classes_in_records(records, split_name):
        """Count beat classes in a list of records"""
        class_counts = Counter()
        total_beats = 0
        
        for record_name in records:
            try:
                record_path = os.path.join(data_dir, record_name)
                annotation = wfdb.rdann(record_path, 'atr')
                
                for symbol in annotation.symbol:
                    if symbol in BEAT_CLASS_MAPPING:
                        class_id = BEAT_CLASS_MAPPING[symbol]
                        class_counts[class_id] += 1
                        total_beats += 1
                        
            except Exception as e:
                print(f"Warning: Could not process {record_name}: {e}")
        
        return class_counts, total_beats
    
    # Analyze each split
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    splits = [
        (train_records, "TRAINING"),
        (val_records, "VALIDATION"),
        (test_records, "TEST")
    ]
    
    all_class_counts = {}
    
    for records, split_name in splits:
        class_counts, total_beats = count_classes_in_records(records, split_name)
        all_class_counts[split_name] = class_counts
        
        print(f"\n{split_name} SET ({len(records)} records, {total_beats:,} beats)")
        print("-" * 80)
        print(f"{'Class':<20} {'Count':>10} {'Percentage':>12} {'Present':>10}")
        print("-" * 80)
        
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_beats * 100) if total_beats > 0 else 0
            present = "✓" if count > 0 else "✗ MISSING"
            
            print(f"{class_name:<20} {count:>10,} {percentage:>11.2f}% {present:>10}")
    
    # Check for missing classes
    print("\n" + "="*80)
    print("MISSING CLASS REPORT")
    print("="*80)
    
    missing_found = False
    for class_id, class_name in enumerate(CLASS_NAMES):
        missing_in = []
        for split_name in ["TRAINING", "VALIDATION", "TEST"]:
            if all_class_counts[split_name].get(class_id, 0) == 0:
                missing_in.append(split_name)
        
        if missing_in:
            missing_found = True
            print(f"⚠️  {class_name}: MISSING in {', '.join(missing_in)}")
    
    if not missing_found:
        print("✅ All classes present in all splits!")
    
    # Imbalance report
    print("\n" + "="*80)
    print("CLASS IMBALANCE ACROSS SPLITS")
    print("="*80)
    print(f"{'Class':<20} {'Train %':>10} {'Val %':>10} {'Test %':>10} {'Max Diff':>12}")
    print("-" * 80)
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        train_total = sum(all_class_counts["TRAINING"].values())
        val_total = sum(all_class_counts["VALIDATION"].values())
        test_total = sum(all_class_counts["TEST"].values())
        
        train_pct = (all_class_counts["TRAINING"].get(class_id, 0) / train_total * 100) if train_total > 0 else 0
        val_pct = (all_class_counts["VALIDATION"].get(class_id, 0) / val_total * 100) if val_total > 0 else 0
        test_pct = (all_class_counts["TEST"].get(class_id, 0) / test_total * 100) if test_total > 0 else 0
        
        max_diff = max(train_pct, val_pct, test_pct) - min(train_pct, val_pct, test_pct)
        
        print(f"{class_name:<20} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}% {max_diff:>11.2f}%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Check class distribution across splits')
    parser.add_argument('--data_dir', type=str, default='../data/mitdb',
                       help='Directory containing MIT-BIH data')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified split')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    if args.stratified:
        print("TESTING STRATIFIED SPLIT")
    else:
        print("TESTING RANDOM SPLIT (Default)")
    print("="*80)
    
    # Create splits
    train_records, val_records, test_records = create_patient_splits(
        data_dir=args.data_dir,
        train_ratio=0.75,
        val_ratio=0.125,
        test_ratio=0.125,
        random_seed=args.seed,
        stratified=args.stratified
    )
    
    # Analyze distribution
    analyze_split_distribution(train_records, val_records, test_records, args.data_dir)
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    if args.stratified:
        print("You used --stratified flag. Compare with non-stratified:")
        print("  python check_split_distribution.py")
    else:
        print("You used random split. Try stratified split for better balance:")
        print("  python check_split_distribution.py --stratified")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

