"""
Script to check class distribution across train/val/test splits

This helps verify that all classes are represented in each split.
"""

import os
import argparse
from collections import Counter
import wfdb
from dataset import BEAT_CLASS_MAPPING, CLASS_NAMES, create_patient_splits, create_curated_hybrid_splits, create_curated_patient_splits


def analyze_split_distribution(train_records, val_records, test_records, data_dir='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0', beat_wise=False, hybrid_mode=False, train_ratio=0.75, val_ratio=0.125, seed=42):
    """
    Analyze and print class distribution for each split
    
    Parameters:
    -----------
    hybrid_mode : bool
        If True, test is patient-wise but train/val are beat-wise (curated hybrid split)
    """
    
    def count_classes_in_records(records, split_name):
        """Count beat classes in a list of records"""
        # Determine if this split should use beat-wise splitting
        if hybrid_mode:
            # HYBRID MODE: train/val are beat-wise, test is patient-wise
            use_beat_wise = (split_name in ["train", "val"])
        else:
            # NORMAL MODE: all splits follow the beat_wise flag
            use_beat_wise = beat_wise
        
        if use_beat_wise:
            # For beat-wise split, use BeatDataset to get actual split
            from dataset import BeatDataset
            dataset = BeatDataset(
                records, 
                data_dir=data_dir,
                beat_wise_split=True,
                split_name=split_name,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                random_seed=seed
            )
            class_counts = Counter(dataset.labels)
            total_beats = len(dataset.labels)
        else:
            # For patient-wise split, count all beats in these records
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
        (train_records, "train"),
        (val_records, "val"),
        (test_records, "test")
    ]
    
    split_display_names = {
        "train": "TRAINING",
        "val": "VALIDATION",
        "test": "TEST"
    }
    
    all_class_counts = {}
    
    for records, split_name in splits:
        class_counts, total_beats = count_classes_in_records(records, split_name)
        display_name = split_display_names[split_name]
        all_class_counts[display_name] = class_counts
        
        print(f"\n{display_name} SET ({len(records)} records, {total_beats:,} beats)")
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
    parser.add_argument('--data_dir', type=str, default='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                       help='Directory containing MIT-BIH data')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified split')
    parser.add_argument('--beat_wise', action='store_true',
                       help='WARNING: Use beat-wise split (shows data leakage impact)')
    parser.add_argument('--curated_test', nargs='+', type=str, default=None,
                       help='HYBRID: Curated test patients (e.g., --curated_test 207 217)')
    parser.add_argument('--curated_val', nargs='+', type=str, default=None,
                       help='Curated validation patients (e.g., --curated_val 203 208)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    if args.curated_val is not None and args.curated_test is not None:
        print("TESTING CURATED PATIENT-WISE SPLIT (Pure Patient-Wise)")
    elif args.curated_test is not None:
        print("TESTING CURATED HYBRID SPLIT")
    elif args.beat_wise:
        print("WARNING: TESTING BEAT-WISE SPLIT (WITH DATA LEAKAGE)")
    elif args.stratified:
        print("TESTING STRATIFIED PATIENT-WISE SPLIT")
    else:
        print("TESTING RANDOM PATIENT-WISE SPLIT (Default)")
    print("="*80)
    
    # Create splits based on mode
    if args.curated_val is not None and args.curated_test is not None:
        # PURE PATIENT-WISE: Curated val and test patients
        train_records, val_records, test_records = create_curated_patient_splits(
            data_dir=args.data_dir,
            val_patients=args.curated_val,
            test_patients=args.curated_test,
            random_seed=args.seed
        )
        use_beat_wise = False
    elif args.curated_test is not None:
        # HYBRID: Curated test (patient-wise) + beat-pooled train/val
        train_records, val_records, test_records = create_curated_hybrid_splits(
            data_dir=args.data_dir,
            test_patients=args.curated_test,
            train_ratio=0.85,  # Of remaining beats
            val_ratio=0.15,
            random_seed=args.seed
        )
        # Use beat_wise=True for train/val to pool beats
        use_beat_wise = True
    else:
        # Regular patient-wise or beat-wise split
        train_records, val_records, test_records = create_patient_splits(
            data_dir=args.data_dir,
            train_ratio=0.75,
            val_ratio=0.125,
            test_ratio=0.125,
            random_seed=args.seed,
            stratified=args.stratified,
            beat_wise=args.beat_wise
        )
        use_beat_wise = args.beat_wise
    
    # Analyze distribution
    # For curated hybrid, adjust ratios for train/val
    if args.curated_val is not None and args.curated_test is not None:
        # Pure patient-wise: no beat-wise splitting anywhere
        analyze_split_distribution(train_records, val_records, test_records, args.data_dir, 
                                  beat_wise=False, hybrid_mode=False,
                                  train_ratio=0.75, val_ratio=0.125, seed=args.seed)
    elif args.curated_test is not None:
        # Hybrid: train/val are beat-wise, test is patient-wise
        analyze_split_distribution(train_records, val_records, test_records, args.data_dir, 
                                  beat_wise=False, hybrid_mode=True,
                                  train_ratio=0.85, val_ratio=0.15, seed=args.seed)
    else:
        # Regular split
        analyze_split_distribution(train_records, val_records, test_records, args.data_dir, 
                                  beat_wise=use_beat_wise, hybrid_mode=False,
                                  train_ratio=0.75, val_ratio=0.125, seed=args.seed)
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    if args.curated_val is not None and args.curated_test is not None:
        print("You used CURATED PATIENT-WISE split!")
        print("")
        print("What this means:")
        print("  ✅ All splits: Pure patient-wise (no leakage anywhere)")
        print("  ✅ Validation: Manually selected diverse patients")
        print("  ✅ Test: Manually selected diverse patients")
        print("  ✅ Training: All remaining patients")
        print("")
        print("When to use:")
        print("  + BEST for publications and clinical validation")
        print("  + Maximum rigor and control over splits")
        print("  + Final model evaluation")
        print("")
        print("Note: May have class imbalance - use --class_weights in training!")
    elif args.curated_test is not None:
        print("You used CURATED HYBRID split!")
        print("")
        print("What this means:")
        print("  ✅ Test set: Patient-wise (valid generalization)")
        print("  ⚠️  Train/Val: Beat-wise (shares patients for class balance)")
        print("")
        print("When to use:")
        print("  + Rare classes cluster in specific patients")
        print("  + Need all classes in test set")
        print("  + Test validity is critical")
        print("  + Train/Val leakage is acceptable (val is just for tuning)")
        print("")
        print("To find diverse patients for test set:")
        print("  python analyze_patient_diversity.py")
    elif args.beat_wise:
        print("WARNING: You used BEAT-WISE split!")
        print("")
        print("This creates DATA LEAKAGE - same patient in train/val/test!")
        print("Use ONLY for:")
        print("  + Quick prototyping")
        print("  + Establishing upper-bound performance")
        print("")
        print("For production/research, consider:")
        print("  python check_split_distribution.py --stratified")
        print("Or for better class balance with valid test:")
        print("  python check_split_distribution.py --curated_test 207 217")
    elif args.stratified:
        print("You used --stratified flag. Compare with non-stratified:")
        print("  python check_split_distribution.py")
        print("")
        print("If you have missing classes in test, try curated hybrid:")
        print("  python check_split_distribution.py --curated_test 207 217")
    else:
        print("You used random split. Try stratified split for better balance:")
        print("  python check_split_distribution.py --stratified")
        print("")
        print("Or try curated hybrid for guaranteed class coverage:")
        print("  python check_split_distribution.py --curated_test 207 217")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

