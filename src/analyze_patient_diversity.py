"""
Analyze patient diversity to find good candidates for curated test set.

This script finds patients with the most diverse class distributions,
which are ideal for creating a test set that covers all arrhythmia types.
"""

import os
import sys
import wfdb
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

# Import from dataset.py
from dataset import BEAT_CLASS_MAPPING, CLASS_NAMES

def get_all_records(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0') -> List[str]:
    """Get list of all available record names"""
    records = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.dat'):
            record_name = filename[:-4]
            records.append(record_name)
    return sorted(records)


def analyze_patient_diversity(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0', 
                              min_beats: int = 100) -> Dict:
    """
    Analyze diversity of each patient record.
    
    Returns dictionary with patient diversity information:
    - num_classes: number of different arrhythmia classes present
    - classes: set of class IDs present
    - counts: Counter of beats per class
    - total_beats: total number of beats
    """
    
    all_records = get_all_records(data_dir)
    print(f"Analyzing {len(all_records)} patient records...\n")
    
    patient_diversity = {}
    
    for record in all_records:
        record_path = os.path.join(data_dir, record)
        
        try:
            # Load annotations
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Count classes
            classes_present = set()
            class_counts = Counter()
            
            for symbol in annotation.symbol:
                if symbol in BEAT_CLASS_MAPPING:
                    class_id = BEAT_CLASS_MAPPING[symbol]
                    classes_present.add(class_id)
                    class_counts[class_id] += 1
            
            total_beats = sum(class_counts.values())
            
            # Only include patients with minimum beat count
            if total_beats >= min_beats:
                patient_diversity[record] = {
                    'num_classes': len(classes_present),
                    'classes': classes_present,
                    'counts': class_counts,
                    'total_beats': total_beats
                }
                
        except Exception as e:
            print(f"Warning: Could not process record {record}: {e}")
    
    return patient_diversity


def print_diversity_report(patient_diversity: Dict, top_n: int = 10):
    """Print a formatted report of patient diversity"""
    
    # Sort by number of classes (descending), then by total beats
    sorted_patients = sorted(
        patient_diversity.items(),
        key=lambda x: (x[1]['num_classes'], x[1]['total_beats']),
        reverse=True
    )
    
    print("=" * 80)
    print(f"TOP {top_n} MOST DIVERSE PATIENTS (Best candidates for curated test set)")
    print("=" * 80)
    
    for i, (record, info) in enumerate(sorted_patients[:top_n], 1):
        print(f"\n{i}. Patient {record}")
        print(f"   Classes present: {info['num_classes']}/6")
        print(f"   Total beats: {info['total_beats']}")
        print(f"   Class distribution:")
        
        for class_id in sorted(info['classes']):
            count = info['counts'][class_id]
            percentage = (count / info['total_beats']) * 100
            print(f"      - {CLASS_NAMES[class_id]:20s} (class {class_id}): {count:5d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)


def suggest_test_patients(patient_diversity: Dict, 
                         num_patients: int = 2) -> List[str]:
    """
    Suggest patients for test set based on diversity.
    
    Strategy:
    1. Find patients with most classes present
    2. Ensure union of selected patients covers all 6 classes
    3. Prefer patients with more balanced distributions
    """
    
    # Sort by diversity
    sorted_patients = sorted(
        patient_diversity.items(),
        key=lambda x: (x[1]['num_classes'], x[1]['total_beats']),
        reverse=True
    )
    
    selected = []
    covered_classes = set()
    
    # Greedy selection to maximize class coverage
    for record, info in sorted_patients:
        if len(selected) >= num_patients:
            break
        
        # Check if this patient adds new classes
        new_classes = info['classes'] - covered_classes
        
        if len(selected) == 0 or len(new_classes) > 0:
            selected.append(record)
            covered_classes.update(info['classes'])
            
            if len(covered_classes) == 6:  # All classes covered
                break
    
    return selected, covered_classes


def print_combined_stats(patient_diversity: Dict, selected_patients: List[str]):
    """Print combined statistics for selected test patients"""
    
    print("\n" + "=" * 80)
    print("COMBINED TEST SET STATISTICS")
    print("=" * 80)
    
    combined_counts = Counter()
    total_beats = 0
    
    for record in selected_patients:
        info = patient_diversity[record]
        combined_counts.update(info['counts'])
        total_beats += info['total_beats']
    
    print(f"\nTest patients: {', '.join(selected_patients)}")
    print(f"Total test beats: {total_beats}")
    print(f"\nClass distribution in test set:")
    
    for class_id in range(6):
        count = combined_counts.get(class_id, 0)
        percentage = (count / total_beats * 100) if total_beats > 0 else 0
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"   {CLASS_NAMES[class_id]:20s} (class {class_id}): {count:5d} ({percentage:5.1f}%) {status}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze patient diversity for curated test set')
    parser.add_argument('--data_dir', type=str, default='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                       help='Path to MIT-BIH data directory')
    parser.add_argument('--num_test_patients', type=int, default=2,
                       help='Number of patients to suggest for test set')
    parser.add_argument('--top_n', type=int, default=48,
                       help='Number of top diverse patients to display')
    
    args = parser.parse_args()
    
    # Analyze diversity
    patient_diversity = analyze_patient_diversity(args.data_dir)
    
    # Print report
    print_diversity_report(patient_diversity, args.top_n)
    
    # Suggest test patients
    print("\n" + "=" * 80)
    print("SUGGESTED TEST SET")
    print("=" * 80)
    
    suggested_patients, covered_classes = suggest_test_patients(
        patient_diversity, 
        args.num_test_patients
    )
    
    print(f"\nSuggested test patients: {suggested_patients}")
    print(f"Classes covered: {len(covered_classes)}/6")
    
    if len(covered_classes) < 6:
        missing_classes = set(range(6)) - covered_classes
        print(f"⚠️  WARNING: Missing classes: {[CLASS_NAMES[c] for c in missing_classes]}")
        print(f"   Consider increasing --num_test_patients")
    
    # Print combined statistics
    print_combined_stats(patient_diversity, suggested_patients)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nUse the following command to train with curated test set:")
    print(f"\npython train.py --model simple_cnn --curated_test {' '.join(suggested_patients)}")
    print(f"\nThis will:")
    print(f"  1. Hold out patients {suggested_patients} for testing (pure patient-wise)")
    print(f"  2. Pool beats from remaining {len(patient_diversity) - len(suggested_patients)} patients")
    print(f"  3. Split pooled beats for train/val (class balance)")
    print(f"  4. Test on held-out patients (true generalization)")
    print()


if __name__ == '__main__':
    main()

