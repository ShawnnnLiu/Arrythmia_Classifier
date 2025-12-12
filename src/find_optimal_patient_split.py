"""
Find optimal patient split for validation and test sets
that best represents the overall class distribution.

This script tries different combinations of patients and scores them
based on how closely their combined class distribution matches
the overall dataset distribution.
"""

import os
import sys
import numpy as np
import wfdb
from collections import Counter
from itertools import combinations

# Ensure src/ is on sys.path so we can import dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import BEAT_CLASS_MAPPING, CLASS_NAMES


# Default MIT-BIH data directory – resolved relative to this script's location
# so it works regardless of which directory you run from.
DEFAULT_DATA_DIR = os.path.join(
    SCRIPT_DIR, "..", "data", "mit-bih-arrhythmia-database-1.0.0", "mit-bih-arrhythmia-database-1.0.0"
)


def load_patient_class_counts(data_dir: str = DEFAULT_DATA_DIR):
    """Load class distribution for each patient"""
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    patient_data = {}
    overall_counts = Counter()
    
    for record_name in all_records:
        try:
            record_path = os.path.join(data_dir, record_name)
            annotation = wfdb.rdann(record_path, 'atr')
            
            class_counts = Counter()
            for symbol in annotation.symbol:
                if symbol in BEAT_CLASS_MAPPING:
                    class_id = BEAT_CLASS_MAPPING[symbol]
                    class_counts[class_id] += 1
                    overall_counts[class_id] += 1
            
            patient_data[record_name] = {
                'counts': class_counts,
                'total': sum(class_counts.values()),
                'num_classes': len(class_counts)
            }
            
        except Exception as e:
            print(f"Warning: Could not process {record_name}: {e}")
    
    return patient_data, overall_counts


def get_distribution(class_counts, num_classes=6):
    """Convert counts to distribution array"""
    total = sum(class_counts.values())
    if total == 0:
        return np.zeros(num_classes)
    
    dist = np.zeros(num_classes)
    for class_id, count in class_counts.items():
        dist[class_id] = count / total
    
    return dist


def score_split(val_patients, test_patients, patient_data, overall_dist, total_beats):
    """
    Score how well a val+test split represents overall distribution AND is balanced
    
    Lower score is better
    
    Considers:
    1. Distribution match (val+test vs overall)
    2. Balance between val and test (should be roughly equal in beat count)
    3. Individual val and test distribution match
    """
    # Get counts for val and test separately
    val_counts = Counter()
    test_counts = Counter()
    
    for patient in val_patients:
        val_counts.update(patient_data[patient]['counts'])
    
    for patient in test_patients:
        test_counts.update(patient_data[patient]['counts'])
    
    # Combined counts
    combined_counts = val_counts + test_counts
    
    # Get beat counts
    val_beats = sum(val_counts.values())
    test_beats = sum(test_counts.values())
    total_val_test = val_beats + test_beats
    
    # Target: each should be ~12.5% of total dataset
    target_val_ratio = 0.125
    target_test_ratio = 0.125
    
    val_ratio = val_beats / total_beats
    test_ratio = test_beats / total_beats
    
    # Penalty for imbalanced val/test sizes (heavily penalize if one is much larger)
    size_imbalance = abs(val_beats - test_beats) / total_val_test
    size_penalty = size_imbalance * 10  # Heavy penalty for imbalance
    
    # Penalty for not meeting target ratios
    ratio_penalty = abs(val_ratio - target_val_ratio) * 50 + abs(test_ratio - target_test_ratio) * 50
    
    # Get distributions
    combined_dist = get_distribution(combined_counts)
    val_dist = get_distribution(val_counts)
    test_dist = get_distribution(test_counts)
    
    # Compute distribution match (combined vs overall)
    epsilon = 1e-10
    combined_dist_smooth = combined_dist + epsilon
    overall_dist_smooth = overall_dist + epsilon
    
    kl_div = np.sum(overall_dist_smooth * np.log(overall_dist_smooth / combined_dist_smooth))
    l1_dist = np.sum(np.abs(combined_dist - overall_dist))
    
    dist_score = kl_div + l1_dist
    
    # Combined score (distribution match + balance penalties)
    score = dist_score + size_penalty + ratio_penalty
    
    return score, combined_dist, combined_counts, val_beats, test_beats


def find_optimal_split(patient_data, overall_counts, 
                       target_val_test_ratio=0.25,
                       num_val=6, num_test=6,
                       top_n_candidates=20):
    """
    Find optimal patient split by trying different combinations
    
    Parameters:
    -----------
    target_val_test_ratio : float
        Target fraction of total beats for val+test (default: 0.25 for 12.5%+12.5%)
    num_val : int
        Number of patients for validation
    num_test : int
        Number of patients for testing
    top_n_candidates : int
        Only consider the top N most diverse patients
    """
    # Sort patients by diversity (number of classes)
    sorted_patients = sorted(patient_data.items(), 
                            key=lambda x: (x[1]['num_classes'], x[1]['total']), 
                            reverse=True)
    
    # Get top candidates (most diverse)
    candidate_patients = [p[0] for p in sorted_patients[:top_n_candidates]]
    
    # Calculate overall distribution
    overall_dist = get_distribution(overall_counts)
    total_beats = sum(overall_counts.values())
    target_beats = total_beats * target_val_test_ratio
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR OPTIMAL VAL/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Total patients: {len(patient_data)}")
    print(f"Candidate patients (most diverse): {len(candidate_patients)}")
    print(f"Target: {num_val} validation + {num_test} test patients")
    print(f"Target beats for val+test: ~{target_beats:.0f} ({target_val_test_ratio*100:.1f}% of total)")
    print(f"\nOverall class distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:<20} {overall_dist[i]*100:>6.2f}%")
    
    # Try combinations (using sampling for efficiency)
    print(f"\nTrying combinations...")
    
    best_score = float('inf')
    best_splits = []
    num_combinations = 0
    
    total_val_test = num_val + num_test
    
    # Calculate total possible combinations
    from math import comb
    total_possible = comb(len(candidate_patients), total_val_test) * comb(total_val_test, num_val)
    print(f"Total possible combinations: {total_possible:,}")
    
    # If too many, use random sampling; otherwise try all
    max_to_try = 50000
    use_sampling = total_possible > max_to_try
    
    if use_sampling:
        print(f"Using random sampling (trying {max_to_try:,} combinations)")
        import random
        random.seed(42)
        
        tried = set()
        while num_combinations < max_to_try:
            # Randomly sample val_test patients
            val_test_combo = tuple(sorted(random.sample(candidate_patients, total_val_test)))
            
            # Randomly split into val and test
            val_patients = tuple(sorted(random.sample(val_test_combo, num_val)))
            test_patients = tuple(sorted(p for p in val_test_combo if p not in val_patients))
            
            # Skip if we've tried this before
            combo_key = (val_patients, test_patients)
            if combo_key in tried:
                continue
            tried.add(combo_key)
            
            # Score this split
            score, split_dist, combined_counts, val_beats, test_beats = score_split(
                val_patients, test_patients, patient_data, overall_dist, total_beats
            )
            
            num_combinations += 1
            
            # Check if this is better
            if score < best_score:
                best_score = score
                best_splits = [(val_patients, test_patients, score, split_dist, combined_counts, val_beats, test_beats)]
            elif abs(score - best_score) < 1e-6:  # Essentially tied
                best_splits.append((val_patients, test_patients, score, split_dist, combined_counts, val_beats, test_beats))
            
            # Print progress
            if num_combinations % 5000 == 0:
                print(f"  Evaluated {num_combinations:,} combinations... (best score: {best_score:.4f})")
    else:
        print(f"Trying all {total_possible:,} combinations")
        
        # Try all combinations
        for val_test_combo in combinations(candidate_patients, total_val_test):
            # For each combination, try different ways to split into val and test
            for val_patients in combinations(val_test_combo, num_val):
                val_set = set(val_patients)
                test_patients = tuple(p for p in val_test_combo if p not in val_set)
                
                # Score this split
                score, split_dist, combined_counts, val_beats, test_beats = score_split(
                    val_patients, test_patients, patient_data, overall_dist, total_beats
                )
                
                num_combinations += 1
                
                # Check if this is better
                if score < best_score:
                    best_score = score
                    best_splits = [(val_patients, test_patients, score, split_dist, combined_counts, val_beats, test_beats)]
                elif abs(score - best_score) < 1e-6:  # Essentially tied
                    best_splits.append((val_patients, test_patients, score, split_dist, combined_counts, val_beats, test_beats))
                
                # Print progress every 10000 combinations
                if num_combinations % 10000 == 0:
                    print(f"  Evaluated {num_combinations:,} combinations... (best score: {best_score:.4f})")
    
    print(f"\nEvaluated {num_combinations:,} total combinations")
    print(f"Found {len(best_splits)} optimal split(s) with score {best_score:.4f}")
    
    return best_splits, overall_dist, total_beats


def print_split_details(split_idx, val_patients, test_patients, score, 
                       split_dist, combined_counts, overall_dist, 
                       patient_data, total_beats):
    """Print detailed information about a split"""
    print(f"\n{'='*80}")
    print(f"OPTIMAL SPLIT #{split_idx}")
    print(f"{'='*80}")
    print(f"Score: {score:.4f} (lower is better)")
    print(f"\nValidation patients ({len(val_patients)}): {', '.join(sorted(val_patients))}")
    print(f"Test patients ({len(test_patients)}):       {', '.join(sorted(test_patients))}")
    
    # Count beats
    val_beats = sum(patient_data[p]['total'] for p in val_patients)
    test_beats = sum(patient_data[p]['total'] for p in test_patients)
    total_val_test_beats = val_beats + test_beats
    
    print(f"\nBeat counts:")
    print(f"  Validation: {val_beats:>6,} beats ({val_beats/total_beats*100:>5.2f}% of total)")
    print(f"  Test:       {test_beats:>6,} beats ({test_beats/total_beats*100:>5.2f}% of total)")
    print(f"  Combined:   {total_val_test_beats:>6,} beats ({total_val_test_beats/total_beats*100:>5.2f}% of total)")
    
    print(f"\nClass distribution comparison:")
    print(f"  {'Class':<20} {'Overall':>10} {'Val+Test':>10} {'Diff':>10}")
    print(f"  {'-'*60}")
    
    total_diff = 0
    for i, class_name in enumerate(CLASS_NAMES):
        overall_pct = overall_dist[i] * 100
        split_pct = split_dist[i] * 100
        diff = abs(overall_pct - split_pct)
        total_diff += diff
        
        print(f"  {class_name:<20} {overall_pct:>9.2f}% {split_pct:>9.2f}% {diff:>9.2f}%")
    
    print(f"  {'-'*60}")
    print(f"  Total absolute difference: {total_diff:.2f}%")
    
    print(f"\nClass counts in val+test:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = combined_counts.get(i, 0)
        if count > 0:
            print(f"  {class_name:<20} {count:>6,} beats")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Search for an optimal curated patient-wise val/test split.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing MIT-BIH data "
             "(default: ../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0)",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=6,
        help="Number of validation patients (default: 6)",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=6,
        help="Number of test patients (default: 6)",
    )
    parser.add_argument(
        "--top_n_candidates",
        type=int,
        default=24,
        help="How many of the most diverse patients to consider when searching (default: 24)",
    )
    parser.add_argument(
        "--target_val_test_ratio",
        type=float,
        default=0.25,
        help="Target fraction of total beats allocated to val+test together (default: 0.25 → 12.5%+12.5%)",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    
    print("Loading patient data...")
    patient_data, overall_counts = load_patient_class_counts(data_dir)
    
    print(f"Loaded {len(patient_data)} patients")
    print(f"Total beats: {sum(overall_counts.values()):,}")
    
    # Find optimal split
    best_splits, overall_dist, total_beats = find_optimal_split(
        patient_data,
        overall_counts,
        target_val_test_ratio=args.target_val_test_ratio,
        num_val=args.num_val,
        num_test=args.num_test,
        top_n_candidates=args.top_n_candidates,
    )
    
    # Print top 5 results (or all if fewer than 5)
    num_to_print = min(5, len(best_splits))
    
    for i in range(num_to_print):
        val_patients, test_patients, score, split_dist, combined_counts, val_beats, test_beats = best_splits[i]
        print_split_details(
            i + 1, val_patients, test_patients, score, 
            split_dist, combined_counts, overall_dist,
            patient_data, total_beats
        )
        
        # Print balance info
        print(f"\nBalance check:")
        print(f"  Val beats:  {val_beats:>6,} ({val_beats/total_beats*100:>5.2f}% of total, target: 12.5%)")
        print(f"  Test beats: {test_beats:>6,} ({test_beats/total_beats*100:>5.2f}% of total, target: 12.5%)")
        print(f"  Imbalance: {abs(val_beats - test_beats):>6,} beats ({abs(val_beats - test_beats)/(val_beats+test_beats)*100:>5.2f}%)")
    
    # Print command to use the best split
    print(f"\n{'='*80}")
    print(f"RECOMMENDED COMMAND")
    print(f"{'='*80}")
    
    val_patients, test_patients, _, _, _, _, _ = best_splits[0]
    val_str = " ".join(sorted(val_patients))
    test_str = " ".join(sorted(test_patients))
    
    print(f"\npython train.py --model simple_cnn \\")
    print(f"  --curated_val {val_str} \\")
    print(f"  --curated_test {test_str} \\")
    print(f"  --class_weights --epochs 20")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
