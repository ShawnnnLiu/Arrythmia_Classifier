"""
Analyze rhythm diversity across patients in MIT-BIH database

This script helps identify which patients have which rhythm types,
enabling the creation of optimal test sets that cover all rhythm classes.
"""

import os
import sys
import wfdb
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pandas as pd

# Import rhythm mappings
from dataset import RHYTHM_CLASS_MAPPING_SIMPLE, CLASS_NAMES


def analyze_patient_rhythms(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0') -> Dict:
    """
    Analyze rhythm distribution across all patients
    
    Returns:
    --------
    patient_rhythms : dict
        Dictionary mapping patient ID to rhythm counts
    """
    # Get all records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    print(f"Analyzing rhythm diversity across {len(all_records)} patients...")
    print("="*70)
    
    patient_rhythms = {}
    rhythm_to_patients = defaultdict(list)
    
    for record_name in all_records:
        try:
            record_path = os.path.join(data_dir, record_name)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Count rhythm types for this patient
            rhythm_counts = Counter()
            rhythm_segments = []
            
            for i in range(len(annotation.sample)):
                aux_note = annotation.aux_note[i]
                if aux_note and aux_note in RHYTHM_CLASS_MAPPING_SIMPLE:
                    rhythm_counts[aux_note] += 1
                    rhythm_segments.append({
                        'sample': annotation.sample[i],
                        'rhythm': aux_note,
                        'class_id': RHYTHM_CLASS_MAPPING_SIMPLE[aux_note]
                    })
            
            if rhythm_counts:
                patient_rhythms[record_name] = {
                    'rhythm_counts': rhythm_counts,
                    'total_rhythms': sum(rhythm_counts.values()),
                    'unique_rhythms': len(rhythm_counts),
                    'segments': rhythm_segments
                }
                
                # Track which patients have which rhythms
                for rhythm in rhythm_counts.keys():
                    rhythm_to_patients[rhythm].append(record_name)
        
        except Exception as e:
            continue
    
    print(f"\nFound rhythm annotations in {len(patient_rhythms)} patients")
    print(f"Patients without rhythm annotations: {len(all_records) - len(patient_rhythms)}")
    
    return patient_rhythms, rhythm_to_patients


def print_rhythm_distribution(patient_rhythms: Dict, rhythm_to_patients: Dict):
    """Print overall rhythm distribution"""
    print("\n" + "="*70)
    print("RHYTHM DISTRIBUTION ACROSS ALL PATIENTS")
    print("="*70)
    
    # Count total occurrences
    total_counts = Counter()
    for patient_data in patient_rhythms.values():
        total_counts.update(patient_data['rhythm_counts'])
    
    print(f"\n{'Rhythm':<15} {'Count':>8} {'Patients':>10} {'Patient IDs'}")
    print("-"*70)
    
    for rhythm, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        n_patients = len(rhythm_to_patients[rhythm])
        patient_ids = ', '.join(rhythm_to_patients[rhythm][:5])
        if len(rhythm_to_patients[rhythm]) > 5:
            patient_ids += '...'
        print(f"{rhythm:<15} {count:>8} {n_patients:>10}     {patient_ids}")


def find_diverse_patients(patient_rhythms: Dict, top_n: int = 10):
    """Find patients with the most diverse rhythm types"""
    print("\n" + "="*70)
    print(f"TOP {top_n} MOST DIVERSE PATIENTS")
    print("="*70)
    print(f"\n{'Patient':<10} {'Unique':>8} {'Total':>8} {'Rhythms Present'}")
    print("-"*70)
    
    # Sort by number of unique rhythms, then by total count
    sorted_patients = sorted(
        patient_rhythms.items(),
        key=lambda x: (x[1]['unique_rhythms'], x[1]['total_rhythms']),
        reverse=True
    )
    
    diverse_patients = []
    for record_name, data in sorted_patients[:top_n]:
        unique = data['unique_rhythms']
        total = data['total_rhythms']
        rhythms = ', '.join(sorted(data['rhythm_counts'].keys()))
        
        print(f"{record_name:<10} {unique:>8} {total:>8}     {rhythms}")
        diverse_patients.append(record_name)
    
    return diverse_patients


def suggest_test_set(patient_rhythms: Dict, rhythm_to_patients: Dict, n_test: int = 6):
    """
    Suggest optimal test set to maximize rhythm coverage
    
    Uses a greedy algorithm to select patients that together cover
    as many rhythm types as possible.
    """
    print("\n" + "="*70)
    print(f"SUGGESTED TEST SET ({n_test} patients)")
    print("="*70)
    
    # Greedy selection
    selected_patients = []
    covered_rhythms = set()
    
    # Start with the patient that has the most unique rhythms
    sorted_patients = sorted(
        patient_rhythms.items(),
        key=lambda x: x[1]['unique_rhythms'],
        reverse=True
    )
    
    for _ in range(n_test):
        best_patient = None
        best_score = -1
        best_new_rhythms = set()
        
        for record_name, data in sorted_patients:
            if record_name in selected_patients:
                continue
            
            # Count how many new rhythms this patient would add
            patient_rhythms_set = set(data['rhythm_counts'].keys())
            new_rhythms = patient_rhythms_set - covered_rhythms
            
            # Score = number of new rhythms + diversity bonus
            score = len(new_rhythms) + 0.1 * len(patient_rhythms_set)
            
            if score > best_score:
                best_score = score
                best_patient = record_name
                best_new_rhythms = new_rhythms
        
        if best_patient:
            selected_patients.append(best_patient)
            covered_rhythms.update(best_new_rhythms)
    
    # Print selected patients
    print("\nSelected patients for test set:")
    print(f"\n{'Patient':<10} {'Rhythms':>8} {'New Rhythms Added'}")
    print("-"*70)
    
    temp_covered = set()
    for patient in selected_patients:
        data = patient_rhythms[patient]
        patient_rhythms_set = set(data['rhythm_counts'].keys())
        new_rhythms = patient_rhythms_set - temp_covered
        rhythms_str = ', '.join(sorted(patient_rhythms_set))
        new_rhythms_str = ', '.join(sorted(new_rhythms)) if new_rhythms else 'None'
        
        print(f"{patient:<10} {len(patient_rhythms_set):>8}     {new_rhythms_str}")
        temp_covered.update(patient_rhythms_set)
    
    print(f"\nTotal rhythms covered: {len(covered_rhythms)}")
    print(f"Test set patients: {selected_patients}")
    
    # Map to class distribution
    class_coverage = defaultdict(int)
    for patient in selected_patients:
        for rhythm, count in patient_rhythms[patient]['rhythm_counts'].items():
            class_id = RHYTHM_CLASS_MAPPING_SIMPLE[rhythm]
            class_coverage[class_id] += count
    
    print("\nClass distribution in test set:")
    for class_id in sorted(class_coverage.keys()):
        class_name = CLASS_NAMES[class_id]
        count = class_coverage[class_id]
        print(f"  {class_name:<25} {count:>6} segments")
    
    return selected_patients


def suggest_val_set(patient_rhythms: Dict, test_patients: List[str], n_val: int = 6):
    """Suggest validation set (excluding test patients)"""
    print("\n" + "="*70)
    print(f"SUGGESTED VALIDATION SET ({n_val} patients)")
    print("="*70)
    
    # Filter out test patients
    available_patients = {
        k: v for k, v in patient_rhythms.items() if k not in test_patients
    }
    
    # Use same greedy algorithm
    selected_patients = []
    covered_rhythms = set()
    
    sorted_patients = sorted(
        available_patients.items(),
        key=lambda x: x[1]['unique_rhythms'],
        reverse=True
    )
    
    for _ in range(n_val):
        best_patient = None
        best_score = -1
        
        for record_name, data in sorted_patients:
            if record_name in selected_patients:
                continue
            
            patient_rhythms_set = set(data['rhythm_counts'].keys())
            new_rhythms = patient_rhythms_set - covered_rhythms
            score = len(new_rhythms) + 0.1 * len(patient_rhythms_set)
            
            if score > best_score:
                best_score = score
                best_patient = record_name
        
        if best_patient:
            selected_patients.append(best_patient)
            covered_rhythms.update(set(available_patients[best_patient]['rhythm_counts'].keys()))
    
    print(f"\nValidation set patients: {selected_patients}")
    
    return selected_patients


def create_summary_table(patient_rhythms: Dict):
    """Create a summary table of all patients"""
    print("\n" + "="*70)
    print("COMPLETE PATIENT RHYTHM SUMMARY")
    print("="*70)
    
    # Create DataFrame
    rows = []
    for patient, data in sorted(patient_rhythms.items()):
        row = {
            'Patient': patient,
            'Unique_Rhythms': data['unique_rhythms'],
            'Total_Annotations': data['total_rhythms']
        }
        
        # Add counts for each rhythm type
        for rhythm in sorted(set([r for d in patient_rhythms.values() for r in d['rhythm_counts'].keys()])):
            row[rhythm] = data['rhythm_counts'].get(rhythm, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by diversity
    df = df.sort_values('Unique_Rhythms', ascending=False)
    
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    output_file = 'rhythm_classification/patient_rhythm_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved detailed analysis to: {output_file}")


def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("RHYTHM DIVERSITY ANALYSIS")
    print("MIT-BIH Arrhythmia Database")
    print("="*70 + "\n")
    
    # Analyze all patients
    patient_rhythms, rhythm_to_patients = analyze_patient_rhythms()
    
    if not patient_rhythms:
        print("\nNo rhythm annotations found in the database!")
        print("Make sure the data is downloaded and accessible.")
        return
    
    # Print distributions
    print_rhythm_distribution(patient_rhythms, rhythm_to_patients)
    
    # Find diverse patients
    diverse_patients = find_diverse_patients(patient_rhythms, top_n=10)
    
    # Suggest optimal test set
    test_patients = suggest_test_set(patient_rhythms, rhythm_to_patients, n_test=6)
    
    # Suggest validation set
    val_patients = suggest_val_set(patient_rhythms, test_patients, n_val=6)
    
    # Create summary table
    create_summary_table(patient_rhythms)
    
    # Print final recommendations
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)
    print("\nTo use these splits in training:")
    print("\n1. Modify dataset.py to add a curated split function")
    print("\n2. Or use the suggested patients manually:")
    print(f"\n   Test patients:  {test_patients}")
    print(f"   Val patients:   {val_patients}")
    print(f"   Train patients: (all remaining)")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

