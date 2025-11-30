"""
PyTorch Dataset for rhythm classification from MIT-BIH Arrhythmia Database

This module provides:
- RhythmDataset: PyTorch Dataset for loading ECG segments with rhythm labels
- Helper functions for patient-wise and segment-wise train/val/test splits
- Rhythm annotation parsing from aux_note field
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from typing import List, Tuple, Dict, Optional
from collections import Counter


# Rhythm class mapping based on MIT-BIH aux_note annotations
# The aux_note field contains rhythm change markers like (AFIB, (N, (VT, etc.
RHYTHM_CLASS_MAPPING = {
    # Normal rhythms
    '(N': 0,      # Normal sinus rhythm
    
    # Atrial arrhythmias
    '(AFIB': 1,   # Atrial fibrillation
    '(AFL': 2,    # Atrial flutter
    '(AB': 1,     # Atrial bigeminy (grouped with AFIB)
    
    # Ventricular arrhythmias
    '(VT': 3,     # Ventricular tachycardia
    '(VFL': 3,    # Ventricular flutter (grouped with VT)
    '(B': 4,      # Ventricular bigeminy
    '(T': 5,      # Ventricular trigeminy
    
    # Bradycardia
    '(SBR': 6,    # Sinus bradycardia
    
    # Other
    '(SVTA': 7,   # Supraventricular tachyarrhythmia
    '(PREX': 8,   # Pre-excitation
    '(NOD': 9,    # Nodal rhythm
}

# Simplified mapping for better class balance
RHYTHM_CLASS_MAPPING_SIMPLE = {
    '(N': 0,      # Normal sinus rhythm
    '(AFIB': 1,   # Atrial fibrillation
    '(AFL': 1,    # Atrial flutter (grouped with AFIB)
    '(AB': 1,     # Atrial bigeminy
    '(VT': 2,     # Ventricular tachycardia
    '(VFL': 2,    # Ventricular flutter
    '(B': 2,      # Ventricular bigeminy (grouped with VT)
    '(T': 2,      # Ventricular trigeminy
    '(SBR': 0,    # Sinus bradycardia (grouped with normal)
    '(SVTA': 1,   # Supraventricular (grouped with atrial)
    '(PREX': 3,   # Pre-excitation
    '(NOD': 0,    # Nodal rhythm (grouped with normal)
}

CLASS_NAMES = [
    'Normal',
    'Atrial_Arrhythmia',
    'Ventricular_Arrhythmia',
    'Pre-excitation'
]


class RhythmDataset(Dataset):
    """
    PyTorch Dataset for rhythm classification
    
    Extracts fixed-length ECG segments and assigns rhythm labels based on
    the rhythm annotation that covers that segment.
    """
    
    def __init__(self,
                 record_names: List[str],
                 data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                 segment_length: float = 10.0,
                 segment_stride: float = 5.0,
                 lead: int = 0,
                 normalize: bool = True,
                 use_simple_mapping: bool = True,
                 segment_wise_split: bool = False,
                 split_name: str = 'train',
                 train_ratio: float = 0.75,
                 val_ratio: float = 0.125,
                 random_seed: int = 42):
        """
        Initialize RhythmDataset
        
        Parameters:
        -----------
        record_names : List[str]
            List of record names to include (e.g., ['100', '101'])
        data_dir : str
            Directory containing MIT-BIH data
        segment_length : float
            Length of each ECG segment in seconds (default: 10s)
        segment_stride : float
            Stride between segments in seconds (default: 5s, 50% overlap)
        lead : int
            Which ECG lead to use (0 or 1)
        normalize : bool
            Whether to normalize each segment to zero mean and unit variance
        use_simple_mapping : bool
            If True, uses simplified 4-class mapping; else uses detailed 10-class
        segment_wise_split : bool
            If True, splits individual segments (not patients) - causes data leakage!
        split_name : str
            'train', 'val', or 'test' - used for segment-wise splitting
        train_ratio : float
            Fraction for training (segment-wise split only)
        val_ratio : float
            Fraction for validation (segment-wise split only)
        random_seed : int
            Random seed for segment-wise split reproducibility
        """
        self.record_names = record_names
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.lead = lead
        self.normalize = normalize
        self.use_simple_mapping = use_simple_mapping
        self.segment_wise_split = segment_wise_split
        self.split_name = split_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Choose mapping
        self.class_mapping = RHYTHM_CLASS_MAPPING_SIMPLE if use_simple_mapping else RHYTHM_CLASS_MAPPING
        
        # Storage for segments and labels
        self.segments = []
        self.labels = []
        self.record_ids = []
        self.rhythm_strings = []  # Store the rhythm annotation string for debugging
        
        # Load all segments
        self._load_all_segments()
        
        # Apply segment-wise filtering if requested
        if self.segment_wise_split:
            self._apply_segment_wise_split()
        
        # Print dataset statistics
        self._print_statistics()
    
    def _get_rhythm_at_sample(self, annotation, sample_idx: int) -> Optional[str]:
        """
        Find the rhythm annotation that covers the given sample index
        
        Rhythm annotations mark the START of a rhythm segment.
        We need to find which rhythm segment contains the given sample.
        """
        # Find all rhythm change points
        rhythm_changes = []
        for i in range(len(annotation.sample)):
            if annotation.aux_note[i] and annotation.aux_note[i] in self.class_mapping:
                rhythm_changes.append({
                    'sample': annotation.sample[i],
                    'rhythm': annotation.aux_note[i]
                })
        
        if not rhythm_changes:
            return None
        
        # Find the rhythm that covers this sample
        # The rhythm at a sample is the most recent rhythm change before it
        current_rhythm = None
        for change in rhythm_changes:
            if change['sample'] <= sample_idx:
                current_rhythm = change['rhythm']
            else:
                break
        
        return current_rhythm
    
    def _load_all_segments(self):
        """Load all ECG segments from all records with rhythm labels"""
        records_with_data = 0
        records_without_rhythms = []
        
        for record_name in self.record_names:
            try:
                record_path = os.path.join(self.data_dir, record_name)
                
                # Load record and annotations
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Get sampling frequency and signal
                fs = record.fs
                signal = record.p_signal[:, self.lead]
                
                # Calculate segment size in samples
                segment_samples = int(self.segment_length * fs)
                stride_samples = int(self.segment_stride * fs)
                
                # Check if this record has any rhythm annotations
                has_rhythms = any(annotation.aux_note[i] in self.class_mapping 
                                 for i in range(len(annotation.sample)))
                
                if not has_rhythms:
                    records_without_rhythms.append(record_name)
                    continue
                
                records_with_data += 1
                segments_from_record = 0
                
                # Extract sliding windows
                for start_idx in range(0, len(signal) - segment_samples, stride_samples):
                    end_idx = start_idx + segment_samples
                    
                    # Get the rhythm at the middle of this segment
                    mid_idx = (start_idx + end_idx) // 2
                    rhythm = self._get_rhythm_at_sample(annotation, mid_idx)
                    
                    if rhythm is None:
                        continue
                    
                    # Extract segment
                    segment = signal[start_idx:end_idx]
                    
                    # Normalize if requested
                    if self.normalize:
                        mean = np.mean(segment)
                        std = np.std(segment)
                        if std > 0:
                            segment = (segment - mean) / std
                    
                    # Store segment and label
                    self.segments.append(segment)
                    self.labels.append(self.class_mapping[rhythm])
                    self.record_ids.append(record_name)
                    self.rhythm_strings.append(rhythm)
                    segments_from_record += 1
                
                if segments_from_record > 0:
                    print(f"  Loaded {segments_from_record:4d} segments from record {record_name}")
                    
            except Exception as e:
                print(f"  Warning: Error loading record {record_name}: {e}")
                continue
        
        if records_without_rhythms:
            print(f"\n  Note: {len(records_without_rhythms)} records had no rhythm annotations")
            print(f"  Records skipped: {records_without_rhythms[:10]}{'...' if len(records_without_rhythms) > 10 else ''}")
        
        print(f"\n  Successfully loaded data from {records_with_data} records")
    
    def _apply_segment_wise_split(self):
        """
        Filter segments based on segment-wise split ratios
        
        WARNING: This creates data leakage! Use only for comparison.
        """
        np.random.seed(self.random_seed)
        
        # Create indices for all segments
        n_segments = len(self.segments)
        indices = np.arange(n_segments)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n_segments * self.train_ratio)
        val_end = train_end + int(n_segments * self.val_ratio)
        
        # Select indices for this split
        if self.split_name == 'train':
            selected_indices = indices[:train_end]
        elif self.split_name == 'val':
            selected_indices = indices[train_end:val_end]
        else:  # test
            selected_indices = indices[val_end:]
        
        # Filter segments, labels, and record_ids
        self.segments = [self.segments[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.record_ids = [self.record_ids[i] for i in selected_indices]
        self.rhythm_strings = [self.rhythm_strings[i] for i in selected_indices]
        
        print(f"\n  WARNING: Segment-wise {self.split_name} split: kept {len(self.segments):,} / {n_segments:,} segments")
    
    def _print_statistics(self):
        """Print dataset statistics"""
        split_indicator = f" ({self.split_name.upper()} split)" if self.segment_wise_split else ""
        print(f"\nRhythm Dataset Statistics{split_indicator}:")
        print(f"  Total segments: {len(self.segments)}")
        print(f"  Records: {len(set(self.record_ids))} unique")
        print(f"  Segment length: {self.segment_length}s")
        print(f"  Segment stride: {self.segment_stride}s")
        
        if len(self.labels) == 0:
            print("  No segments loaded!")
            return
        
        # Count segments per class
        label_counts = Counter(self.labels)
        print(f"\n  Class distribution:")
        for class_id in sorted(label_counts.keys()):
            count = label_counts[class_id]
            percentage = (count / len(self.labels)) * 100
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            print(f"    {class_name:25s} (class {class_id}): {count:6d} ({percentage:5.2f}%)")
        
        # Show rhythm annotation distribution
        rhythm_counts = Counter(self.rhythm_strings)
        if rhythm_counts:
            print(f"\n  Rhythm annotation distribution:")
            for rhythm, count in rhythm_counts.most_common():
                percentage = (count / len(self.rhythm_strings)) * 100
                print(f"    {rhythm:10s}: {count:6d} ({percentage:5.2f}%)")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Get a single ECG segment and rhythm label
        
        Returns:
        --------
        signal : torch.Tensor
            ECG segment of shape [1, T] where T is segment length in samples
        label : torch.Tensor
            Integer class label (rhythm type)
        """
        # Get segment and label
        segment = self.segments[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        # Add channel dimension: [T] -> [1, T]
        signal = torch.from_numpy(segment).float().unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label
    
    def get_num_classes(self):
        """Return the number of classes"""
        return len(CLASS_NAMES) if self.use_simple_mapping else 10


def create_segment_wise_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                               train_ratio: float = 0.75,
                               val_ratio: float = 0.125,
                               test_ratio: float = 0.125,
                               random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create segment-wise train/val/test splits by pooling ALL segments then splitting
    
    ⚠️  WARNING: This creates DATA LEAKAGE! Same patient's segments appear in multiple splits.
    ⚠️  Use ONLY for comparison or to establish upper-bound performance.
    
    For production/research, use create_patient_splits() instead!
    """
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    print("\n" + "="*70)
    print("WARNING: SEGMENT-WISE SPLIT ENABLED")
    print("="*70)
    print("This splits individual segments, NOT patients!")
    print("Same patient's segments will appear in train/val/test.")
    print("")
    print("Consequences:")
    print("  X Data leakage - model sees same patient in train & test")
    print("  X Overly optimistic performance estimates")
    print("  X NOT suitable for clinical validation")
    print("")
    print("Use ONLY for:")
    print("  + Quick prototyping")
    print("  + Establishing upper-bound performance")
    print("  + Comparison with patient-wise split")
    print("="*70 + "\n")
    
    # Return all records for all splits
    # The actual segment-level split will happen in the dataset loader
    return all_records, all_records, all_records


def create_patient_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                         train_ratio: float = 0.75,
                         val_ratio: float = 0.125,
                         test_ratio: float = 0.125,
                         random_seed: int = 42,
                         stratified: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """
    Create patient-wise train/val/test splits for rhythm classification
    
    Ensures no patient appears in multiple splits, which is crucial for
    evaluating generalization to new patients.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIT-BIH data
    train_ratio : float
        Fraction of patients for training
    val_ratio : float
        Fraction of patients for validation
    test_ratio : float
        Fraction of patients for testing
    random_seed : int
        Random seed for reproducibility
    stratified : bool
        If True, attempts to balance rhythm distribution across splits
        
    Returns:
    --------
    train_records : List[str]
        List of record names for training
    val_records : List[str]
        List of record names for validation
    test_records : List[str]
        List of record names for testing
    """
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    if not stratified:
        # Simple random split
        records = np.array(all_records)
        np.random.shuffle(records)
        
        n_records = len(records)
        train_end = int(n_records * train_ratio)
        val_end = train_end + int(n_records * val_ratio)
        
        train_records = records[:train_end].tolist()
        val_records = records[train_end:val_end].tolist()
        test_records = records[val_end:].tolist()
    else:
        # Stratified split: analyze rhythm distribution per record
        print("\n  Analyzing rhythm distribution per record for stratified split...")
        
        record_rhythm_counts = {}
        for record_name in all_records:
            try:
                record_path = os.path.join(data_dir, record_name)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Count rhythm types for this record
                rhythm_counts = Counter()
                for i in range(len(annotation.sample)):
                    if annotation.aux_note[i] and annotation.aux_note[i] in RHYTHM_CLASS_MAPPING_SIMPLE:
                        rhythm_counts[RHYTHM_CLASS_MAPPING_SIMPLE[annotation.aux_note[i]]] += 1
                
                record_rhythm_counts[record_name] = rhythm_counts
            except Exception as e:
                print(f"  Warning: Could not analyze record {record_name}: {e}")
                record_rhythm_counts[record_name] = Counter()
        
        # Greedy stratification
        split_rhythm_counts = {
            'train': Counter(),
            'val': Counter(),
            'test': Counter()
        }
        split_records = {
            'train': [],
            'val': [],
            'test': []
        }
        target_counts = {
            'train': int(len(all_records) * train_ratio),
            'val': int(len(all_records) * val_ratio),
            'test': len(all_records)
        }
        
        # Sort records by total rhythm annotations (larger records first)
        sorted_records = sorted(all_records,
                              key=lambda r: sum(record_rhythm_counts[r].values()),
                              reverse=True)
        
        for record_name in sorted_records:
            # Determine which split needs this record most
            best_split = None
            best_score = float('inf')
            
            for split_name in ['train', 'val', 'test']:
                # Skip if split is full
                if len(split_records[split_name]) >= target_counts[split_name]:
                    continue
                
                # Calculate imbalance score
                temp_counts = split_rhythm_counts[split_name].copy()
                temp_counts.update(record_rhythm_counts[record_name])
                
                imbalance = 0
                for class_id in range(len(CLASS_NAMES)):
                    train_count = split_rhythm_counts['train'][class_id]
                    val_count = split_rhythm_counts['val'][class_id]
                    test_count = split_rhythm_counts['test'][class_id]
                    
                    if split_name == 'train':
                        train_count = temp_counts[class_id]
                    elif split_name == 'val':
                        val_count = temp_counts[class_id]
                    else:
                        test_count = temp_counts[class_id]
                    
                    total_class = train_count + val_count + test_count
                    if total_class > 0:
                        ideal_train = total_class * train_ratio
                        ideal_val = total_class * val_ratio
                        ideal_test = total_class * test_ratio
                        
                        imbalance += (train_count - ideal_train)**2
                        imbalance += (val_count - ideal_val)**2
                        imbalance += (test_count - ideal_test)**2
                
                if imbalance < best_score:
                    best_score = imbalance
                    best_split = split_name
            
            if best_split is None:
                best_split = 'train'
            
            split_records[best_split].append(record_name)
            split_rhythm_counts[best_split].update(record_rhythm_counts[record_name])
        
        train_records = split_records['train']
        val_records = split_records['val']
        test_records = split_records['test']
        
        # Shuffle within each split
        np.random.shuffle(train_records)
        np.random.shuffle(val_records)
        np.random.shuffle(test_records)
        
        # Print stratification results
        print(f"\n  Stratified rhythm distribution:")
        print(f"  {'Class':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print(f"  {'-'*59}")
        for class_id, class_name in enumerate(CLASS_NAMES):
            train_count = split_rhythm_counts['train'][class_id]
            val_count = split_rhythm_counts['val'][class_id]
            test_count = split_rhythm_counts['test'][class_id]
            print(f"  {class_name:<25} {train_count:>10,} {val_count:>10,} {test_count:>10,}")
    
    print(f"\nPatient-wise split for rhythm classification:")
    print(f"  Training:   {len(train_records)} records ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_records)} records ({val_ratio*100:.0f}%)")
    print(f"  Test:       {len(test_records)} records ({test_ratio*100:.0f}%)")
    print(f"\n  Train records: {train_records[:5]}... (showing first 5)")
    print(f"  Val records:   {val_records[:3]}... (showing first 3)")
    print(f"  Test records:  {test_records[:3]}... (showing first 3)")
    
    return train_records, val_records, test_records


def create_dataloaders(train_records: List[str],
                       val_records: List[str],
                       test_records: List[str],
                       data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                       batch_size: int = 32,
                       num_workers: int = 4,
                       segment_wise_split: bool = False,
                       train_ratio: float = 0.75,
                       val_ratio: float = 0.125,
                       random_seed: int = 42,
                       **dataset_kwargs,) -> Tuple:
    """
    Create PyTorch DataLoaders for train/val/test sets
    
    Parameters:
    -----------
    train_records : List[str]
        Training record names
    val_records : List[str]
        Validation record names
    test_records : List[str]
        Test record names
    data_dir : str
        Directory containing MIT-BIH data
    batch_size : int
        Batch size for DataLoader
    num_workers : int
        Number of worker processes for data loading
    segment_wise_split : bool
        If True, uses segment-wise splitting (data leakage!)
    **dataset_kwargs : dict
        Additional arguments to pass to RhythmDataset
        
    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
    num_classes : int
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = RhythmDataset(train_records, data_dir=data_dir,
                                  segment_wise_split=segment_wise_split,
                                  split_name='train',
                                  train_ratio=train_ratio,
                                  val_ratio=val_ratio,
                                  random_seed=random_seed,
                                  **dataset_kwargs)
    
    print("\nCreating validation dataset...")
    val_dataset = RhythmDataset(val_records, data_dir=data_dir,
                                segment_wise_split=segment_wise_split,
                                split_name='val',
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                random_seed=random_seed,
                                **dataset_kwargs)
    
    print("\nCreating test dataset...")
    test_dataset = RhythmDataset(test_records, data_dir=data_dir,
                                 segment_wise_split=segment_wise_split,
                                 split_name='test',
                                 train_ratio=train_ratio,
                                 val_ratio=val_ratio,
                                 random_seed=random_seed,
                                 **dataset_kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = train_dataset.get_num_classes()
    
    return train_loader, val_loader, test_loader, num_classes


# Example usage
if __name__ == "__main__":
    print("Testing RhythmDataset...")
    print("="*70)
    
    # Create patient-wise splits
    train_records, val_records, test_records = create_patient_splits()
    
    # Create a small dataset for testing
    print("\n" + "="*70)
    print("Testing RhythmDataset with first 3 training records...")
    print("="*70)
    
    dataset = RhythmDataset(train_records[:3], segment_length=10.0, segment_stride=5.0, lead=0)
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    if len(dataset) > 0:
        # Get a sample
        signal, label = dataset[0]
        print(f"\nSample signal shape: {signal.shape}")
        print(f"Sample label: {label} ({CLASS_NAMES[label]})")