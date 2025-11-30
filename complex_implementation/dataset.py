"""
PyTorch Dataset for per-beat classification from MIT-BIH Arrhythmia Database

This module provides:
- BeatDataset: PyTorch Dataset for loading individual beat windows
- Helper functions for patient-wise train/val/test splits
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from typing import List, Tuple, Dict
from collections import Counter


# Beat annotation mapping based on AAMI EC57 standard
# We'll group similar beat types into broader classes for better model performance
BEAT_CLASS_MAPPING = {
    # Normal beats
    'N': 0,  # Normal beat
    'L': 0,  # Left bundle branch block beat (often grouped with normal)
    'R': 0,  # Right bundle branch block beat (often grouped with normal)
    'e': 0,  # Atrial escape beat
    'j': 0,  # Nodal (junctional) escape beat
    
    # Supraventricular ectopic beats
    'A': 1,  # Atrial premature beat
    'a': 1,  # Aberrated atrial premature beat
    'J': 1,  # Nodal (junctional) premature beat
    'S': 1,  # Supraventricular premature beat
    
    # Ventricular ectopic beats
    'V': 2,  # Premature ventricular contraction
    'E': 2,  # Ventricular escape beat
    
    # Fusion beats
    'F': 3,  # Fusion of ventricular and normal beat
    
    # Paced beats
    '/': 4,  # Paced beat
    'f': 4,  # Fusion of paced and normal beat
    
    # Unknown/Unclassifiable
    'Q': 5,  # Unclassifiable beat
    '?': 5,  # Beat not classified during learning
}

CLASS_NAMES = [
    'Normal',
    'Supraventricular',
    'Ventricular',
    'Fusion',
    'Paced',
    'Unknown'
]


class BeatDataset(Dataset):
    """
    PyTorch Dataset for per-beat ECG classification
    
    Extracts fixed-length windows around R-peaks and returns beat labels
    """
    
    def __init__(self, 
                 record_names: List[str],
                 data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                 window_size: float = 0.8,
                 lead: int = 0,
                 normalize: bool = True,
                 beat_wise_split: bool = False,
                 split_name: str = 'train',
                 train_ratio: float = 0.75,
                 val_ratio: float = 0.125,
                 random_seed: int = 42):
        """
        Initialize BeatDataset
        
        Parameters:
        -----------
        record_names : List[str]
            List of record names to include (e.g., ['100', '101'])
        data_dir : str
            Directory containing MIT-BIH data
        window_size : float
            Size of the window around R-peak in seconds (default: 0.8s)
        lead : int
            Which ECG lead to use (0 or 1)
        normalize : bool
            Whether to normalize each beat window to zero mean and unit variance
        beat_wise_split : bool
            If True, splits individual beats (not patients)
        split_name : str
            'train', 'val', or 'test' - used for beat-wise splitting
        train_ratio : float
            Fraction for training (beat-wise split only)
        val_ratio : float
            Fraction for validation (beat-wise split only)
        random_seed : int
            Random seed for beat-wise split reproducibility
        """
        self.record_names = record_names
        self.data_dir = data_dir
        self.window_size = window_size
        self.lead = lead
        self.normalize = normalize
        self.beat_wise_split = beat_wise_split
        self.split_name = split_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Storage for beat windows and labels
        self.beats = []
        self.labels = []
        self.record_ids = []  # Track which record each beat came from
        
        # Load all beats
        self._load_all_beats()
        
        # Apply beat-wise filtering if requested
        if self.beat_wise_split:
            self._apply_beat_wise_split()
        
        # Print dataset statistics
        self._print_statistics()
    
    def _load_all_beats(self):
        """Load all beat windows from all records"""
        for record_name in self.record_names:
            try:
                record_path = os.path.join(self.data_dir, record_name)
                
                # Load record and annotations
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Get sampling frequency and signal
                fs = record.fs
                signal = record.p_signal[:, self.lead]
                
                # Calculate window size in samples
                window_samples = int(self.window_size * fs)
                half_window = window_samples // 2
                
                # Process each annotation
                for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                    # Skip if beat type not in our mapping
                    if symbol not in BEAT_CLASS_MAPPING:
                        continue
                    
                    # Extract window around R-peak
                    start_idx = sample_idx - half_window
                    end_idx = sample_idx + half_window
                    
                    # Skip if window is out of bounds
                    if start_idx < 0 or end_idx >= len(signal):
                        continue
                    
                    # Extract beat window
                    beat_window = signal[start_idx:end_idx]
                    
                    # Normalize if requested
                    if self.normalize:
                        mean = np.mean(beat_window)
                        std = np.std(beat_window)
                        if std > 0:
                            beat_window = (beat_window - mean) / std
                    
                    # Store beat and label
                    self.beats.append(beat_window)
                    self.labels.append(BEAT_CLASS_MAPPING[symbol])
                    self.record_ids.append(record_name)
                    
            except Exception as e:
                print(f"Warning: Error loading record {record_name}: {e}")
                continue
    
    def _apply_beat_wise_split(self):
        """
        Filter beats based on beat-wise split ratios
        
        WARNING: This creates data leakage! Use only for prototyping.
        """
        np.random.seed(self.random_seed)
        
        # Create indices for all beats
        n_beats = len(self.beats)
        indices = np.arange(n_beats)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n_beats * self.train_ratio)
        val_end = train_end + int(n_beats * self.val_ratio)
        
        # Debug: print what we're doing
        print(f"    [DEBUG] Split {self.split_name}: train_ratio={self.train_ratio}, val_ratio={self.val_ratio}")
        print(f"    [DEBUG] n_beats={n_beats}, train_end={train_end}, val_end={val_end}")
        print(f"    [DEBUG] indices shape: {indices.shape}, first 5: {indices[:5]}")
        
        # Select indices for this split
        print(f"    [DEBUG] Checking split_name: '{self.split_name}' (type: {type(self.split_name)})")
        if self.split_name == 'train':
            selected_indices = indices[:train_end]
            print(f"    [DEBUG] MATCHED TRAIN - Train slice: [:{train_end}] → {len(selected_indices)} indices")
        elif self.split_name == 'val':
            selected_indices = indices[train_end:val_end]
            print(f"    [DEBUG] MATCHED VAL - Val slice: [{train_end}:{val_end}] → {len(selected_indices)} indices")
        else:  # test
            selected_indices = indices[val_end:]
            print(f"    [DEBUG] MATCHED TEST - Test slice: [{val_end}:] → {len(selected_indices)} indices")
        
        # Filter beats, labels, and record_ids
        self.beats = [self.beats[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.record_ids = [self.record_ids[i] for i in selected_indices]
        
        print(f"  WARNING: Beat-wise {self.split_name} split: kept {len(self.beats):,} / {n_beats:,} beats")
    
    def _print_statistics(self):
        """Print dataset statistics"""
        split_indicator = f" ({self.split_name.upper()} split)" if self.beat_wise_split else ""
        print(f"\nDataset Statistics{split_indicator}:")
        print(f"  Total beats: {len(self.beats)}")
        print(f"  Records: {len(self.record_names)}")
        
        if len(self.labels) == 0:
            print("  No beats loaded!")
            return
        
        # Count beats per class
        label_counts = Counter(self.labels)
        print(f"\n  Class distribution:")
        for class_id in sorted(label_counts.keys()):
            count = label_counts[class_id]
            percentage = (count / len(self.labels)) * 100
            print(f"    {CLASS_NAMES[class_id]:20s} (class {class_id}): {count:6d} ({percentage:5.2f}%)")
    
    def __len__(self):
        return len(self.beats)
    
    def __getitem__(self, idx):
        """
        Get a single beat window and label
        
        Returns:
        --------
        signal : torch.Tensor
            Beat window of shape [1, T] where T is window length in samples
        label : torch.Tensor
            Integer class label
        """
        # Get beat window and label
        beat = self.beats[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        # Add channel dimension: [T] -> [1, T]
        signal = torch.from_numpy(beat).float().unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label
    
    def get_num_classes(self):
        """Return the number of classes"""
        return len(CLASS_NAMES)


def create_beat_wise_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                           train_ratio: float = 0.75,
                           val_ratio: float = 0.125,
                           test_ratio: float = 0.125,
                           random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create beat-wise train/val/test splits by pooling ALL beats then splitting
    
    ⚠️  WARNING: This creates DATA LEAKAGE! Same patient's beats appear in multiple splits.
    ⚠️  Use ONLY for prototyping or to establish upper-bound performance.
    ⚠️  NOT suitable for clinical validation or publication.
    
    For production/research, use create_patient_splits() instead!
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIT-BIH data
    train_ratio : float
        Fraction of BEATS for training
    val_ratio : float
        Fraction of BEATS for validation
    test_ratio : float
        Fraction of BEATS for testing
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_records : List[str]
        ALL record names (for compatibility)
    val_records : List[str]
        ALL record names (for compatibility)
    test_records : List[str]
        ALL record names (for compatibility)
        
    Note: The actual split happens at the beat level in BeatDataset,
          this function returns all records for all splits to enable beat-level splitting.
    """
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    print("\n" + "="*70)
    print("WARNING: BEAT-WISE SPLIT ENABLED")
    print("="*70)
    print("This splits individual beats, NOT patients!")
    print("Same patient's beats will appear in train/val/test.")
    print("")
    print("Consequences:")
    print("  X Data leakage - model sees same patient in train & test")
    print("  X Overly optimistic performance estimates")
    print("  X NOT suitable for clinical validation")
    print("  X NOT publishable in medical journals")
    print("")
    print("Use ONLY for:")
    print("  + Quick prototyping")
    print("  + Establishing upper-bound performance")
    print("  + Debugging models")
    print("="*70 + "\n")
    
    # Return all records for all splits
    # The actual beat-level split will happen in the dataset loader
    return all_records, all_records, all_records


def create_curated_hybrid_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                                  test_patients: List[str] = None,
                                  train_ratio: float = 0.85,
                                  val_ratio: float = 0.15,
                                  random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create HYBRID curated test split:
    - Hold out specific patients for testing (pure patient-wise, no leakage)
    - Pool beats from remaining patients for train/val (beat-level split for class balance)
    
    This approach balances:
    ✅ Test generalization (patient-wise test set)
    ✅ Class balance (beat-wise train/val split)
    ⚠️  Train/val share patients (acceptable since val is just for tuning)
    
    Use cases:
    - When rare classes cluster in specific patients
    - When you need all classes represented in test set
    - When test validity is critical but train/val leakage is acceptable
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIT-BIH data
    test_patients : List[str]
        List of patient record IDs to hold out for testing (e.g., ['207', '217'])
        If None, will suggest patients based on diversity analysis
    train_ratio : float
        Fraction of remaining BEATS for training (default: 0.85)
    val_ratio : float
        Fraction of remaining BEATS for validation (default: 0.15)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_records : List[str]
        All non-test record names (beats will be pooled and split)
    val_records : List[str]
        All non-test record names (same as train, beats will be pooled)
    test_records : List[str]
        Held-out test patient record names (pure patient-wise)
        
    Example:
    --------
    >>> # Hold out patients 207 and 217 for testing
    >>> train, val, test = create_curated_hybrid_splits(
    ...     data_dir='data/mitdb',
    ...     test_patients=['207', '217'],
    ...     train_ratio=0.85,
    ...     val_ratio=0.15
    ... )
    >>> # Now train and val will contain all other patients (beats pooled)
    >>> # test will contain only patients 207 and 217 (patient-wise)
    """
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    # If no test patients specified, suggest some
    if test_patients is None or len(test_patients) == 0:
        print("\n" + "="*70)
        print("ERROR: No test patients specified!")
        print("="*70)
        print("Please specify test patients using --curated_test argument.")
        print("\nTo find diverse patients, run:")
        print("  python analyze_patient_diversity.py")
        print("\nExample:")
        print("  python train.py --model simple_cnn --curated_test 207 217")
        print("="*70 + "\n")
        raise ValueError("Must specify test_patients for curated hybrid split")
    
    # Validate test patients exist
    invalid_patients = [p for p in test_patients if p not in all_records]
    if invalid_patients:
        raise ValueError(f"Invalid test patients (not in dataset): {invalid_patients}")
    
    # Separate test patients from remaining patients
    test_records = sorted(test_patients)
    remaining_records = sorted([r for r in all_records if r not in test_patients])
    
    print("\n" + "="*70)
    print("CURATED HYBRID SPLIT")
    print("="*70)
    print(f"Test patients (held out, patient-wise): {test_records}")
    print(f"Remaining patients (will be beat-pooled): {len(remaining_records)} patients")
    print("")
    print("Strategy:")
    print("  ✅ Test set: Pure patient-wise (no leakage)")
    print("  ⚠️  Train/Val: Beat-wise split (shares patients for class balance)")
    print("")
    print("Why this works:")
    print("  - Test results are valid (true generalization to new patients)")
    print("  - Train/Val have better class balance (from beat pooling)")
    print("  - Val is only used for tuning (acceptable to share patients)")
    print("  - Final reported metric (test) is clinically valid ✅")
    print("="*70 + "\n")
    
    # For train and val, return all remaining records
    # The beat-level split will happen in BeatDataset
    train_records = remaining_records
    val_records = remaining_records
    
    print(f"Curated hybrid split:")
    print(f"  Test (patient-wise):  {len(test_records)} patients")
    print(f"  Train/Val (beat-wise): {len(remaining_records)} patients")
    print(f"    - Training beats:   {train_ratio*100:.0f}% of pooled beats")
    print(f"    - Validation beats: {val_ratio*100:.0f}% of pooled beats")
    print(f"\n  Test records: {test_records}")
    print(f"  Pooled records: {remaining_records[:5]}... (showing first 5)")
    
    return train_records, val_records, test_records


def create_curated_patient_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                                   val_patients: List[str] = None,
                                   test_patients: List[str] = None,
                                   random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create PURE PATIENT-WISE split with manually specified val and test patients.
    
    This is the most rigorous approach:
    ✅ Pure patient-wise (no data leakage anywhere)
    ✅ Manually curated validation and test sets
    ✅ Clinically valid for publications
    ⚠️  May have class imbalance (depends on patient selection)
    
    Use cases:
    - When you want complete control over which patients go where
    - When you've identified diverse patients using analyze_patient_diversity.py
    - For final publication-ready models
    - When you need clinically valid results across all splits
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIT-BIH data
    val_patients : List[str]
        List of patient record IDs for validation (e.g., ['203', '208', '213'])
    test_patients : List[str]
        List of patient record IDs for testing (e.g., ['215', '205', '210'])
    random_seed : int
        Random seed (not used, but kept for API consistency)
        
    Returns:
    --------
    train_records : List[str]
        Training patient IDs (all remaining patients)
    val_records : List[str]
        Validation patient IDs (specified val_patients)
    test_records : List[str]
        Test patient IDs (specified test_patients)
        
    Examples:
    ---------
    >>> train, val, test = create_curated_patient_splits(
    ...     data_dir='data/mitdb',
    ...     val_patients=['203', '208', '213', '233', '104', '223'],
    ...     test_patients=['215', '205', '210', '200', '214', '219']
    ... )
    >>> # val contains exactly the 6 specified patients
    >>> # test contains exactly the 6 specified patients
    >>> # train contains all remaining patients
    """
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    # Validate inputs
    if val_patients is None or len(val_patients) == 0:
        raise ValueError("Must specify val_patients for curated patient-wise split")
    if test_patients is None or len(test_patients) == 0:
        raise ValueError("Must specify test_patients for curated patient-wise split")
    
    # Check for overlap
    overlap = set(val_patients) & set(test_patients)
    if overlap:
        raise ValueError(f"Patients cannot be in both val and test: {overlap}")
    
    # Validate all patients exist
    invalid_val = [p for p in val_patients if p not in all_records]
    if invalid_val:
        raise ValueError(f"Invalid validation patients (not in dataset): {invalid_val}")
    
    invalid_test = [p for p in test_patients if p not in all_records]
    if invalid_test:
        raise ValueError(f"Invalid test patients (not in dataset): {invalid_test}")
    
    # Create splits
    val_records = sorted(val_patients)
    test_records = sorted(test_patients)
    train_records = sorted([r for r in all_records if r not in val_patients and r not in test_patients])
    
    # Print summary
    print(f"\nCurated patient-wise split:")
    print(f"  Training patients:   {len(train_records)}")
    print(f"  Validation patients: {len(val_records)}")
    print(f"  Test patients:       {len(test_records)}")
    print(f"  Total patients:      {len(all_records)}")
    print(f"\n  Train records: {train_records[:5]}... (showing first 5)")
    print(f"  Val records:   {val_records}")
    print(f"  Test records:  {test_records}")
    
    return train_records, val_records, test_records


def create_patient_splits(data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                         train_ratio: float = 0.75,
                         val_ratio: float = 0.125,
                         test_ratio: float = 0.125,
                         random_seed: int = 42,
                         stratified: bool = True,
                         beat_wise: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """
    Create patient-wise train/val/test splits
    
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
        If True, attempts to balance class distribution across splits
    beat_wise : bool
        If True, splits beats instead of patients (⚠️ creates data leakage!)
        
    Returns:
    --------
    train_records : List[str]
        List of record names for training
    val_records : List[str]
        List of record names for validation
    test_records : List[str]
        List of record names for testing
    """
    # If beat-wise split requested, use different function
    if beat_wise:
        return create_beat_wise_splits(data_dir, train_ratio, val_ratio, test_ratio, random_seed)
    # Get all available records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    if not stratified:
        # Original simple random split
        records = np.array(all_records)
        np.random.shuffle(records)
        
        n_records = len(records)
        train_end = int(n_records * train_ratio)
        val_end = train_end + int(n_records * val_ratio)
        
        train_records = records[:train_end].tolist()
        val_records = records[train_end:val_end].tolist()
        test_records = records[val_end:].tolist()
    else:
        # Stratified split: analyze class distribution per record
        print("\n  Analyzing class distribution per record for stratified split...")
        
        record_class_counts = {}
        for record_name in all_records:
            try:
                record_path = os.path.join(data_dir, record_name)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Count beats per class for this record
                class_counts = Counter()
                for symbol in annotation.symbol:
                    if symbol in BEAT_CLASS_MAPPING:
                        class_counts[BEAT_CLASS_MAPPING[symbol]] += 1
                
                record_class_counts[record_name] = class_counts
            except Exception as e:
                print(f"  Warning: Could not analyze record {record_name}: {e}")
                record_class_counts[record_name] = Counter()
        
        # Greedy stratification: iteratively assign patients to splits
        # Goal: balance ALL class counts across splits as much as possible
        
        # Initialize split accumulators
        split_class_counts = {
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
            'test': len(all_records)  # Will fill remaining
        }
        
        # Sort records by total beats (larger records assigned first)
        sorted_records = sorted(all_records, 
                              key=lambda r: sum(record_class_counts[r].values()), 
                              reverse=True)
        
        for record_name in sorted_records:
            # Determine which split needs this record most
            # Prioritize splits that are under capacity and have lowest class representation
            
            best_split = None
            best_score = float('inf')
            
            for split_name in ['train', 'val', 'test']:
                # Skip if split is full
                if len(split_records[split_name]) >= target_counts[split_name]:
                    continue
                
                # Calculate imbalance score if we add this record to this split
                # Score = sum of squared differences from ideal distribution per class
                temp_counts = split_class_counts[split_name].copy()
                temp_counts.update(record_class_counts[record_name])
                
                # Compute imbalance (variance across splits for each class)
                imbalance = 0
                for class_id in range(6):  # All 6 classes
                    # How many of this class in each split if we add this record
                    train_count = split_class_counts['train'][class_id]
                    val_count = split_class_counts['val'][class_id]
                    test_count = split_class_counts['test'][class_id]
                    
                    if split_name == 'train':
                        train_count = temp_counts[class_id]
                    elif split_name == 'val':
                        val_count = temp_counts[class_id]
                    else:
                        test_count = temp_counts[class_id]
                    
                    # Ideal: proportional to split size
                    total_class = train_count + val_count + test_count
                    if total_class > 0:
                        ideal_train = total_class * train_ratio
                        ideal_val = total_class * val_ratio
                        ideal_test = total_class * test_ratio
                        
                        # Squared differences
                        imbalance += (train_count - ideal_train)**2
                        imbalance += (val_count - ideal_val)**2
                        imbalance += (test_count - ideal_test)**2
                
                if imbalance < best_score:
                    best_score = imbalance
                    best_split = split_name
            
            # Assign to best split (or train if all full)
            if best_split is None:
                best_split = 'train'
            
            split_records[best_split].append(record_name)
            split_class_counts[best_split].update(record_class_counts[record_name])
        
        train_records = split_records['train']
        val_records = split_records['val']
        test_records = split_records['test']
        
        # Shuffle within each split
        np.random.shuffle(train_records)
        np.random.shuffle(val_records)
        np.random.shuffle(test_records)
        
        # Print stratification results
        print(f"\n  Stratified class distribution:")
        print(f"  {'Class':<20} {'Train':>10} {'Val':>10} {'Test':>10}")
        print(f"  {'-'*54}")
        for class_id, class_name in enumerate(CLASS_NAMES):
            train_count = split_class_counts['train'][class_id]
            val_count = split_class_counts['val'][class_id]
            test_count = split_class_counts['test'][class_id]
            print(f"  {class_name:<20} {train_count:>10,} {val_count:>10,} {test_count:>10,}")
    
    print(f"\nPatient-wise split:")
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
                       batch_size: int = 64,
                       num_workers: int = 4,
                       return_train_dataset: bool = False,
                       beat_wise_split: bool = False,
                       hybrid_mode: bool = False,
                       train_ratio: float = 0.75,
                       val_ratio: float = 0.125,
                       random_seed: int = 42,
                       **dataset_kwargs) -> Tuple:
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
    return_train_dataset : bool
        If True, also returns the training dataset object
    beat_wise_split : bool
        If True, uses beat-wise splitting (data leakage!)
    hybrid_mode : bool
        If True, test is patient-wise but train/val are beat-wise (curated hybrid)
    **dataset_kwargs : dict
        Additional arguments to pass to BeatDataset
        
    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
    num_classes : int
    train_dataset : BeatDataset (optional, if return_train_dataset=True)
    """
    from torch.utils.data import DataLoader
    
    # Determine beat_wise_split for each dataset
    if hybrid_mode:
        # HYBRID: train/val use beat-wise, test uses patient-wise
        train_beat_wise = True
        val_beat_wise = True
        test_beat_wise = False
    else:
        # NORMAL: all use the same setting
        train_beat_wise = beat_wise_split
        val_beat_wise = beat_wise_split
        test_beat_wise = beat_wise_split
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = BeatDataset(train_records, data_dir=data_dir, 
                                beat_wise_split=train_beat_wise,
                                split_name='train',
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                random_seed=random_seed,
                                **dataset_kwargs)
    
    print("\nCreating validation dataset...")
    val_dataset = BeatDataset(val_records, data_dir=data_dir,
                             beat_wise_split=val_beat_wise,
                             split_name='val',
                             train_ratio=train_ratio,
                             val_ratio=val_ratio,
                             random_seed=random_seed,
                             **dataset_kwargs)
    
    print("\nCreating test dataset...")
    test_dataset = BeatDataset(test_records, data_dir=data_dir,
                              beat_wise_split=test_beat_wise,
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
        pin_memory=True
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
    
    if return_train_dataset:
        return train_loader, val_loader, test_loader, num_classes, train_dataset
    else:
        return train_loader, val_loader, test_loader, num_classes


# Example usage
if __name__ == "__main__":
    # Create patient-wise splits
    train_records, val_records, test_records = create_patient_splits()
    
    # Create a small dataset for testing
    print("\n" + "="*70)
    print("Testing BeatDataset with first 3 training records...")
    print("="*70)
    
    dataset = BeatDataset(train_records[:3], window_size=0.8, lead=0)
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    # Get a sample
    signal, label = dataset[0]
    print(f"\nSample signal shape: {signal.shape}")
    print(f"Sample label: {label} ({CLASS_NAMES[label]})")

