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
                 data_dir: str = 'data/mitdb',
                 window_size: float = 0.8,
                 lead: int = 0,
                 normalize: bool = True):
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
        """
        self.record_names = record_names
        self.data_dir = data_dir
        self.window_size = window_size
        self.lead = lead
        self.normalize = normalize
        
        # Storage for beat windows and labels
        self.beats = []
        self.labels = []
        self.record_ids = []  # Track which record each beat came from
        
        # Load all beats
        self._load_all_beats()
        
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
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"  Total beats: {len(self.beats)}")
        print(f"  Records: {len(self.record_names)}")
        
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


def create_patient_splits(data_dir: str = 'data/mitdb',
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
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
    
    # Shuffle records
    records = np.array(all_records)
    np.random.shuffle(records)
    
    # Calculate split points
    n_records = len(records)
    train_end = int(n_records * train_ratio)
    val_end = train_end + int(n_records * val_ratio)
    
    # Split records
    train_records = records[:train_end].tolist()
    val_records = records[train_end:val_end].tolist()
    test_records = records[val_end:].tolist()
    
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
                       data_dir: str = 'data/mitdb',
                       batch_size: int = 64,
                       num_workers: int = 4,
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
    **dataset_kwargs : dict
        Additional arguments to pass to BeatDataset
        
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
    train_dataset = BeatDataset(train_records, data_dir=data_dir, **dataset_kwargs)
    
    print("\nCreating validation dataset...")
    val_dataset = BeatDataset(val_records, data_dir=data_dir, **dataset_kwargs)
    
    print("\nCreating test dataset...")
    test_dataset = BeatDataset(test_records, data_dir=data_dir, **dataset_kwargs)
    
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

