"""
Rhythm-Bounded Segmentation Dataset

This is an improved version of RhythmDataset that uses rhythm-bounded
non-overlapping segmentation to prevent patient-level bias.

Key difference from original:
- Original: Sliding window across entire record (many overlapping segments)
- This version: Non-overlapping segments within each rhythm annotation boundary
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from typing import List, Tuple, Dict, Optional
from collections import Counter

# Import from original dataset
from dataset import (
    RHYTHM_CLASS_MAPPING,
    RHYTHM_CLASS_MAPPING_SIMPLE,
    CLASS_NAMES,
    create_patient_splits,
    create_segment_wise_splits
)


class RhythmDatasetBounded(Dataset):
    """
    PyTorch Dataset for rhythm classification using rhythm-bounded segmentation
    
    Unlike the original RhythmDataset which uses overlapping sliding windows,
    this version creates non-overlapping segments within each rhythm annotation
    boundary to prevent over-representation of long stable rhythms.
    """
    
    def __init__(self,
                 record_names: List[str],
                 data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                 segment_length: float = 10.0,
                 lead: int = 0,
                 normalize: bool = True,
                 use_simple_mapping: bool = True,
                 allow_overlap: bool = False,
                 overlap_ratio: float = 0.5):
        """
        Initialize RhythmDatasetBounded
        
        Parameters:
        -----------
        record_names : List[str]
            List of record names to include
        data_dir : str
            Directory containing MIT-BIH data
        segment_length : float
            Length of each ECG segment in seconds
        lead : int
            Which ECG lead to use (0 or 1)
        normalize : bool
            Whether to normalize each segment
        use_simple_mapping : bool
            If True, uses simplified 4-class mapping
        allow_overlap : bool
            If True, allows overlap within rhythm annotations
        overlap_ratio : float
            If allow_overlap=True, ratio of overlap (0.5 = 50% overlap)
        """
        self.record_names = record_names
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.lead = lead
        self.normalize = normalize
        self.use_simple_mapping = use_simple_mapping
        self.allow_overlap = allow_overlap
        self.overlap_ratio = overlap_ratio
        
        # Choose mapping
        self.class_mapping = RHYTHM_CLASS_MAPPING_SIMPLE if use_simple_mapping else RHYTHM_CLASS_MAPPING
        
        # Storage for segments and labels
        self.segments = []
        self.labels = []
        self.record_ids = []
        self.rhythm_strings = []
        self.annotation_ids = []  # Track which annotation each segment came from
        
        # Load all segments using rhythm-bounded approach
        self._load_rhythm_bounded_segments()
        
        # Print dataset statistics
        self._print_statistics()
    
    def _load_rhythm_bounded_segments(self):
        """
        Load ECG segments using rhythm-bounded non-overlapping segmentation
        
        For each rhythm annotation:
        1. Find start and end samples
        2. Create non-overlapping (or minimally overlapping) segments within bounds
        3. Skip segments that would cross rhythm boundaries
        """
        records_with_data = 0
        records_without_rhythms = []
        
        print(f"\n{'='*70}")
        print("RHYTHM-BOUNDED SEGMENTATION")
        print(f"{'='*70}")
        print(f"Segment length: {self.segment_length}s")
        print(f"Overlap: {'Yes' if self.allow_overlap else 'No'}")
        if self.allow_overlap:
            print(f"Overlap ratio: {self.overlap_ratio:.0%}")
        print(f"{'='*70}\n")
        
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
                
                # Calculate stride
                if self.allow_overlap:
                    stride_samples = int(segment_samples * (1 - self.overlap_ratio))
                else:
                    stride_samples = segment_samples  # No overlap
                
                # Find all rhythm change points
                rhythm_changes = []
                for i in range(len(annotation.sample)):
                    if annotation.aux_note[i] and annotation.aux_note[i] in self.class_mapping:
                        rhythm_changes.append({
                            'sample': annotation.sample[i],
                            'rhythm': annotation.aux_note[i]
                        })
                
                if not rhythm_changes:
                    records_without_rhythms.append(record_name)
                    continue
                
                records_with_data += 1
                segments_from_record = 0
                
                # Process each rhythm annotation
                for annotation_idx, change in enumerate(rhythm_changes):
                    start_sample = change['sample']
                    rhythm = change['rhythm']
                    
                    # Determine end of this rhythm segment
                    if annotation_idx < len(rhythm_changes) - 1:
                        end_sample = rhythm_changes[annotation_idx + 1]['sample']
                    else:
                        end_sample = len(signal)
                    
                    # Calculate duration of this rhythm annotation
                    annotation_duration_samples = end_sample - start_sample
                    annotation_duration_seconds = annotation_duration_samples / fs
                    
                    # Skip if annotation is too short
                    if annotation_duration_samples < segment_samples:
                        continue
                    
                    # Extract segments from within this rhythm annotation
                    current_pos = start_sample
                    segments_from_annotation = 0
                    
                    while current_pos + segment_samples <= end_sample:
                        # Extract segment
                        segment = signal[current_pos:current_pos + segment_samples]
                        
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
                        self.annotation_ids.append(f"{record_name}_ann{annotation_idx}")
                        
                        segments_from_record += 1
                        segments_from_annotation += 1
                        
                        # Move to next position
                        current_pos += stride_samples
                    
                    # Optional: print per-annotation stats for debugging
                    # print(f"    {rhythm} annotation ({annotation_duration_seconds:.1f}s) → {segments_from_annotation} segments")
                
                if segments_from_record > 0:
                    print(f"  Loaded {segments_from_record:4d} segments from record {record_name}")
                    
            except Exception as e:
                print(f"  Warning: Error loading record {record_name}: {e}")
                continue
        
        if records_without_rhythms:
            print(f"\n  Note: {len(records_without_rhythms)} records had no rhythm annotations")
            print(f"  Records skipped: {records_without_rhythms[:10]}{'...' if len(records_without_rhythms) > 10 else ''}")
        
        print(f"\n  Successfully loaded data from {records_with_data} records")
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n{'='*70}")
        print("RHYTHM DATASET STATISTICS (Rhythm-Bounded)")
        print(f"{'='*70}")
        print(f"Total segments: {len(self.segments):,}")
        print(f"Records: {len(set(self.record_ids))} unique")
        print(f"Rhythm annotations: {len(set(self.annotation_ids))} unique")
        print(f"Segment length: {self.segment_length}s")
        
        if len(self.labels) == 0:
            print("No segments loaded!")
            return
        
        # Count segments per class
        label_counts = Counter(self.labels)
        print(f"\nClass distribution:")
        for class_id in sorted(label_counts.keys()):
            count = label_counts[class_id]
            percentage = (count / len(self.labels)) * 100
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            print(f"  {class_name:25s} (class {class_id}): {count:6,} ({percentage:5.2f}%)")
        
        # Show rhythm annotation distribution
        rhythm_counts = Counter(self.rhythm_strings)
        if rhythm_counts:
            print(f"\nRhythm annotation distribution:")
            for rhythm, count in rhythm_counts.most_common():
                percentage = (count / len(self.rhythm_strings)) * 100
                print(f"  {rhythm:10s}: {count:6,} ({percentage:5.2f}%)")
        
        # Show segments per patient
        record_counts = Counter(self.record_ids)
        print(f"\nSegments per patient:")
        print(f"  Mean: {np.mean(list(record_counts.values())):.1f}")
        print(f"  Std:  {np.std(list(record_counts.values())):.1f}")
        print(f"  Min:  {min(record_counts.values())}")
        print(f"  Max:  {max(record_counts.values())}")
        
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Get a single ECG segment and rhythm label
        
        Returns:
        --------
        signal : torch.Tensor
            ECG segment of shape [1, T]
        label : torch.Tensor
            Integer class label (rhythm type)
        """
        segment = self.segments[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        signal = torch.from_numpy(segment).float().unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label
    
    def get_num_classes(self):
        """Return the number of classes"""
        return len(CLASS_NAMES) if self.use_simple_mapping else 10


def create_dataloaders_bounded(train_records: List[str],
                               val_records: List[str],
                               test_records: List[str],
                               data_dir: str = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                               batch_size: int = 32,
                               num_workers: int = 4,
                               **dataset_kwargs) -> Tuple:
    """
    Create PyTorch DataLoaders using rhythm-bounded segmentation
    
    Parameters:
    -----------
    train_records, val_records, test_records : List[str]
        Record names for each split
    data_dir : str
        Directory containing MIT-BIH data
    batch_size : int
        Batch size for DataLoader
    num_workers : int
        Number of worker processes
    **dataset_kwargs : dict
        Additional arguments for RhythmDatasetBounded
        
    Returns:
    --------
    train_loader, val_loader, test_loader, num_classes
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    print("\n" + "="*70)
    print("Creating TRAINING dataset...")
    print("="*70)
    train_dataset = RhythmDatasetBounded(train_records, data_dir=data_dir, **dataset_kwargs)
    
    print("\n" + "="*70)
    print("Creating VALIDATION dataset...")
    print("="*70)
    val_dataset = RhythmDatasetBounded(val_records, data_dir=data_dir, **dataset_kwargs)
    
    print("\n" + "="*70)
    print("Creating TEST dataset...")
    print("="*70)
    test_dataset = RhythmDatasetBounded(test_records, data_dir=data_dir, **dataset_kwargs)
    
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
    print("\n" + "="*70)
    print("Testing Rhythm-Bounded Segmentation")
    print("="*70 + "\n")
    
    # Create patient-wise splits
    train_records, val_records, test_records = create_patient_splits()
    
    # Test non-overlapping approach
    print("\n" + "="*70)
    print("OPTION 1: Non-Overlapping (Recommended)")
    print("="*70)
    dataset_no_overlap = RhythmDatasetBounded(
        train_records[:3],
        segment_length=10.0,
        allow_overlap=False
    )
    
    # Test with minimal overlap
    print("\n" + "="*70)
    print("OPTION 2: 50% Overlap (More segments)")
    print("="*70)
    dataset_with_overlap = RhythmDatasetBounded(
        train_records[:3],
        segment_length=10.0,
        allow_overlap=True,
        overlap_ratio=0.5
    )
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Non-overlapping: {len(dataset_no_overlap):,} segments")
    print(f"50% overlap:     {len(dataset_with_overlap):,} segments")
    print(f"Difference:      {len(dataset_with_overlap) - len(dataset_no_overlap):,} segments")
    print(f"Ratio:           {len(dataset_with_overlap) / len(dataset_no_overlap):.2f}x")
    print("="*70 + "\n")









