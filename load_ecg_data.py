"""
Helper script to load and visualize ECG data from MIT-BIH database
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def load_ecg_record(record_name: str, data_dir: str = 'data/mitdb') -> Tuple:
    """
    Load an ECG record from the local MIT-BIH database
    
    Parameters:
    -----------
    record_name : str
        Name of the record (e.g., '100', '101')
    data_dir : str
        Directory containing the MIT-BIH data
        
    Returns:
    --------
    record : wfdb.Record
        Record object containing signal data
    annotation : wfdb.Annotation
        Annotation object containing beat labels
    """
    import os
    
    # Construct the full path to the record
    record_path = os.path.join(data_dir, record_name)
    
    # Load the record from local directory
    record = wfdb.rdrecord(record_path)
    
    # Load the annotations from local directory
    annotation = wfdb.rdann(record_path, 'atr')
    
    return record, annotation


def visualize_ecg(record_name: str, 
                  data_dir: str = 'data/mitdb',
                  start_sec: float = 0,
                  duration_sec: float = 10,
                  show_annotations: bool = True):
    """
    Visualize ECG signal with annotations
    
    Parameters:
    -----------
    record_name : str
        Name of the record (e.g., '100', '101')
    data_dir : str
        Directory containing the MIT-BIH data
    start_sec : float
        Start time in seconds
    duration_sec : float
        Duration to display in seconds
    show_annotations : bool
        Whether to show beat annotations
    """
    # Load data
    record, annotation = load_ecg_record(record_name, data_dir)
    
    # Calculate sample indices
    fs = record.fs  # Sampling frequency
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    
    # Create time array
    time = np.arange(start_sample, end_sample) / fs
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot both leads
    for i, lead_name in enumerate(record.sig_name):
        signal = record.p_signal[start_sample:end_sample, i]
        axes[i].plot(time, signal, linewidth=0.5)
        axes[i].set_ylabel(f'{lead_name} ({record.units[i]})', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add annotations if requested
        if show_annotations:
            # Find annotations in the time window
            mask = (annotation.sample >= start_sample) & (annotation.sample < end_sample)
            ann_samples = annotation.sample[mask]
            ann_symbols = np.array(annotation.symbol)[mask]
            
            # Plot annotation markers
            for sample, symbol in zip(ann_samples, ann_symbols):
                ann_time = sample / fs
                axes[i].axvline(ann_time, color='red', alpha=0.3, linewidth=0.5)
                if i == 0:  # Only show symbol on top plot
                    axes[i].text(ann_time, axes[i].get_ylim()[1] * 0.95, symbol, 
                               ha='center', fontsize=8, color='red')
    
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    fig.suptitle(f'ECG Record {record_name} ({start_sec}s - {start_sec + duration_sec}s)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def get_annotation_statistics(record_name: str, data_dir: str = 'data/mitdb'):
    """
    Get statistics about beat annotations in a record
    
    Parameters:
    -----------
    record_name : str
        Name of the record (e.g., '100', '101')
    data_dir : str
        Directory containing the MIT-BIH data
    """
    _, annotation = load_ecg_record(record_name, data_dir)
    
    # Count different beat types
    unique, counts = np.unique(annotation.symbol, return_counts=True)
    
    print(f"\nAnnotation Statistics for Record {record_name}")
    print("=" * 50)
    print(f"Total beats: {len(annotation.symbol)}")
    print("\nBeat type distribution:")
    
    # Common beat type descriptions
    beat_types = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block beat',
        'R': 'Right bundle branch block beat',
        'B': 'Bundle branch block beat (unspecified)',
        'A': 'Atrial premature beat',
        'a': 'Aberrated atrial premature beat',
        'J': 'Nodal (junctional) premature beat',
        'S': 'Supraventricular premature beat',
        'V': 'Premature ventricular contraction',
        'r': 'R-on-T premature ventricular contraction',
        'F': 'Fusion of ventricular and normal beat',
        'e': 'Atrial escape beat',
        'j': 'Nodal (junctional) escape beat',
        'n': 'Supraventricular escape beat',
        'E': 'Ventricular escape beat',
        '/': 'Paced beat',
        'f': 'Fusion of paced and normal beat',
        'Q': 'Unclassifiable beat',
        '?': 'Beat not classified during learning'
    }
    
    for symbol, count in zip(unique, counts):
        description = beat_types.get(symbol, 'Unknown')
        percentage = (count / len(annotation.symbol)) * 100
        print(f"  {symbol}: {count:5d} ({percentage:5.2f}%) - {description}")


def list_available_records(data_dir: str = 'data/mitdb'):
    """
    List all available records in the database
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the MIT-BIH data
    """
    import os
    
    # Get all .hea files (header files)
    files = os.listdir(data_dir)
    records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    print(f"\nAvailable records in {data_dir}:")
    print("=" * 50)
    for i, record in enumerate(records, 1):
        print(f"{i:2d}. {record}", end="  ")
        if i % 8 == 0:
            print()
    print()
    
    return records


# Example usage
if __name__ == "__main__":
    # List available records
    records = list_available_records()
    
    # Example: Analyze record 100
    record_name = '100'
    print(f"\n\nAnalyzing Record {record_name}...")
    get_annotation_statistics(record_name)
    
    # Visualize a 10-second segment
    print(f"\nVisualizing Record {record_name}...")
    visualize_ecg(record_name, start_sec=0, duration_sec=10)

