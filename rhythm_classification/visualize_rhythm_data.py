"""
Visualize rhythm annotations from MIT-BIH Arrhythmia Database

This script provides comprehensive visualization of:
- Which records have rhythm annotations
- Distribution of rhythm types
- Timeline of rhythm changes
- Example ECG segments with rhythm labels
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
from dataset import RHYTHM_CLASS_MAPPING_SIMPLE, CLASS_NAMES

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


def analyze_all_records(data_dir='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'):
    """Analyze rhythm annotations across all records"""
    
    # Get all records
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    print(f"Analyzing {len(all_records)} MIT-BIH records for rhythm annotations...")
    print("="*80)
    
    records_with_rhythms = []
    rhythm_data = {}
    all_rhythm_types = Counter()
    
    for record_name in all_records:
        try:
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Find rhythm annotations
            rhythm_changes = []
            for i in range(len(annotation.sample)):
                aux_note = annotation.aux_note[i]
                if aux_note and aux_note in RHYTHM_CLASS_MAPPING_SIMPLE:
                    rhythm_changes.append({
                        'sample': annotation.sample[i],
                        'time': annotation.sample[i] / record.fs,
                        'rhythm': aux_note,
                        'class_id': RHYTHM_CLASS_MAPPING_SIMPLE[aux_note]
                    })
                    all_rhythm_types[aux_note] += 1
            
            if rhythm_changes:
                records_with_rhythms.append(record_name)
                
                # Calculate rhythm durations
                record_duration = len(record.p_signal) / record.fs
                rhythm_durations = []
                
                for i, change in enumerate(rhythm_changes):
                    if i < len(rhythm_changes) - 1:
                        duration = rhythm_changes[i+1]['time'] - change['time']
                    else:
                        duration = record_duration - change['time']
                    
                    rhythm_durations.append({
                        **change,
                        'duration': duration
                    })
                
                rhythm_data[record_name] = {
                    'changes': rhythm_changes,
                    'durations': rhythm_durations,
                    'record_duration': record_duration,
                    'fs': record.fs
                }
        
        except Exception as e:
            print(f"Error reading {record_name}: {e}")
            continue
    
    return records_with_rhythms, rhythm_data, all_rhythm_types


def print_summary(records_with_rhythms, rhythm_data, all_rhythm_types):
    """Print summary statistics"""
    
    print(f"\n{'='*80}")
    print(f"RHYTHM ANNOTATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nRecords with rhythm annotations: {len(records_with_rhythms)}/48")
    print(f"Records: {', '.join(records_with_rhythms)}")
    
    print(f"\n{'='*80}")
    print(f"RHYTHM TYPES DISTRIBUTION")
    print(f"{'='*80}")
    print(f"\n{'Rhythm':12} {'Count':>8} {'Class':>8}  {'Class Name'}")
    print("-"*80)
    
    for rhythm, count in sorted(all_rhythm_types.items(), key=lambda x: -x[1]):
        class_id = RHYTHM_CLASS_MAPPING_SIMPLE[rhythm]
        class_name = CLASS_NAMES[class_id]
        print(f"{rhythm:12} {count:>8} {class_id:>8}  {class_name}")
    
    print(f"\n{'='*80}")
    print(f"PER-RECORD DETAILS")
    print(f"{'='*80}")
    
    for record_name in sorted(records_with_rhythms):
        data = rhythm_data[record_name]
        print(f"\nRecord {record_name} ({data['record_duration']/60:.1f} minutes):")
        
        # Count rhythms in this record
        rhythm_counts = Counter()
        total_duration = {}
        
        for dur_info in data['durations']:
            rhythm_counts[dur_info['rhythm']] += 1
            total_duration[dur_info['rhythm']] = total_duration.get(dur_info['rhythm'], 0) + dur_info['duration']
        
        print(f"  {'Rhythm':12} {'Changes':>8} {'Total Time':>12} {'% of Record':>12}")
        print("  " + "-"*50)
        
        for rhythm in sorted(rhythm_counts.keys()):
            count = rhythm_counts[rhythm]
            duration = total_duration[rhythm]
            pct = (duration / data['record_duration']) * 100
            print(f"  {rhythm:12} {count:>8} {duration:>10.1f}s {pct:>11.1f}%")


def plot_rhythm_timeline(record_name, rhythm_data, save_dir='rhythm_classification/visualizations'):
    """Plot rhythm timeline for a record"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    data = rhythm_data[record_name]
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Define colors for each class
    colors = {
        0: '#2ecc71',  # Normal - green
        1: '#e74c3c',  # Atrial - red
        2: '#9b59b6',  # Ventricular - purple
        3: '#f39c12'   # Paced/Pre-excitation - orange
    }
    
    # Plot rhythm segments
    for dur_info in data['durations']:
        start_time = dur_info['time']
        duration = dur_info['duration']
        class_id = dur_info['class_id']
        rhythm = dur_info['rhythm']
        
        ax.barh(0, duration, left=start_time, height=0.8, 
               color=colors.get(class_id, '#95a5a6'),
               edgecolor='black', linewidth=0.5,
               label=CLASS_NAMES[class_id] if CLASS_NAMES[class_id] not in ax.get_legend_handles_labels()[1] else "")
        
        # Add rhythm label in the middle of the segment
        mid_time = start_time + duration / 2
        ax.text(mid_time, 0, rhythm, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlim(0, data['record_duration'])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Record {record_name} - Rhythm Timeline ({data["record_duration"]/60:.1f} minutes)', 
                fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'rhythm_timeline_{record_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved timeline to: {save_path}")
    
    return save_path


def plot_ecg_with_rhythm(record_name, rhythm_data, 
                         start_sec=0, duration_sec=30,
                         data_dir='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                         save_dir='rhythm_classification/visualizations'):
    """Plot ECG signal with rhythm annotation overlay"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load record
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    fs = record.fs
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    
    # Get signal
    time = np.arange(start_sample, end_sample) / fs
    signal = record.p_signal[start_sample:end_sample, 0]  # Lead 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot ECG
    ax1.plot(time, signal, 'b-', linewidth=0.8)
    ax1.set_ylabel(f'{record.sig_name[0]} ({record.units[0]})', fontsize=12, fontweight='bold')
    ax1.set_title(f'Record {record_name} - ECG with Rhythm Annotations ({start_sec}s - {start_sec+duration_sec}s)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add beat markers
    beat_mask = (annotation.sample >= start_sample) & (annotation.sample < end_sample)
    beat_samples = annotation.sample[beat_mask]
    beat_symbols = np.array(annotation.symbol)[beat_mask]
    
    for sample, symbol in zip(beat_samples, beat_symbols):
        beat_time = sample / fs
        ax1.axvline(beat_time, color='red', alpha=0.2, linewidth=0.5)
        ax1.text(beat_time, ax1.get_ylim()[1] * 0.95, symbol,
                ha='center', fontsize=7, color='red', alpha=0.7)
    
    # Plot rhythm timeline
    data = rhythm_data[record_name]
    colors = {0: '#2ecc71', 1: '#e74c3c', 2: '#9b59b6', 3: '#f39c12'}
    
    for dur_info in data['durations']:
        rhythm_start = dur_info['time']
        rhythm_end = rhythm_start + dur_info['duration']
        
        # Only plot if it overlaps with our time window
        if rhythm_end < start_sec or rhythm_start > (start_sec + duration_sec):
            continue
        
        plot_start = max(rhythm_start, start_sec)
        plot_end = min(rhythm_end, start_sec + duration_sec)
        
        class_id = dur_info['class_id']
        rhythm = dur_info['rhythm']
        
        ax2.barh(0, plot_end - plot_start, left=plot_start, height=0.6,
                color=colors.get(class_id, '#95a5a6'),
                edgecolor='black', linewidth=1,
                label=CLASS_NAMES[class_id] if CLASS_NAMES[class_id] not in [h.get_label() for h in ax2.get_legend_handles_labels()[0]] else "")
        
        # Add rhythm label
        mid_time = (plot_start + plot_end) / 2
        ax2.text(mid_time, 0, rhythm, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    
    ax2.set_xlim(start_sec, start_sec + duration_sec)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rhythm', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'ecg_with_rhythm_{record_name}_{start_sec}s.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved ECG plot to: {save_path}")
    
    return save_path


def plot_class_distribution(rhythm_data, save_dir='rhythm_classification/visualizations'):
    """Plot overall rhythm class distribution"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Count total duration per class across all records
    class_durations = Counter()
    
    for record_name, data in rhythm_data.items():
        for dur_info in data['durations']:
            class_id = dur_info['class_id']
            duration = dur_info['duration']
            class_durations[class_id] += duration
    
    # Convert to percentages
    total_duration = sum(class_durations.values())
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = []
    durations = []
    percentages = []
    colors_list = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    for class_id in range(len(CLASS_NAMES)):
        duration = class_durations.get(class_id, 0)
        pct = (duration / total_duration) * 100 if total_duration > 0 else 0
        
        classes.append(CLASS_NAMES[class_id])
        durations.append(duration)
        percentages.append(pct)
    
    bars = ax.bar(classes, percentages, color=colors_list[:len(classes)], 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, pct, dur) in enumerate(zip(bars, percentages, durations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%\n({dur/60:.1f} min)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage of Total Rhythm Duration', fontsize=12, fontweight='bold')
    ax.set_title('Rhythm Class Distribution Across All Records', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(percentages) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rhythm_class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved class distribution to: {save_path}")
    
    return save_path


def plot_records_heatmap(rhythm_data, save_dir='rhythm_classification/visualizations'):
    """Create heatmap showing which records have which rhythms"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique rhythm types
    all_rhythms = set()
    for data in rhythm_data.values():
        for change in data['changes']:
            all_rhythms.add(change['rhythm'])
    
    all_rhythms = sorted(all_rhythms)
    records = sorted(rhythm_data.keys())
    
    # Create matrix: records x rhythms
    matrix = np.zeros((len(records), len(all_rhythms)))
    
    for i, record_name in enumerate(records):
        for dur_info in rhythm_data[record_name]['durations']:
            j = all_rhythms.index(dur_info['rhythm'])
            matrix[i, j] += dur_info['duration']
    
    # Convert to percentages per record
    for i in range(len(records)):
        total = matrix[i, :].sum()
        if total > 0:
            matrix[i, :] = (matrix[i, :] / total) * 100
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(all_rhythms)))
    ax.set_yticks(np.arange(len(records)))
    ax.set_xticklabels(all_rhythms, fontsize=10, fontweight='bold')
    ax.set_yticklabels(records, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Record Duration', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(records)):
        for j in range(len(all_rhythms)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                             ha="center", va="center", color="black" if matrix[i, j] < 50 else "white",
                             fontsize=8)
    
    ax.set_title('Rhythm Distribution Across Records\n(% of each record\'s duration)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Rhythm Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Record ID', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rhythm_records_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap to: {save_path}")
    
    return save_path


def main():
    """Main visualization function"""
    
    data_dir = '../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'
    
    print("\n" + "="*80)
    print("RHYTHM ANNOTATION VISUALIZATION")
    print("="*80 + "\n")
    
    # Analyze all records
    records_with_rhythms, rhythm_data, all_rhythm_types = analyze_all_records(data_dir)
    
    if not records_with_rhythms:
        print("\n❌ No rhythm annotations found!")
        print("Make sure the data has been downloaded with aux_note field.")
        return
    
    # Print summary
    print_summary(records_with_rhythms, rhythm_data, all_rhythm_types)
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Plot class distribution
    print("1. Creating rhythm class distribution plot...")
    plot_class_distribution(rhythm_data)
    
    # Plot heatmap
    print("\n2. Creating records heatmap...")
    plot_records_heatmap(rhythm_data)
    
    # Plot timelines for a few interesting records
    print("\n3. Creating rhythm timelines...")
    for i, record_name in enumerate(records_with_rhythms[:5]):
        print(f"   {record_name}...")
        plot_rhythm_timeline(record_name, rhythm_data)
    
    # Plot ECG examples for a few records
    print("\n4. Creating ECG plots with rhythm annotations...")
    for i, record_name in enumerate(records_with_rhythms[:3]):
        print(f"   {record_name}...")
        # Find an interesting segment (where rhythm changes)
        data = rhythm_data[record_name]
        if len(data['changes']) > 1:
            # Start at the first rhythm change
            start_time = max(0, data['changes'][1]['time'] - 10)
        else:
            start_time = 0
        
        plot_ecg_with_rhythm(record_name, rhythm_data, 
                            start_sec=start_time, duration_sec=30,
                            data_dir=data_dir)
    
    print(f"\n{'='*80}")
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nVisualization files saved to: rhythm_classification/visualizations/")
    print("\nGenerated files:")
    print("  - rhythm_class_distribution.png    (Overall class distribution)")
    print("  - rhythm_records_heatmap.png       (Which records have which rhythms)")
    print("  - rhythm_timeline_*.png            (Timeline for each record)")
    print("  - ecg_with_rhythm_*.png            (ECG with rhythm overlay)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

