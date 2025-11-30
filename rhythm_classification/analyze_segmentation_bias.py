"""
Analyze segmentation bias in rhythm classification dataset

This script compares the current overlapping sliding window approach
with a rhythm-bounded non-overlapping approach to show how different
segmentation strategies affect data distribution.
"""

import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from dataset import RHYTHM_CLASS_MAPPING_SIMPLE, CLASS_NAMES


def analyze_current_approach(data_dir='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                             segment_length=10.0,
                             segment_stride=5.0):
    """Analyze current overlapping sliding window approach"""
    
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    record_stats = {}
    total_segments = Counter()
    segments_per_record = {}
    segments_per_rhythm_annotation = defaultdict(list)
    
    for record_name in all_records:
        try:
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            fs = record.fs
            signal = record.p_signal[:, 0]
            
            segment_samples = int(segment_length * fs)
            stride_samples = int(segment_stride * fs)
            
            # Find rhythm annotations
            rhythm_changes = []
            for i in range(len(annotation.sample)):
                if annotation.aux_note[i] and annotation.aux_note[i] in RHYTHM_CLASS_MAPPING_SIMPLE:
                    rhythm_changes.append({
                        'sample': annotation.sample[i],
                        'rhythm': annotation.aux_note[i],
                        'class_id': RHYTHM_CLASS_MAPPING_SIMPLE[annotation.aux_note[i]]
                    })
            
            if not rhythm_changes:
                continue
            
            # Calculate rhythm durations
            record_duration = len(signal) / fs
            rhythm_durations = []
            
            for i, change in enumerate(rhythm_changes):
                if i < len(rhythm_changes) - 1:
                    duration = (rhythm_changes[i+1]['sample'] - change['sample']) / fs
                else:
                    duration = record_duration - (change['sample'] / fs)
                
                rhythm_durations.append({
                    **change,
                    'duration': duration
                })
            
            # Current approach: sliding window
            record_segment_count = Counter()
            annotation_segment_counts = []
            
            for start_idx in range(0, len(signal) - segment_samples, stride_samples):
                end_idx = start_idx + segment_samples
                mid_idx = (start_idx + end_idx) // 2
                
                # Find rhythm at midpoint
                current_rhythm = None
                annotation_idx = None
                for idx, change in enumerate(rhythm_changes):
                    if change['sample'] <= mid_idx:
                        current_rhythm = change['rhythm']
                        annotation_idx = idx
                    else:
                        break
                
                if current_rhythm:
                    record_segment_count[current_rhythm] += 1
            
            # Count segments per annotation
            for i, dur_info in enumerate(rhythm_durations):
                # Count how many sliding windows have midpoint in this annotation
                start_sample = dur_info['sample']
                if i < len(rhythm_durations) - 1:
                    end_sample = rhythm_durations[i + 1]['sample']
                else:
                    end_sample = len(signal)
                
                count = 0
                for start_idx in range(0, len(signal) - segment_samples, stride_samples):
                    mid_idx = (start_idx + start_idx + segment_samples) // 2
                    if start_sample <= mid_idx < end_sample:
                        count += 1
                
                annotation_segment_counts.append(count)
                segments_per_rhythm_annotation[dur_info['rhythm']].append({
                    'duration': dur_info['duration'],
                    'segments': count,
                    'record': record_name
                })
            
            segments_per_record[record_name] = sum(record_segment_count.values())
            total_segments.update(record_segment_count)
            
            record_stats[record_name] = {
                'rhythm_changes': rhythm_changes,
                'durations': rhythm_durations,
                'segment_count': record_segment_count,
                'annotation_counts': annotation_segment_counts
            }
            
        except Exception as e:
            continue
    
    return record_stats, total_segments, segments_per_record, segments_per_rhythm_annotation


def analyze_rhythm_bounded_approach(data_dir='../data/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0',
                                    segment_length=10.0):
    """Analyze rhythm-bounded non-overlapping approach"""
    
    files = os.listdir(data_dir)
    all_records = sorted(set([f.split('.')[0] for f in files if f.endswith('.hea')]))
    
    record_stats = {}
    total_segments = Counter()
    segments_per_record = {}
    segments_per_rhythm_annotation = defaultdict(list)
    
    for record_name in all_records:
        try:
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            fs = record.fs
            signal = record.p_signal[:, 0]
            
            segment_samples = int(segment_length * fs)
            
            # Find rhythm annotations
            rhythm_changes = []
            for i in range(len(annotation.sample)):
                if annotation.aux_note[i] and annotation.aux_note[i] in RHYTHM_CLASS_MAPPING_SIMPLE:
                    rhythm_changes.append({
                        'sample': annotation.sample[i],
                        'rhythm': annotation.aux_note[i],
                        'class_id': RHYTHM_CLASS_MAPPING_SIMPLE[annotation.aux_note[i]]
                    })
            
            if not rhythm_changes:
                continue
            
            # Calculate rhythm durations
            record_duration = len(signal) / fs
            rhythm_durations = []
            
            for i, change in enumerate(rhythm_changes):
                if i < len(rhythm_changes) - 1:
                    duration = (rhythm_changes[i+1]['sample'] - change['sample']) / fs
                    end_sample = rhythm_changes[i+1]['sample']
                else:
                    duration = record_duration - (change['sample'] / fs)
                    end_sample = len(signal)
                
                rhythm_durations.append({
                    **change,
                    'duration': duration,
                    'end_sample': end_sample
                })
            
            # New approach: non-overlapping within each rhythm annotation
            record_segment_count = Counter()
            
            for i, change in enumerate(rhythm_changes):
                start_sample = change['sample']
                if i < len(rhythm_changes) - 1:
                    end_sample = rhythm_changes[i + 1]['sample']
                else:
                    end_sample = len(signal)
                
                segment_duration_samples = end_sample - start_sample
                
                if segment_duration_samples < segment_samples:
                    continue
                
                # Non-overlapping segments within this annotation
                count = 0
                current_pos = start_sample
                while current_pos + segment_samples <= end_sample:
                    count += 1
                    current_pos += segment_samples  # No overlap
                
                record_segment_count[change['rhythm']] += count
                segments_per_rhythm_annotation[change['rhythm']].append({
                    'duration': rhythm_durations[i]['duration'],
                    'segments': count,
                    'record': record_name
                })
            
            segments_per_record[record_name] = sum(record_segment_count.values())
            total_segments.update(record_segment_count)
            
            record_stats[record_name] = {
                'rhythm_changes': rhythm_changes,
                'durations': rhythm_durations,
                'segment_count': record_segment_count
            }
            
        except Exception as e:
            continue
    
    return record_stats, total_segments, segments_per_record, segments_per_rhythm_annotation


def plot_comparison(current_stats, rhythm_bounded_stats, save_path='rhythm_classification/segmentation_comparison.png'):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Total segments per class
    ax1 = fig.add_subplot(gs[0, 0])
    current_total = current_stats[1]
    bounded_total = rhythm_bounded_stats[1]
    
    classes = list(range(len(CLASS_NAMES)))
    current_counts = [current_total.get(cls, 0) for cls in range(len(CLASS_NAMES))]
    bounded_counts = [bounded_total.get(cls, 0) for cls in range(len(CLASS_NAMES))]
    
    # Reconstruct by rhythm string
    current_by_rhythm = defaultdict(int)
    bounded_by_rhythm = defaultdict(int)
    for rhythm, class_id in RHYTHM_CLASS_MAPPING_SIMPLE.items():
        for r_str, count in current_stats[1].items():
            if r_str == rhythm:
                current_by_rhythm[rhythm] = count
        for r_str, count in rhythm_bounded_stats[1].items():
            if r_str == rhythm:
                bounded_by_rhythm[rhythm] = count
    
    rhythms = sorted(set(list(current_by_rhythm.keys()) + list(bounded_by_rhythm.keys())))
    x = np.arange(len(rhythms))
    width = 0.35
    
    current_rhythm_counts = [current_by_rhythm.get(r, 0) for r in rhythms]
    bounded_rhythm_counts = [bounded_by_rhythm.get(r, 0) for r in rhythms]
    
    ax1.bar(x - width/2, current_rhythm_counts, width, label='Current (Overlapping)', alpha=0.8)
    ax1.bar(x + width/2, bounded_rhythm_counts, width, label='Rhythm-Bounded (Non-overlapping)', alpha=0.8)
    ax1.set_xlabel('Rhythm Type', fontweight='bold')
    ax1.set_ylabel('Number of Segments', fontweight='bold')
    ax1.set_title('Total Segments by Rhythm Type', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rhythms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Segments per record
    ax2 = fig.add_subplot(gs[0, 1])
    records = sorted(set(list(current_stats[2].keys()) + list(rhythm_bounded_stats[2].keys())))
    x = np.arange(len(records))
    
    current_record_counts = [current_stats[2].get(r, 0) for r in records]
    bounded_record_counts = [rhythm_bounded_stats[2].get(r, 0) for r in records]
    
    ax2.bar(x - width/2, current_record_counts, width, label='Current', alpha=0.8)
    ax2.bar(x + width/2, bounded_record_counts, width, label='Rhythm-Bounded', alpha=0.8)
    ax2.set_xlabel('Patient Record', fontweight='bold')
    ax2.set_ylabel('Number of Segments', fontweight='bold')
    ax2.set_title('Segments per Patient Record', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(records, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Segments per annotation (scatter plot)
    ax3 = fig.add_subplot(gs[1, :])
    
    for rhythm in rhythms:
        if rhythm in current_stats[3]:
            data = current_stats[3][rhythm]
            durations = [d['duration'] for d in data]
            segments = [d['segments'] for d in data]
            ax3.scatter(durations, segments, alpha=0.6, s=50, label=f'{rhythm} (Current)')
    
    for rhythm in rhythms:
        if rhythm in rhythm_bounded_stats[3]:
            data = rhythm_bounded_stats[3][rhythm]
            durations = [d['duration'] for d in data]
            segments = [d['segments'] for d in data]
            ax3.scatter(durations, segments, alpha=0.6, s=50, marker='s', label=f'{rhythm} (Bounded)')
    
    ax3.set_xlabel('Annotation Duration (seconds)', fontweight='bold')
    ax3.set_ylabel('Number of Segments Generated', fontweight='bold')
    ax3.set_title('Segments Generated per Rhythm Annotation', fontweight='bold', fontsize=14)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    
    # 4. Distribution statistics
    ax4 = fig.add_subplot(gs[2, 0])
    
    stats_text = "CURRENT APPROACH (Overlapping Sliding Window):\n"
    stats_text += f"Total segments: {sum(current_stats[1].values()):,}\n"
    stats_text += f"Records with data: {len(current_stats[2])}\n\n"
    
    stats_text += "RHYTHM-BOUNDED APPROACH (Non-overlapping):\n"
    stats_text += f"Total segments: {sum(rhythm_bounded_stats[1].values()):,}\n"
    stats_text += f"Records with data: {len(rhythm_bounded_stats[2])}\n\n"
    
    reduction_pct = (1 - sum(rhythm_bounded_stats[1].values()) / sum(current_stats[1].values())) * 100
    stats_text += f"Reduction: {reduction_pct:.1f}%\n\n"
    
    stats_text += "BIAS ANALYSIS:\n"
    stats_text += "Current approach creates more segments from:\n"
    stats_text += "- Patients with long stable rhythms\n"
    stats_text += "- Large rhythm annotations (linear growth)\n\n"
    stats_text += "Rhythm-bounded approach:\n"
    stats_text += "- More balanced patient representation\n"
    stats_text += "- Each annotation contributes proportionally\n"
    stats_text += "- Reduces redundant near-duplicate segments\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    
    # 5. Class imbalance comparison
    ax5 = fig.add_subplot(gs[2, 1])
    
    current_total_sum = sum(current_stats[1].values())
    bounded_total_sum = sum(rhythm_bounded_stats[1].values())
    
    current_percentages = [(current_by_rhythm.get(r, 0) / current_total_sum * 100) if current_total_sum > 0 else 0 
                          for r in rhythms]
    bounded_percentages = [(bounded_by_rhythm.get(r, 0) / bounded_total_sum * 100) if bounded_total_sum > 0 else 0
                          for r in rhythms]
    
    x = np.arange(len(rhythms))
    ax5.bar(x - width/2, current_percentages, width, label='Current', alpha=0.8)
    ax5.bar(x + width/2, bounded_percentages, width, label='Rhythm-Bounded', alpha=0.8)
    ax5.set_xlabel('Rhythm Type', fontweight='bold')
    ax5.set_ylabel('Percentage of Total Segments', fontweight='bold')
    ax5.set_title('Class Distribution Comparison', fontweight='bold', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(rhythms, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Segmentation Strategy Comparison: Current vs Rhythm-Bounded', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to: {save_path}")
    plt.close()


def main():
    """Run full comparison analysis"""
    
    print("\n" + "="*80)
    print("RHYTHM SEGMENTATION STRATEGY COMPARISON")
    print("="*80 + "\n")
    
    print("Analyzing current approach (overlapping sliding windows)...")
    current_stats = analyze_current_approach()
    
    print("\nAnalyzing rhythm-bounded approach (non-overlapping within annotations)...")
    rhythm_bounded_stats = analyze_rhythm_bounded_approach()
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Current Approach (10s window, 5s stride):")
    print(f"  Total segments: {sum(current_stats[1].values()):,}")
    print(f"  Records: {len(current_stats[2])}")
    print(f"\n  Distribution by rhythm:")
    for rhythm, count in sorted(current_stats[1].items(), key=lambda x: -x[1]):
        pct = (count / sum(current_stats[1].values())) * 100
        print(f"    {rhythm:10s}: {count:6,} ({pct:5.2f}%)")
    
    print(f"\n\nRhythm-Bounded Approach (non-overlapping 10s segments):")
    print(f"  Total segments: {sum(rhythm_bounded_stats[1].values()):,}")
    print(f"  Records: {len(rhythm_bounded_stats[2])}")
    print(f"\n  Distribution by rhythm:")
    for rhythm, count in sorted(rhythm_bounded_stats[1].items(), key=lambda x: -x[1]):
        pct = (count / sum(rhythm_bounded_stats[1].values())) * 100
        print(f"    {rhythm:10s}: {count:6,} ({pct:5.2f}%)")
    
    reduction = sum(current_stats[1].values()) - sum(rhythm_bounded_stats[1].values())
    reduction_pct = (reduction / sum(current_stats[1].values())) * 100
    print(f"\n  Segment reduction: {reduction:,} ({reduction_pct:.1f}%)")
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80 + "\n")
    
    plot_comparison(current_stats, rhythm_bounded_stats)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    print("✅ RHYTHM-BOUNDED APPROACH is recommended because:")
    print("  1. More balanced patient representation")
    print("  2. Reduces bias from long stable rhythms")
    print("  3. Cleaner labels (segments don't cross rhythm boundaries)")
    print("  4. Less redundancy (fewer near-duplicate segments)")
    print("  5. More similar to beat classification methodology")
    print("\n❌ CURRENT APPROACH issues:")
    print("  1. Over-represents patients with long stable rhythms")
    print("  2. Creates many highly-overlapping near-duplicates")
    print("  3. Segments can span rhythm boundaries (noisy labels)")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()

