"""
Script to download MIT-BIH Arrhythmia Database from PhysioNet
"""

import os
import wfdb
from tqdm import tqdm

def download_mitdb_dataset(output_dir='data/mitdb'):
    """
    Download the MIT-BIH Arrhythmia Database
    
    Parameters:
    -----------
    output_dir : str
        Directory where data will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading MIT-BIH Arrhythmia Database to {output_dir}...")
    
    # List of all record names in MIT-BIH database
    # These are the 48 half-hour excerpts from 47 subjects
    record_names = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # Download each record with progress bar
    successful_downloads = 0
    
    for record in tqdm(record_names, desc="Downloading records"):
        try:
            # Download and read record from PhysioNet
            record_data = wfdb.rdrecord(record, pn_dir='mitdb')
            
            # Write record to local directory
            wfdb.wrsamp(
                record_name=record,
                fs=record_data.fs,
                units=record_data.units,
                sig_name=record_data.sig_name,
                p_signal=record_data.p_signal,
                fmt=record_data.fmt,
                write_dir=output_dir
            )
            
            # Download and read annotations
            ann_data = wfdb.rdann(record, 'atr', pn_dir='mitdb')
            
            # Write annotations to local directory
            wfdb.wrann(
                record_name=record,
                extension='atr',
                sample=ann_data.sample,
                symbol=ann_data.symbol,
                write_dir=output_dir
            )
            
            successful_downloads += 1
            
        except Exception as e:
            print(f"\nError downloading record {record}: {e}")
            continue
    
    print("\n✓ Download complete!")
    print(f"Data saved to: {os.path.abspath(output_dir)}")
    print(f"Successfully downloaded: {successful_downloads}/{len(record_names)} records")
    
    return output_dir

if __name__ == "__main__":
    download_mitdb_dataset()
