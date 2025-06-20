#!/usr/bin/env python3
"""
Enhanced data preprocessing script for E4 wearable sensor data.
Converts raw sensor CSV files into processed DataFrames for ML training.
Supports both signal-centric and participant-centric output formats.
Includes HRV feature extraction from IBI data.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Default sampling rates for resampling (Hz)
DEFAULT_SAMPLING_RATES = {
    "ACC": 32,    # Original: 32Hz
    "HR": 1,      # Original: 1Hz  
    "EDA": 4,     # Original: 4Hz
    "TEMP": 4,    # Original: 4Hz
    "BVP": 64,    # Original: 64Hz
    "IBI": 1,     # Irregular, will be interpolated to 1Hz
    "HRV": 1      # HRV features extracted from IBI at 1Hz
}


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )


def discover_data_files(root_dir: str) -> List[Dict]:
    """
    Discover all CSV files in the dataset directory.
    
    Args:
        root_dir: Root directory containing participant folders
        
    Returns:
        List of dictionaries containing file metadata
    """
    data_files = []
    
    for part_dir in glob.glob(os.path.join(root_dir, "*")):
        participant = os.path.basename(part_dir)
        
        for session_dir in glob.glob(os.path.join(part_dir, "*")):
            session_name = os.path.basename(session_dir)
            
            if "_" in session_name:
                session_ts = session_name.split("_")[1]
            else:
                continue  # skip if folder doesn't match expected pattern
            
            for file_path in glob.glob(os.path.join(session_dir, "*.csv")):
                sensor = os.path.splitext(os.path.basename(file_path))[0]
                
                data_files.append({
                    "participant": participant,
                    "session_ts": int(session_ts),
                    "sensor": sensor,
                    "path": file_path
                })
    
    logging.info(f"Discovered {len(data_files)} files across {len(set(f['participant'] for f in data_files))} participants")
    return data_files


def load_e4_csv(path: str, sensor: str) -> pd.DataFrame:
    """
    Load standard E4 sensor CSV files (ACC, HR, EDA, TEMP, BVP).
    
    Args:
        path: Path to CSV file
        sensor: Sensor type
        
    Returns:
        DataFrame with timestamp column and sensor data
    """
    try:
        # Step 1: Read timestamp (t0) and sampling frequency (fs)
        t0, fs = pd.read_csv(path, header=None, nrows=2)[0].values
        fs = float(fs)
        
        # Step 2: Read data values
        data = pd.read_csv(path, header=None, skiprows=2)
        
        # Step 3: Create timestamps
        n = len(data)
        timestamps = pd.to_datetime(t0 + np.arange(n) / fs, unit="s")
        
        # Step 4: Set column names
        if sensor == "ACC":
            data.columns = ["ACC_X", "ACC_Y", "ACC_Z"]
        else:
            data.columns = [sensor]
        
        # Step 5: Insert timestamp
        data.insert(0, "timestamp", timestamps)
        
        logging.debug(f"Successfully loaded {sensor} data: {len(data)} records from {path}")
        return data
        
    except Exception as e:
        logging.error(f"Failed to load E4 file {path}: {e}")
        return pd.DataFrame()


def load_ibi_csv(path: str) -> pd.DataFrame:
    """
    Load IBI (Inter-Beat Interval) CSV files with special handling for various formats.
    
    Args:
        path: Path to IBI CSV file
        
    Returns:
        DataFrame with timestamp and IBI columns
    """
    try:
        with open(path, "r") as f:
            lines = f.read().splitlines()

        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) < 2:
            logging.warning(f"{path}: Not enough lines ({len(lines)} found)")
            return pd.DataFrame()

        # Parse the first line (start timestamp) - handle "timestamp, IBI" format
        first_line = lines[0].strip()
        if "," in first_line:
            # Handle cases like "1594920460.000000, IBI"
            start_time_str = first_line.split(",")[0].strip()
        else:
            # Handle cases with just the timestamp
            start_time_str = first_line
        
        start_time = float(start_time_str)
        
        data = []
        for line_num, line in enumerate(lines[1:], start=2):  # Skip the first line (header)
            line = line.strip()
            if not line:  # skip empty lines
                continue
                
            parts = line.split(",")
            if len(parts) != 2:
                logging.warning(f"{path}: Line {line_num} malformed: {line}")
                continue
            try:
                offset, ibi = float(parts[0].strip()), float(parts[1].strip())
                data.append((start_time + offset, ibi))
            except ValueError as e:
                logging.warning(f"{path}: Line {line_num} parsing error: {line} - {e}")
                continue

        if not data:
            logging.warning(f"{path}: No valid IBI entries found")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp", "IBI"])
        
        # Convert timestamp to datetime (like load_e4_csv does)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        
        logging.debug(f"Successfully loaded IBI data: {len(df)} records from {path}")
        return df

    except Exception as e:
        logging.error(f"Failed to load IBI file {path}: {e}")
        return pd.DataFrame()


def calculate_hrv_features(ibi_values: List[float]) -> Dict[str, float]:
    """
    Calculate HRV features from IBI values.
    
    Args:
        ibi_values: List of IBI values in seconds
        
    Returns:
        Dictionary containing HRV features
    """
    if len(ibi_values) < 2:
        return {"RMSSD": np.nan, "SDNN": np.nan, "PNN50": np.nan}
    
    ibi_array = np.array(ibi_values)
    
    # Convert to milliseconds for standard HRV calculations
    ibi_ms = ibi_array * 1000
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(ibi_ms, ddof=1)
    
    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(ibi_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    
    # PNN50: Percentage of successive NN intervals that differ by more than 50ms
    pnn50 = (np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)) * 100
    
    return {
        "RMSSD": rmssd,
        "SDNN": sdnn,
        "PNN50": pnn50
    }


def extract_hrv_features_from_ibi(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Extract HRV features from IBI data using time windows.
    
    Args:
        df: IBI DataFrame with 'timestamp' and 'IBI' columns
        window_size: Window size in seconds (default: 60)
        
    Returns:
        DataFrame with HRV features at regular intervals
    """
    if df.empty or 'IBI' not in df.columns:
        return pd.DataFrame()
    
    # Sort by timestamp to ensure proper ordering
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create time windows
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    # Generate window boundaries
    window_starts = pd.date_range(
        start=start_time.floor('min'),  # Round down to nearest minute
        end=end_time,
        freq=f'{window_size}s'
    )
    
    hrv_data = []
    
    for i, window_start in enumerate(window_starts[:-1]):  # Exclude last boundary
        window_end = window_starts[i + 1]
        
        # Get IBI values within this window
        window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] < window_end)
        window_ibi = df.loc[window_mask, 'IBI'].tolist()
        
        # Skip windows with insufficient data
        if len(window_ibi) < 2:
            continue
        
        # Calculate HRV features for this window
        hrv_features = calculate_hrv_features(window_ibi)
        
        # Add timestamp and metadata
        hrv_row = {
            'timestamp': window_start,
            **hrv_features
        }
        
        # Add participant and session_ts if they exist in original data
        if 'participant' in df.columns:
            hrv_row['participant'] = df['participant'].iloc[0]
        if 'session_ts' in df.columns:
            hrv_row['session_ts'] = df['session_ts'].iloc[0]
            
        hrv_data.append(hrv_row)
    
    if not hrv_data:
        return pd.DataFrame()
    
    hrv_df = pd.DataFrame(hrv_data)
    
    logging.debug(f"Extracted HRV features: {len(hrv_df)} windows from {len(df)} IBI records")
    return hrv_df


def resample_sensor_data(df: pd.DataFrame, sensor: str, target_fs: float) -> pd.DataFrame:
    """
    Resample sensor data to target frequency.
    
    Args:
        df: Input DataFrame with timestamp column
        sensor: Sensor type
        target_fs: Target sampling frequency in Hz
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    try:
        # Step 1: Inspect DataFrame columns and identify what we have
        all_cols = df.columns.tolist()
        logging.debug(f"Input DataFrame columns for {sensor}: {all_cols}")
        logging.debug(f"DataFrame dtypes: {dict(df.dtypes)}")
        
        # Step 2: Store metadata columns before processing
        metadata_cols = ['participant', 'session_ts']
        metadata_values = {}
        for col in metadata_cols:
            if col in df.columns:
                metadata_values[col] = df[col].iloc[0]  # Take first value
        
        # Step 3: Identify sensor value columns (exclude timestamp and metadata)
        sensor_value_cols = [col for col in df.columns 
                           if col not in ['timestamp'] + metadata_cols]
        
        logging.debug(f"Sensor value columns for {sensor}: {sensor_value_cols}")
        
        # Step 4: Create clean DataFrame with only timestamp + sensor values
        clean_df = df[['timestamp'] + sensor_value_cols].copy()
        
        # Step 5: Ensure sensor columns are numeric
        for col in sensor_value_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            logging.debug(f"Column {col} converted to numeric, NaN count: {clean_df[col].isna().sum()}")
        
        # Step 6: Set timestamp as index for resampling
        clean_df = clean_df.set_index('timestamp')
        
        # Step 7: Calculate resampling rule (period)
        period = f"{int(1000/target_fs)}ms"  # Convert Hz to milliseconds
        
        # Step 8: Resample based on sensor type
        if sensor in ["IBI", "HRV"]:
            # For IBI and HRV, interpolate irregular data to regular intervals
            resampled_df = clean_df.resample(period).mean().interpolate(method='linear')
        else:
            # For regular sensors, use mean aggregation
            resampled_df = clean_df.resample(period).mean()
        
        # Step 9: Reset index to get timestamp back as column
        resampled_df = resampled_df.reset_index()
        
        # Step 10: Add back metadata columns
        for col, value in metadata_values.items():
            resampled_df[col] = value
        
        # Step 11: Remove rows where all sensor values are NaN
        resampled_df = resampled_df.dropna(subset=sensor_value_cols, how='all')
        
        logging.debug(f"Resampled {sensor} from {len(df)} to {len(resampled_df)} records at {target_fs}Hz")
        return resampled_df
        
    except Exception as e:
        logging.error(f"Error resampling {sensor} data: {e}")
        logging.error(f"DataFrame columns: {df.columns.tolist()}")
        logging.error(f"DataFrame dtypes: {dict(df.dtypes)}")
        logging.error(f"Sample data:\n{df.head()}")
        return df


def process_ibi_to_hrv(ibi_df: pd.DataFrame, target_fs: float = 1.0, window_size: int = 60) -> pd.DataFrame:
    """
    Process single IBI DataFrame to extract HRV features and resample to target frequency.
    
    Args:
        ibi_df: IBI DataFrame
        target_fs: Target sampling frequency for HRV features (default: 1Hz)
        window_size: Window size in seconds for HRV calculation (default: 60)
        
    Returns:
        Processed HRV DataFrame
    """
    if ibi_df.empty:
        return pd.DataFrame()
        
    # Extract HRV features
    hrv_df = extract_hrv_features_from_ibi(ibi_df, window_size)
    
    if hrv_df.empty:
        return pd.DataFrame()
    
    # Resample HRV features to target frequency
    resampled_hrv = resample_sensor_data(hrv_df, "HRV", target_fs)
    
    return resampled_hrv


def process_sensor_files(data_files: List[Dict], sampling_rates: Dict[str, float] = None, 
                        extract_hrv: bool = True, hrv_window_size: int = 60) -> Dict[str, List[pd.DataFrame]]:
    """
    Process all sensor files and organize by sensor type, with optional HRV feature extraction.
    
    Args:
        data_files: List of file metadata dictionaries
        sampling_rates: Target sampling rates for each sensor
        extract_hrv: Whether to extract HRV features from IBI data
        hrv_window_size: Window size in seconds for HRV calculation
        
    Returns:
        Dictionary mapping sensor types to lists of DataFrames
    """
    if sampling_rates is None:
        sampling_rates = DEFAULT_SAMPLING_RATES
        
    sensor_data = defaultdict(list)
    processed_count = 0
    error_count = 0
    
    logging.info(f"Processing {len(data_files)} files...")
    
    for item in data_files:
        sensor = item["sensor"]
        path = item["path"]

        try:
            if sensor == "IBI":
                df = load_ibi_csv(path)
                if df.empty:
                    logging.warning(f"Empty IBI DataFrame for {path}")
                    continue
                
                # Add metadata before processing
                df["participant"] = item["participant"]
                df["session_ts"] = item["session_ts"]
                
                # Store original IBI data if requested
                if "IBI" in sampling_rates:
                    ibi_resampled = resample_sensor_data(df, "IBI", sampling_rates["IBI"])
                    if not ibi_resampled.empty:
                        sensor_data["IBI"].append(ibi_resampled)
                
                # Extract HRV features if requested
                if extract_hrv:
                    hrv_target_fs = sampling_rates.get("HRV", 1.0)
                    hrv_df = process_ibi_to_hrv(df, hrv_target_fs, hrv_window_size)
                    if not hrv_df.empty:
                        sensor_data["HRV"].append(hrv_df)
                        logging.debug(f"Extracted HRV features from {path}: {len(hrv_df)} records")
                
                processed_count += 1
                    
            elif sensor in {"ACC", "HR", "EDA", "TEMP", "BVP"}:
                df = load_e4_csv(path, sensor)
                if df.empty:
                    logging.warning(f"Empty DataFrame for {path}")
                    continue
                
                # Add metadata before resampling
                df["participant"] = item["participant"]
                df["session_ts"] = item["session_ts"]
                
                # Resample if target frequency is specified
                if sensor in sampling_rates:
                    df = resample_sensor_data(df, sensor, sampling_rates[sensor])
                    if not df.empty:
                        sensor_data[sensor].append(df)
                        processed_count += 1
            else:
                logging.debug(f"Skipping unknown sensor: {sensor}")
                continue

        except Exception as e:
            logging.error(f"Failed to process {path}: {e}")
            error_count += 1

    logging.info(f"Processing complete: {processed_count} successful, {error_count} errors")
    
    # Log summary statistics
    for sensor, dfs in sensor_data.items():
        total_records = sum(len(df) for df in dfs)
        participant_count = len(set(df['participant'].iloc[0] for df in dfs if not df.empty))
        logging.info(f"  {sensor}: {len(dfs)} files, {total_records} total records, {participant_count} participants")
    
    return dict(sensor_data)


def concatenate_sensor_data(sensor_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Concatenate all DataFrames for each sensor type.
    
    Args:
        sensor_data: Dictionary mapping sensor types to lists of DataFrames
        
    Returns:
        Dictionary mapping sensor types to concatenated DataFrames
    """
    concatenated_data = {}
    
    for sensor, dfs in sensor_data.items():
        if not dfs:
            logging.warning(f"No data available for sensor: {sensor}")
            continue
            
        try:
            df = pd.concat(dfs, ignore_index=True)
            concatenated_data[sensor] = df
            logging.info(f"Concatenated {sensor}: {len(df)} total records")
        except Exception as e:
            logging.error(f"Error concatenating {sensor} data: {e}")
    
    return concatenated_data


def merge_participant_data(sensor_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Merge sensor data by participant, creating one DataFrame per participant.
    
    Args:
        sensor_data: Dictionary mapping sensor types to lists of DataFrames
        
    Returns:
        Dictionary mapping participant IDs to merged DataFrames
    """
    # First, organize data by participant
    participant_sensor_data = defaultdict(lambda: defaultdict(list))
    
    for sensor, dfs in sensor_data.items():
        for df in dfs:
            if not df.empty:
                participant = df['participant'].iloc[0]
                participant_sensor_data[participant][sensor].append(df)
    
    # Now merge data for each participant
    participant_data = {}
    
    for participant, sensors in participant_sensor_data.items():
        logging.info(f"Merging data for participant {participant}")
        
        # Concatenate each sensor's data for this participant
        participant_sensors = {}
        for sensor, dfs in sensors.items():
            if dfs:
                concatenated = pd.concat(dfs, ignore_index=True)
                # Sort by timestamp
                concatenated = concatenated.sort_values('timestamp').reset_index(drop=True)
                participant_sensors[sensor] = concatenated
        
        if not participant_sensors:
            logging.warning(f"No sensor data found for participant {participant}")
            continue
        
        # Merge all sensors for this participant
        try:
            merged_df = merge_sensors_for_participant(participant_sensors, participant)
            if not merged_df.empty:
                participant_data[participant] = merged_df
                logging.info(f"Merged data for participant {participant}: {len(merged_df)} records")
        except Exception as e:
            logging.error(f"Error merging data for participant {participant}: {e}")
    
    return participant_data


def merge_sensors_for_participant(sensor_dfs: Dict[str, pd.DataFrame], participant: str) -> pd.DataFrame:
    """
    Merge multiple sensor DataFrames for a single participant based on timestamps.
    
    Args:
        sensor_dfs: Dictionary mapping sensor names to DataFrames
        participant: Participant ID
        
    Returns:
        Merged DataFrame with all sensor data
    """
    if not sensor_dfs:
        return pd.DataFrame()
    
    # Start with the first sensor as base
    sensor_names = list(sensor_dfs.keys())
    merged_df = sensor_dfs[sensor_names[0]].copy()
    
    # Remove participant and session_ts columns from sensor data (we'll add them back later)
    sensor_cols_to_drop = ['participant', 'session_ts']
    for col in sensor_cols_to_drop:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])
    
    # Merge with remaining sensors
    for sensor in sensor_names[1:]:
        df = sensor_dfs[sensor].copy()
        
        # Remove metadata columns
        for col in sensor_cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Merge on timestamp using outer join to keep all timestamps
        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer', suffixes=('', f'_{sensor}'))
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    # Add back participant information
    #merged_df['participant'] = participant
    
    # Identify all sensor value columns (excluding timestamp and participant)
    sensor_value_cols = [col for col in merged_df.columns if col not in ['timestamp', 'participant']]
    
    # CRITICAL: Remove rows where ALL sensor values are NaN
    # This handles the case where timestamps exist but no sensor has data at that time
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna(subset=sensor_value_cols, how='all')
    dropped_empty_rows = initial_rows - len(merged_df)
    
    if dropped_empty_rows > 0:
        logging.info(f"Participant {participant}: Dropped {dropped_empty_rows} rows with no sensor data")
    
    # Forward fill missing values for remaining rows (reasonable for physiological signals over short gaps)
    # Only apply to numeric columns to avoid issues with any categorical data
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].ffill()
    
    # Final check: remove any remaining rows where all sensor values are still NaN
    # (this can happen if the first few rows are all NaN and forward fill doesn't help)
    final_initial_rows = len(merged_df)
    merged_df = merged_df.dropna(subset=sensor_value_cols, how='all')
    final_dropped_rows = final_initial_rows - len(merged_df)
    
    if final_dropped_rows > 0:
        logging.info(f"Participant {participant}: Dropped {final_dropped_rows} additional rows after forward fill")
    
    return merged_df


def save_processed_data(data: Dict[str, pd.DataFrame], output_dir: str, mode: str) -> None:
    """
    Save processed DataFrames to files.
    
    Args:
        data: Dictionary of processed DataFrames
        output_dir: Output directory path
        mode: Processing mode ('signal' or 'participant')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for key, df in data.items():
        if mode == 'signal':
            filename = f"{key.lower()}_processed.csv"
        else:  # participant mode
            filename = f"participant_{key}_merged.csv"
            
        filepath = output_path / filename
        
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"Saved {key} data to {filepath} ({len(df)} records)")
        except Exception as e:
            logging.error(f"Failed to save {key} data: {e}")

def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description="Process E4 wearable sensor data with HRV feature extraction")
    parser.add_argument("--input-dir", "-i", default="data/raw", 
                       help="Input directory containing raw data")
    parser.add_argument("--output-dir", "-o", default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--mode", choices=["signal", "participant"], default="signal",
                       help="Processing mode: 'signal' for sensor-centric output, 'participant' for participant-centric output")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--sensors", nargs="+", 
                       default=["ACC", "HR", "EDA", "TEMP", "BVP", "IBI"],
                       help="Sensors to process (e.g., --sensors ACC HR EDA to exclude IBI and others)")
    parser.add_argument("--exclude-sensors", nargs="+", default=[],
                       help="Sensors to exclude from processing (e.g., --exclude-sensors IBI TAG)")
    parser.add_argument("--sampling-rates", type=str,
                       help="JSON string of custom sampling rates, e.g., '{\"ACC\": 16, \"HR\": 1}'")
    parser.add_argument("--extract-hrv", action="store_true", default=True,
                       help="Extract HRV features from IBI data (default: True)")
    parser.add_argument("--no-hrv", action="store_true", default=False,
                       help="Disable HRV feature extraction")
    parser.add_argument("--hrv-window-size", type=int, default=60,
                       help="Window size in seconds for HRV feature extraction (default: 60)")
    
    args = parser.parse_args()
    
    # Handle HRV extraction flag
    if args.no_hrv:
        args.extract_hrv = False
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info(f"Starting data preprocessing pipeline in {args.mode} mode")
    logging.info(f"HRV extraction: {'enabled' if args.extract_hrv else 'disabled'}")
    
    # Handle custom sampling rates
    sampling_rates = DEFAULT_SAMPLING_RATES.copy()
    if args.sampling_rates:
        try:
            import json
            custom_rates = json.loads(args.sampling_rates)
            sampling_rates.update(custom_rates)
            logging.info(f"Using custom sampling rates: {custom_rates}")
        except Exception as e:
            logging.error(f"Error parsing custom sampling rates: {e}")
            logging.info("Using default sampling rates")
    
    logging.info(f"Sampling rates: {sampling_rates}")
    
    # Step 1: Discover data files
    data_files = discover_data_files(args.input_dir)
    
    # Filter by requested sensors and exclude specified sensors
    if args.exclude_sensors:
        logging.info(f"Excluding sensors: {args.exclude_sensors}")
        args.sensors = [s for s in args.sensors if s not in args.exclude_sensors]
    
    data_files = [f for f in data_files if f["sensor"] in args.sensors]
    logging.info(f"Processing {len(data_files)} files for sensors: {args.sensors}")
    
    # Step 2: Process sensor files
    sensor_data = process_sensor_files(data_files, sampling_rates, 
                                     args.extract_hrv, args.hrv_window_size)
    
    # Step 3: Process based on mode
    if args.mode == "signal":
        # Signal-centric: concatenate all data by sensor type
        processed_data = concatenate_sensor_data(sensor_data)
        
        # Print summary
        print("\n" + "="*70)
        print("SIGNAL-CENTRIC PREPROCESSING SUMMARY")
        print("="*70)
        for sensor, df in processed_data.items():
            participants = df['participant'].nunique()
            sessions = df['session_ts'].nunique() if 'session_ts' in df.columns else 0
            if sensor == "HRV":
                print(f"{sensor:>5}: {len(df):>8,} records | {participants:>3} participants | {sessions:>3} sessions | HRV features")
            else:
                print(f"{sensor:>5}: {len(df):>8,} records | {participants:>3} participants | {sessions:>3} sessions")
        print("="*70)
        
    else:  # participant mode
        # Participant-centric: merge all sensors by participant
        processed_data = merge_participant_data(sensor_data)
        
        # Print summary
        print("\n" + "="*70)
        print("PARTICIPANT-CENTRIC PREPROCESSING SUMMARY")
        print("="*70)
        total_records = 0
        for participant, df in processed_data.items():
            records = len(df)
            total_records += records
            sensor_cols = [col for col in df.columns if col not in ['timestamp', 'participant']]
            print(f"Participant {participant:>6}: {records:>8,} records | {len(sensor_cols):>2} sensor features")
        print("-" * 70)
        print(f"{'TOTAL':>15}: {total_records:>8,} records | {len(processed_data)} participants")
        print("="*70)
    
    # Step 4: Save processed data
    if processed_data:
        save_processed_data(processed_data, args.output_dir, args.mode)
        logging.info(f"Processing complete! Output saved to {args.output_dir}")
        
        # Additional statistics
        if args.mode == "signal":
            total_records = sum(len(df) for df in processed_data.values())
            all_participants = set()
            for df in processed_data.values():
                all_participants.update(df['participant'].unique())
            
            print(f"\nOverall Statistics:")
            print(f"  Total records across all sensors: {total_records:,}")
            print(f"  Unique participants: {len(all_participants)}")
            print(f"  Sensors processed: {list(processed_data.keys())}")
            
        else:  # participant mode
            total_records = sum(len(df) for df in processed_data.values())
            all_sensors = set()
            for df in processed_data.values():
                sensor_cols = [col for col in df.columns if col not in ['timestamp', 'participant']]
                all_sensors.update(sensor_cols)
            
            print(f"\nOverall Statistics:")
            print(f"  Total records: {total_records:,}")
            print(f"  Participants: {len(processed_data)}")
            print(f"  Unique sensor features: {len(all_sensors)}")
    
    else:
        logging.warning("No data was processed successfully")
        print("\nNo data was processed. Please check:")
        print("  1. Input directory path is correct")
        print("  2. Data files are in the expected format")
        print("  3. Requested sensors exist in the data")
        print("  4. Check the log file for detailed error messages")


if __name__ == "__main__":
    main()