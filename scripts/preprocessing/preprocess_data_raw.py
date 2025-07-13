#!/usr/bin/env python3
"""
Enhanced data preprocessing script for E4 wearable sensor data.
Converts raw sensor CSV files into processed DataFrames for ML training.
Supports both signal-centric and participant-centric output formats.
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
    "IBI": 1      # regular, will be interpolated to 1Hz
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


def discover_data_files(root_dir: str, max_participants: Optional[int] = None) -> List[Dict]:
    """
    Discover all CSV files in the dataset directory.
    
    Args:
        root_dir: Root directory containing participant folders
        max_participants: Maximum number of participants to process (None for all)
        
    Returns:
        List of dictionaries containing file metadata
    """
    data_files = []
    
    # Get list of participant directories and limit if needed
    part_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
    if max_participants is not None:
        part_dirs = part_dirs[:max_participants]
    
    for part_dir in part_dirs:
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


def resample_sensor_data(df: pd.DataFrame, sensor: str, target_fs: float) -> pd.DataFrame:
    """
    Resample sensor data to target frequency without interpolation.
    
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
        
        # Step 8: Resample without interpolation
        resampled_df = clean_df.resample(period).mean()
        
        # Step 9: Reset index to get timestamp back as column
        resampled_df = resampled_df.reset_index()
        
        # Step 10: Add back metadata columns
        for col, value in metadata_values.items():
            resampled_df[col] = value
        
        logging.debug(f"Resampled {sensor} from {len(df)} to {len(resampled_df)} records at {target_fs}Hz")
        return resampled_df
        
    except Exception as e:
        logging.error(f"Error resampling {sensor} data: {e}")
        logging.error(f"DataFrame columns: {df.columns.tolist()}")
        logging.error(f"DataFrame dtypes: {dict(df.dtypes)}")
        logging.error(f"Sample data:\n{df.head()}")
        return df


def process_sensor_files(data_files: List[Dict], sampling_rates: Dict[str, float] = None) -> Dict[str, List[pd.DataFrame]]:
    """
    Process all sensor files and organize by sensor type.
    
    Args:
        data_files: List of file metadata dictionaries
        sampling_rates: Target sampling rates for each sensor
        
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
            elif sensor in {"ACC", "HR", "EDA", "TEMP", "BVP"}:
                df = load_e4_csv(path, sensor)
            else:
                logging.debug(f"Skipping unknown sensor: {sensor}")
                continue  # skip unknown or TAG files

            if df.empty:
                logging.warning(f"Empty DataFrame for {path}")
                continue  # skip bad/empty data

            # Add metadata before resampling
            df["participant"] = item["participant"]
            df["session_ts"] = item["session_ts"]
            
            # Resample if target frequency is specified
            if sensor in sampling_rates:
                df = resample_sensor_data(df, sensor, sampling_rates[sensor])
                if df.empty:
                    logging.warning(f"Empty DataFrame after resampling for {path}")
                    continue
            
            sensor_data[sensor].append(df)
            processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process {path}: {e}")
            error_count += 1

    logging.info(f"Processing complete: {processed_count} successful, {error_count} errors")
    
    # Log summary statistics
    for sensor, dfs in sensor_data.items():
        total_records = sum(len(df) for df in dfs)
        logging.info(f"  {sensor}: {len(dfs)} files, {total_records} total records")
    
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
    merged_df['participant'] = participant
    
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
    parser = argparse.ArgumentParser(description="Process E4 wearable sensor data")
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
    parser.add_argument("--max-participants", type=int,
                       help="Maximum number of participants to process (processes first N participants alphabetically)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info(f"Starting data preprocessing pipeline in {args.mode} mode")
    
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
    data_files = discover_data_files(args.input_dir, args.max_participants)
    
    # Filter by requested sensors and exclude specified sensors
    if args.exclude_sensors:
        logging.info(f"Excluding sensors: {args.exclude_sensors}")
        args.sensors = [s for s in args.sensors if s not in args.exclude_sensors]
    
    data_files = [f for f in data_files if f["sensor"] in args.sensors]
    logging.info(f"Processing {len(data_files)} files for sensors: {args.sensors}")
    
    # Step 2: Process sensor files
    sensor_data = process_sensor_files(data_files, sampling_rates)
    
    # Step 3: Process based on mode
    if args.mode == "signal":
        # Signal-centric: concatenate all data by sensor type
        processed_data = concatenate_sensor_data(sensor_data)
        
        # Print summary
        print("\n" + "="*60)
        print("SIGNAL-CENTRIC PREPROCESSING SUMMARY")
        print("="*60)
        for sensor, df in processed_data.items():
            participants = df['participant'].nunique()
            sessions = df['session_ts'].nunique()
            print(f"{sensor:>5}: {len(df):>8,} records | {participants:>3} participants | {sessions:>3} sessions")
        
    else:  # participant mode
        # Participant-centric: merge all sensors for each participant
        processed_data = merge_participant_data(sensor_data)
        
        # Print summary
        print("\n" + "="*60)
        print("PARTICIPANT-CENTRIC PREPROCESSING SUMMARY")
        print("="*60)
        for participant, df in processed_data.items():
            sensor_cols = [col for col in df.columns if col not in ['timestamp', 'participant']]
            print(f"Participant {participant:>8}: {len(df):>8,} records | {len(sensor_cols):>2} sensor features")
    
    print("="*60)
    
    # Step 4: Save processed data
    save_processed_data(processed_data, args.output_dir, args.mode)
    
    logging.info("Data preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()