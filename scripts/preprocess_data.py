#!/usr/bin/env python3
"""
Data preprocessing script for E4 wearable sensor data.
Converts raw sensor CSV files into processed DataFrames for ML training.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


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
            data.columns = ["X", "Y", "Z"]
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


def process_sensor_files(data_files: List[Dict]) -> Dict[str, List[pd.DataFrame]]:
    """
    Process all sensor files and organize by sensor type.
    
    Args:
        data_files: List of file metadata dictionaries
        
    Returns:
        Dictionary mapping sensor types to lists of DataFrames
    """
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

            # Add metadata
            df["participant"] = item["participant"]
            df["session_ts"] = item["session_ts"]
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


def save_processed_data(data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save processed DataFrames to files.
    
    Args:
        data: Dictionary of processed DataFrames
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sensor, df in data.items():
        filename = f"{sensor.lower()}_processed.csv"
        filepath = output_path / filename
        
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"Saved {sensor} data to {filepath} ({len(df)} records)")
        except Exception as e:
            logging.error(f"Failed to save {sensor} data: {e}")


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description="Process E4 wearable sensor data")
    parser.add_argument("--input-dir", "-i", default="data/raw", 
                       help="Input directory containing raw data")
    parser.add_argument("--output-dir", "-o", default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--sensors", nargs="+", 
                       default=["ACC", "HR", "EDA", "TEMP", "BVP", "IBI"],
                       help="Sensors to process")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting data preprocessing pipeline")
    
    # Step 1: Discover data files
    data_files = discover_data_files(args.input_dir)
    
    # Filter by requested sensors
    data_files = [f for f in data_files if f["sensor"] in args.sensors]
    logging.info(f"Processing {len(data_files)} files for sensors: {args.sensors}")
    
    # Step 2: Process sensor files
    sensor_data = process_sensor_files(data_files)
    
    # Step 3: Concatenate data
    concatenated_data = concatenate_sensor_data(sensor_data)
    
    # Step 4: Save processed data
    save_processed_data(concatenated_data, args.output_dir)
    
    logging.info("Data preprocessing pipeline completed successfully")
    
    # Print final summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    for sensor, df in concatenated_data.items():
        participants = df['participant'].nunique()
        sessions = df['session_ts'].nunique()
        print(f"{sensor:>5}: {len(df):>8,} records | {participants:>3} participants | {sessions:>3} sessions")
    print("="*50)


if __name__ == "__main__":
    main()