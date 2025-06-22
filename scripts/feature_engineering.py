import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_hrv_features(ibi_series):
    """
    Calculates HRV features from an IBI series.
    """
    # RMSSD
    nn_intervals = ibi_series.dropna()
    if len(nn_intervals) < 2:
        return np.nan, np.nan, np.nan
    
    diff_nni = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    
    # SDNN
    sdnn = np.std(nn_intervals)
    
    # pNN50
    pnn50 = np.sum(np.abs(diff_nni) > 0.05) / len(diff_nni) * 100 if len(diff_nni) > 0 else 0
    
    return rmssd, sdnn, pnn50


def calculate_temp_features(temp_series):
    """
    Calculates temperature features.
    """
    return temp_series.mean(), temp_series.std(), temp_series.max(), temp_series.min()


def calculate_acc_features(acc_x, acc_y, acc_z):
    """
    Calculates motion features from accelerometer data.
    """
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return acc_mag.mean(), acc_mag.std(), acc_mag.max()

def process_participant_file(file_path, window_size, output_dir):
    """
    Processes a single participant file to engineer features.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    df = df.set_index('timestamp')
    
    # Resample and apply feature engineering functions
    window = f"{window_size}s"
    
    # HRV features
    ibi = df['IBI'].resample(window).apply(list)
    hrv_features = ibi.apply(lambda x: calculate_hrv_features(pd.Series(x)))
    hrv_df = pd.DataFrame(hrv_features.tolist(), index=hrv_features.index, columns=['rmssd', 'sdnn', 'pnn50'])
    
    # Temperature features
    temp_mean = df['TEMP'].resample(window).mean()
    temp_std = df['TEMP'].resample(window).std()
    temp_max = df['TEMP'].resample(window).max()
    temp_min = df['TEMP'].resample(window).min()
    temp_df = pd.concat([temp_mean, temp_std, temp_max, temp_min], axis=1)
    temp_df.columns = ['temp_mean', 'temp_std', 'temp_max', 'temp_min']
    
    # Motion features
    acc_x_mean = df['ACC_X'].resample(window).mean()
    acc_y_mean = df['ACC_Y'].resample(window).mean()
    acc_z_mean = df['ACC_Z'].resample(window).mean()
    
    acc_mag = np.sqrt(df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)
    acc_mean = acc_mag.resample(window).mean()
    acc_std = acc_mag.resample(window).std()
    acc_max = acc_mag.resample(window).max()
    
    acc_df = pd.concat([acc_mean, acc_std, acc_max], axis=1)
    acc_df.columns = ['acc_mean', 'acc_std', 'acc_magnitude']
    
    # Merge all features
    features_df = pd.concat([hrv_df, temp_df, acc_df], axis=1)
    
    # Save to output directory
    output_filename = Path(file_path).stem + "_features.csv"
    output_path = os.path.join(output_dir, output_filename)
    features_df.to_csv(output_path)
    logging.info(f"Saved features to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Engineer features from sensor data.")
    parser.add_argument("--input-dir", type=str, default="data/test", help="Directory containing the raw CSV files.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save the processed files.")
    parser.add_argument("--window-size", type=int, default=10, help="Window size in seconds for feature extraction.")
    parser.add_argument("--participant-id", type=str, default=None, help="The ID of the participant to process. If not provided, all participants will be processed.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.participant_id:
        filename = f"participant_{args.participant_id}_merged.csv"
        file_path = os.path.join(args.input_dir, filename)
        if os.path.exists(file_path):
            process_participant_file(file_path, args.window_size, args.output_dir)
        else:
            logging.error(f"File not found for participant {args.participant_id} at {file_path}")
    else:
        for filename in os.listdir(args.input_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(args.input_dir, filename)
                process_participant_file(file_path, args.window_size, args.output_dir)

if __name__ == "__main__":
    main()
