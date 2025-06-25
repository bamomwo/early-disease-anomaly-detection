import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import scipy.signal
from scipy.signal import butter, filtfilt
import scipy.interpolate

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

def bandpass_filter(signal, fs, lowcut=0.5, highcut=8.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_bvp_morph_features(df, window_size, buffer=2, fs=64):
    """
    Extract BVP morphological features for each window.
    Returns a DataFrame indexed by window start time.
    """
    bvp = df['BVP']
    times = pd.to_datetime(df.index)
    window = f"{window_size}s"
    buffer_samples = int(buffer * fs)
    results = []
    index = []
    # Resample index for windows
    window_starts = pd.date_range(start=times.min(), end=times.max(), freq=window)
    logging.info(f"Extracting BVP morphological features: {len(window_starts)} windows to process.")
    for start in window_starts:
        win_start = start
        win_end = start + pd.Timedelta(seconds=window_size)
        buf_start = win_start - pd.Timedelta(seconds=buffer)
        buf_end = win_end + pd.Timedelta(seconds=buffer)
        mask = (times >= buf_start) & (times < buf_end)
        bvp_buf = bvp[mask]
        t_buf = times[mask]
        if len(bvp_buf) < 5:
            logging.warning(f"Not enough BVP data in buffer window starting at {win_start}. Outputting NaN.")
            results.append([np.nan]*8)
            index.append(win_start)
            continue
        try:
            bvp_filt = bandpass_filter(bvp_buf.ffill().bfill(), fs)
        except Exception as e:
            logging.warning(f"Filtering failed for window starting at {win_start}: {e}. Using unfiltered signal.")
            bvp_filt = bvp_buf.ffill().bfill()
        peaks, _ = scipy.signal.find_peaks(bvp_filt, distance=int(0.4*fs))
        logging.debug(f"Window {win_start}: Found {len(peaks)} peaks.")
        feet = []
        for peak in peaks:
            search = bvp_filt[max(0, peak-int(0.5*fs)):peak]
            if len(search) == 0:
                feet.append(None)
                continue
            foot = np.argmin(search) + max(0, peak-int(0.5*fs))
            feet.append(foot)
        ends = []
        for peak in peaks:
            search = bvp_filt[peak:peak+int(0.5*fs)]
            if len(search) == 0:
                ends.append(None)
                continue
            end = np.argmin(search) + peak
            ends.append(end)
        win_mask = (t_buf >= win_start) & (t_buf < win_end)
        win_indices = np.where(win_mask)[0]
        if len(win_indices) == 0:
            logging.warning(f"No data in main window starting at {win_start}. Outputting NaN.")
            results.append([np.nan]*8)
            index.append(win_start)
            continue
        win_start_idx, win_end_idx = win_indices[0], win_indices[-1]
        systolic_amps, pulse_widths, rise_times, aucs = [], [], [], []
        for i, peak in enumerate(peaks):
            foot = feet[i]
            end = ends[i]
            if foot is None or end is None:
                continue
            if foot < win_start_idx or end > win_end_idx:
                continue
            peak_val = bvp_filt[peak]
            foot_val = bvp_filt[foot]
            systolic_amps.append(peak_val - foot_val)
            pulse_widths.append((t_buf[end] - t_buf[foot]).total_seconds())
            rise_times.append((t_buf[peak] - t_buf[foot]).total_seconds())
            aucs.append(np.trapezoid(bvp_filt[foot:end+1], (t_buf[foot:end+1]-t_buf[foot]).total_seconds()))
        if len(systolic_amps) == 0:
            logging.info(f"No complete pulses in window starting at {win_start}. Outputting NaN.")
        def agg(x):
            return np.nanmean(x) if len(x) > 0 else np.nan, np.nanstd(x) if len(x) > 0 else np.nan
        amp_mean, amp_std = agg(systolic_amps)
        width_mean, width_std = agg(pulse_widths)
        rise_mean, rise_std = agg(rise_times)
        auc_mean, auc_std = agg(aucs)
        results.append([amp_mean, amp_std, width_mean, width_std, rise_mean, rise_std, auc_mean, auc_std])
        index.append(win_start)
    columns = [
        'bvp_systolic_amp_mean', 'bvp_systolic_amp_std',
        'bvp_pulse_width_mean', 'bvp_pulse_width_std',
        'bvp_rise_time_mean', 'bvp_rise_time_std',
        'bvp_auc_mean', 'bvp_auc_std',
    ]
    logging.info("Finished extracting BVP morphological features.")
    return pd.DataFrame(results, index=index, columns=columns)

def process_data_block(df_block, window_size):
    """
    Processes a single contiguous data block to engineer features.
    """
    logging.info(f"Processing a data block of size {len(df_block)} from {df_block.index.min()} to {df_block.index.max()}")
    
    # --- Create a dense DataFrame by upsampling to the BVP timeline ---
    # 1. Get the clean BVP timeline (where BVP is not NaN)
    bvp_timeline_df = df_block[['BVP']].dropna()
    
    # 2. Interpolate other sensors to this master timeline
    sensors_to_interpolate = ['ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'HR', 'EDA', 'IBI']
    for sensor in sensors_to_interpolate:
        # Use existing data for interpolation
        source_data = df_block[[sensor]].dropna()
        if source_data.empty:
            bvp_timeline_df[sensor] = np.nan
            continue
        
        # Create an interpolator function
        interpolator = scipy.interpolate.interp1d(
            source_data.index.astype(np.int64),
            source_data[sensor],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Apply interpolator to the BVP timeline
        bvp_timeline_df[sensor] = interpolator(bvp_timeline_df.index.astype(np.int64))

    # Now, `dense_df` is our new gold standard for this block
    dense_df = bvp_timeline_df
    logging.info(f"Created dense data block of size {len(dense_df)}.")

    # --- Log dropped data statistics ---
    dropped_rows = df_block[df_block['BVP'].isna()]
    sensors = ['ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'HR', 'EDA', 'IBI']
    dropped_stats = {sensor: dropped_rows[sensor].notna().sum() for sensor in sensors}
    logging.info(f"Dropped {len(dropped_rows)} rows due to missing BVP.")
    for sensor, count in dropped_stats.items():
        logging.info(f"  Of these, {count} rows had valid data for {sensor}.")

    window = f"{window_size}s"
    
    # HRV features
    ibi = dense_df['IBI'].resample(window).apply(list)
    hrv_features = ibi.apply(lambda x: calculate_hrv_features(pd.Series(x)))
    hrv_df = pd.DataFrame(hrv_features.tolist(), index=hrv_features.index, columns=['rmssd', 'sdnn', 'pnn50'])
    
    # Temperature features
    temp_mean = dense_df['TEMP'].resample(window).mean()
    temp_std = dense_df['TEMP'].resample(window).std()
    temp_max = dense_df['TEMP'].resample(window).max()
    temp_min = dense_df['TEMP'].resample(window).min()
    temp_df = pd.concat([temp_mean, temp_std, temp_max, temp_min], axis=1)
    temp_df.columns = ['temp_mean', 'temp_std', 'temp_max', 'temp_min']

    # Motion features
    acc_mag = np.sqrt(dense_df['ACC_X']**2 + dense_df['ACC_Y']**2 + dense_df['ACC_Z']**2)
    acc_mean = acc_mag.resample(window).mean()
    acc_std = acc_mag.resample(window).std()
    acc_max = acc_mag.resample(window).max()
    acc_df = pd.concat([acc_mean, acc_std, acc_max], axis=1)
    acc_df.columns = ['acc_mean', 'acc_std', 'acc_magnitude']

    # Context-aware features
    epsilon = 1e-6
    # Note: We now use acc_mag directly as it's dense
    eda_acc_ratio = dense_df['EDA'] / (acc_mag + epsilon)
    hr_acc_ratio = dense_df['HR'] / (acc_mag + epsilon)
    
    eda_acc_mean = eda_acc_ratio.resample(window).mean()
    eda_acc_std = eda_acc_ratio.resample(window).std()
    hr_acc_mean = hr_acc_ratio.resample(window).mean()
    hr_acc_std = hr_acc_ratio.resample(window).std()
    context_df = pd.concat([eda_acc_mean, eda_acc_std, hr_acc_mean, hr_acc_std], axis=1)
    context_df.columns = ['eda_acc_mean', 'eda_acc_std', 'hr_acc_mean', 'hr_acc_std']

    # BVP morphological features - uses the original sparse df for bvp extraction
    bvp_morph_df = extract_bvp_morph_features(df_block, window_size, buffer=2, fs=64)

    # Merge all features
    features_df = pd.concat([hrv_df, temp_df, acc_df, context_df, bvp_morph_df], axis=1)
    return features_df

def process_participant_file(file_path, window_size, output_dir):
    """
    Processes a single participant file to engineer features.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.set_index('timestamp')
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    logging.info(f"Data loaded. {len(df)} rows.")

    # --- Block-based processing ---
    # Identify contiguous blocks of data by finding large time gaps
    df_sorted = df.sort_index()
    time_diffs = df_sorted.index.to_series().diff()
    
    # A new block starts where the gap is larger than the window size (a sensible threshold)
    block_starts = time_diffs > pd.Timedelta(seconds=window_size)
    block_ids = block_starts.cumsum()
    
    all_features = []
    
    logging.info(f"Found {block_ids.nunique()} data blocks to process.")

    for block_id in range(block_ids.max() + 1):
        block_df = df_sorted[block_ids == block_id]
        
        # Skip small, noisy blocks
        if len(block_df) < window_size * 32: # Require at least one window of ACC data
            logging.info(f"Skipping block {block_id} due to small size: {len(block_df)} rows.")
            continue
            
        block_features = process_data_block(block_df, window_size)
        all_features.append(block_features)

    if not all_features:
        logging.warning("No data blocks were processed for this participant.")
        return
        
    # Combine features from all blocks
    final_features_df = pd.concat(all_features)
    logging.info(f"All features merged. Shape: {final_features_df.shape}")

    # Save to output directory
    output_filename = Path(file_path).stem + "_features.csv"
    output_path = os.path.join(output_dir, output_filename)
    final_features_df.to_csv(output_path)
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
