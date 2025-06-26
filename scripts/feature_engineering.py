import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import scipy.signal
from scipy.signal import butter, filtfilt

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

def process_participant_file(file_path, window_size, output_dir):
    """
    Processes a single participant file to engineer features using windowed aggregation (no block splitting).
    For each window, BVP NaNs are dropped before morphological feature extraction. Windows with large BVP gaps or too few samples are dropped.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    df = df.set_index('timestamp')
    logging.info(f"Data loaded. {len(df)} rows.")

    window = f"{window_size}s"
    min_bvp_samples = window_size * 32 // 2  # Require at least half the expected BVP samples (32Hz)
    max_bvp_gap = pd.Timedelta(seconds=2)    # If any gap >2s in a window, drop the window
    dropped_windows = 0
    feature_rows = []
    window_starts = pd.date_range(start=df.index.min().floor(window), end=df.index.max().ceil(window), freq=window)

    for start in window_starts:
        win_start = start
        win_end = start + pd.Timedelta(seconds=window_size)
        window_df = df[(df.index >= win_start) & (df.index < win_end)]
        if window_df.empty:
            continue

        # --- BVP morphological features ---
        bvp_win = window_df['BVP'].dropna()
        # Check for enough BVP samples
        if len(bvp_win) < min_bvp_samples:
            dropped_windows += 1
            feature_rows.append([np.nan]*20)  # 20 = total number of features below
            continue
        # Check for large time gaps in BVP
        bvp_time_diffs = bvp_win.index.to_series().diff()
        if (bvp_time_diffs > max_bvp_gap).any():
            dropped_windows += 1
            feature_rows.append([np.nan]*20)
            continue
        # Extract BVP morphological features for this window
        bvp_morph = extract_bvp_morph_features(window_df, window_size, buffer=0, fs=64).iloc[0].tolist()

        # --- Other features (resample as before) ---
        temp_mean = window_df['TEMP'].mean()
        temp_std = window_df['TEMP'].std()
        temp_max = window_df['TEMP'].max()
        temp_min = window_df['TEMP'].min()
        acc_mag = np.sqrt(window_df['ACC_X']**2 + window_df['ACC_Y']**2 + window_df['ACC_Z']**2)
        acc_mean = acc_mag.mean()
        acc_std = acc_mag.std()
        acc_max = acc_mag.max()
        # Context-aware features (interpolate EDA/HR to ACC timestamps, then aggregate)
        acc_mask = window_df[['ACC_X', 'ACC_Y', 'ACC_Z']].notnull().any(axis=1)
        acc_mag_full = np.sqrt(window_df['ACC_X'].fillna(0)**2 + window_df['ACC_Y'].fillna(0)**2 + window_df['ACC_Z'].fillna(0)**2)
        acc_mag_full[~acc_mask] = np.nan
        eda_interp = window_df['EDA'].interpolate(method='time').reindex(window_df.index, method='nearest')
        hr_interp = window_df['HR'].interpolate(method='time').reindex(window_df.index, method='nearest')
        epsilon = 1e-6
        eda_acc_ratio = eda_interp / (acc_mag_full + epsilon)
        hr_acc_ratio = hr_interp / (acc_mag_full + epsilon)
        eda_acc_mean = eda_acc_ratio.mean()
        eda_acc_std = eda_acc_ratio.std()
        hr_acc_mean = hr_acc_ratio.mean()
        hr_acc_std = hr_acc_ratio.std()
        # HRV features
        ibi_list = window_df['IBI'].dropna().tolist()
        hrv_rmssd, hrv_sdnn, hrv_pnn50 = calculate_hrv_features(pd.Series(ibi_list))
        # Collect all features for this window
        feature_rows.append([
            hrv_rmssd, hrv_sdnn, hrv_pnn50,
            temp_mean, temp_std, temp_max, temp_min,
            acc_mean, acc_std, acc_max,
            eda_acc_mean, eda_acc_std, hr_acc_mean, hr_acc_std,
            *bvp_morph
        ])

    # Build DataFrame
    columns = [
        'rmssd', 'sdnn', 'pnn50',
        'temp_mean', 'temp_std', 'temp_max', 'temp_min',
        'acc_mean', 'acc_std', 'acc_magnitude',
        'eda_acc_mean', 'eda_acc_std', 'hr_acc_mean', 'hr_acc_std',
        'bvp_systolic_amp_mean', 'bvp_systolic_amp_std',
        'bvp_pulse_width_mean', 'bvp_pulse_width_std',
        'bvp_rise_time_mean', 'bvp_rise_time_std',
        'bvp_auc_mean', 'bvp_auc_std',
    ]
    features_df = pd.DataFrame(feature_rows, columns=columns, index=window_starts[:len(feature_rows)])
    logging.info(f"Dropped {dropped_windows} windows due to insufficient BVP data or large gaps.")
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
