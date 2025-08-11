import pandas as pd
import numpy as np
import os
import logging
import argparse
from pathlib import Path
import scipy.signal
from scipy.signal import butter, filtfilt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BiosignalPreprocessor:
    """
    Optimized preprocessing pipeline for multimodal biosignal data
    using session-aware window-based aggregation without resampling.
    """
    
    def __init__(self, window_size: int = 10, min_data_threshold: float = 0.6):
        """
        Initialize preprocessor.
        
        Args:
            window_size: Window size in seconds for feature extraction
            min_data_threshold: Minimum fraction of expected data points required per window
        """
        self.window_size = window_size
        self.min_data_threshold = min_data_threshold
        
        # Expected sampling rates
        self.sampling_rates = {
            'HR': 1, 'IBI': 'irregular', 'EDA': 4, 'TEMP': 4, 'BVP': 64, 
            'ACC_X': 32, 'ACC_Y': 32, 'ACC_Z': 32
        }
        
    def load_and_merge_sessions(self, participant_dir: str, max_sessions: Optional[int] = None) -> pd.DataFrame:
        """
        Load and merge sessions for a participant without resampling.
        Maintains original timestamps and handles gaps properly.
        Each session is kept separate for proper windowing.
        
        Args:
            participant_dir: Path to participant directory
            max_sessions: Maximum number of sessions to process (None for all sessions)
        """
        logging.info(f"Processing participant directory: {participant_dir}")
        
        all_data = []
        session_dirs = [d for d in os.listdir(participant_dir) 
                       if os.path.isdir(os.path.join(participant_dir, d))]
        
        # Sort sessions to ensure consistent ordering
        session_dirs = sorted(session_dirs)
        
        # Limit sessions if specified
        if max_sessions is not None:
            original_count = len(session_dirs)
            session_dirs = session_dirs[:max_sessions]
            logging.info(f"Limited to {len(session_dirs)} sessions out of {original_count} available")
        
        for session_dir in session_dirs:
            session_path = os.path.join(participant_dir, session_dir)
            session_data = self._load_session_data(session_path)
            
            if session_data is not None:
                # Add session identifier
                session_data['session'] = session_dir
                all_data.append(session_data)
                logging.info(f"Loaded session {session_dir}: {len(session_data)} records")
        
        if not all_data:
            raise ValueError(f"No valid session data found in {participant_dir}")
        
        # Concatenate all sessions but keep session information
        merged_data = pd.concat(all_data, ignore_index=True)
        # Sort by session first, then by timestamp within each session
        merged_data = merged_data.sort_values(['session', 'timestamp']).reset_index(drop=True)
        
        logging.info(f"Merged data shape: {merged_data.shape}")
        logging.info(f"Sessions included: {merged_data['session'].unique()}")
        return merged_data
    
    def _load_session_data(self, session_path: str) -> Optional[pd.DataFrame]:
        """Load data from a single session directory with proper format handling."""
        sensor_files = {
            'HR': 'HR.csv', 'IBI': 'IBI.csv', 'EDA': 'EDA.csv',
            'TEMP': 'TEMP.csv', 'BVP': 'BVP.csv', 'ACC': 'ACC.csv'
        }
        
        session_data = []
        
        for sensor, filename in sensor_files.items():
            file_path = os.path.join(session_path, filename)
            if not os.path.exists(file_path):
                logging.warning(f"Missing {filename} in {session_path}")
                continue
                
            try:
                if sensor == 'IBI':
                    # IBI has different format: 
                    # First row: start_timestamp, "IBI"
                    # Subsequent rows: time_offset_seconds, ibi_value
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) < 2:
                        logging.warning(f"IBI file {file_path} has insufficient data")
                        continue
                    
                    # Parse header row
                    header_parts = lines[0].strip().split(',')
                    start_timestamp = float(header_parts[0])
                    
                    # Parse data rows
                    timestamps = []
                    ibi_values = []
                    
                    for line in lines[1:]:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                time_offset = float(parts[0])
                                ibi_value = float(parts[1])
                                
                                # Calculate absolute timestamp
                                absolute_timestamp = start_timestamp + time_offset
                                timestamps.append(absolute_timestamp)
                                ibi_values.append(ibi_value)
                        except (ValueError, IndexError):
                            continue
                    
                    if not timestamps:
                        logging.warning(f"No valid IBI data found in {file_path}")
                        continue
                    
                    # Create dataframe
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        'IBI': ibi_values,
                        'sensor': 'IBI'
                    })
                    session_data.append(df)
                
                elif sensor == 'ACC':
                    # ACC file format: first row is timestamps, second row is sampling rates
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse start timestamp and sampling rate
                    start_timestamps = [float(x.strip()) for x in lines[0].split(',')]
                    sampling_rates = [float(x.strip()) for x in lines[1].split(',')]
                    
                    # Use the first timestamp as start time and first sampling rate
                    start_time = start_timestamps[0]
                    sampling_rate = sampling_rates[0]
                    
                    # Read the actual data (skip first 2 rows)
                    acc_data = pd.read_csv(file_path, skiprows=2, header=None)
                    acc_data.columns = ['ACC_X', 'ACC_Y', 'ACC_Z']
                    
                    # Generate timestamps
                    n_samples = len(acc_data)
                    timestamps = start_time + np.arange(n_samples) / sampling_rate
                    
                    # Create separate entries for each axis
                    for i, axis in enumerate(['X', 'Y', 'Z']):
                        axis_df = pd.DataFrame({
                            'timestamp': timestamps,
                            f'ACC_{axis}': acc_data.iloc[:, i],
                            'sensor': f'ACC_{axis}'
                        })
                        session_data.append(axis_df)
                
                else:
                    # HR, EDA, TEMP, BVP format: first row is timestamp, second row is sampling rate
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse start timestamp and sampling rate
                    start_time = float(lines[0].strip())
                    sampling_rate = float(lines[1].strip())
                    
                    # Read the actual data (skip first 2 rows)
                    sensor_data = []
                    for line in lines[2:]:
                        try:
                            value = float(line.strip())
                            sensor_data.append(value)
                        except ValueError:
                            continue
                    
                    if not sensor_data:
                        logging.warning(f"No valid data found in {file_path}")
                        continue
                    
                    # Generate timestamps
                    n_samples = len(sensor_data)
                    timestamps = start_time + np.arange(n_samples) / sampling_rate
                    
                    # Create dataframe
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        sensor: sensor_data,
                        'sensor': sensor
                    })
                    session_data.append(df)
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                continue
        
        if not session_data:
            return None
        
        # Combine all sensor data
        combined = pd.concat(session_data, ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], unit='s')
        
        # Pivot to wide format
        # Get the value column (should be the column that's not 'timestamp' or 'sensor')
        value_cols = [col for col in combined.columns if col not in ['timestamp', 'sensor']]
        
        # Melt the dataframe to handle multiple value columns properly
        melted_data = []
        for _, row in combined.iterrows():
            for col in value_cols:
                if not pd.isna(row[col]):
                    melted_data.append({
                        'timestamp': row['timestamp'],
                        'sensor': row['sensor'],
                        'value': row[col]
                    })
        
        if not melted_data:
            return None
        
        melted_df = pd.DataFrame(melted_data)
        
        # Pivot to wide format
        pivot_data = melted_df.pivot_table(
            index='timestamp', 
            columns='sensor', 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        return pivot_data
    
    def extract_window_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features using session-aware window-based aggregation.
        Windows are generated separately for each session to prevent
        cross-session contamination.
        """
        # Process each session separately
        all_features = []
        
        for session in data['session'].unique():
            logging.info(f"Processing session: {session}")
            session_data = data[data['session'] == session].copy()
            session_features = self._extract_session_features(session_data, session)
            
            if not session_features.empty:
                all_features.append(session_features)
                logging.info(f"Extracted {len(session_features)} windows from session {session}")
        
        if not all_features:
            raise ValueError("No valid features extracted from any session")
        
        # Combine all session features
        combined_features = pd.concat(all_features, ignore_index=True)
        logging.info(f"Total features extracted: {len(combined_features)} windows across {len(all_features)} sessions")
        
        return combined_features
    
    def _extract_session_features(self, session_data: pd.DataFrame, session_id: str) -> pd.DataFrame:
        """
        Extract features from a single session using window-based aggregation.
        This ensures windows never span across sessions.
        """
        # Remove session column for processing, but keep session_id for later
        data = session_data.drop(columns=['session']).set_index('timestamp')
        
        if data.empty:
            logging.warning(f"No data found for session {session_id}")
            return pd.DataFrame()
        
        # Generate windows for this session only
        start_time = data.index.min().floor(f'{self.window_size}s')
        end_time = data.index.max().ceil(f'{self.window_size}s')
        windows = pd.date_range(start=start_time, end=end_time, 
                               freq=f'{self.window_size}s')
        
        features_list = []
        valid_windows = 0
        
        logging.info(f"Processing {len(windows)} windows for session {session_id}...")
        
        for i, window_start in enumerate(windows[:-1]):
            window_end = window_start + pd.Timedelta(seconds=self.window_size)
            window_data = data[(data.index >= window_start) & (data.index < window_end)]
            
            if window_data.empty:
                continue
            
            # Quality check for window
            if not self._is_window_valid(window_data):
                continue
            
            # Extract features for this window
            window_features = self._extract_window_features_single(
                window_data, window_start
            )
            
            if window_features is not None:
                # Add session information to the features
                window_features['session'] = session_id
                features_list.append(window_features)
                valid_windows += 1
            
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(windows)} windows for session {session_id}")
        
        logging.info(f"Valid windows for session {session_id}: {valid_windows}/{len(windows)}")
        
        if not features_list:
            logging.warning(f"No valid windows found for session {session_id}")
            return pd.DataFrame()
        
        return pd.DataFrame(features_list)
    
    def _is_window_valid(self, window_data: pd.DataFrame) -> bool:
        """
        Check if window has sufficient data quality for feature extraction.
        """
        # Check BVP data quality (most critical for morphological features)
        if 'BVP' in window_data.columns:
            bvp_data = window_data['BVP'].dropna()
            expected_bvp_samples = self.window_size * self.sampling_rates['BVP']
            
            if len(bvp_data) < expected_bvp_samples * self.min_data_threshold:
                return False
            
            # Check for large gaps in BVP data
            if len(bvp_data) > 1:
                time_diffs = bvp_data.index.to_series().diff().dt.total_seconds()
                max_expected_gap = 2.0 / self.sampling_rates['BVP']  # 2x normal sampling interval
                if (time_diffs > max_expected_gap).sum() > 5:  # Allow few gaps
                    return False
        
        # Check EDA data quality
        if 'EDA' in window_data.columns:
            eda_data = window_data['EDA'].dropna()
            expected_eda_samples = self.window_size * self.sampling_rates['EDA']
            
            if len(eda_data) < expected_eda_samples * self.min_data_threshold:
                return False
        
        # Check ACC data quality
        acc_cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
        if all(col in window_data.columns for col in acc_cols):
            # We can check any of the axes, e.g., ACC_X, as they share timestamps
            acc_data = window_data['ACC_X'].dropna()
            expected_acc_samples = self.window_size * self.sampling_rates['ACC_X']
            
            if len(acc_data) < expected_acc_samples * self.min_data_threshold:
                return False
        
        return True
    
    def _extract_window_features_single(self, window_data: pd.DataFrame, 
                                      window_start: pd.Timestamp) -> Optional[Dict]:
        """Extract features from a single window."""
        features = {'timestamp': window_start}
        
        try:
            # HRV Features
            if 'IBI' in window_data.columns:
                ibi_values = window_data['IBI'].dropna()
                if len(ibi_values) >= 2:
                    hrv_features = self._calculate_hrv_features(ibi_values)
                    features.update(hrv_features)
            
            # Heart Rate Features
            if 'HR' in window_data.columns:
                hr_data = window_data['HR'].dropna()
                if len(hr_data) > 0:
                    features.update({
                        'hr_mean': hr_data.mean(),
                        'hr_std': hr_data.std(),
                        'hr_min': hr_data.min(),
                        'hr_max': hr_data.max(),
                        'hr_range': hr_data.max() - hr_data.min()
                    })
            
            # Temperature Features
            if 'TEMP' in window_data.columns:
                temp_data = window_data['TEMP'].dropna()
                if len(temp_data) > 0:
                    features.update({
                        'temp_mean': temp_data.mean(),
                        'temp_std': temp_data.std(),
                        'temp_min': temp_data.min(),
                        'temp_max': temp_data.max(),
                        'temp_range': temp_data.max() - temp_data.min()
                    })
            
            # Accelerometer Features
            acc_cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
            if all(col in window_data.columns for col in acc_cols):
                acc_features = self._calculate_acc_features(window_data[acc_cols])
                features.update(acc_features)
            
            # EDA Features
            if 'EDA' in window_data.columns:
                eda_data = window_data['EDA'].dropna()
                if len(eda_data) > 1:  # Need at least 2 points for advanced features
                    eda_features = self._calculate_eda_features(eda_data)
                    features.update(eda_features)
            
            # BVP Morphological Features (optimized)
            if 'BVP' in window_data.columns:
                bvp_data = window_data['BVP'].dropna()
                if len(bvp_data) >= self.window_size * 32:  # Minimum samples for reliable morphology
                    bvp_features = self._calculate_bvp_morphological_features(bvp_data)
                    features.update(bvp_features)
            
            # Context-aware features
            context_features = self._calculate_context_aware_features(window_data)
            features.update(context_features)
            
            return features
            
        except Exception as e:
            logging.warning(f"Error extracting features for window {window_start}: {e}")
            return None
    
    def _calculate_hrv_features(self, ibi_series: pd.Series) -> Dict:
        """Calculate HRV features from IBI data."""
        ibi_values = ibi_series.values
        
        # Basic statistics
        mean_ibi = np.mean(ibi_values)
        
        # RMSSD
        diff_ibi = np.diff(ibi_values)
        rmssd = np.sqrt(np.mean(diff_ibi ** 2)) if len(diff_ibi) > 0 else np.nan
        
        # SDNN
        sdnn = np.std(ibi_values)
        
        # pNN50
        pnn50 = (np.abs(diff_ibi) > 0.05).sum() / len(diff_ibi) * 100 if len(diff_ibi) > 0 else 0
        
        return {
            'hrv_mean_ibi': mean_ibi,
            'hrv_rmssd': rmssd,
            'hrv_sdnn': sdnn,
            'hrv_pnn50': pnn50
        }
    
    def _calculate_acc_features(self, acc_data: pd.DataFrame) -> Dict:
        """
        Calculate accelerometer features - only magnitude-based features are retained.
        Ensures magnitude is calculated per sample, then aggregated over the window.
        
        Note: Raw ACC data is in 1/64g units, so we convert to g-force units first.
        """
        # Verify we have all required columns
        required_cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
        if not all(col in acc_data.columns for col in required_cols):
            return {}
        
        # Convert raw accelerometer data from 1/64g units to g-force units
        # Raw data format: 1/64g (so divide by 64 to get actual g values)
        acc_x_g = acc_data['ACC_X'].fillna(0) / 64.0
        acc_y_g = acc_data['ACC_Y'].fillna(0) / 64.0
        acc_z_g = acc_data['ACC_Z'].fillna(0) / 64.0
        
        # Calculate magnitude per sample in g-force units
        # For each time point: magnitude = sqrt(x² + y² + z²)
        acc_magnitude = np.sqrt(acc_x_g**2 + acc_y_g**2 + acc_z_g**2)
        
        # Remove zero padding (samples where all axes were NaN/0)
        acc_magnitude = acc_magnitude[acc_magnitude > 0]
        
        if len(acc_magnitude) == 0:
            return {}
        
        # Calculate only the important magnitude-based features (now in g-force units)
        features = {
            'acc_magnitude_mean': acc_magnitude.mean(),
            'acc_magnitude_std': acc_magnitude.std(),
            'acc_magnitude_max': acc_magnitude.max(),
            'acc_magnitude_min': acc_magnitude.min(),
            'acc_activity_level': self._classify_activity_level(acc_magnitude.mean())
        }
        
        return features
    
    def _classify_activity_level(self, acc_mean: float) -> int:
        """
        Classify activity level based on accelerometer magnitude in g-force units.
        
        Args:
            acc_mean: Mean acceleration magnitude in g-force units
            
        Returns:
            Activity level: 0=Resting, 1=Light, 2=Moderate, 3=High
            
        Note: Thresholds are based on research literature for wearable accelerometers.
        - Resting: ~1.0g (gravitational baseline with minimal movement)
        - Light: 1.0-1.2g (walking slowly, gentle movements)  
        - Moderate: 1.2-1.8g (normal walking, moderate activities)
        - High: >1.8g (running, vigorous activities)
        """
        if acc_mean < 1.2:
            return 0  # Resting/Sedentary
        elif acc_mean < 1.8:
            return 1  # Light activity
        elif acc_mean < 2.5:
            return 2  # Moderate activity
        else:
            return 3  # High activity

    
    def _calculate_eda_features(self, eda_data: pd.Series) -> Dict:
        """
        Calculate advanced EDA features including tonic and phasic components.
        """
        features = {
            'eda_mean': eda_data.mean(),
            'eda_std': eda_data.std(),
            'eda_max': eda_data.max(),
            'eda_min': eda_data.min(),
            'eda_range': eda_data.max() - eda_data.min()
        }
        
        fs = self.sampling_rates['EDA']
        signal = eda_data.values
        
        try:
            # Tonic component (SCL) - low-pass filter to get the slow-moving trend
            nyq = 0.5 * fs
            low = 0.1 / nyq  # Cutoff frequency for SCL
            b, a = butter(2, low, btype='low')
            scl = filtfilt(b, a, signal)
            
            # EDA slope
            time_seconds = len(signal) / fs
            if time_seconds > 0:
                features['eda_slope'] = (scl[-1] - scl[0]) / time_seconds

            # Phasic component (SCR) - band-pass filter to isolate rapid changes
            low_scr = 0.05 / nyq
            # The high cutoff must be strictly less than the Nyquist frequency (nyq).
            # The original value of 2.0 was equal to nyq for EDA (fs=4Hz), causing an error.
            # We set it to a value slightly below nyq.
            high_scr_freq = 1.99
            high_scr = high_scr_freq / nyq
            b_scr, a_scr = butter(4, [low_scr, high_scr], btype='band')
            scr = filtfilt(b_scr, a_scr, signal)

            # Find peaks (SCRs)
            peaks, properties = scipy.signal.find_peaks(
                scr, 
                height=0.01,  # Min height in micro-siemens
                distance=fs,  # Min 1-second distance between peaks
                width=True
            )
            
            if len(peaks) > 0:
                features['num_scr_peaks'] = len(peaks)
                features['mean_scr_amplitude'] = np.mean(properties['peak_heights'])
                
                # Width is in samples, convert to seconds
                peak_widths_sec = properties['widths'] / fs
                features['mean_scr_peak_width'] = np.mean(peak_widths_sec)
                
                # Area under curve (amplitude * width)
                areas = properties['peak_heights'] * peak_widths_sec
                features['mean_scr_area'] = np.mean(areas)
            else:
                # If no peaks, set features to 0
                features['num_scr_peaks'] = 0
                features['mean_scr_amplitude'] = 0
                features['mean_scr_peak_width'] = 0
                features['mean_scr_area'] = 0

        except Exception as e:
            logging.warning(f"Could not extract advanced EDA features for window: {e}")
            # Fallback to NaN for advanced features if calculation fails
            features['eda_slope'] = np.nan
            features['num_scr_peaks'] = 0
            features['mean_scr_amplitude'] = 0
            features['mean_scr_peak_width'] = 0
            features['mean_scr_area'] = 0
            
        return features

    
    def _calculate_bvp_morphological_features(self, bvp_data: pd.Series) -> Dict:
        """
        Optimized BVP morphological feature extraction.
        Only processes windows with sufficient data quality.
        """
        try:
            # Apply bandpass filter
            fs = 64  # BVP sampling rate
            filtered_bvp = self._bandpass_filter(bvp_data.values, fs)
            
            # Basic statistical features first
            features = {
                'bvp_mean': np.mean(filtered_bvp),
                'bvp_std': np.std(filtered_bvp),
                'bvp_max': np.max(filtered_bvp),
                'bvp_min': np.min(filtered_bvp),
                'bvp_range': np.max(filtered_bvp) - np.min(filtered_bvp)
            }
            
            # Find peaks with adaptive parameters
            min_distance = int(0.4 * fs)  # Minimum 40ms between peaks
            peaks, _ = scipy.signal.find_peaks(
                filtered_bvp, 
                distance=min_distance,
                prominence=np.std(filtered_bvp) * 0.1
            )
            
            if len(peaks) < 2:
                return features
            
            # Calculate morphological features for each pulse
            pulse_features = self._extract_pulse_features(filtered_bvp, peaks, fs)
            
            if not pulse_features or not pulse_features['amplitudes']:
                return features
            
            # Aggregate morphological features
            features.update({
                'bvp_systolic_amp_mean': np.mean(pulse_features['amplitudes']),
                'bvp_systolic_amp_std': np.std(pulse_features['amplitudes']),
                'bvp_pulse_width_mean': np.mean(pulse_features['widths']),
                'bvp_pulse_width_std': np.std(pulse_features['widths']),
                'bvp_rise_time_mean': np.mean(pulse_features['rise_times']),
                'bvp_rise_time_std': np.std(pulse_features['rise_times']),
                'bvp_pulse_rate': len(peaks) / (len(bvp_data) / fs) * 60  # beats per minute
            })
            
            return features
            
        except Exception as e:
            logging.warning(f"BVP feature extraction failed: {e}")
            return {
                'bvp_mean': np.mean(bvp_data.values),
                'bvp_std': np.std(bvp_data.values),
                'bvp_max': np.max(bvp_data.values),
                'bvp_min': np.min(bvp_data.values)
            }
    
    def _extract_pulse_features(self, signal: np.ndarray, peaks: np.ndarray, 
                               fs: int) -> Dict[str, List]:
        """Extract features for individual pulses."""
        amplitudes, widths, rise_times = [], [], []
        
        for peak in peaks:
            try:
                # Find foot (minimum before peak)
                search_start = max(0, peak - int(0.5 * fs))
                foot_idx = np.argmin(signal[search_start:peak]) + search_start
                
                # Find end (minimum after peak)
                search_end = min(len(signal), peak + int(0.5 * fs))
                end_idx = np.argmin(signal[peak:search_end]) + peak
                
                if foot_idx >= peak or end_idx <= peak:
                    continue
                
                # Calculate features
                amplitude = signal[peak] - signal[foot_idx]
                width = (end_idx - foot_idx) / fs
                rise_time = (peak - foot_idx) / fs
                
                # Quality checks
                if amplitude > 0 and 0.1 < width < 2.0 and 0.05 < rise_time < 1.0:
                    amplitudes.append(amplitude)
                    widths.append(width)
                    rise_times.append(rise_time)
                    
            except (IndexError, ValueError):
                continue
        
        return {
            'amplitudes': amplitudes,
            'widths': widths,
            'rise_times': rise_times
        }
    
    def _bandpass_filter(self, signal: np.ndarray, fs: int, 
                        lowcut: float = 0.5, highcut: float = 8.0) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def _calculate_context_aware_features(self, window_data: pd.DataFrame) -> Dict:
        """Calculate context-aware features for activity normalization."""
        features = {}
        
        # Calculate ACC magnitude for normalization
        acc_cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
        if all(col in window_data.columns for col in acc_cols):
            acc_mag = np.sqrt(
                window_data['ACC_X'].fillna(0)**2 + 
                window_data['ACC_Y'].fillna(0)**2 + 
                window_data['ACC_Z'].fillna(0)**2
            )
            acc_mag = acc_mag[acc_mag > 0]
            
            if len(acc_mag) > 0:
                acc_mean = acc_mag.mean()
                
                # EDA to ACC ratio
                if 'EDA' in window_data.columns:
                    eda_mean = window_data['EDA'].mean()
                    if not pd.isna(eda_mean) and acc_mean > 0.01:
                        features['eda_acc_ratio'] = eda_mean / acc_mean
                
                # HR to ACC ratio
                if 'HR' in window_data.columns:
                    hr_mean = window_data['HR'].mean()
                    if not pd.isna(hr_mean) and acc_mean > 0.01:
                        features['hr_acc_ratio'] = hr_mean / acc_mean
        
        return features

def process_participant(participant_dir: str, output_dir: str, 
                       window_size: int = 10, max_sessions: Optional[int] = None) -> None:
    """
    Process a single participant's data.
    
    Args:
        participant_dir: Path to participant directory
        output_dir: Output directory for processed features
        window_size: Window size in seconds for feature extraction
        max_sessions: Maximum number of sessions to process (None for all)
    """
    participant_id = os.path.basename(participant_dir)
    
    # Add session limit to output filename if specified
    if max_sessions is not None:
        output_file = os.path.join(output_dir, f"{participant_id}_{max_sessions}sessions.csv")
    else:
        output_file = os.path.join(output_dir, f"{participant_id}.csv")
    
    try:
        preprocessor = BiosignalPreprocessor(window_size=window_size)
        
        # Load and merge sessions with session limit
        merged_data = preprocessor.load_and_merge_sessions(participant_dir, max_sessions=max_sessions)
        
        # Extract features (now session-aware)
        features_df = preprocessor.extract_window_features(merged_data)
        
        # Reorder columns to put session and timestamp first
        column_order = ['session', 'timestamp'] + [col for col in features_df.columns if col not in ['session', 'timestamp']]
        features_df = features_df[column_order]
        
        # Save results
        features_df.to_csv(output_file, index=False)
        
        sessions_info = f" (limited to {max_sessions} sessions)" if max_sessions else ""
        logging.info(f"Saved features for participant {participant_id}{sessions_info}: {len(features_df)} windows")
        logging.info(f"Sessions processed: {features_df['session'].unique()}")
        logging.info(f"Windows per session: {features_df['session'].value_counts().to_dict()}")
        
    except Exception as e:
        logging.error(f"Failed to process participant {participant_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process multimodal biosignal data')
    parser.add_argument('--participant', type=str, help='Specific participant ID to process')
    parser.add_argument('--data-dir', type=str, default='../../data/raw', 
                       help='Path to raw data directory (default: ../../data/raw)')
    parser.add_argument('--output-dir', type=str, default='../../data/processed',
                       help='Output directory for processed features (default: processed_features)')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Window size in seconds (default: 10)')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='Maximum number of sessions to process per participant (default: all sessions)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert relative paths to absolute paths based on script location
    script_dir = Path(__file__).parent
    raw_data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    # Verify data directory exists
    if not raw_data_dir.exists():
        logging.error(f"Data directory does not exist: {raw_data_dir}")
        logging.info(f"Script is located at: {script_dir}")
        logging.info(f"Looking for data at: {raw_data_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Data directory: {raw_data_dir}")
    logging.info(f"Output directory: {output_dir}")
    if args.max_sessions:
        logging.info(f"Session limit: {args.max_sessions}")
    
    if args.participant:
        # Process specific participant
        participant_path = raw_data_dir / args.participant
        if not participant_path.exists():
            logging.error(f"Participant directory does not exist: {participant_path}")
            available_participants = [d.name for d in raw_data_dir.iterdir() if d.is_dir()]
            logging.info(f"Available participants: {available_participants}")
            return
        
        logging.info(f"Processing single participant: {args.participant}")
        process_participant(str(participant_path), str(output_dir), args.window_size, args.max_sessions)
    else:
        # Process all participants
        logging.info("Processing all participants...")
        for participant_dir in raw_data_dir.iterdir():
            if participant_dir.is_dir():
                process_participant(str(participant_dir), str(output_dir), args.window_size, args.max_sessions)

if __name__ == "__main__":
    main()