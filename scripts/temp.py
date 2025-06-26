import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def debug_temperature_processing(session_path: str):
    """
    Debug function to analyze temperature data processing step by step
    """
    print("=== TEMPERATURE PROCESSING DEBUG ===\n")
    
    # 1. Read raw TEMP.csv file
    temp_file = f"{session_path}/TEMP.csv"
    print(f"1. Reading raw temperature file: {temp_file}")
    
    try:
        with open(temp_file, 'r') as f:
            lines = f.readlines()
        
        print(f"   Total lines in file: {len(lines)}")
        print(f"   First 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"   Line {i}: {line.strip()}")
        
        # 2. Parse header information
        print(f"\n2. Parsing header information:")
        start_time = float(lines[0].strip())
        sampling_rate = float(lines[1].strip())
        
        print(f"   Start timestamp (Unix): {start_time}")
        print(f"   Start time (readable): {datetime.fromtimestamp(start_time)}")
        print(f"   Sampling rate: {sampling_rate} Hz")
        
        # 3. Parse temperature data
        print(f"\n3. Parsing temperature values:")
        temp_values = []
        for i, line in enumerate(lines[2:], start=2):
            try:
                value = float(line.strip())
                temp_values.append(value)
            except ValueError:
                print(f"   Warning: Could not parse line {i}: {line.strip()}")
                continue
        
        print(f"   Successfully parsed {len(temp_values)} temperature values")
        print(f"   First 10 values: {temp_values[:10]}")
        print(f"   Last 10 values: {temp_values[-10:]}")
        
        # 4. Basic statistics
        print(f"\n4. Raw temperature statistics:")
        temp_array = np.array(temp_values)
        print(f"   Min: {np.min(temp_array):.4f}")
        print(f"   Max: {np.max(temp_array):.4f}")
        print(f"   Mean: {np.mean(temp_array):.4f}")
        print(f"   Std: {np.std(temp_array):.4f}")
        print(f"   Median: {np.median(temp_array):.4f}")
        
        # 5. Check for anomalies
        print(f"\n5. Anomaly detection:")
        zero_count = np.sum(temp_array == 0)
        negative_count = np.sum(temp_array < 0)
        very_low_count = np.sum(temp_array < 20)  # Below 20°C is suspicious
        very_high_count = np.sum(temp_array > 50)  # Above 50°C is suspicious
        
        print(f"   Zero values: {zero_count} ({zero_count/len(temp_array)*100:.2f}%)")
        print(f"   Negative values: {negative_count} ({negative_count/len(temp_array)*100:.2f}%)")
        print(f"   Values < 20°C: {very_low_count} ({very_low_count/len(temp_array)*100:.2f}%)")
        print(f"   Values > 50°C: {very_high_count} ({very_high_count/len(temp_array)*100:.2f}%)")
        
        # 6. Generate timestamps as done in original code
        print(f"\n6. Timestamp generation:")
        n_samples = len(temp_values)
        timestamps = start_time + np.arange(n_samples) / sampling_rate
        
        print(f"   Number of samples: {n_samples}")
        print(f"   Expected duration: {n_samples / sampling_rate / 60:.2f} minutes")
        print(f"   First timestamp: {timestamps[0]} ({datetime.fromtimestamp(timestamps[0])})")
        print(f"   Last timestamp: {timestamps[-1]} ({datetime.fromtimestamp(timestamps[-1])})")
        
        # 7. Create DataFrame as in original code
        print(f"\n7. Creating DataFrame:")
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'TEMP': temp_values,
            'sensor': 'TEMP'
        })
        
        print(f"   DataFrame shape: {df.shape}")
        print(f"   DataFrame info:")
        print(df.info())
        print(f"\n   First 5 rows:")
        print(df.head())
        print(f"\n   Temperature column statistics:")
        print(df['TEMP'].describe())
        
        # 8. Simulate window processing
        print(f"\n8. Simulating window processing:")
        window_size = 10  # seconds
        df_indexed = df.set_index('timestamp')
        
        # Generate windows
        start_time_dt = df_indexed.index.min().floor(f'{window_size}s')
        end_time_dt = df_indexed.index.max().ceil(f'{window_size}s')
        windows = pd.date_range(start=start_time_dt, end=end_time_dt, freq=f'{window_size}s')
        
        print(f"   Number of windows: {len(windows)-1}")
        print(f"   Window start: {start_time_dt}")
        print(f"   Window end: {end_time_dt}")
        
        # Process first few windows
        print(f"\n   Processing first 3 windows:")
        for i in range(min(3, len(windows)-1)):
            window_start = windows[i]
            window_end = windows[i] + pd.Timedelta(seconds=window_size)
            
            window_data = df_indexed[(df_indexed.index >= window_start) & (df_indexed.index < window_end)]
            temp_data = window_data['TEMP'].dropna()
            
            print(f"   Window {i+1}: {window_start} to {window_end}")
            print(f"     Data points: {len(temp_data)}")
            if len(temp_data) > 0:
                print(f"     Min: {temp_data.min():.4f}")
                print(f"     Max: {temp_data.max():.4f}")
                print(f"     Mean: {temp_data.mean():.4f}")
                print(f"     Sample values: {temp_data.head().tolist()}")
            else:
                print(f"     No data in this window")
        
        return df
        
    except Exception as e:
        print(f"Error during debug: {e}")
        return None

def plot_temperature_data(df: pd.DataFrame, max_points: int = 10000):
    """
    Plot temperature data to visualize patterns
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Sample data if too large
    if len(df) > max_points:
        df_plot = df.sample(n=max_points).sort_values('timestamp')
        print(f"Plotting sample of {max_points} points from {len(df)} total points")
    else:
        df_plot = df
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['timestamp'], df_plot['TEMP'], alpha=0.7)
    plt.title('Temperature Data Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['TEMP'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.axvline(df['TEMP'].mean(), color='red', linestyle='--', label=f'Mean: {df["TEMP"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
df = debug_temperature_processing("/path/to/session/directory")
plot_temperature_data(df)