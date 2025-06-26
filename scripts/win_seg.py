def debug_window_segmentation_temperature(merged_data: pd.DataFrame, window_size: int = 10):
    """
    Debug the window segmentation process specifically for temperature data
    """
    print("=== WINDOW SEGMENTATION DEBUG FOR TEMPERATURE ===\n")
    
    # Filter for temperature data
    if 'TEMP' not in merged_data.columns:
        print("ERROR: No TEMP column found in merged data")
        print(f"Available columns: {list(merged_data.columns)}")
        return
    
    print(f"1. Input data analysis:")
    print(f"   Total rows: {len(merged_data)}")
    print(f"   Temperature data points: {merged_data['TEMP'].notna().sum()}")
    print(f"   Temperature NaN values: {merged_data['TEMP'].isna().sum()}")
    print(f"   Temperature range: {merged_data['TEMP'].min():.4f} to {merged_data['TEMP'].max():.4f}")
    print(f"   Temperature mean: {merged_data['TEMP'].mean():.4f}")
    
    # Check timestamp range
    print(f"\n2. Timestamp analysis:")
    print(f"   Timestamp range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
    print(f"   Duration: {merged_data['timestamp'].max() - merged_data['timestamp'].min()}")
    
    # Set timestamp as index for window processing
    data_indexed = merged_data.set_index('timestamp')
    
    # Generate windows (as done in original code)
    print(f"\n3. Window generation:")
    start_time = data_indexed.index.min().floor(f'{window_size}s')
    end_time = data_indexed.index.max().ceil(f'{window_size}s')
    windows = pd.date_range(start=start_time, end=end_time, freq=f'{window_size}s')
    
    print(f"   Window size: {window_size} seconds")
    print(f"   Start time: {start_time}")
    print(f"   End time: {end_time}")
    print(f"   Total windows: {len(windows)-1}")
    
    # Detailed analysis of first 5 windows
    print(f"\n4. Detailed window analysis (first 5 windows):")
    
    window_features = []
    
    for i in range(min(5, len(windows)-1)):
        window_start = windows[i]
        window_end = windows[i] + pd.Timedelta(seconds=window_size)
        
        print(f"\n   Window {i+1}: {window_start} to {window_end}")
        
        # Extract window data
        window_data = data_indexed[(data_indexed.index >= window_start) & 
                                  (data_indexed.index < window_end)]
        
        print(f"     Total data points in window: {len(window_data)}")
        
        # Focus on temperature
        if 'TEMP' in window_data.columns:
            temp_data = window_data['TEMP'].dropna()
            print(f"     Temperature data points: {len(temp_data)}")
            
            if len(temp_data) > 0:
                temp_stats = {
                    'window': i+1,
                    'start_time': window_start,
                    'count': len(temp_data),
                    'mean': temp_data.mean(),
                    'std': temp_data.std(),
                    'min': temp_data.min(),
                    'max': temp_data.max(),
                    'first_value': temp_data.iloc[0] if len(temp_data) > 0 else None,
                    'last_value': temp_data.iloc[-1] if len(temp_data) > 0 else None
                }
                window_features.append(temp_stats)
                
                print(f"     Temperature stats:")
                print(f"       Mean: {temp_stats['mean']:.4f}")
                print(f"       Std: {temp_stats['std']:.4f}")
                print(f"       Min: {temp_stats['min']:.4f}")
                print(f"       Max: {temp_stats['max']:.4f}")
                print(f"       First value: {temp_stats['first_value']:.4f}")
                print(f"       Last value: {temp_stats['last_value']:.4f}")
                
                # Show some sample values
                sample_values = temp_data.head(10).tolist()
                print(f"       Sample values: {[f'{v:.4f}' for v in sample_values]}")
                
                # Check timestamps of temperature data in this window
                temp_timestamps = temp_data.index[:5]
                print(f"       Sample timestamps: {temp_timestamps.tolist()}")
            else:
                print(f"     No temperature data in this window")
        else:
            print(f"     No TEMP column in window data")
    
    # Summary statistics across windows
    if window_features:
        print(f"\n5. Summary across analyzed windows:")
        means = [w['mean'] for w in window_features]
        counts = [w['count'] for w in window_features]
        
        print(f"   Window means: {[f'{m:.4f}' for m in means]}")
        print(f"   Window counts: {counts}")
        print(f"   Average window mean: {np.mean(means):.4f}")
        print(f"   Average window count: {np.mean(counts):.1f}")
    
    return window_features

def compare_original_vs_windowed_temp(merged_data: pd.DataFrame, window_size: int = 10):
    """
    Compare original temperature data vs windowed features
    """
    print("=== COMPARING ORIGINAL VS WINDOWED TEMPERATURE ===\n")
    
    # Original data stats
    original_temp = merged_data['TEMP'].dropna()
    print(f"Original temperature data:")
    print(f"   Count: {len(original_temp)}")
    print(f"   Mean: {original_temp.mean():.4f}")
    print(f"   Std: {original_temp.std():.4f}")
    print(f"   Min: {original_temp.min():.4f}")
    print(f"   Max: {original_temp.max():.4f}")
    
    # Process windows and extract features
    data_indexed = merged_data.set_index('timestamp')
    start_time = data_indexed.index.min().floor(f'{window_size}s')
    end_time = data_indexed.index.max().ceil(f'{window_size}s')
    windows = pd.date_range(start=start_time, end=end_time, freq=f'{window_size}s')
    
    windowed_means = []
    windowed_counts = []
    
    for i in range(len(windows)-1):
        window_start = windows[i]
        window_end = windows[i] + pd.Timedelta(seconds=window_size)
        
        window_data = data_indexed[(data_indexed.index >= window_start) & 
                                  (data_indexed.index < window_end)]
        
        if 'TEMP' in window_data.columns:
            temp_data = window_data['TEMP'].dropna()
            if len(temp_data) > 0:
                windowed_means.append(temp_data.mean())
                windowed_counts.append(len(temp_data))
    
    print(f"\nWindowed temperature features:")
    print(f"   Valid windows: {len(windowed_means)}")
    print(f"   Mean of window means: {np.mean(windowed_means):.4f}")
    print(f"   Std of window means: {np.std(windowed_means):.4f}")
    print(f"   Min window mean: {np.min(windowed_means):.4f}")
    print(f"   Max window mean: {np.max(windowed_means):.4f}")
    print(f"   Average points per window: {np.mean(windowed_counts):.1f}")
    
    # Check if there's a significant difference
    diff = abs(original_temp.mean() - np.mean(windowed_means))
    print(f"\nDifference between original mean and windowed mean: {diff:.4f}")
    
    if diff > 1.0:  # If difference is more than 1 degree
        print("⚠️  WARNING: Significant difference detected!")
        print("   This suggests an issue with the window processing logic.")
    else:
        print("✅ Difference is within acceptable range.")

# Example usage:
# window_features = debug_window_segmentation_temperature(merged_data)
# compare_original_vs_windowed_temp(merged_data)