"""
compare_signals.py

Module to compare raw vs. processed physiological signals using:
1. Visual comparison (line and box plots)
2. Statistical metrics (MSE, Pearson correlation, DTW)
3. Spectral analysis (Power Spectral Density)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.signal import welch

try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False


def align_dataframes(df_raw, df_processed, column='timestamp'):
    """Aligns raw and processed DataFrames to common time range."""
    common_start = max(df_raw[column].min(), df_processed[column].min())
    common_end = min(df_raw[column].max(), df_processed[column].max())
    
    df_raw_aligned = df_raw[(df_raw[column] >= common_start) & (df_raw[column] <= common_end)].copy()
    df_processed_aligned = df_processed[(df_processed[column] >= common_start) & (df_processed[column] <= common_end)].copy()
    return df_raw_aligned.reset_index(drop=True), df_processed_aligned.reset_index(drop=True)


def compare_visuals(df_raw, df_processed, feature='HR', title_prefix=''):
    """Plot line plot and boxplot comparing raw vs. processed data."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ✅ Reset indices to avoid duplicate index issues during concat
    df_raw = df_raw.reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Line plot
    axes[0].plot(df_raw['timestamp'], df_raw[feature], label='Raw', alpha=0.6)
    axes[0].plot(df_processed['timestamp'], df_processed[feature], label='Processed', alpha=0.6)
    axes[0].set_title(f"{title_prefix} Time Series Comparison: {feature}")
    axes[0].legend()

    # Boxplot
    melted = pd.concat([
        df_raw[[feature]].assign(Source='Raw'),
        df_processed[[feature]].assign(Source='Processed')
    ], ignore_index=True)  # ✅ Added ignore_index=True to prevent index conflicts
    sns.boxplot(x='Source', y=feature, data=melted, ax=axes[1])
    axes[1].set_title(f"{title_prefix} Boxplot: {feature}")

    plt.tight_layout()
    plt.show()


def compare_visuals_debug(df_raw, df_processed, feature='HR', title_prefix=''):
    """Enhanced version with debugging and better signal alignment."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Reset indices
    df_raw = df_raw.reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)
    
    # 🔍 DEBUGGING: Print data info
    print(f"\n=== DEBUGGING {feature} ===")
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Raw {feature} - Non-null count: {df_raw[feature].notna().sum()}")
    print(f"Processed {feature} - Non-null count: {df_processed[feature].notna().sum()}")
    print(f"Raw timestamp range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    print(f"Processed timestamp range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    # Check for actual data presence
    raw_has_data = df_raw[feature].notna().any()
    proc_has_data = df_processed[feature].notna().any()
    print(f"Raw has {feature} data: {raw_has_data}")
    print(f"Processed has {feature} data: {proc_has_data}")
    
    if raw_has_data:
        print(f"Raw {feature} range: {df_raw[feature].min():.2f} to {df_raw[feature].max():.2f}")
    if proc_has_data:
        print(f"Processed {feature} range: {df_processed[feature].min():.2f} to {df_processed[feature].max():.2f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 🔧 ENHANCED LINE PLOT with better handling
    if raw_has_data and proc_has_data:
        # Plot raw data
        raw_mask = df_raw[feature].notna()
        axes[0].plot(df_raw.loc[raw_mask, 'timestamp'], 
                    df_raw.loc[raw_mask, feature], 
                    label='Raw', alpha=0.7, linewidth=1.5, color='blue')
        
        # Plot processed data
        proc_mask = df_processed[feature].notna()
        axes[0].plot(df_processed.loc[proc_mask, 'timestamp'], 
                    df_processed.loc[proc_mask, feature], 
                    label='Processed', alpha=0.7, linewidth=1.5, color='red')
        
        axes[0].set_title(f"{title_prefix} Time Series Comparison: {feature}")
        axes[0].set_ylabel(feature)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add data point counts to legend
        raw_count = raw_mask.sum()
        proc_count = proc_mask.sum()
        axes[0].text(0.02, 0.98, f'Raw: {raw_count} points\nProcessed: {proc_count} points', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    elif raw_has_data:
        raw_mask = df_raw[feature].notna()
        axes[0].plot(df_raw.loc[raw_mask, 'timestamp'], 
                    df_raw.loc[raw_mask, feature], 
                    label='Raw', alpha=0.7, linewidth=1.5, color='blue')
        axes[0].set_title(f"{title_prefix} Time Series: {feature} (Raw Only)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
    elif proc_has_data:
        proc_mask = df_processed[feature].notna()
        axes[0].plot(df_processed.loc[proc_mask, 'timestamp'], 
                    df_processed.loc[proc_mask, feature], 
                    label='Processed', alpha=0.7, linewidth=1.5, color='red')
        axes[0].set_title(f"{title_prefix} Time Series: {feature} (Processed Only)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, f'No data available for {feature}', 
                    transform=axes[0].transAxes, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        axes[0].set_title(f"{title_prefix} Time Series: {feature} (No Data)")

    # 🔧 ENHANCED BOXPLOT with better data handling
    valid_data = []
    if raw_has_data:
        raw_valid = df_raw[df_raw[feature].notna()][[feature]].assign(Source='Raw')
        valid_data.append(raw_valid)
    
    if proc_has_data:
        proc_valid = df_processed[df_processed[feature].notna()][[feature]].assign(Source='Processed')
        valid_data.append(proc_valid)
    
    if valid_data:
        melted = pd.concat(valid_data, ignore_index=True)
        sns.boxplot(x='Source', y=feature, data=melted, ax=axes[1])
        axes[1].set_title(f"{title_prefix} Distribution Comparison: {feature}")
        
        # Add sample sizes to boxplot
        for i, source in enumerate(melted['Source'].unique()):
            count = len(melted[melted['Source'] == source])
            axes[1].text(i, axes[1].get_ylim()[1] * 0.95, f'n={count}', 
                        ha='center', va='top', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, f'No valid data for boxplot: {feature}', 
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        axes[1].set_title(f"{title_prefix} Distribution: {feature} (No Data)")

    plt.tight_layout()
    plt.show()
    print("=== END DEBUG ===\n")


def compare_visuals_fixed(df_raw, df_processed, feature='HR', title_prefix=''):
    """Fixed version focusing on proper signal alignment and visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Clean data preparation
    df_raw = df_raw.reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)
    
    # Remove rows where the feature is NaN
    raw_clean = df_raw[df_raw[feature].notna()].copy()
    proc_clean = df_processed[df_processed[feature].notna()].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 🎯 IMPROVED LINE PLOT
    if len(raw_clean) > 0:
        axes[0].plot(raw_clean['timestamp'], raw_clean[feature], 
                    label=f'Raw ({len(raw_clean)} points)', 
                    alpha=0.6, linewidth=1.2, color='steelblue', marker='o', markersize=1)
    
    if len(proc_clean) > 0:
        axes[0].plot(proc_clean['timestamp'], proc_clean[feature], 
                    label=f'Processed ({len(proc_clean)} points)', 
                    alpha=0.8, linewidth=1.5, color='orangered', marker='s', markersize=1)
    
    axes[0].set_title(f"{title_prefix} Time Series Comparison: {feature}")
    axes[0].set_ylabel(feature)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Set reasonable x-axis limits if data exists
    if len(raw_clean) > 0 or len(proc_clean) > 0:
        all_timestamps = []
        if len(raw_clean) > 0:
            all_timestamps.extend(raw_clean['timestamp'].tolist())
        if len(proc_clean) > 0:
            all_timestamps.extend(proc_clean['timestamp'].tolist())
        
        if all_timestamps:
            axes[0].set_xlim(min(all_timestamps), max(all_timestamps))

    # 🎯 IMPROVED BOXPLOT
    plot_data = []
    if len(raw_clean) > 0:
        plot_data.append(raw_clean[[feature]].assign(Source='Raw'))
    if len(proc_clean) > 0:
        plot_data.append(proc_clean[[feature]].assign(Source='Processed'))
    
    if plot_data:
        melted = pd.concat(plot_data, ignore_index=True)
        sns.boxplot(x='Source', y=feature, data=melted, ax=axes[1])
        axes[1].set_title(f"{title_prefix} Distribution Comparison: {feature}")
        
        # Add statistical info
        for i, source in enumerate(['Raw', 'Processed']):
            if source in melted['Source'].values:
                subset = melted[melted['Source'] == source][feature]
                stats_text = f'μ={subset.mean():.2f}\nσ={subset.std():.2f}'
                axes[1].text(i, axes[1].get_ylim()[0], stats_text, 
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        axes[1].text(0.5, 0.5, f'No valid data available for {feature}', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=12)
        axes[1].set_title(f"{title_prefix} Distribution: {feature} (No Data)")

    plt.tight_layout()
    plt.show()


def diagnose_signal_data(df_raw, df_processed, feature):
    """Diagnose why signals might not be appearing in plots."""
    print(f"\n🔍 DIAGNOSING {feature.upper()} SIGNAL ISSUES:")
    print("=" * 50)
    
    # Basic data info
    print(f"📊 DATA OVERVIEW:")
    print(f"  Raw DataFrame: {df_raw.shape[0]} rows")
    print(f"  Processed DataFrame: {df_processed.shape[0]} rows")
    
    # Feature-specific analysis
    raw_feature_data = df_raw[feature] if feature in df_raw.columns else pd.Series(dtype=float)
    proc_feature_data = df_processed[feature] if feature in df_processed.columns else pd.Series(dtype=float)
    
    print(f"\n📈 {feature.upper()} FEATURE ANALYSIS:")
    print(f"  Raw - Total values: {len(raw_feature_data)}")
    print(f"  Raw - Non-null values: {raw_feature_data.notna().sum()}")
    print(f"  Raw - Null values: {raw_feature_data.isna().sum()}")
    
    print(f"  Processed - Total values: {len(proc_feature_data)}")
    print(f"  Processed - Non-null values: {proc_feature_data.notna().sum()}")
    print(f"  Processed - Null values: {proc_feature_data.isna().sum()}")
    
    # Value ranges
    if raw_feature_data.notna().any():
        print(f"\n📊 RAW {feature.upper()} STATISTICS:")
        print(f"  Min: {raw_feature_data.min():.3f}")
        print(f"  Max: {raw_feature_data.max():.3f}")
        print(f"  Mean: {raw_feature_data.mean():.3f}")
        print(f"  Std: {raw_feature_data.std():.3f}")
    else:
        print(f"\n❌ NO VALID RAW {feature.upper()} DATA")
    
    if proc_feature_data.notna().any():
        print(f"\n📊 PROCESSED {feature.upper()} STATISTICS:")
        print(f"  Min: {proc_feature_data.min():.3f}")
        print(f"  Max: {proc_feature_data.max():.3f}")
        print(f"  Mean: {proc_feature_data.mean():.3f}")
        print(f"  Std: {proc_feature_data.std():.3f}")
    else:
        print(f"\n❌ NO VALID PROCESSED {feature.upper()} DATA")
    
    # Timestamp analysis
    if 'timestamp' in df_raw.columns and 'timestamp' in df_processed.columns:
        print(f"\n⏰ TIMESTAMP ANALYSIS:")
        print(f"  Raw timestamp range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        print(f"  Processed timestamp range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
        
        # Check overlap
        raw_ts_set = set(df_raw['timestamp'].dropna())
        proc_ts_set = set(df_processed['timestamp'].dropna())
        overlap = raw_ts_set.intersection(proc_ts_set)
        print(f"  Overlapping timestamps: {len(overlap)}")
        
        if len(overlap) == 0:
            print("  ⚠️  WARNING: No overlapping timestamps found!")
        
    print("=" * 50)


def compare_stats(df_raw, df_processed, feature):
    """Compute MSE and Pearson correlation between raw and processed."""
    raw_values = df_raw[feature].dropna()
    proc_values = df_processed[feature].dropna()
    common_len = min(len(raw_values), len(proc_values))

    mse = mean_squared_error(raw_values[:common_len], proc_values[:common_len])
    corr, _ = pearsonr(raw_values[:common_len], proc_values[:common_len])

    results = {
        'MSE': mse,
        'Pearson Correlation': corr
    }

    if HAS_DTW:
        dtw_distance = dtw.distance(raw_values[:common_len].values, proc_values[:common_len].values)
        results['DTW Distance'] = dtw_distance

    return results


def compare_spectrum(df_raw, df_processed, feature, fs=4):
    """Plot Power Spectral Density of raw and processed data."""
    raw = df_raw[feature].dropna().values
    proc = df_processed[feature].dropna().values

    f_raw, Pxx_raw = welch(raw, fs=fs)
    f_proc, Pxx_proc = welch(proc, fs=fs)

    plt.figure(figsize=(10, 4))
    plt.semilogy(f_raw, Pxx_raw, label='Raw')
    plt.semilogy(f_proc, Pxx_proc, label='Processed')
    plt.title(f"Power Spectral Density: {feature}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("This is a module. Import into a notebook or script to use.")