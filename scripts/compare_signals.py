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
    """Robust version that handles any index issues."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create clean copies with sequential indices
    raw_clean = df_raw.copy().reset_index(drop=True)
    proc_clean = df_processed.copy().reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Line plot
    axes[0].plot(raw_clean['timestamp'], raw_clean[feature], label='Raw', alpha=0.6)
    axes[0].plot(proc_clean['timestamp'], proc_clean[feature], label='Processed', alpha=0.8)
    axes[0].set_title(f"{title_prefix} Time Series Comparison: {feature}")
    axes[0].set_ylabel(feature)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Create boxplot data manually to avoid any concatenation issues
    raw_data = pd.DataFrame({
        feature: raw_clean[feature].values,
        'Source': ['Raw'] * len(raw_clean)
    })
    
    proc_data = pd.DataFrame({
        feature: proc_clean[feature].values,
        'Source': ['Processed'] * len(proc_clean)
    })
    
    # Combine with fresh indices
    melted = pd.concat([raw_data, proc_data], ignore_index=True)
    
    sns.boxplot(x='Source', y=feature, data=melted, ax=axes[1])
    axes[1].set_title(f"{title_prefix} Boxplot: {feature}")

    plt.tight_layout()
    plt.show()


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

if __name__ == "__main__":
    print("This is a module. Import into a notebook or script to use.")
