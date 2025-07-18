import os
import json
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/normalized"
STATS_FILE = "stats/norm_stats.json"

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

RANDOM_SEED = 42  # only relevant if you decide to shuffle in the future
ID_COLUMN = 'session'      # columns to exclude from normalization
TIME_COLUMN = 'timestamp'  # timestamp column to preserve time order
FEATURE_COLUMNS = None     # will be inferred from the first file

# List of features to exclude from normalization

# ---------- FUNCTIONS ----------

def normalize_participant(file_path, participant_id):
    print(f"Normalizing {participant_id}...")
    df = pd.read_csv(file_path)

    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is None:
        # Automatically infer feature columns (numeric only, excluding ID/timestamp)
        FEATURE_COLUMNS = df.select_dtypes(include=[np.number]).columns.tolist()
        FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col not in [ID_COLUMN, TIME_COLUMN]]

    # Time-ordered slicing
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # Compute mean and std from train only
    train_values = train_df[FEATURE_COLUMNS]
    means = train_values.mean(skipna=True)
    stds = train_values.std(skipna=True)

    # Z-score normalization using train stats
    def z_score(df_chunk):
        return (df_chunk[FEATURE_COLUMNS] - means) / stds

    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()

    train_norm[FEATURE_COLUMNS] = z_score(train_df)
    val_norm[FEATURE_COLUMNS] = z_score(val_df)
    test_norm[FEATURE_COLUMNS] = z_score(test_df)

    # Save stats for reproducibility
    stats = {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'features': FEATURE_COLUMNS
    }

    return train_norm, val_norm, test_norm, stats


# ---------- MAIN ----------

def main(participant_id=None, output_dir=OUTPUT_DIR):
    # Output subdirectories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs("stats", exist_ok=True)

    norm_stats = {}
    files_to_process = []

    if participant_id:
        filename = f"{participant_id}.csv"
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.exists(filepath):
            files_to_process.append((filename, participant_id, filepath))
        else:
            print(f"Participant file {filename} not found in {INPUT_DIR}.")
            return
    else:
        for filename in os.listdir(INPUT_DIR):
            if filename.endswith(".csv"):
                pid = filename.replace(".csv", "")
                filepath = os.path.join(INPUT_DIR, filename)
                files_to_process.append((filename, pid, filepath))

    for filename, pid, filepath in files_to_process:
        train_norm, val_norm, test_norm, stats = normalize_participant(filepath, pid)

        # Save to output directories
        train_norm.to_csv(os.path.join(train_dir, f"{pid}_train.csv"), index=False)
        val_norm.to_csv(os.path.join(val_dir, f"{pid}_val.csv"), index=False)
        test_norm.to_csv(os.path.join(test_dir, f"{pid}_test.csv"), index=False)

        # Save normalization stats
        norm_stats[pid] = stats

    # Write all stats to JSON
    with open(STATS_FILE, "w") as f:
        json.dump(norm_stats, f, indent=4)

    print("Normalization complete.")

# Run script if executed directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize participant data (v2).")
    parser.add_argument(
        "-p", "--participant", type=str, default=None,
        help="Participant ID to process (without .csv). If not provided, process all participants."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=OUTPUT_DIR,
        help="Parent directory to save normalized data. Default is 'data/normalized'."
    )
    args = parser.parse_args()
    main(participant_id=args.participant, output_dir=args.output_dir)
