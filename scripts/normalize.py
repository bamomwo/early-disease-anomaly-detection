import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

# ---------- CONFIG ----------
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/normalized"
STATS_FILE = "stats/norm_stats.json"
FEATURE_COLUMNS = None  # will infer from data
TEST_SIZE = 0.2
RANDOM_SEED = 42
ID_COLUMN = 'session'  # columns to exclude from normalization
TIME_COLUMN = 'timestamp'

# ---------- FUNCTIONS ----------

def normalize_participant(file_path, participant_id):
    print(f"Normalizing {participant_id}...")
    df = pd.read_csv(file_path)

    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = df.select_dtypes(include=[np.number]).columns.tolist()

    # Split into train and test (time-based)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED)

    # Compute mean and std from train data (ignore NaNs)
    train_values = train_df[FEATURE_COLUMNS]
    means = train_values.mean(skipna=True)
    stds = train_values.std(skipna=True)

    # Normalize both train and test using train stats — PRESERVE NaNs!
    def z_score(df_chunk):
        return (df_chunk[FEATURE_COLUMNS] - means) / stds

    train_norm = train_df.copy()
    test_norm = test_df.copy()

    train_norm[FEATURE_COLUMNS] = z_score(train_df)
    test_norm[FEATURE_COLUMNS] = z_score(test_df)

    # Save stats
    stats = {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'features': FEATURE_COLUMNS
    }

    return train_norm, test_norm, stats


# ---------- MAIN ----------

def main(participant_id=None, output_dir=OUTPUT_DIR):
    # Determine train and test output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
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
        train_norm, test_norm, stats = normalize_participant(filepath, pid)

        # Save normalized CSVs in respective subfolders
        train_norm.to_csv(os.path.join(train_dir, f"{pid}_train.csv"), index=False)
        test_norm.to_csv(os.path.join(test_dir, f"{pid}_test.csv"), index=False)

        # Save stats
        norm_stats[pid] = stats

    # Save all stats to JSON
    with open(STATS_FILE, "w") as f:
        json.dump(norm_stats, f, indent=4)

    print(f"Normalization complete. Train files saved to: {train_dir}, Test files saved to: {test_dir}. Stats saved to: {STATS_FILE}")

# ---------- RUN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize participant data.")
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
