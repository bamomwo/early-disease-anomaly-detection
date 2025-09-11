import os
import json
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_DIR = "data/labelled"  # Changed from processed to labelled
tRAIN_VAL_RATIO = 0.8  # Portion of data for train+val (A)
TRAIN_RATIO_WITHIN_A = 0.7  # Portion of A for train (rest for val)
OUTPUT_DIR = "data/normalized"
STATS_FILE = "stats/norm_stats.json"

RANDOM_SEED = 42  # only relevant if you decide to shuffle in the future
ID_COLUMN = 'session'      # columns to exclude from normalization
TIME_COLUMN = 'timestamp'  # timestamp column to preserve time order
FEATURE_COLUMNS = None     # will be inferred from the first file
STRESS_COLUMN = 'stress_level'

# ---------- FUNCTIONS ----------
def create_mask(df, feature_cols):
    return (~df[feature_cols].isna()).astype(int)

def normalize_participant(file_path, participant_id):
    print(f"Normalizing {participant_id}...")
    df = pd.read_csv(file_path)

    # Load selected features from config
    try:
        with open("config/selected_features.json", 'r') as f:
            selected_features = json.load(f)['features']
    except FileNotFoundError:
        print("Warning: 'config/selected_features.json' not found. Falling back to all numeric features.")
        selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = [col for col in selected_features if col not in [ID_COLUMN, TIME_COLUMN, STRESS_COLUMN]]

    # Immediately subset the DataFrame to only the selected features + essential columns
    essential_cols = [col for col in [ID_COLUMN, TIME_COLUMN, STRESS_COLUMN] if col in df.columns]
    df = df[essential_cols + selected_features]

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = selected_features

    n = len(df)
    a_end = int(n * tRAIN_VAL_RATIO)
    df_A = df.iloc[:a_end].copy()
    df_B = df.iloc[a_end:].copy()  # test set (all stress levels)

    # The data is no longer filtered for stress levels here.
    # That logic is moved to the PhysiologicalDataLoader.
    n_A = len(df_A)
    train_end = int(n_A * TRAIN_RATIO_WITHIN_A)
    train_df = df_A.iloc[:train_end].reset_index(drop=True)
    val_df = df_A.iloc[train_end:].reset_index(drop=True)
    test_df = df_B.reset_index(drop=True)  # test set: all samples in B

    # Compute mean and std from train only
    train_values = train_df[FEATURE_COLUMNS]
    means = train_values.mean(skipna=True)
    stds = train_values.std(skipna=True)

    # Z-score normalization using train stats
    def z_score(df_chunk):
        return (df_chunk[FEATURE_COLUMNS] - means) / stds

    def process_split(df_chunk):
        norm = df_chunk.copy()
        norm[FEATURE_COLUMNS] = z_score(df_chunk)
        filled = norm.copy()
        filled[FEATURE_COLUMNS] = filled[FEATURE_COLUMNS].fillna(0)
        mask = create_mask(norm, FEATURE_COLUMNS)
        return norm, filled, mask

    train_norm, train_filled, train_mask = process_split(train_df)
    val_norm, val_filled, val_mask = process_split(val_df)
    test_norm, test_filled, test_mask = process_split(test_df)

    # Save stats for reproducibility
    stats = {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'features': FEATURE_COLUMNS
    }

    return (train_norm, train_filled, train_mask,
            val_norm, val_filled, val_mask,
            test_norm, test_filled, test_mask,
            stats)

# ---------- MAIN ----------
def main(participant_id=None, output_dir=OUTPUT_DIR, force=False):
    # Output subdirectories
    for split in ["train", "val", "test"]:
        for kind in ["norm", "filled", "mask"]:
            os.makedirs(os.path.join(output_dir, split, kind), exist_ok=True)
    os.makedirs("stats", exist_ok=True)

    # Load existing stats if available
    norm_stats = {}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                norm_stats = json.load(f)
            print(f"Loaded existing statistics for {len(norm_stats)} participants")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load existing stats file: {e}")
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
        # Check if participant has already been processed
        if pid in norm_stats and not force:
            print(f"Participant {pid} already processed. Skipping... (use --force to reprocess)")
            continue
            
        (train_norm, train_filled, train_mask,
         val_norm, val_filled, val_mask,
         test_norm, test_filled, test_mask,
         stats) = normalize_participant(filepath, pid)

        # Save to output directories
        train_norm.to_csv(os.path.join(output_dir, "train", "norm", f"{pid}_train_norm.csv"), index=False)
        train_filled.to_csv(os.path.join(output_dir, "train", "filled", f"{pid}_train_filled.csv"), index=False)
        train_mask.to_csv(os.path.join(output_dir, "train", "mask", f"{pid}_train_mask.csv"), index=False)

        val_norm.to_csv(os.path.join(output_dir, "val", "norm", f"{pid}_val_norm.csv"), index=False)
        val_filled.to_csv(os.path.join(output_dir, "val", "filled", f"{pid}_val_filled.csv"), index=False)
        val_mask.to_csv(os.path.join(output_dir, "val", "mask", f"{pid}_val_mask.csv"), index=False)

        test_norm.to_csv(os.path.join(output_dir, "test", "norm", f"{pid}_test_norm.csv"), index=False)
        test_filled.to_csv(os.path.join(output_dir, "test", "filled", f"{pid}_test_filled.csv"), index=False)
        test_mask.to_csv(os.path.join(output_dir, "test", "mask", f"{pid}_test_mask.csv"), index=False)

        # Save normalization stats
        norm_stats[pid] = stats

    # Write all stats to JSON
    with open(STATS_FILE, "w") as f:
        json.dump(norm_stats, f, indent=4)

    print(f"Normalization complete. Statistics saved for {len(norm_stats)} participants.")

# Run script if executed directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize participant data (v2) with stress-level-aware splitting.")
    parser.add_argument(
        "-p", "--participant", type=str, default=None,
        help="Participant ID to process (without .csv). If not provided, process all participants."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=OUTPUT_DIR,
        help="Parent directory to save normalized data. Default is 'data/normalized'."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force reprocessing even if participant has already been processed."
    )
    args = parser.parse_args()
    main(participant_id=args.participant, output_dir=args.output_dir, force=args.force)
