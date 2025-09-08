import os
import json
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_DIR = "data/labelled"
TRAIN_NORMAL_RATIO = 0.8  # Portion of normal sessions for training
VAL_TEST_SPLIT_RATIO = 0.5 # Portion of the remaining data for validation
OUTPUT_DIR = "data/normalized_stratified"
STATS_FILE = "stats/norm_stats_stratified.json"
SESSION_STATUS_FILE = "stats/session_stress_status.json"


RANDOM_SEED = 42
ID_COLUMN = 'session'
TIME_COLUMN = 'timestamp'
FEATURE_COLUMNS = None
STRESS_COLUMN = 'stress_level'

# ---------- FUNCTIONS ----------
def create_mask(df, feature_cols):
    return (~df[feature_cols].isna()).astype(int)

def normalize_participant(file_path, participant_id):
    print(f"Normalizing {participant_id}...")
    df = pd.read_csv(file_path)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = df.select_dtypes(include=[np.number]).columns.tolist()
        FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col not in [ID_COLUMN, TIME_COLUMN, STRESS_COLUMN]]

    # 1. Identify sessions and their stress status, preserving order
    session_groups = df.groupby(ID_COLUMN)
    normal_sessions = []
    stressed_sessions = []
    for session_id, session_df in session_groups:
        if (session_df[STRESS_COLUMN] > 0).any():
            stressed_sessions.append(session_id)
        else:
            normal_sessions.append(session_id)

    session_status_info = {
        'normal_sessions': normal_sessions,
        'stressed_sessions': stressed_sessions
    }
    # Sessions are already sorted by their first appearance because the df is sorted

    # 2. Create training set from earliest normal sessions
    train_sessions_count = int(len(normal_sessions) * TRAIN_NORMAL_RATIO)
    train_session_ids = normal_sessions[:train_sessions_count]
    train_df = df[df[ID_COLUMN].isin(train_session_ids)].reset_index(drop=True)

    # 3. Create validation and test pools from the rest
    remaining_normal_ids = normal_sessions[train_sessions_count:]
    
    # Sort stressed sessions chronologically
    stressed_session_first_timestamp = {sid: df[df[ID_COLUMN] == sid][TIME_COLUMN].min() for sid in stressed_sessions}
    stressed_sessions_sorted = sorted(stressed_sessions, key=lambda sid: stressed_session_first_timestamp[sid])

    # Split remaining normal sessions
    val_normal_count = int(len(remaining_normal_ids) * VAL_TEST_SPLIT_RATIO)
    val_normal_ids = remaining_normal_ids[:val_normal_count]
    test_normal_ids = remaining_normal_ids[val_normal_count:]

    # Split stressed sessions
    val_stressed_count = int(len(stressed_sessions_sorted) * VAL_TEST_SPLIT_RATIO)
    val_stressed_ids = stressed_sessions_sorted[:val_stressed_count]
    test_stressed_ids = stressed_sessions_sorted[val_stressed_count:]

    # Combine to form final val and test sets
    val_session_ids = val_normal_ids + val_stressed_ids
    test_session_ids = test_normal_ids + test_stressed_ids

    val_df = df[df[ID_COLUMN].isin(val_session_ids)].reset_index(drop=True)
    test_df = df[df[ID_COLUMN].isin(test_session_ids)].reset_index(drop=True)


    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Warning: Could not create one or more data splits for participant {participant_id}. Check data distribution. Skipping.")
        return None

    # 4. Compute mean and std from train only
    train_values = train_df[FEATURE_COLUMNS]
    means = train_values.mean(skipna=True)
    stds = train_values.std(skipna=True)

    def z_score(df_chunk):
        return (df_chunk[FEATURE_COLUMNS] - means) / stds

    def process_split(df_chunk):
        norm = df_chunk.copy()
        norm[FEATURE_COLUMNS] = z_score(df_chunk)
        filled = norm.copy()
        filled[FEATURE_COLUMNS] = filled[FEATURE_COLUMNS].fillna(0)
        mask = create_mask(norm, FEATURE_COLUMNS)
        return filled, mask

    train_filled, train_mask = process_split(train_df)
    val_filled, val_mask = process_split(val_df)
    test_filled, test_mask = process_split(test_df)

    stats = {
        'mean': means.tolist(),
        'std': stds.tolist(),
        'features': FEATURE_COLUMNS
    }

    return (train_filled, train_mask,
            val_filled, val_mask,
            test_filled, test_mask,
            stats, session_status_info)

# ---------- MAIN ----------
def main(participant_id=None, output_dir=OUTPUT_DIR, force=False):
    for split in ["train", "val", "test"]:
        for kind in ["filled", "mask"]:
            os.makedirs(os.path.join(output_dir, split, kind), exist_ok=True)
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

    norm_stats = {}
    session_statuses = {}
    if os.path.exists(STATS_FILE) and not force:
        try:
            with open(STATS_FILE, "r") as f:
                norm_stats = json.load(f)
            print(f"Loaded existing statistics for {len(norm_stats)} participants")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load existing stats file: {e}")
            norm_stats = {}
    
    if os.path.exists(SESSION_STATUS_FILE) and not force:
        try:
            with open(SESSION_STATUS_FILE, "r") as f:
                session_statuses = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load existing session status file: {e}")
            session_statuses = {}


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
        if pid in norm_stats and not force:
            print(f"Participant {pid} already processed. Skipping... (use --force to reprocess)")
            continue
            
        result = normalize_participant(filepath, pid)
        if result is None:
            continue

        (train_filled, train_mask,
         val_filled, val_mask,
         test_filled, test_mask,
         stats, session_status_info) = result

        train_filled.to_csv(os.path.join(output_dir, "train", "filled", f"{pid}_train_filled.csv"), index=False)
        train_mask.to_csv(os.path.join(output_dir, "train", "mask", f"{pid}_train_mask.csv"), index=False)

        val_filled.to_csv(os.path.join(output_dir, "val", "filled", f"{pid}_val_filled.csv"), index=False)
        val_mask.to_csv(os.path.join(output_dir, "val", "mask", f"{pid}_val_mask.csv"), index=False)

        test_filled.to_csv(os.path.join(output_dir, "test", "filled", f"{pid}_test_filled.csv"), index=False)
        test_mask.to_csv(os.path.join(output_dir, "test", "mask", f"{pid}_test_mask.csv"), index=False)

        norm_stats[pid] = stats
        session_statuses[pid] = session_status_info


    with open(STATS_FILE, "w") as f:
        json.dump(norm_stats, f, indent=4)
    
    with open(SESSION_STATUS_FILE, "w") as f:
        json.dump(session_statuses, f, indent=4)


    print(f"Normalization complete. Statistics saved for {len(norm_stats)} participants.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize participant data with session-based chronological and stratified splitting.")
    parser.add_argument(
        "-p", "--participant", type=str, default=None,
        help="Participant ID to process. If not provided, process all."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=OUTPUT_DIR,
        help=f"Parent directory to save normalized data. Default is '{OUTPUT_DIR}'."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force reprocessing even if participant has already been processed."
    )
    args = parser.parse_args()
    main(participant_id=args.participant, output_dir=args.output_dir, force=args.force)
