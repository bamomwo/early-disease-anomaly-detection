import os
import argparse
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
NORMALIZED_DIR = "../data/normalized"
MASKED_INPUT_DIR = "../data/filled"
MASK_DIR = "../data/masks"

# Ensure train/test subfolders exist in output directories
for split in ["train", "test"]:
    os.makedirs(os.path.join(MASKED_INPUT_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(MASK_DIR, split), exist_ok=True)

def create_mask_and_fill(dataframe: pd.DataFrame, feature_cols: list):
    """
    Returns:
      - input_filled: NaNs replaced with 0.0 (z-score neutral)
      - mask: binary mask (1 = real, 0 = NaN)
    """
    mask = ~dataframe[feature_cols].isna()           # DataFrame of True/False
    input_filled = dataframe.copy()
    input_filled[feature_cols] = input_filled[feature_cols].fillna(0.0)
    return input_filled, mask.astype(np.uint8)

def process_participant(participant_id: str):
    print(f"Processing missing values for {participant_id}...")

    for split in ["train", "test"]:
        input_path = os.path.join(NORMALIZED_DIR, split, f"{participant_id}_{split}.csv")
        if not os.path.exists(input_path):
            print(f"  ✖ File not found: {input_path}")
            continue

        df = pd.read_csv(input_path)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Create filled input and binary mask
        input_filled, mask = create_mask_and_fill(df, feature_cols)

        # Save filled data
        filled_output_path = os.path.join(MASKED_INPUT_DIR, split, f"{participant_id}_{split}_filled.csv")
        input_filled.to_csv(filled_output_path, index=False)

        # Save mask as .npy
        mask_output_path = os.path.join(MASK_DIR, split, f"{participant_id}_{split}_mask.npy")
        np.save(mask_output_path, mask.values)

        # print(f"  ✓ Saved: {filled_output_path}")
        print(f"  ✓ Saved mask: {mask_output_path}")

    print(f"✅ Done processing {participant_id}\n")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle missing values by masking NaNs per participant.")
    parser.add_argument("--participant", type=str, required=True, help="Participant ID (e.g., participant_001)")

    args = parser.parse_args()
    process_participant(args.participant)
