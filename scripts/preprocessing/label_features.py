import os
import argparse
import pandas as pd
from datetime import datetime

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Annotate processed data with stress levels from survey.')
    parser.add_argument('--participant_id', type=str, default=None, help='Participant ID to process (e.g., 15 or 6B). If not provided, process all participants.')
    return parser.parse_args()

# --- Main annotation function ---
def annotate_participant(participant_id, survey_path, processed_dir, labelled_dir):
    processed_path = os.path.join(processed_dir, f"{participant_id}.csv")
    if not os.path.exists(processed_path):
        print(f"Processed file not found for participant {participant_id}: {processed_path}")
        return
    df_train = pd.read_csv(processed_path)
    df_survey = pd.read_csv(survey_path)

    # --- 1. Convert 'timestamp' in df_train to datetime ---
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])

    # --- 2. Clean and convert df_survey times ---
    df_survey['Stress level'] = df_survey['Stress level'].replace('na', pd.NA)
    df_survey['Stress level'] = pd.to_numeric(df_survey['Stress level'], errors='coerce')
    df_survey['Stress level'] = df_survey['Stress level'].fillna(0).astype(int)
    df_survey['start_datetime'] = pd.to_datetime(df_survey['date'] + ' ' + df_survey['Start time'])
    df_survey['end_datetime'] = pd.to_datetime(df_survey['date'] + ' ' + df_survey['End time'])

    # --- 3. Add a new column to df_train ---
    df_train['stress_level'] = 0  # default: no stress

    # --- 4. Annotate training data based on intervals ---
    for _, row in df_survey.iterrows():
        start = row['start_datetime']
        end = row['end_datetime']
        level = row['Stress level']
        mask = (df_train['timestamp'] >= start) & (df_train['timestamp'] <= end)
        df_train.loc[mask, 'stress_level'] = level

    # --- 5. Save result ---
    os.makedirs(labelled_dir, exist_ok=True)
    output_path = os.path.join(labelled_dir, f"{participant_id}.csv")
    df_train.to_csv(output_path, index=False)
    print(f"Annotated file saved: {output_path}")


def main():
    args = parse_args()
    processed_dir = os.path.join('data', 'processed')
    labelled_dir = os.path.join('data', 'labelled')
    survey_path = os.path.join('data', 'raw', 'surveyresult.csv')

    if args.participant_id:
        annotate_participant(args.participant_id, survey_path, processed_dir, labelled_dir)
    else:
        # Process all participants
        for fname in os.listdir(processed_dir):
            if fname.endswith('.csv'):
                participant_id = os.path.splitext(fname)[0]
                annotate_participant(participant_id, survey_path, processed_dir, labelled_dir)

if __name__ == "__main__":
    main()
