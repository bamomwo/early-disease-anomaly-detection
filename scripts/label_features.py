import pandas as pd
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_survey_data(survey_file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the survey data, focusing on stress level."""
    logging.info(f"Loading survey data from: {survey_file_path}")
    survey_df = pd.read_excel(survey_file_path)

    # Combine date and time columns to create full datetime objects
    survey_df['start_datetime'] = pd.to_datetime(survey_df['date'].dt.strftime('%Y-%m-%d') + ' ' + survey_df['Start time'].astype(str))
    survey_df['end_datetime'] = pd.to_datetime(survey_df['date'].dt.strftime('%Y-%m-%d') + ' ' + survey_df['End time'].astype(str))

    # Rename ID column for consistency
    survey_df.rename(columns={'ID': 'participant_id'}, inplace=True)
    
    # Select only relevant columns for stress labeling
    survey_df = survey_df[['participant_id', 'start_datetime', 'end_datetime', 'Stress level']]
    
    logging.info(f"Loaded {len(survey_df)} survey entries for stress labeling.")
    return survey_df

def label_participant_features(participant_features_path: str, survey_data: pd.DataFrame, output_dir: Path) -> None:
    """Labels a single participant's feature data with stress level information."""
    participant_id = participant_features_path.stem.replace('participant_', '').replace('_features', '')
    logging.info(f"Processing participant: {participant_id}")

    # Load participant features
    features_df = pd.read_csv(participant_features_path)
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    
    # Filter survey data for the current participant
    participant_survey = survey_data[survey_data['participant_id'] == participant_id].copy()

    if participant_survey.empty:
        logging.warning(f"No survey data found for participant {participant_id}. Skipping labeling.")
        # Save original features to labeled directory if no survey data, or handle as per project needs
        # For now, we'll just skip saving if no survey data for labeling
        return

    # Sort both dataframes by timestamp for merge_asof
    features_df.sort_values('timestamp', inplace=True)
    participant_survey.sort_values('start_datetime', inplace=True)

    # Perform merge_asof to assign labels
    # We merge on 'timestamp' from features_df and 'start_datetime' from participant_survey
    # direction='backward' means for each feature timestamp, it looks for the latest start_datetime
    # that is less than or equal to the feature timestamp.
    labeled_features_df = pd.merge_asof(
        features_df,
        participant_survey[['start_datetime', 'end_datetime', 'Stress level']],
        left_on='timestamp',
        right_on='start_datetime',
        direction='backward'
    )

    # Assign stress level only if the feature timestamp falls within the survey event duration
    labeled_features_df['stress_level'] = labeled_features_df.apply(
        lambda row: row['Stress level'] if row['timestamp'] >= row['start_datetime'] and row['timestamp'] <= row['end_datetime'] else None,
        axis=1
    )

    # Drop temporary merge columns
    labeled_features_df.drop(columns=['start_datetime', 'end_datetime', 'Stress level'], inplace=True)

    # Save the labeled data
    output_file_path = output_dir / f"participant_{participant_id}_labeled_features.csv"
    labeled_features_df.to_csv(output_file_path, index=False)
    logging.info(f"Saved labeled features for participant {participant_id} to: {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Label preprocessed biosignal features with survey data.')
    parser.add_argument('--input-dir', type=str, default='../data/processed',
                        help='Path to the directory containing preprocessed feature CSVs (default: ../data/processed)')
    parser.add_argument('--output-dir', type=str, default='../data/labeled',
                        help='Path to the directory to save labeled feature CSVs (default: ../data/labeled)')
    parser.add_argument('--survey-file', type=str, default='../data/raw/surveyresult.xlsx',
                        help='Path to the survey results Excel file (default: ../data/raw/surveyresult.xlsx)')
    
    args = parser.parse_args()

    # Resolve paths relative to the script location
    script_dir = Path(__file__).parent
    input_dir = (script_dir / args.input_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    survey_file = (script_dir / args.survey_file).resolve()

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load survey data once
    survey_data = load_survey_data(survey_file)

    # Process each participant's feature file
    for feature_file in input_dir.glob('participant_*.csv'):
        label_participant_features(feature_file, survey_data, output_dir)

if __name__ == "__main__":
    main()