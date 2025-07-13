import pandas as pd
import os
from datetime import datetime, time
import numpy as np

def create_survey_datetime(date_str, time_str):
    """Combine survey date and time into a single datetime object"""
    try:
        # Parse date
        if '/' in str(date_str):
            date_part = datetime.strptime(str(date_str), '%Y/%m/%d').date()
        else:
            date_part = datetime.strptime(str(date_str), '%Y-%m-%d').date()
        
        # Handle time - could be string or already a time object
        time_str = str(time_str).strip()
        if ':' in time_str:
            time_parts = time_str.split(':')
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
                time_part = time(hour, minute, second)
            else:
                return None
        else:
            return None
        
        # Combine date and time
        return datetime.combine(date_part, time_part)
    except Exception as e:
        print(f"    Error creating datetime from '{date_str}' and '{time_str}': {e}")
        return None

def extract_participant_id(filename):
    """Extract participant ID from filename in format '6B.csv'"""
    base_name = os.path.splitext(filename)[0]  # Remove .csv extension
    return base_name  # Returns '6B' from '6B.csv'

def populate_stress_levels(participant_folder, survey_file_path, output_folder=None):
    """
    Populate stress levels for all participant files based on survey data
    
    Args:
        participant_folder: Path to folder containing participant CSV files
        survey_file_path: Path to the survey CSV file
        output_folder: Path to save updated files (if None, overwrites original files)
    """
    
    # Load survey data
    try:
        survey_data = pd.read_csv(survey_file_path)
        survey_data = survey_data.dropna()
        print(f"Loaded survey data with {len(survey_data)} records")
    except Exception as e:
        print(f"Error loading survey file: {e}")
        return
    
    # Clean column names
    survey_data.columns = survey_data.columns.str.strip()
    
    # Get list of participant files
    participant_files = [f for f in os.listdir(participant_folder) if f.endswith('.csv')]
    print(f"Found {len(participant_files)} participant files")
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in participant_files:
        try:
            print(f"\nProcessing {file}...")
            
            # Extract participant ID from filename
            participant_id = extract_participant_id(file)
            
            # Load participant timeseries data
            participant_path = os.path.join(participant_folder, file)
            participant_data = pd.read_csv(participant_path)
            
            # Clean column names
            participant_data.columns = participant_data.columns.str.strip()
            
            # Initialize stress level column
            participant_data['stress_level'] = np.nan
            
            # Find survey records for this participant
            # You may need to adjust the column name used for matching participant ID
            participant_surveys = survey_data[
                survey_data['participant_id'] == participant_id  # Adjust column name as needed
            ].copy() if 'participant_id' in survey_data.columns else survey_data.copy()
            
            if len(participant_surveys) == 0:
                print(f"  No survey data found for participant {participant_id}")
                continue
            
            print(f"  Found {len(participant_surveys)} survey sessions for participant {participant_id}")
            
            # Process each timestamp in participant data
            for idx, row in participant_data.iterrows():
                timestamp_str = str(row['timestamp'])  # Adjust column name as needed
                try:
                    timestamp_dt = pd.to_datetime(timestamp_str)
                except:
                    continue
                
                # Check against each survey session
                for _, survey_row in participant_surveys.iterrows():
                    try:
                        # Create survey start and end datetime objects
                        survey_start_dt = create_survey_datetime(
                            survey_row['date'], 
                            survey_row['Start time']  # Adjust column name
                        )
                        survey_end_dt = create_survey_datetime(
                            survey_row['date'], 
                            survey_row['End time']    # Adjust column name
                        )
                        
                        if survey_start_dt is None or survey_end_dt is None:
                            continue
                        
                        # Simple datetime comparison
                        if survey_start_dt <= timestamp_dt <= survey_end_dt:
                            stress_level = survey_row['Stress level']  # Adjust column name
                            participant_data.at[idx, 'stress_level'] = stress_level
                            break  # Found matching session, move to next timestamp
                            
                    except Exception as e:
                        print(f"    Error processing survey row: {e}")
                        continue
            
            # Save updated participant data
            if output_folder:
                output_path = os.path.join(output_folder, file)
            else:
                output_path = participant_path  # Overwrite original
            
            participant_data.to_csv(output_path, index=False)
            
            # Print summary
            stress_assigned = participant_data['stress_level'].notna().sum()
            total_rows = len(participant_data)
            print(f"  Assigned stress levels to {stress_assigned}/{total_rows} timestamps")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    print("\nProcessing completed!")

# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    participant_folder = "../data/processed"
    survey_file = "../data/raw/surveyresult.csv"
    output_folder = "../data/labeled"  # Optional, set to None to overwrite original files
    
    populate_stress_levels(participant_folder, survey_file, output_folder)
    
    # If you want to process just one file for testing:
    # populate_stress_levels("participant_files", "survey_data.csv", "output_files")