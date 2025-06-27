
import pandas as pd

file_path = "/Users/mac/Documents/teak/env/code/early-disease-anomaly-detection/data/raw/surveyresult.xlsx"

try:
    df = pd.read_excel(file_path)
    print(df.head().to_markdown(index=False))
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
