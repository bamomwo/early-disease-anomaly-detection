# Early Disease Anomaly Detection

This project implements an anomaly detection model for early disease detection using machine learning techniques.

## Project Structure

```
early-disease-anomaly-detection/
├── README.md
├── requirements.txt                 # Python dependencies
├── config/                         # Configuration files
├── data/                           # Data directory
│   ├── raw/                        # Original dataset
│   ├── processed/                  # Processed/cleaned data
│   └── interim/                    # Intermediate processing steps
├── src/                           # Source code
├── scripts/                       # Executable scripts
├── notebooks/                     # Jupyter notebooks
├── models/                        # Saved model artifacts
├── results/                       # Model outputs, metrics, plots
└── tests/                         # Unit tests
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python scripts/preprocess_data.py
```

2. Model Training:
```bash
python scripts/train_model.py
```

3. Model Evaluation:
```bash
python scripts/evaluate_model.py
```

## Development

- Add new features in the `src` directory
- Write tests in the `tests` directory
- Use notebooks in the `notebooks` directory for exploration
