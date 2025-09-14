# Anomaly Detection in Physiological Time-Series Data

This project provides a flexible framework for detecting anomalies in physiological time-series data, with the goal of identifying early signs of disease or stress. It utilizes a variety of deep learning models, including LSTMs, TCNs, and Transformers, designed to learn and reconstruct normal physiological patterns. Anomalies are detected when the model's reconstruction error for a given data sequence exceeds a learned threshold, indicating a deviation from normal behavior.

The system is designed to be flexible, supporting both generalized models trained on data from multiple subjects and personalized models trained on data from a single individual.

## Key Features

- **Multiple Autoencoder Architectures:** Implements several models to capture temporal dependencies, including:
    - Long Short-Term Memory (LSTM), with a Bidirectional option
    - Temporal Convolutional Network (TCN)
    - Transformer
- **Advanced Feature Engineering:** Extracts a comprehensive set of features from raw biosignals, including:
    - **Time-Domain HRV:** RMSSD, SDNN, pNN50, etc.
    - **Frequency-Domain HRV:** LF, HF, and LF/HF ratio to measure autonomic nervous system balance.
    - **Non-Linear HRV:** Poincaré plot descriptors (SD1, SD2) and Sample Entropy to capture signal complexity.
    - **EDA and BVP analysis:** Features from electrodermal activity and blood volume pulse.
- **Masking for Missing Data:** The models and loss functions are designed to handle missing data points (NaNs), a common issue with sensor data.
- **Modular Data Pipeline:** A structured, single-script data processing pipeline prepares the raw data for training.
- **Comprehensive Training and Evaluation:** Includes scripts for training, evaluation, and hyperparameter management.

## Directory Structure

```
early-disease-anomaly-detection/
├── README.md
├── requirements.txt                 # Python dependencies
├── config/
│   └── *.yaml / *.json              # Configuration files for models
├── data/
│   ├── raw/                         # Raw physiological data
│   ├── processed/                   # Processed data ready for the model
│   └── ...
├── src/
│   ├── models/
│   │   ├── lstm_ae.py               # Model definitions
│   │   ├── tcn_ae.py
│   │   └── transformer_ae.py
│   ├── data/
│   │   └── physiological_loader.py  # Data loader for PyTorch
│   └── utils/
│       └── train_utils.py           # Training and validation helpers
├── scripts/
│   ├── preprocessing/
│   │   └── initial_preprocessing.py # Main script for data prep and feature extraction
│   ├── training/
│   │   └── general/                 # Scripts for training general models
│   └── evaluation/
│       └── evaluate_tcn_ae.py       # Scripts for evaluating trained models
├── models/                          # Saved model checkpoints
└── results/                         # Training results, logs, and plots
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd early-disease-anomaly-detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The data preparation process involves several steps, executed by scripts in the `scripts/preprocessing` directory.

1.  **Initial Preprocessing & Feature Extraction:**
    This step reads the raw data, performs cleaning, segments the data into windows, and extracts a comprehensive feature set.

    ```bash
    python scripts/preprocessing/initial_preprocessing.py --participant [participant_id]
    ```
    *(You can also use `initial_preprocessing_2.py` for the experimental 60s window and advanced HRV features.)*

2.  **Feature Labeling:**
    This step labels the extracted features based on the session type (e.g., stress vs. normal).

    ```bash
    python scripts/preprocessing/label_features.py --participant [participant_id]
    ```
    *(YStress data is used as proxies to determine non-baseline physiological states, consequenctly enabling model evaluation with AUC-PR, and F1-Scores(with precision and recall))*
3.  **Normalization and Data Splitting:**
    This final step normalizes the labeled features and splits the data into training, validation, and test sets.

    ```bash
    python scripts/preprocessing/data_preparation.py --participant [participant_id]
    ```

*Note: You may need to adjust the paths and parameters in the scripts or config files to match your dataset.*

## Training the Models

You can train generalized models on data from multiple participants or personalized models for each individual. Scripts for each model type are located in `scripts/training/`.

### Example: Training a General TCN Model

```bash
python scripts/training/general/general_tcn_ae.py
```

This will train the model using the participants specified in the script and save the checkpoints and results in the `results/tcn_ae/general/` directory. Similar scripts exist for LSTM and Transformer models.

## Evaluation

To evaluate a trained model, use the corresponding evaluation script from `scripts/evaluation/`. You will need to provide the path to the model checkpoint.

```bash
python scripts/evaluation/evaluate_tcn_ae.py --model_path path/to/your/model.pth
```

## Dependencies

The main dependencies for this project are listed in `requirements.txt` and include:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `torch`
-   `pyyaml`
-   `pyhrv`
-   `pytest`
-   `jupyter`
