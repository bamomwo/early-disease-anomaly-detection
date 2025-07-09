# Early Disease Anomaly Detection using LSTM Autoencoder

This project provides a framework for detecting anomalies in physiological time-series data, with the goal of identifying early signs of disease. It utilizes a `MaskedLSTMAutoencoder`, a deep learning model built with PyTorch, designed to learn and reconstruct normal physiological patterns. Anomalies are detected when the model's reconstruction error for a given data sequence exceeds a learned threshold, indicating a deviation from normal behavior.

The system is designed to be flexible, supporting both generalized models trained on data from multiple subjects and personalized models trained on data from a single individual.

## Key Features

- **LSTM Autoencoder:** Employs a Long Short-Term Memory (LSTM) autoencoder to effectively model temporal dependencies in physiological data.
- **Masking for Missing Data:** The model and loss function are designed to handle missing data points (NaNs), a common issue with sensor data, ensuring robust performance.
- **Generalized and Personalized Models:** Supports both a general model trained on a diverse dataset and personalized models tailored to individual-specific patterns.
- **Modular Data Pipeline:** A structured, multi-step data processing pipeline prepares the raw data for training.
- **Comprehensive Training and Evaluation:** Includes scripts for training, evaluation, and hyperparameter management, with features like early stopping and model checkpointing.

## System Architecture

The project follows a standard machine learning project structure:

1.  **Data Processing:** Raw data is processed through a series of scripts that handle cleaning, normalization, feature engineering, and resampling.
2.  **Model Training:** The `MaskedLSTMAutoencoder` is trained on the processed data. The training process aims to minimize the reconstruction error on normal data.
3.  **Anomaly Detection:** Once trained, the model is used to reconstruct new data. A high reconstruction error indicates a potential anomaly.

## Directory Structure

```
early-disease-anomaly-detection/
├── README.md
├── requirements.txt                 # Python dependencies
├── config/
│   └── config.yaml                  # Configuration file for hyperparameters
├── data/
│   ├── raw/                         # Raw physiological data
│   ├── processed/                   # Processed data ready for the model
│   └── ...                          # Other intermediate data directories
├── src/
│   ├── models/
│   │   └── lstm_ae.py               # LSTM Autoencoder model definition
│   ├── data/
│   │   └── physiological_loader.py  # Data loader for PyTorch
│   ├── utils/
│   │   ├── train_utils.py           # Training and validation helper functions
│   │   └── ...
│   └── ...
├── scripts/
│   ├── preprocess_data_raw.py       # Script for initial data preprocessing
│   ├── general_lstm_ae.py           # Script to train the general model
│   ├── pure_lstm_ae.py              # Script to train personalized models
│   └── ...
├── models/                          # Saved model checkpoints
├── results/                         # Training results, logs, and plots
└── tests/                           # Unit and integration tests
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

The data preparation process involves several steps, executed by scripts in the `scripts/` directory. The exact sequence and configuration will depend on the specifics of your raw data.

1.  **Initial Preprocessing:**
    ```bash
    python scripts/preprocess_data_raw.py
    ```
2.  **Feature Engineering :**
    ```bash
    python scripts/preprocess_feature_engineering.py
    ```
3.  **Normalization:**
    ```bash
    python scripts/normalize.py
    ```

*Note: You may need to adjust the paths and parameters in the scripts or the `config/config.yaml` file to match your dataset.*

## Training the Model

You can train either a general model or personalized models.

### General Model

To train a model on data from multiple participants, run the `general_lstm_ae.py` script:

```bash
python scripts/general_lstm_ae.py
```

This will train the model using the participants specified in the script and save the checkpoints, results, and logs in the `results/lstm_ae/general/` directory.

### Personalized Models

To train a separate model for each participant, use the `pure_lstm_ae.py` script. This script will iterate through the specified participants and train a dedicated model for each one.

```bash
python scripts/pure_lstm_ae.py
```

Checkpoints and results for personalized models will be saved in subdirectories within `results/lstm_ae/pure/`.

## Evaluation

To evaluate the performance of a trained model, use the `evaluate_lstm_ae.py` script. You will need to provide the path to the model checkpoint and the data you want to evaluate.

```bash
python scripts/evaluate_lstm_ae.py --model_path path/to/your/model.pth --data_path path/to/your/data
```

## Configuration

The `config/config.yaml` file is used to manage hyperparameters and settings for the training and data processing scripts. This allows for easy experimentation and consistent configuration across the project.

## Dependencies

The main dependencies for this project are listed in `requirements.txt` and include:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `torch`
-   `pyyaml`