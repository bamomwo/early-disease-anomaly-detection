# LSTM Autoencoder Evaluation Script

This script evaluates trained LSTM autoencoder models for both personalized and general approaches.

## Features

- **Personalized Models**: Evaluate models trained on individual participants
- **General Models**: Evaluate models trained on multiple participants
- **Flexible Configuration**: Support for custom paths and parameters
- **Comprehensive Evaluation**: Generate ROC curves, PR curves, confusion matrices, and reconstruction visualizations

## Usage

### Basic Usage

#### Personalized Model Evaluation
```bash
python scripts/evaluation/evaluate_lstm_ae.py \
    --model-type personalized \
    --participant 001
```

#### General Model Evaluation
```bash
python scripts/evaluation/evaluate_lstm_ae.py \
    --model-type general \
    --participants 001 002 003
```

### Advanced Usage

#### Custom Paths
```bash
python scripts/evaluation/evaluate_lstm_ae.py \
    --model-type general \
    --participants 001 002 003 004 \
    --model-path custom/path/to/model.pth \
    --figs-dir custom/path/to/figures \
    --input-size 45
```

#### Personalized with Custom Paths
```bash
python scripts/evaluation/evaluate_lstm_ae.py \
    --model-type personalized \
    --participant 005 \
    --model-path results/custom_model.pth \
    --figs-dir results/custom_figs
```

## Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model-type` | str | Yes | `"personalized"` or `"general"` |
| `--participant` | str | For personalized | Single participant ID |
| `--participants` | list | For general | List of participant IDs |
| `--model-path` | str | No | Path to model checkpoint (auto-detected if not provided) |
| `--model-dir` | str | No | Directory containing model (for general models) |
| `--data-path` | str | No | Path to normalized data (default: "data/normalized") |
| `--figs-dir` | str | No | Directory to save figures (auto-detected if not provided) |
| `--input-size` | int | No | Model input size (auto-detected if not provided) |

## Default Paths

### Personalized Models
- **Model**: `results/lstm_ae/pure/{participant}/final_model_{participant}.pth`
- **Config**: `results/lstm_ae/pure/{participant}/config.json`
- **Figures**: `results/lstm_ae/pure/{participant}/figs/`

### General Models
- **Model**: `results/lstm_ae/general/final_model_general.pth`
- **Config**: `results/lstm_ae/general/config.json`
- **Figures**: `results/lstm_ae/general/figs/`

## Generated Outputs

The script generates the following evaluation figures:

1. **Reconstruction Error Distribution** (`recon_error_dist.png/pdf`)
   - Histograms of reconstruction errors for normal vs. stressed sequences
   - 95th percentile threshold for anomaly detection

2. **ROC Curve** (`roc_curve.png/pdf`)
   - Receiver Operating Characteristic curve with AUC score

3. **Precision-Recall Curve** (`pr_curve.png/pdf`)
   - Precision-Recall curve with AUC score

4. **Confusion Matrix** (`confusion_matrix.png/pdf`)
   - Confusion matrix for binary classification

5. **Time-Series Reconstructions** (`time_series_reconstructions.png/pdf`)
   - Example reconstructions of input sequences

## Configuration

The script automatically loads model configuration from the JSON file:
- `config/lstm_config.json`

If this file doesn't exist, it falls back to:
- `config/best_config.json`

## Error Handling

- Validates required arguments based on model type
- Checks for file existence before loading
- Provides clear error messages for missing files or invalid configurations
- Handles missing participants gracefully

## Examples

See `example_usage.py` for complete examples of how to use the script.

## Backward Compatibility

The script maintains full backward compatibility with existing personalized model evaluations. The original usage patterns still work without modification. 