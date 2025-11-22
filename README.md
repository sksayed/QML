# Quantum Machine Learning: Transformer + AutoML with 5G-NIDD Dataset

This project implements a Quantum Machine Learning (QML) model using:
- **Quantum Transformer Architecture** (using PennyLane)
- **AutoML** (using Optuna for hyperparameter optimization)
- **5G-NIDD Dataset** (Network Intrusion Detection)

## Project Structure

```
QML/
├── data/
│   └── data.csv              # 5G-NIDD dataset
├── lesson/                   # Previous lesson files
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── data_preprocessor.py  # Data preprocessing
│   ├── quantum_attention.py  # Quantum attention mechanism
│   ├── quantum_transformer.py # Quantum transformer model
│   ├── automl_optimizer.py   # Optuna-based AutoML
│   └── trainer.py            # Training pipeline
├── models/                   # Saved models (created automatically)
├── results/                  # Results and plots (created automatically)
├── main.py                   # Main execution script
└── requirements.txt          # Python dependencies
```

## Installation

1. Ensure you have Python 3.8+ installed
2. Activate your virtual environment (if using venv)
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### Basic Execution

Run the main script:

```powershell
python main.py
```

### Configuration

You can modify the following parameters in `main.py`:

- **n_samples**: Number of samples to load from dataset (default: 100000)
- **n_trials**: Number of Optuna optimization trials (default: 20)
- **n_features**: Number of features to use (default: 64)

### Outputs

The script will generate:

1. **models/best_model.pth**: Best trained model weights
2. **results/training_history.png**: Training/validation curves
3. **results/results_summary.json**: Summary of best hyperparameters and test metrics

## Features

### Quantum Transformer
- Quantum self-attention mechanism using PennyLane
- Hybrid quantum-classical architecture
- Configurable number of qubits, layers, and heads

### AutoML Integration
- Automated hyperparameter optimization using Optuna
- Optimizes: embedding dimension, number of heads/layers, qubits, dropout, learning rate, batch size, epochs
- Uses TPE (Tree-structured Parzen Estimator) sampler
- Includes pruning for early stopping

### Data Processing
- Automatic feature selection
- Standardization and normalization
- Label encoding
- Train/validation/test splitting

## Notes

- The dataset is loaded in chunks for large files
- Quantum circuits use angle encoding for input features
- The model falls back to classical attention if quantum execution fails
- For Windows PowerShell, all paths use forward slashes or raw strings

## Troubleshooting

1. **Memory Issues**: Reduce `n_samples` in `main.py`
2. **Slow Training**: Reduce `n_trials` or use GPU if available
3. **Quantum Device Errors**: The code includes fallback to classical operations

## Next Steps

- Experiment with different quantum circuit architectures
- Increase number of Optuna trials for better hyperparameters
- Try different quantum devices (e.g., `lightning.qubit` for faster execution)
- Add more evaluation metrics and visualizations

