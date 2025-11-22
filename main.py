"""
Main script for QML Transformer + AutoML with 5G-NIDD Dataset
"""
import os
import numpy as np
import torch
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.quantum_transformer import QuantumTransformer
from src.automl_optimizer import AutoMLOptimizer
from src.trainer import Trainer

def main():
    print("="*70)
    print("Quantum Machine Learning: Transformer + AutoML")
    print("Dataset: 5G-NIDD")
    print("="*70)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("Step 1: Loading 5G-NIDD Dataset")
    print("="*70)
    data_path = 'data/data.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    loader = DataLoader(data_path)
    # Load samples for training (adjust as needed)
    df = loader.load_data(n_samples=1000)
    X, y = loader.get_features_labels()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Step 2: Preprocess data
    print("\n" + "="*70)
    print("Step 2: Preprocessing Data")
    print("="*70)
    preprocessor = DataPreprocessor()
    
    # Determine number of features (limit to reasonable number for quantum circuits)
    n_features = min(16, X_train.shape[1])  # Use up to 16 features
    X_train, X_val, X_test = preprocessor.preprocess_features(
        X_train, X_val, X_test, n_features=n_features, y_train=y_train
    )
    y_train, y_val, y_test = preprocessor.encode_labels(y_train, y_val, y_test)
    
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 3: AutoML Optimization
    print("\n" + "="*70)
    print("Step 3: AutoML Hyperparameter Optimization")
    print("="*70)
    
    # Enhanced GPU detection and setup
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        # Set memory fraction to avoid OOM
        torch.cuda.empty_cache()
        print(f"\nNote: Classical layers will use GPU, quantum circuits run on CPU")
    else:
        device = 'cpu'
        print("⚠ CUDA not available, using CPU for all operations")
    
    print(f"\nUsing device: {device}")
    
    # Number of Optuna trials for hyperparameter optimization
    n_trials = 5  # Moderate number of trials
    
    automl = AutoMLOptimizer(
        QuantumTransformer,
        X_train, y_train,
        X_val, y_val,
        device=device,
        n_trials=n_trials
    )
    
    best_params = automl.optimize(input_dim, n_classes)
    print(f"\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Step 4: Train with best hyperparameters
    print("\n" + "="*70)
    print("Step 4: Training Quantum Transformer with Best Hyperparameters")
    print("="*70)
    
    model = QuantumTransformer(
        input_dim=input_dim,
        embed_dim=best_params['embed_dim'],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        n_qubits=best_params['n_qubits'],
        n_classes=n_classes,
        dropout=best_params['dropout']
    )
    
    trainer = Trainer(model, device=device)
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=best_params['batch_size'],
        n_epochs=best_params['n_epochs'],
        learning_rate=best_params['learning_rate']
    )
    
    # Step 5: Evaluate
    print("\n" + "="*70)
    print("Step 5: Evaluating on Test Set")
    print("="*70)
    results = trainer.evaluate(X_test, y_test)
    
    # Step 6: Plot results
    trainer.plot_history()
    
    # Save results
    import json
    results_summary = {
        'best_hyperparameters': best_params,
        'test_accuracy': float(results['accuracy']),
        'test_f1': float(results['f1']),
        'test_precision': float(results['precision']),
        'test_recall': float(results['recall'])
    }
    
    with open('results/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best model saved to: models/best_model.pth")
    print(f"Results saved to: results/")

if __name__ == "__main__":
    main()

