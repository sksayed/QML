import os
import sys
import numpy as np
import torch
from src.data_loader import NIDDDataLoader
from src.data_preprocessor import DataPreprocessor
from src.quantum_transformer import QuantumTransformer
from src.automl_optimizer import AutoMLOptimizer
from src.trainer import Trainer
from src.log_capture import LogCapture

def main():
    # Initialize log capture - all print output will be saved to results/training_log_*.txt
    with LogCapture(results_dir='results') as logger:
        log_path = logger.get_log_path()
        print(f"📝 All output will be logged to: {log_path}")
        print(f"{'='*70}\n")
        
        _run_training()
        
        print(f"\n✅ Log file saved to: {log_path}")

def _run_training():
    print("="*70)
    print("Quantum Machine Learning: Transformer + AutoML")
    print("Dataset: 5G-NIDD")
    print("="*70)
    
    # ========================================================================
    # SECTION 0: GPU CHECK (Always run this first)
    # ========================================================================
    print("\n" + "="*70)
    print("GPU Availability Check")
    print("="*70)
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA/GPU is not available!")
        print("   This script requires a GPU to run.")
        print("   Please ensure:")
        print("   1. You have a CUDA-compatible GPU")
        print("   2. CUDA drivers are installed")
        print("   3. PyTorch with CUDA support is installed")
        print("\n   Exiting...")
        sys.exit(1)
    
    # GPU is available - proceed with setup
    device = torch.device('cuda:0')  # Explicitly use first GPU
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Verify GPU is actually accessible
    torch.cuda.empty_cache()
    try:
        test_tensor = torch.randn(10, 10).to(device)
        print(f"  ✓ GPU Test: Successfully created tensor on {test_tensor.device}")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ GPU Test Failed: {e}")
        print("   Exiting...")
        sys.exit(1)
    
    print(f"\n🚀 GPU Ready: {device}")
    print(f"Note: Classical layers will use GPU, quantum circuits run on CPU")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ========================================================================
    # SECTION 1: TEST DATA LOADER
    # ========================================================================
    print("\n" + "="*70)
    print("SECTION 1: Testing Data Loader")
    print("="*70)
    
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    loader = NIDDDataLoader(data_path)
    # Test with 50,000 samples to check for any problems
    test_sample_size = 50000
    print(f"📊 Using TEST dataset ({test_sample_size:,} samples) for quick validation")
    df = loader.load_data(n_samples=test_sample_size)  # Load 50k samples for testing
    print(f"✓ Data loaded: {df.shape}")
    
    X, y = loader.get_features_labels()
    print(f"✓ Features extracted: X={X.shape}, y={y.shape}")
  
    
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    print(f"✓ Data split complete")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ========================================================================
    # SECTION 2: TEST DATA PREPROCESSOR
    # ========================================================================
    print("\n" + "="*70)
    print("SECTION 2: Testing Data Preprocessor")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    
    # Determine number of features (must match n_qubits=6)
    n_features = min(6, X_train.shape[1])  # Use 6 features to match n_qubits=6
    print(f"Reducing to {n_features} features (to match n_qubits=6)...")
    
    X_train, X_val, X_test = preprocessor.preprocess_features(
        X_train, X_val, X_test, n_features=n_features, y_train=y_train
    )
    print(f"✓ Features preprocessed: {X_train.shape}")
    
    y_train, y_val, y_test = preprocessor.encode_labels(y_train, y_val, y_test)
    print(f"✓ Labels encoded")
    
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # ========================================================================
    # SECTION 3: TEST QUANTUM ATTENTION (Standalone) - COMMENTED OUT
    # ========================================================================
    # print("\n" + "="*70)
    # print("SECTION 3: Testing Quantum Attention (Standalone)")
    # print("="*70)
    # 
    # from src.quantum_attention import BatchedQuantumAttention
    # 
    # # Create a small test
    # n_qubits = 2
    # batch_size = 4
    # seq_len = 1
    # embed_dim = 8
    # 
    # print(f"Creating quantum attention with n_qubits={n_qubits}...")
    # qattn = BatchedQuantumAttention(n_qubits=n_qubits, n_quantum_layers=2)
    # print("✓ Quantum attention created")
    # 
    # # Create dummy inputs
    # query = torch.randn(batch_size, seq_len, n_qubits)
    # key = torch.randn(batch_size, seq_len, n_qubits)
    # value = torch.randn(batch_size, seq_len, embed_dim)
    # 
    # print(f"Testing forward pass with shapes: q={query.shape}, k={key.shape}, v={value.shape}")
    # try:
    #     context, attn_weights = qattn(query, key, value)
    #     print(f"✓ Forward pass successful!")
    #     print(f"  Context shape: {context.shape}")
    #     print(f"  Attention weights shape: {attn_weights.shape}")
    # except Exception as e:
    #     print(f"❌ Forward pass failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return
    
    # ========================================================================
    # SECTION 4: TEST QUANTUM TRANSFORMER (Standalone) - COMMENTED OUT
    # ========================================================================
    # print("\n" + "="*70)
    # print("SECTION 4: Testing Quantum Transformer (Standalone)")
    # print("="*70)
    # 
    # print(f"Creating transformer with input_dim={input_dim}, n_classes={n_classes}...")
    # try:
    #     model = QuantumTransformer(
    #         input_dim=input_dim,
    #         embed_dim=16,
    #         n_transformer_layers=2,
    #         n_quantum_layers=2,
    #         n_qubits=2,
    #         n_classes=n_classes,
    #         dropout=0.1
    #     )
    #     print("✓ Transformer created")
    #     
    #     # Test forward pass
    #     test_input = torch.randn(2, 1, input_dim)  # (batch=2, seq=1, features)
    #     print(f"Testing forward pass with input shape: {test_input.shape}")
    #     
    #     logits = model(test_input)
    #     print(f"✓ Forward pass successful!")
    #     print(f"  Output logits shape: {logits.shape}")
    #     print(f"  Expected: (2, {n_classes})")
    #     
    # except Exception as e:
    #     print(f"❌ Transformer test failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return
    
    # ========================================================================
    # SECTION 5: TEST TRAINER (Minimal Training) - COMMENTED OUT
    # ========================================================================
    # print("\n" + "="*70)
    # print("SECTION 5: Testing Trainer (Minimal Training)")
    # print("="*70)
    # 
    # print("Creating trainer...")
    # trainer = Trainer(model, device=device)
    # print("✓ Trainer created")
    # 
    # print("Starting minimal training (2 epochs, small batch)...")
    # try:
    #     trainer.train(
    #         X_train[:20], y_train[:20],  # Use only 20 samples
    #         X_val[:10], y_val[:10],      # Use only 10 samples
    #         batch_size=4,
    #         n_epochs=2,
    #         learning_rate=0.001
    #     )
    #     print("✓ Training completed successfully!")
    # except Exception as e:
    #     print(f"❌ Training failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return
    
    # ========================================================================
    # SECTION 7: FULL PIPELINE
    # ========================================================================
    print("\n" + "="*70)
    print("SECTION 7: Full Pipeline")
    print("="*70)
    
    # Step 3: AutoML Optimization
    print("\n" + "="*70)
    print("Step 3: AutoML Hyperparameter Optimization")
    print("="*70)
    print(f"Using device: {device}")
    print(f"Training on TEST dataset (50,000 samples) for validation")
    
    n_trials = 30  # higher number of trials for better hyperparameter search
    
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
    
    model_kwargs = {
        'input_dim': input_dim,
        'embed_dim': best_params['embed_dim'],
        'n_qubits': best_params['n_qubits'],
        'n_classes': n_classes,
        'dropout': best_params['dropout']
    }
    
    if 'n_transformer_layers' in best_params:
        model_kwargs['n_transformer_layers'] = best_params['n_transformer_layers']
        model_kwargs['n_quantum_layers'] = best_params['n_quantum_layers']
    else:
        model_kwargs['n_heads'] = best_params['n_heads']
        model_kwargs['n_layers'] = best_params['n_layers']
    
    model = QuantumTransformer(**model_kwargs)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    trainer = Trainer(model, device=device)
    n_epochs = min(best_params['n_epochs'], 50)  # Moderate cap for better training
    print(f"\n📊 Training Configuration:")
    print(f"  Dataset: TEST dataset (50,000 samples)")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Learning rate: {best_params['learning_rate']:.6f}")
    print(f"  Label smoothing: 0.1 (for better generalization)")
    print(f"  LR Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=best_params['batch_size'],
        n_epochs=n_epochs,
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
