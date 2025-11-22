"""
AutoML optimizer using Optuna for Quantum/Classical Hybrid Models.
Optimized for performance and reproducibility.
"""
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np

class AutoMLOptimizer:
    """
    AutoML optimizer using Optuna for Quantum/Classical Hybrid Models.
    Optimized for performance and reproducibility.
    """
    
    def __init__(self, model_class, X_train, y_train, X_val, y_val, 
                 device='cpu', n_trials=100, seed=42):
        self.model_class = model_class
        # Normalize device to torch.device object for consistent handling
        # This handles both string ('cuda', 'cpu') and torch.device objects
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.n_trials = n_trials
        self.seed = seed
        self.study = None
        
        # --- OPTIMIZATION 1: Pre-process Data Once ---
        # Move data to tensors immediately to avoid overhead during trials
        # Note: We keep tensors on CPU initially to avoid VRAM clutter, 
        # moving to GPU only during the training batch.
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
    
    def get_dataloaders(self, batch_size):
        """Efficiently generates dataloaders for a specific batch size"""
        pin_memory = (self.device.type == 'cuda')
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0
        )
        return train_loader, val_loader
    
    def create_objective(self, input_dim, n_classes):
        """Create objective function for Optuna"""
        
        def objective(trial):
            # --- BEST PRACTICE: Seeding ---
            # Ensure two trials with same params yield same result
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            # Suggest hyperparameters
            # Note: Using new parameter names (n_transformer_layers, n_quantum_layers)
            # but also keeping old names in params dict for backward compatibility
            params = {
                'embed_dim': trial.suggest_int('embed_dim', 16, 64, step=16),
                'n_transformer_layers': trial.suggest_int('n_transformer_layers', 2, 4),
                'n_quantum_layers': trial.suggest_int('n_quantum_layers', 1, 3),
                'n_qubits': trial.suggest_categorical('n_qubits', [2, 4, 6]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                # FIX: suggest_loguniform is deprecated, use suggest_float with log=True
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'n_epochs': trial.suggest_int('n_epochs', 5, 15),
                # Backward compatibility: also include old names
                'n_heads': None,  # Will be set from n_transformer_layers
                'n_layers': None  # Will be set from n_quantum_layers
            }
            # Map new names to old names for backward compatibility
            params['n_heads'] = params['n_transformer_layers']
            params['n_layers'] = params['n_quantum_layers']
            
            model = None
            try:
                # Create DataLoaders specific to this trial's batch_size
                train_loader, val_loader = self.get_dataloaders(params['batch_size'])
                
                # Create model using new parameter names
                # The model supports both new and old names via backward compatibility
                model = self.model_class(
                    input_dim=input_dim,
                    embed_dim=params['embed_dim'],
                    n_transformer_layers=params['n_transformer_layers'],
                    n_quantum_layers=params['n_quantum_layers'],
                    n_qubits=params['n_qubits'],
                    n_classes=n_classes,
                    dropout=params['dropout']
                ).to(self.device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                criterion = nn.CrossEntropyLoss()
                
                best_val_acc = 0.0
                
                for epoch in range(params['n_epochs']):
                    # Train
                    model.train()
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        # Robust reshaping: Only unsqueeze if rank is 2 (Batch, Features)
                        # If your model expects (Batch, Seq, Feat), this converts (B, F) -> (B, 1, F)
                        if batch_x.dim() == 2:
                            batch_x = batch_x.unsqueeze(1)
                        
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                    
                    # Validate
                    model.eval()
                    val_preds = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            
                            if batch_x.dim() == 2:
                                batch_x = batch_x.unsqueeze(1)
                            
                            logits = model(batch_x)
                            preds = torch.argmax(logits, dim=1)
                            
                            # Move to cpu numpy efficiently
                            val_preds.extend(preds.cpu().numpy())
                            val_targets.extend(batch_y.cpu().numpy())
                    
                    val_acc = accuracy_score(val_targets, val_preds)
                    best_val_acc = max(best_val_acc, val_acc)
                    
                    # Pruning Hook
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return best_val_acc
                
            except optuna.TrialPruned:
                raise  # Re-raise pruning exception so Optuna handles it correctly
            except Exception as e:
                print(f"[Trial Failed] Error: {e}")
                # Optional: print traceback if needed, but keep log clean
                return 0.0
            finally:
                # CLEANUP: Critical for GPU memory management
                if model is not None:
                    del model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return objective
    
    def optimize(self, input_dim, n_classes, direction='maximize'):
        """Run optimization"""
        objective = self.create_objective(input_dim, n_classes)
        
        self.study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        print(f"Starting Optuna optimization on {self.device} with {self.n_trials} trials...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\nBest trial value: {self.study.best_value:.4f}")
        print("Best params:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
        
        return self.study.best_params
    
    def get_best_params(self):
        """Get best hyperparameters"""
        if self.study:
            return self.study.best_params
        return None
