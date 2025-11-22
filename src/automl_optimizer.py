"""
AutoML optimizer using Optuna for hyperparameter tuning
"""
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np

class AutoMLOptimizer:
    """AutoML optimizer using Optuna"""
    
    def __init__(self, model_class, X_train, y_train, X_val, y_val, 
                 device='cpu', n_trials=100):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device
        self.n_trials = n_trials
        self.study = None
        
    def create_objective(self, input_dim, n_classes):
        """Create objective function for Optuna"""
        def objective(trial):
            # Suggest hyperparameters
            embed_dim = trial.suggest_int('embed_dim', 16, 64, step=16)
            n_heads = trial.suggest_int('n_heads', 2, 4)
            n_layers = trial.suggest_int('n_layers', 1, 3)
            n_qubits = trial.suggest_categorical('n_qubits', [2, 4, 6])
            dropout = trial.suggest_float('dropout', 0.1, 0.4)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            n_epochs = trial.suggest_int('n_epochs', 5, 15)
            
            try:
                # Clear GPU cache before creating new model
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Create model
                model = self.model_class(
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    n_classes=n_classes,
                    dropout=dropout
                ).to(self.device)
                
                # Move model to device explicitly
                if self.device == 'cuda':
                    model = model.cuda()
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(self.X_train),
                    torch.LongTensor(self.y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(self.X_val),
                    torch.LongTensor(self.y_val)
                )
                
                # Use pin_memory for faster GPU transfer if using CUDA
                pin_memory = (self.device == 'cuda')
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    pin_memory=pin_memory,
                    num_workers=0  # Set to 0 for Windows compatibility
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    pin_memory=pin_memory,
                    num_workers=0
                )
                
                # Training
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                best_val_acc = 0
                for epoch in range(n_epochs):
                    # Train
                    model.train()
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        # Reshape for transformer (add sequence dimension)
                        if len(batch_x.shape) == 2:
                            batch_x = batch_x.unsqueeze(1)  # (batch, 1, features)
                        
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
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            
                            if len(batch_x.shape) == 2:
                                batch_x = batch_x.unsqueeze(1)
                            
                            logits = model(batch_x)
                            preds = torch.argmax(logits, dim=1)
                            val_preds.extend(preds.cpu().numpy())
                            val_targets.extend(batch_y.cpu().numpy())
                    
                    val_acc = accuracy_score(val_targets, val_preds)
                    best_val_acc = max(best_val_acc, val_acc)
                    
                    # Pruning
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Clean up GPU memory
                if self.device == 'cuda':
                    del model
                    torch.cuda.empty_cache()
                
                return best_val_acc
                
            except Exception as e:
                import traceback
                print(f"\n{'='*70}")
                print(f"Trial failed with error:")
                print(f"{'='*70}")
                print(f"Error message: {e}")
                print(f"\nFull traceback:")
                traceback.print_exc()
                print(f"{'='*70}\n")
                # Clean up on error
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return 0.0  # Return worst possible score
        
        return objective
    
    def optimize(self, input_dim, n_classes, direction='maximize'):
        """Run optimization"""
        objective = self.create_objective(input_dim, n_classes)
        self.study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print(f"Starting Optuna optimization with {self.n_trials} trials...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\nBest trial:")
        print(f"  Value: {self.study.best_value:.4f}")
        print(f"  Params: {self.study.best_params}")
        
        return self.study.best_params
    
    def get_best_params(self):
        """Get best hyperparameters"""
        if self.study:
            return self.study.best_params
        return None

