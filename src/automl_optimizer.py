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
import os

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
        
        # GPU Compatibility Check: Test if CUDA kernels are available
        is_cuda = self.device.type == 'cuda'
        if is_cuda:
            capability = None
            gpu_name = "Unknown"
            try:
                capability = torch.cuda.get_device_capability(0)
                capability_major = capability[0]
                capability_minor = capability[1]
                gpu_name = torch.cuda.get_device_name(0)
                
                print(f"🎮 AutoML GPU: {gpu_name} (Compute Capability: {capability_major}.{capability_minor})")
                
                # Check for unsupported newer GPUs (sm_100+)
                if capability_major >= 10:
                    print(f"⚠️  WARNING: GPU with compute capability {capability_major}.{capability_minor} detected")
                    print("   This GPU architecture may not be supported by current PyTorch builds.")
                    if capability_major == 12:
                        print("   ⚠️  RTX 5060 or similar Blackwell architecture GPU detected (sm_120)")
                        print("   Testing compatibility...")
            except Exception:
                pass
            
            try:
                # Test basic CUDA operation to verify kernel compatibility
                test_tensor = torch.zeros(1, device=self.device)
                _ = test_tensor + 1  # Simple operation to test kernel availability
                del test_tensor
                torch.cuda.empty_cache()
                print(f"✅ AutoML GPU compatibility check passed")
            except (torch.cuda.CudaError, RuntimeError) as e:
                error_msg = str(e)
                if "no kernel image" in error_msg.lower() or "cudaerror" in error_msg.lower():
                    # Get GPU info for better error message (if not already got)
                    if capability is None:
                        try:
                            capability = torch.cuda.get_device_capability(0)
                            capability_major = capability[0]
                            gpu_name = torch.cuda.get_device_name(0)
                        except:
                            capability_major = None
                            gpu_name = "Unknown"
                    else:
                        capability_major = capability[0]
                    
                    print(f"\n❌ AutoML CUDA Kernel Compatibility Error")
                    print(f"   GPU: {gpu_name}")
                    if capability:
                        print(f"   Compute Capability: {capability[0]}.{capability[1]}")
                    
                    # Provide specific guidance based on GPU
                    if capability_major and capability_major >= 12:
                        print("\n   🔍 DIAGNOSIS: RTX 5060 or newer GPU (sm_120+)")
                        print("   Current PyTorch build doesn't include kernels for this GPU architecture.")
                        print("\n   💡 SOLUTIONS:")
                        print("   1. Try PyTorch nightly build:")
                        print("      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
                        print("   2. Use CPU mode (automatic fallback - slower but works)")
                    elif capability_major and capability_major >= 10:
                        print("\n   🔍 DIAGNOSIS: Newer GPU architecture (sm_100+)")
                        print("   May need newer PyTorch build or nightly version.")
                        print("\n   💡 SOLUTIONS:")
                        print("   1. Try PyTorch nightly build")
                        print("   2. Use CPU mode (automatic fallback)")
                    else:
                        print("\n   🔍 DIAGNOSIS: GPU architecture mismatch")
                        print("   PyTorch was compiled for different GPU architecture.")
                        print("\n   💡 SOLUTIONS:")
                        print("   1. Reinstall PyTorch compatible with your GPU")
                        print("   2. Use CPU mode (automatic fallback)")
                    
                    print("\n   ⚙️  AutoML falling back to CPU mode...")
                    self.device = torch.device('cpu')
                    print("   ✅ AutoML will use CPU (slower but will work)")
                else:
                    raise  # Re-raise if it's a different CUDA error
        
        self.n_trials = n_trials
        self.seed = seed
        self.study = None
        
        # Track best model state for saving
        self.best_model_state_dict = None
        self.best_trial_value = None
        self.best_trial_number = None
        
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
    
    def _save_best_model_callback(self, study, trial):
        """
        Callback function to save the best model's state_dict.
        Called after each trial completes.
        """
        # Check if this trial is the best one so far
        if study.best_trial.number == trial.number:
            # This trial is the new best
            model_state_dict = trial.user_attrs.get('model_state_dict')
            trial_best_val_acc = trial.user_attrs.get('trial_best_val_acc')
            trial_best_val_loss = trial.user_attrs.get('trial_best_val_loss', None)
            
            if model_state_dict is not None:
                # Save the best model state_dict
                self.best_model_state_dict = model_state_dict
                self.best_trial_value = trial_best_val_acc
                self.best_trial_number = trial.number
                if trial_best_val_loss is not None:
                    print(f"[Best Model Saved] Trial {trial.number} - Val Acc: {trial_best_val_acc:.4f}, Val Loss: {trial_best_val_loss:.4f}")
                else:
                    print(f"[Best Model Saved] Trial {trial.number} with validation accuracy: {trial_best_val_acc:.4f}")
    
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
                'embed_dim': trial.suggest_int('embed_dim', 32, 128, step=16),  # Moderate range
                'n_transformer_layers': trial.suggest_int('n_transformer_layers', 2, 6),  # Moderate depth
                'n_quantum_layers': trial.suggest_int('n_quantum_layers', 1, 4),  # Moderate quantum depth
                'n_qubits': trial.suggest_categorical('n_qubits', [5]),  # Fixed to 5 qubits for amplitude encoding (2^5=32 features)
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                # FIX: suggest_loguniform is deprecated, use suggest_float with log=True
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),  # Moderate batch sizes
                'n_epochs': trial.suggest_int('n_epochs', 5, 15),  # Balanced range: works for both small and large datasets
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
                # Use label smoothing for consistency with final training (0.1 is standard)
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                
                best_val_acc = 0.0
                best_val_loss = float('inf')
                best_epoch_model_state = None  # Track model state at best epoch within this trial
                patience = 3  # Early stopping patience
                patience_counter = 0
                
                for epoch in range(params['n_epochs']):
                    # Train
                    model.train()
                    train_loss_sum = 0.0
                    train_count = 0
                    
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
                        
                        # Gradient clipping for training stability (especially important for quantum models)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        train_loss_sum += loss.item() * batch_y.size(0)
                        train_count += batch_y.size(0)
                    
                    train_loss = train_loss_sum / train_count if train_count > 0 else 0.0
                    
                    # Validate
                    model.eval()
                    val_preds = []
                    val_targets = []
                    val_loss_sum = 0.0
                    val_count = 0
                    
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            
                            if batch_x.dim() == 2:
                                batch_x = batch_x.unsqueeze(1)
                            
                            logits = model(batch_x)
                            loss = criterion(logits, batch_y)
                            
                            preds = torch.argmax(logits, dim=1)
                            val_preds.extend(preds.cpu().numpy())
                            val_targets.extend(batch_y.cpu().numpy())
                            
                            val_loss_sum += loss.item() * batch_y.size(0)
                            val_count += batch_y.size(0)
                    
                    val_acc = accuracy_score(val_targets, val_preds)
                    val_loss = val_loss_sum / val_count if val_count > 0 else float('inf')
                    
                    # Early stopping: Stop if validation loss doesn't improve
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        patience_counter = 0
                        # Save model state if this is the best epoch in this trial
                        best_epoch_model_state = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Early stopping triggered
                            print(f"[Early Stop] Trial {trial.number} stopped at epoch {epoch+1}/{params['n_epochs']} - "
                                  f"Val loss not improving (best: {best_val_loss:.4f}, current: {val_loss:.4f}, "
                                  f"best acc: {best_val_acc:.4f})")
                            break
                    
                    # Pruning Hook: Report both accuracy and loss to Optuna
                    # Optuna uses accuracy for pruning (since direction='maximize'),
                    # but we also store loss for better tracking
                    trial.report(val_acc, epoch)
                    trial.set_user_attr(f'val_loss_epoch_{epoch}', val_loss)
                    trial.set_user_attr(f'val_acc_epoch_{epoch}', val_acc)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # After trial completes, check if this is the global best
                # We'll use a callback to handle this properly with Optuna's study
                # Store the model state temporarily for the callback
                trial.set_user_attr('model_state_dict', best_epoch_model_state)
                trial.set_user_attr('trial_best_val_acc', best_val_acc)
                trial.set_user_attr('trial_best_val_loss', best_val_loss)
                
                return best_val_acc
                
            except optuna.TrialPruned:
                raise  # Re-raise pruning exception so Optuna handles it correctly
            except (torch.cuda.CudaError, RuntimeError) as e:
                error_msg = str(e)
                if "no kernel image" in error_msg.lower() or "cudaerror" in error_msg.lower():
                    # CUDA kernel error during training - this shouldn't happen if __init__ check worked
                    # But handle it gracefully anyway
                    print(f"[Trial Failed] CUDA kernel error: {e}")
                    print("   This trial will be skipped. Consider using CPU mode for AutoML.")
                    return 0.0
                else:
                    # Different CUDA error - re-raise
                    raise
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
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,      # Start pruning after 3 trials
                n_warmup_steps=2,         # Wait 2 steps before pruning
                interval_steps=1           # Check pruning every step
            ),
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        print(f"Starting Optuna optimization on {self.device} with {self.n_trials} trials...")
        self.study.optimize(
            objective, 
            n_trials=self.n_trials, 
            show_progress_bar=True,
            callbacks=[self._save_best_model_callback]
        )
        
        print(f"\nBest trial value: {self.study.best_value:.4f}")
        print("Best params:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
        
        if self.best_model_state_dict is not None:
            print(f"\nBest model state_dict saved from trial {self.best_trial_number}")
        else:
            print("\nWarning: No best model state_dict was saved.")
        
        return self.study.best_params
    
    def get_best_params(self):
        """Get best hyperparameters"""
        if self.study:
            return self.study.best_params
        return None
    
    def get_best_model_state_dict(self):
        """
        Get the state_dict of the best model from optimization.
        
        Returns:
            dict: Model state_dict if available, None otherwise
        """
        return self.best_model_state_dict
    
    def save_best_model(self, filepath):
        """
        Save the best model's state_dict to disk.
        
        Args:
            filepath: Path to save the model (e.g., 'best_model.pth')
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.best_model_state_dict is None:
            print("Warning: No best model state_dict available to save.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # Save state_dict along with metadata
            save_dict = {
                'state_dict': self.best_model_state_dict,
                'best_params': self.study.best_params if self.study else None,
                'best_value': self.best_trial_value,
                'trial_number': self.best_trial_number
            }
            
            torch.save(save_dict, filepath)
            print(f"Best model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving best model: {e}")
            return False
    
    def load_best_model(self, filepath, model):
        """
        Load the best model's state_dict from disk into a model instance.
        
        Args:
            filepath: Path to the saved model file
            model: Model instance to load the state_dict into
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Best model loaded from {filepath}")
                if 'best_value' in checkpoint:
                    print(f"  Best validation accuracy: {checkpoint['best_value']:.4f}")
                return True
            else:
                # Assume it's a direct state_dict
                model.load_state_dict(checkpoint)
                print(f"Model state_dict loaded from {filepath}")
                return True
        except Exception as e:
            print(f"Error loading best model: {e}")
            return False
