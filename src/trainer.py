"""
High-Performance Hybrid Trainer for Quantum-Classical Models
Optimized with quantum-specific features, parallel data loading, and comprehensive evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
from tqdm import tqdm

class Trainer:
    """
    Hybrid Trainer optimized for Quantum-Classical Models.
    
    Quantum-Specific Optimizations:
    1. No DataParallel: Prevents quantum context serialization errors
    2. torch.compile() support: Automatic compilation with proper wrapper handling
    3. set_to_none=True: Faster gradient zeroing
    4. Quantum gradient clipping: Prevents barren plateaus
    
    Comprehensive Features:
    1. History tracking: Full training history for analysis
    2. Detailed evaluation: F1, precision, recall, confusion matrix
    3. Visualization: Training history plots
    4. Optimized data loading: Multi-process, pin_memory, persistent workers
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 use_data_parallel=False, use_compile=True):
        """
        Initialize Trainer.
        
        Args:
            model: The PyTorch/PennyLane model
            device: 'cuda', 'cpu', or torch.device. If None, auto-detects.
            use_data_parallel: If True, use DataParallel for multi-GPU (NOT recommended for quantum models)
            use_compile: If True, attempt torch.compile() for performance (PyTorch 2.0+)
        """
        # Normalize device to torch.device object for consistent handling
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Check if device is CUDA
        is_cuda = self.device.type == 'cuda'
        
        # GPU Compatibility Check: Test if CUDA kernels are available
        if is_cuda:
            # First, check compute capability BEFORE testing operations
            capability = None
            gpu_name = "Unknown"
            try:
                capability = torch.cuda.get_device_capability(0)
                capability_major = capability[0]
                capability_minor = capability[1]
                gpu_name = torch.cuda.get_device_name(0)
                
                print(f"🎮 GPU: {gpu_name} (Compute Capability: {capability_major}.{capability_minor})")
                
                # Check for unsupported newer GPUs (sm_100+)
                if capability_major >= 10:
                    print(f"\n⚠️  WARNING: GPU with compute capability {capability_major}.{capability_minor} detected")
                    print("   This GPU architecture may not be supported by current PyTorch builds.")
                    
                    # Specific detection for RTX 5060 (sm_120)
                    if capability_major == 12:
                        print("   ⚠️  RTX 5060 or similar Blackwell architecture GPU detected (sm_120)")
                        print("   Official stable PyTorch releases don't support this GPU yet.")
                        print("\n   💡 RECOMMENDED SOLUTIONS:")
                        print("   1. Try PyTorch nightly build (may have sm_120 support):")
                        print("      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
                        print("   2. Use CPU mode (automatic fallback - slower but works)")
                        print("   3. Wait for official PyTorch release with sm_120 support")
                        print("\n   Testing compatibility...")
                    else:
                        print(f"   GPU compute capability {capability_major}.{capability_minor} may require newer PyTorch")
                        print("   Testing compatibility...")
            except Exception:
                # If we can't get capability info, continue with test
                pass
            
            try:
                # Test basic CUDA operation to verify kernel compatibility
                test_tensor = torch.zeros(1, device=self.device)
                _ = test_tensor + 1  # Simple operation to test kernel availability
                del test_tensor
                torch.cuda.empty_cache()
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
                    
                    print(f"\n❌ CUDA Kernel Compatibility Error")
                    print(f"   GPU: {gpu_name}")
                    if capability:
                        print(f"   Compute Capability: {capability[0]}.{capability[1]}")
                    
                    # Provide specific guidance based on GPU
                    if capability_major and capability_major >= 12:
                        print("\n   🔍 DIAGNOSIS: RTX 5060 or newer GPU (sm_120+)")
                        print("   Current PyTorch build doesn't include kernels for this GPU architecture.")
                        print("\n   💡 SOLUTIONS (in order of recommendation):")
                        print("   1. Try PyTorch nightly build:")
                        print("      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
                        print("      Then restart your script.")
                        print("   2. Use CPU mode (automatic - slower but works):")
                        print("      The trainer will automatically fall back to CPU.")
                        print("   3. Build PyTorch from source with sm_120 support (advanced)")
                        print("   4. Wait for official PyTorch release")
                    elif capability_major and capability_major >= 10:
                        print("\n   🔍 DIAGNOSIS: Newer GPU architecture (sm_100+)")
                        print("   May need newer PyTorch build or nightly version.")
                        print("\n   💡 SOLUTIONS:")
                        print("   1. Try PyTorch nightly build")
                        print("   2. Reinstall PyTorch from: https://pytorch.org/get-started/locally/")
                        print("   3. Use CPU mode (automatic fallback)")
                    else:
                        print("\n   🔍 DIAGNOSIS: GPU architecture mismatch")
                        print("   PyTorch was compiled for different GPU architecture.")
                        print("\n   💡 SOLUTIONS:")
                        print("   1. Reinstall PyTorch compatible with your GPU:")
                        print("      Visit: https://pytorch.org/get-started/locally/")
                        print("   2. Use CPU mode (automatic fallback)")
                    
                    print("\n   ⚙️  Falling back to CPU mode...")
                    self.device = torch.device('cpu')
                    is_cuda = False
                    # Disable compilation when using CPU (Windows requires C++ compiler)
                    use_compile = False
                    print("   ℹ️  Compilation disabled (CPU mode on Windows requires C++ compiler)")
                else:
                    raise  # Re-raise if it's a different CUDA error
        
        # Print final device info
        if is_cuda:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            gpu_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            print(f"✅ Using GPU: {gpu_name} (Compute: {gpu_capability[0]}.{gpu_capability[1]})")
        else:
            print(f"✅ Using CPU (GPU unavailable or incompatible)")
        
        # CRITICAL: For quantum models, avoid DataParallel to prevent context serialization errors
        # Quantum circuits (PennyLane) often fail when serialized across processes
        if use_data_parallel and is_cuda and torch.cuda.device_count() > 1:
            print(f"⚠️  WARNING: DataParallel enabled. This may cause issues with quantum models!")
            print(f"🚀 Parallelizing model across {torch.cuda.device_count()} GPUs!")
            model = model.to(self.device)
            self.model = nn.DataParallel(model)
        else:
            # Standard single-GPU or CPU setup (recommended for quantum models)
            self.model = model.to(self.device)
            
        # Optional: PyTorch 2.0+ Compilation (Try/Except safely)
        # Note: CPU compilation on Windows requires Visual Studio C++ compiler
        # Only compile on GPU - CPU compilation is disabled to avoid compiler requirements
        self._compiled = False
        if use_compile and is_cuda:  # Only compile on GPU
            # Check for Triton availability on CUDA (required for GPU compilation)
            try:
                import triton
                triton_available = True
            except ImportError:
                triton_available = False
                print("⚠️  Triton not available. GPU compilation requires Triton.")
                print("   Note: Triton is not available on Windows via pip.")
                print("   Proceeding without compilation (model will work normally)...")
                use_compile = False
            
            if use_compile:
                try:
                    # Use 'reduce-overhead' mode which is more compatible
                    # This mode is less aggressive and works better with quantum models
                    compile_mode = 'reduce-overhead'
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self._compiled = True
                    print(f"✅ Model compiled with torch.compile() on {self.device} (mode={compile_mode})")
                except Exception as e:
                    # Compilation failed - disable it and use original model
                    self._compiled = False
                    print(f"⚠️  Compilation failed, using uncompiled model: {e}")
                    if "Triton" in str(e) or "triton" in str(e).lower():
                        print("   Tip: Triton is required for GPU compilation. Install with: pip install triton")
                    elif "Compiler" in str(e) or "cl" in str(e).lower() or "compiler" in str(e).lower():
                        print("   Tip: CPU compilation requires Visual Studio C++ compiler on Windows")
        elif use_compile and not is_cuda:
            # CPU mode - compilation disabled (requires C++ compiler on Windows)
            print(f"✅ Model initialized on {self.device} (Compilation disabled - CPU mode)")
        else:
            print(f"✅ Model initialized on {self.device} (Compilation disabled)")
            
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        # Hardware Detection
        self.cpu_count = multiprocessing.cpu_count()
        print(f"✅ Hardware detected: Device={self.device}, CPUs={self.cpu_count}")
        
    def _get_loader_kwargs(self, batch_size):
        """
        Configures the fastest possible DataLoader settings based on OS and Hardware.
        
        Args:
            batch_size: Batch size for data loading
        
        Returns:
            Dictionary of DataLoader configuration parameters
        """
        # Windows often has issues with multiprocessing spawn, so we default to 0 there
        # On Linux/Mac, we use roughly 50-75% of available cores
        if os.name == 'nt':
            print("⚠️ Windows detected: Forced num_workers=0 (Multiprocessing disabled for stability)")
            num_workers = 0
            persistent = False
        else:
            num_workers = min(self.cpu_count, 8)  # Cap at 8 to prevent overhead
            persistent = True
            
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': (self.device.type == 'cuda'),  # Faster CPU->GPU transfer
            'persistent_workers': persistent,       # Don't kill workers after epoch
            'prefetch_factor': 2 if num_workers > 0 else None,  # Load 2 batches ahead
        }
        return kwargs

    def train(self, X_train=None, y_train=None, X_val=None, y_val=None,
              train_loader=None, val_loader=None,
              batch_size=32, 
              n_epochs=50, 
              learning_rate=0.001,
              patience=5,
              grad_clip=1.0,
              save_path='models/best_model.pth'):
        """
        Train with parallel data loading and advanced features.
        
        Args:
            X_train: Training features (optional if train_loader provided)
            y_train: Training labels (optional if train_loader provided)
            X_val: Validation features (optional if val_loader provided)
            y_val: Validation labels (optional if val_loader provided)
            train_loader: Pre-configured DataLoader for training (optional)
            val_loader: Pre-configured DataLoader for validation (optional)
            batch_size: Batch size for training (used if loaders not provided)
            n_epochs: Number of training epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience (epochs without improvement)
            grad_clip: Gradient clipping threshold (critical for quantum models)
            save_path: Path to save best model
        """
        print("\n--- Starting Quantum-Hybrid Training ---")
        
        # 1. Prepare Data - Support both raw data and pre-configured DataLoaders
        if train_loader is None or val_loader is None:
            if X_train is None or y_train is None or X_val is None or y_val is None:
                raise ValueError("Either provide (X_train, y_train, X_val, y_val) or (train_loader, val_loader)")
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            
            # Get optimized loader settings
            loader_kwargs = self._get_loader_kwargs(batch_size)
            print(f"⚙️ DataLoader Config: workers={loader_kwargs['num_workers']}, "
                  f"pin_memory={loader_kwargs['pin_memory']}")
            
            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        else:
            print("✅ Using provided DataLoaders")
        
        # 2. Optimizer Setup
        # AdamW with weight decay is generally better than Adam
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # 3. Training Loop
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0
            
            # TQDM with smoothing=0 to see instantaneous speed changes
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", smoothing=0)
            
            for batch_x, batch_y in loop:
                # Non-blocking transfer: Allows CPU to preload next batch while GPU calculates
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Shape Safety: Ensure (Batch, Seq, Feature) layout
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                # Optimization: set_to_none is slightly faster than zeroing tensor memory
                optimizer.zero_grad(set_to_none=True)
                
                # Forward Pass
                # Handle both models with and without return_attention parameter
                # Also handle compilation errors (e.g., TritonMissing) that occur during execution
                try:
                    try:
                        logits = self.model(batch_x, return_attention=False)
                    except TypeError:
                        # Model doesn't support return_attention parameter
                        logits = self.model(batch_x)
                except Exception as e:
                    # Check for various error types and handle appropriately
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    # Handle CUDA kernel compatibility errors
                    if ("no kernel image" in error_str.lower() or 
                        "cudaerror" in error_str.lower() or 
                        "AcceleratorError" in error_type):
                        print(f"\n❌ CUDA Kernel Error during training: {e}")
                        print("   This indicates a GPU compatibility issue.")
                        print("   The PyTorch build doesn't support your GPU architecture.")
                        print("\n   Solutions:")
                        print("   1. Reinstall PyTorch compatible with your GPU:")
                        print("      Visit: https://pytorch.org/get-started/locally/")
                        print("   2. Use CPU instead:")
                        print("      Set device='cpu' when creating the Trainer")
                        print("   3. Check GPU compute capability compatibility")
                        raise RuntimeError(
                            f"CUDA kernel compatibility error. "
                            f"PyTorch was compiled for a different GPU architecture. "
                            f"Original error: {e}"
                        ) from e
                    
                    # Handle compilation errors (TritonMissing, Compiler errors, etc.)
                    elif ("Triton" in error_str or "triton" in error_str.lower() or "TritonMissing" in error_str or
                          "Compiler" in error_str or "cl is not found" in error_str.lower() or 
                          "InductorError" in error_type or "compiler" in error_str.lower()):
                        if self._compiled:
                            print(f"\n⚠️  Compilation error detected during execution: {e}")
                            
                            # Check if it's a compiler error
                            if "Compiler" in error_str or "cl" in error_str.lower() or "compiler" in error_str.lower():
                                print("   This is a compiler error (CPU compilation requires C++ compiler on Windows).")
                                print("   Disabling compilation and retrying with uncompiled model...")
                            else:
                                print("   Disabling compilation and retrying with uncompiled model...")
                            
                            # Get the original model (unwrap torch.compile)
                            if hasattr(self.model, '_orig_mod'):
                                self.model = self.model._orig_mod
                            self._compiled = False
                            
                            # Retry forward pass with uncompiled model
                            try:
                                logits = self.model(batch_x, return_attention=False)
                            except TypeError:
                                logits = self.model(batch_x)
                        else:
                            raise  # Re-raise if not a compilation issue
                    else:
                        raise  # Re-raise other errors
                
                loss = criterion(logits, batch_y)
                
                # Backward
                loss.backward()
                
                # CRITICAL: Gradient Clipping for Quantum Models
                # Quantum gradients can be spiky. Clipping prevents barren plateaus and instability.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                
                optimizer.step()
                train_loss += loss.item()
                
                loop.set_postfix(loss=loss.item())
            
            # 4. Validation
            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            
            # 5. Early Stopping & Model Saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                
                # Handle different model wrappers when saving state_dict
                if isinstance(self.model, nn.DataParallel):
                    # DataParallel wrapper
                    save_dict = self.model.module.state_dict()
                elif hasattr(self.model, '_orig_mod'):
                    # torch.compile() wrapper
                    save_dict = self.model._orig_mod.state_dict()
                else:
                    # Standard model
                    save_dict = self.model.state_dict()
                
                torch.save(save_dict, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
                    
        print(f"✅ Training complete. Best model saved to {save_path}")

    def _validate(self, loader, criterion):
        """
        Efficient validation that counts correct predictions instead of storing all.
        
        Args:
            loader: DataLoader for validation data
            criterion: Loss function
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Shape Safety: Ensure (Batch, Seq, Feature) layout
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                # Handle both models with and without return_attention parameter
                try:
                    logits = self.model(batch_x, return_attention=False)
                except TypeError:
                    # Model doesn't support return_attention parameter
                    logits = self.model(batch_x)
                
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                
                # Calculate Accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        return total_loss / len(loader), correct / total

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Comprehensive evaluation on test set with detailed metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary containing accuracy, f1, precision, recall, predictions, targets, and confusion matrix
        """
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        loader_kwargs = self._get_loader_kwargs(batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        print("\n" + "="*70)
        print("Running Final Evaluation...")
        print("="*70)
        
        # Quick validation pass for accuracy
        _, acc = self._validate(test_loader, nn.CrossEntropyLoss())
        
        # Generate Full Report with all predictions
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Shape Safety: Ensure (Batch, Seq, Feature) layout
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                # Handle both models with and without return_attention parameter
                try:
                    logits = self.model(batch_x, return_attention=False)
                except TypeError:
                    # Model doesn't support return_attention parameter
                    logits = self.model(batch_x)
                
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate detailed metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds)
        
        # Print results
        print(f"\nTest Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_targets, all_preds, digits=4))
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'targets': all_targets,
            'confusion_matrix': cm
        }
    
    def plot_history(self):
        """
        Plot training history with loss, accuracy, and learning rate curves.
        """
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        # Plot 1: Training and Validation Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Plot 3: Learning Rate Schedule
        if len(self.history['lr']) > 0:
            axes[2].plot(self.history['lr'], label='Learning Rate', linewidth=2, color='orange')
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
        else:
            axes[2].text(0.5, 0.5, 'No LR history', ha='center', va='center', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
        print("📊 Training history plot saved to results/training_history.png")
        plt.close()
