"""
High-Performance Parallelized Training Pipeline for Quantum Transformer
Optimized with multi-GPU support, parallel data loading, and advanced training features.
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
    Optimized Trainer with Multi-Process Data Loading and GPU Acceleration.
    Combines performance optimizations with comprehensive evaluation and visualization.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Normalize device to torch.device object for consistent handling
        # This handles both string ('cuda', 'cpu') and torch.device objects
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Check if device is CUDA (works for both string and torch.device)
        is_cuda = self.device.type == 'cuda'
        
        # Handle Multi-GPU (DataParallel) automatically
        # IMPORTANT: Always move model to device BEFORE wrapping with DataParallel
        # DataParallel expects the model to already be on the target device
        if is_cuda and torch.cuda.device_count() > 1:
            print(f"🚀 Parallelizing model across {torch.cuda.device_count()} GPUs!")
            # Move model to device first, then wrap with DataParallel
            model = model.to(self.device)
            self.model = nn.DataParallel(model)
        else:
            self.model = model.to(self.device)
            
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

    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, 
              n_epochs=50, 
              learning_rate=0.001,
              patience=5,
              grad_clip=1.0):
        """
        Train with parallel data loading and advanced features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience (epochs without improvement)
            grad_clip: Gradient clipping threshold
        """
        print("\n--- Starting Optimized Training ---")
        
        # 1. Prepare Data
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        # Get optimized loader settings
        loader_kwargs = self._get_loader_kwargs(batch_size)
        print(f"⚙️ DataLoader Config: workers={loader_kwargs['num_workers']}, "
              f"pin_memory={loader_kwargs['pin_memory']}")
        
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        
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
                # Non-blocking transfer allows CPU to fetch next batch while GPU processes this one
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Handle 2D input (add sequence dimension if needed)
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Forward Pass
                # NOTE: We explicitly pass return_attention=False to prevent DataParallel unpacking errors
                logits = self.model(batch_x, return_attention=False)
                
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # Gradient Clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
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
            
            # 5. Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                os.makedirs('models', exist_ok=True)
                # Handle DataParallel saving (save .module state_dict)
                save_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save(save_dict, 'models/best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

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
                
                # Handle 2D input (add sequence dimension if needed)
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                logits = self.model(batch_x, return_attention=False)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                
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
                
                # Handle 2D input (add sequence dimension if needed)
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                logits = self.model(batch_x, return_attention=False)
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
