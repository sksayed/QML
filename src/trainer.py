"""
Training pipeline for Quantum Transformer
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

class Trainer:
    """Training pipeline"""
    
    def __init__(self, model, device='cpu'):
        self.device = device
        # Move model to device
        if device == 'cuda' and torch.cuda.is_available():
            self.model = model.cuda()
            torch.cuda.empty_cache()
        else:
            self.model = model.to(device)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, n_epochs=50, learning_rate=0.001):
        """Train the model"""
        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
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
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if len(batch_x.shape) == 2:
                        batch_x = batch_x.unsqueeze(1)
                    
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_targets, val_preds)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate on test set"""
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        pin_memory = (self.device == 'cuda')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0
        )
        
        self.model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(test_targets, test_preds)
        f1 = f1_score(test_targets, test_preds, average='weighted')
        precision = precision_score(test_targets, test_preds, average='weighted')
        recall = recall_score(test_targets, test_preds, average='weighted')
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, test_preds)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': test_preds,
            'targets': test_targets,
            'confusion_matrix': cm
        }
    
    def plot_history(self):
        """Plot training history"""
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150)
        print("Training history plot saved to results/training_history.png")
        plt.close()

