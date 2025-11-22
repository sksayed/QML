"""
Data preprocessing for quantum machine learning
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_min = None
        self.feature_max = None
        
    def preprocess_features(self, X_train, X_val, X_test, n_features=None, y_train=None):
        """Normalize and optionally select features"""
        # Handle NaN and infinite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle any NaN that might have been created by scaling (e.g., zero variance features)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection (optional)
        if n_features and n_features < X_train_scaled.shape[1] and y_train is not None:
            self.feature_selector = SelectKBest(f_classif, k=min(n_features, X_train_scaled.shape[1]))
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
            print(f"Selected {n_features} features from {X_train.shape[1]} original features")
        
        # Store min/max for quantum encoding
        self.feature_min = X_train_scaled.min(axis=0, keepdims=True)
        self.feature_max = X_train_scaled.max(axis=0, keepdims=True)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def encode_labels(self, y_train, y_val, y_test):
        """Encode labels to numerical format"""
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def prepare_quantum_features(self, X, n_qubits=None):
        """Prepare features for quantum encoding (normalize to [0, π])"""
        # Normalize to [0, π] for angle encoding
        if self.feature_min is None or self.feature_max is None:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
        else:
            X_min = self.feature_min
            X_max = self.feature_max
        
        # Avoid division by zero
        X_range = X_max - X_min + 1e-8
        X_normalized = (X - X_min) / X_range * np.pi
        
        # If n_qubits specified and features don't match, pad or truncate
        if n_qubits and X_normalized.shape[1] != n_qubits:
            if X_normalized.shape[1] < n_qubits:
                # Pad with zeros
                padding = np.zeros((X_normalized.shape[0], n_qubits - X_normalized.shape[1]))
                X_normalized = np.hstack([X_normalized, padding])
            else:
                # Truncate
                X_normalized = X_normalized[:, :n_qubits]
        
        return X_normalized

