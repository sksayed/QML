"""
Data preprocessing for quantum machine learning.
Optimized for performance and reproducibility with support for both PCA and feature selection.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class DataPreprocessor:
    """
    Data preprocessor for quantum machine learning.
    Supports standard scaling, dimensionality reduction (PCA or SelectKBest), and quantum feature encoding.
    """
    
    def __init__(self):
        self.standard_scaler = StandardScaler()
        # Custom scaler for Angle Encoding [0, PI]
        self.quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None
        
    def _clean_data(self, X):
        """Helper to clean NaN and Inf values"""
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    def preprocess_and_reduce(self, X_train, X_val, X_test, target_qubits=None, y_train=None, method='pca'):
        """
        Standardize and adjust dimensions to match the number of available qubits.
        Can both reduce (if target < current) or expand (if target > current) dimensions.
        
        Args:
            X_train, X_val, X_test: Training, validation, and test features
            target_qubits: Target number of features (qubits) after adjustment
            y_train: Training labels (required for 'select_k_best' method)
            method: 'pca' or 'select_k_best' for dimensionality adjustment
                   Note: Expansion only uses PCA (requires sufficient samples)
        
        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        # 1. Clean Data
        X_train = self._clean_data(X_train)
        X_val = self._clean_data(X_val)
        X_test = self._clean_data(X_test)
        
        # 2. Standard Scaling (Zero Mean, Unit Variance)
        X_train_std = self.standard_scaler.fit_transform(X_train)
        X_val_std = self.standard_scaler.transform(X_val)
        X_test_std = self.standard_scaler.transform(X_test)
        
        # 3. Handle any NaN that might have been created by scaling (e.g., zero variance features)
        X_train_std = self._clean_data(X_train_std)
        X_val_std = self._clean_data(X_val_std)
        X_test_std = self._clean_data(X_test_std)
        
        # 4. Dimensionality Adjustment (Fit to Qubits)
        # Handle both reduction and expansion to match target_qubits
        if target_qubits and target_qubits != X_train_std.shape[1]:
            original_dim = X_train_std.shape[1]  # Store original dimension
            
            if target_qubits < X_train_std.shape[1]:
                # REDUCTION: Reduce dimensions
                print(f"Reducing dimensions from {original_dim} to {target_qubits} using {method}...")
                
                if method == 'pca':
                    self.pca = PCA(n_components=target_qubits)
                    X_train_std = self.pca.fit_transform(X_train_std)
                    X_val_std = self.pca.transform(X_val_std)
                    X_test_std = self.pca.transform(X_test_std)
                    print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
                    
                elif method == 'select_k_best' and y_train is not None:
                    self.feature_selector = SelectKBest(f_classif, k=target_qubits)
                    X_train_std = self.feature_selector.fit_transform(X_train_std, y_train)
                    X_val_std = self.feature_selector.transform(X_val_std)
                    X_test_std = self.feature_selector.transform(X_test_std)
                    print(f"Selected {target_qubits} best features from {original_dim} original features")
                else:
                    raise ValueError("For 'select_k_best' method, y_train must be provided")
            
            elif target_qubits > X_train_std.shape[1]:
                # EXPANSION: Expand dimensions using PCA
                # PCA can create up to min(n_samples, n_features) components
                max_components = min(X_train_std.shape[0], X_train_std.shape[1])
                if target_qubits > max_components:
                    raise ValueError(
                        f"Cannot create {target_qubits} components from {X_train_std.shape[1]} features "
                        f"with {X_train_std.shape[0]} samples. Maximum possible: {max_components}. "
                        f"Please ensure you have at least {target_qubits} samples."
                    )
                print(f"Expanding dimensions from {original_dim} to {target_qubits} using PCA...")
                self.pca = PCA(n_components=target_qubits)
                X_train_std = self.pca.fit_transform(X_train_std)
                X_val_std = self.pca.transform(X_val_std)
                X_test_std = self.pca.transform(X_test_std)
                print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
                print(f"✓ Successfully expanded to {target_qubits} features for amplitude encoding")
        
        # 5. Fit Quantum Scaler (0 to PI) on the reduced/standardized data
        # We fit this here so we can reuse it later in get_quantum_features()
        self.quantum_scaler.fit(X_train_std)
        
        return X_train_std, X_val_std, X_test_std
    
    def get_quantum_features(self, X, pad_qubits=None):
        """
        Scale to [0, π] for Angle Encoding.
        Optionally pad with zeros if you have more qubits than features.
        
        Args:
            X: Features to transform
            pad_qubits: If specified and larger than current dimensions, pad with zeros
        
        Returns:
            Features scaled to [0, π] range
        """
        # Ensure data is clean
        X = self._clean_data(X)
        
        # Transform to [0, pi] using the scaler fitted in preprocess_and_reduce()
        X_q = self.quantum_scaler.transform(X)
        
        # Padding (useful if you want to use 8 qubits but only have 6 features)
        current_dim = X_q.shape[1]
        if pad_qubits and pad_qubits > current_dim:
            padding = np.zeros((X_q.shape[0], pad_qubits - current_dim))
            X_q = np.hstack([X_q, padding])
            
        return X_q
    
    def encode_labels(self, y_train, y_val, y_test):
        """
        Encode labels to numerical format.
        
        Returns:
            Tuple of (y_train_encoded, y_val_encoded, y_test_encoded)
        """
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_test_enc = self.label_encoder.transform(y_test)
        
        # Print both classes and label mapping for clarity
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return y_train_enc, y_val_enc, y_test_enc
    
    # ========== BACKWARD COMPATIBILITY METHODS ==========
    
    def preprocess_features(self, X_train, X_val, X_test, n_features=None, y_train=None):
        """
        Backward compatibility wrapper for preprocess_and_reduce().
        Uses 'select_k_best' method by default to match old behavior.
        """
        method = 'select_k_best' if y_train is not None else 'pca'
        return self.preprocess_and_reduce(
            X_train, X_val, X_test,
            target_qubits=n_features,
            y_train=y_train,
            method=method
        )
    
    def prepare_quantum_features(self, X, n_qubits=None):
        """
        Backward compatibility wrapper for get_quantum_features().
        """
        return self.get_quantum_features(X, pad_qubits=n_qubits)
