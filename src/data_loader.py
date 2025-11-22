"""
Data loader for 5G-NIDD dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self, n_samples=None):
        """Load 5G-NIDD dataset"""
        print(f"Loading data from {self.data_path}...")
        # Read in chunks if file is large
        try:
            if n_samples:
                self.df = pd.read_csv(self.data_path, nrows=n_samples)
            else:
                # Try reading first chunk to check size
                chunk_size = 100000
                chunks = []
                for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) == 1:  # Read first chunk only for now
                        break
                self.df = chunks[0] if chunks else pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try reading with different encoding
            self.df = pd.read_csv(self.data_path, encoding='latin-1', nrows=n_samples if n_samples else None)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"First few rows:")
        print(self.df.head())
        return self.df
    
    def get_features_labels(self, label_column=None):
        """Extract features and labels"""
        if label_column is None:
            # Try to find label column (common names)
            possible_labels = ['label', 'Label', 'target', 'Target', 'class', 'Class', 'attack', 'Attack']
            label_column = None
            for col in possible_labels:
                if col in self.df.columns:
                    label_column = col
                    break
            
            if label_column is None:
                # Assume last column is label
                label_column = self.df.columns[-1]
                print(f"No explicit label column found, using last column: {label_column}")
        
        # Exclude label column and other non-numeric columns
        exclude_cols = [label_column, 'Attack Type', 'Attack Tool', 'Unnamed: 0']
        
        # Get numeric columns only
        feature_cols = []
        for col in self.df.columns:
            if col not in exclude_cols:
                # Check if column is numeric
                if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
                else:
                    # Try to convert to numeric, if fails, skip
                    try:
                        pd.to_numeric(self.df[col], errors='raise')
                        feature_cols.append(col)
                    except:
                        print(f"Skipping non-numeric column: {col}")
        
        X = self.df[feature_cols].values.astype(np.float32)
        y = self.df[label_column].values
        
        print(f"Selected {len(feature_cols)} numeric features")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split into train/val/test"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=42, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

