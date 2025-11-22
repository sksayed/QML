"""
Data loader for 5G-NIDD dataset.
Optimized for large datasets with robust error handling and data cleaning.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NIDDDataLoader:
    """
    Data loader for 5G-NIDD dataset.
    Handles large CSV files efficiently with proper type conversion and data cleaning.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self, n_samples=None):
        """
        Load 5G-NIDD dataset efficiently.
        
        Args:
            n_samples: If specified, only load first n_samples rows. If None, load full dataset.
        
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {self.data_path}...")
        
        try:
            # Optimization: specify engine and low_memory for large CSVs
            if n_samples:
                self.df = pd.read_csv(self.data_path, nrows=n_samples, low_memory=False)
            else:
                # LOAD FULL DATASET
                self.df = pd.read_csv(self.data_path, low_memory=False)
                
        except UnicodeDecodeError:
            print("UTF-8 failed. Retrying with latin-1...")
            self.df = pd.read_csv(
                self.data_path, 
                encoding='latin-1', 
                nrows=n_samples if n_samples else None,
                low_memory=False
            )
        
        # Strip whitespace from column names (common issue in NIDD datasets)
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns ({len(self.df.columns)}): {list(self.df.columns[:10])}{'...' if len(self.df.columns) > 10 else ''}")
        return self.df
    
    def get_features_labels(self, label_col_name=None):
        """
        Extract features and labels with proper type conversion.
        
        Args:
            label_col_name: Name of the label column. If None, attempts auto-detection.
        
        Returns:
            Tuple of (X, y) where X is feature matrix and y is label array
        
        Raises:
            ValueError: If label column cannot be detected or specified.
        """
        
        # 1. IDENTIFY LABEL
        if label_col_name is None:
            # Priority list for 5G-NIDD
            candidates = ['Label', 'Attack Type', 'Attack Tool', 'target', 'class', 
                         'label', 'Target', 'Class', 'attack', 'Attack']
            for col in candidates:
                if col in self.df.columns:
                    label_col_name = col
                    break
        
        if not label_col_name or label_col_name not in self.df.columns:
            raise ValueError(
                f"Could not detect label column. Available columns: {list(self.df.columns)}. "
                "Please specify label_col_name parameter."
            )
            
        print(f"Target Column detected: '{label_col_name}'")
        
        # 2. DEFINE EXCLUSIONS
        # We remove the label, and distinct metadata columns found in 5G-NIDD
        # Note: 'BS' (Base Station) and 'UE' (User Equipment) IDs are often not useful for ML generalization
        metadata_cols = ['Unnamed: 0', 'Seq', 'Date', 'Time', 'Attack Tool', 
                        'Attack Type', label_col_name]
        
        # 3. PROCESS NUMERICS
        feature_cols = []
        for col in self.df.columns:
            if col in metadata_cols:
                continue
                
            # Force conversion to numeric.
            # 'coerce' turns non-parseable strings into NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Check if the column is effectively empty (all NaNs)
            if self.df[col].isna().sum() > (0.5 * len(self.df)):
                print(f"Dropping sparse column: {col} (>50% NaN)")
            else:
                feature_cols.append(col)
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found after processing!")
        
        # 4. CLEANUP
        # Drop rows where features became NaN after coercion
        # (This is better than filling with 0 for network traffic data,
        # as 0 often has specific meaning like 'no packet')
        X_df = self.df[feature_cols]
        y_series = self.df[label_col_name]
        
        # Remove rows with NaN features
        valid_indices = ~X_df.isna().any(axis=1)
        n_dropped = (~valid_indices).sum()
        
        if n_dropped > 0:
            print(f"Dropping {n_dropped} rows with NaN features ({100*n_dropped/len(X_df):.2f}%)")
        
        X = X_df[valid_indices].values.astype(np.float32)
        y = y_series[valid_indices].values
        
        print(f"Final Features shape: {X.shape}")
        print(f"Final Labels shape: {y.shape}")
        print(f"Selected {len(feature_cols)} feature columns")
        print(f"Unique labels ({len(np.unique(y))}): {np.unique(y)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split into Train/Val/Test with stratification.
        
        Args:
            X: Feature matrix
            y: Label array
            test_size: Fraction of total dataset for test set (default: 0.2 = 20%)
            val_size: Fraction of total dataset for validation set (default: 0.1 = 10%)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
        Note:
            val_size is the percentage of the ORIGINAL dataset to be used for validation.
            The split is: Test = test_size, Val = val_size, Train = 1 - test_size - val_size
        """
        # 1. Split Test off
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # 2. Calculate validation size relative to the remaining (Train + Val) set
        # If we want 10% of total for Val, and Test was 20%,
        # then Val is 12.5% of the remaining 80%.
        # Formula: val_relative = val_absolute / (1 - test_absolute)
        val_relative = val_size / (1.0 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_relative, 
            random_state=random_state, 
            stratify=y_train_val
        )
        
        print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

# Backward compatibility alias
DataLoader = NIDDDataLoader
