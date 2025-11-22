"""
Exploratory Data Analysis (EDA) and Training Parameter Recommendations
Analyzes data.csv and recommends optimal n_samples and epochs for training.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_dataset(data_path='data/data.csv'):
    """
    Perform comprehensive EDA and recommend training parameters.
    
    Args:
        data_path: Path to the CSV file
    """
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA) & TRAINING RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return None
    
    # Get file size
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"\n📁 File Information:")
    print(f"   Path: {data_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    # Load data
    print(f"\n📊 Loading dataset...")
    try:
        # First, get row count without loading full dataset
        print("   Counting total rows...")
        row_count = sum(1 for _ in open(data_path)) - 1  # -1 for header
        print(f"   Total rows (excluding header): {row_count:,}")
        
        # Load a sample for analysis (first 100k rows for quick analysis)
        sample_size = min(100000, row_count)
        print(f"   Loading sample ({sample_size:,} rows) for detailed analysis...")
        df = pd.read_csv(data_path, nrows=sample_size, low_memory=False)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Basic statistics
    print(f"\n{'='*80}")
    print("DATASET OVERVIEW")
    print(f"{'='*80}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n📋 Column Information:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Identify label column
    label_col = None
    label_candidates = ['Label', 'Attack Type', 'Attack Tool', 'target', 'class', 
                       'label', 'Target', 'Class', 'attack', 'Attack']
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        print(f"\n🎯 Label Column: '{label_col}'")
        
        # Class distribution
        class_counts = df[label_col].value_counts()
        n_classes = len(class_counts)
        
        print(f"\n📊 Class Distribution:")
        print(f"   Number of classes: {n_classes}")
        for class_name, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {class_name}: {count:,} ({percentage:.2f}%)")
        
        # Class imbalance analysis
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        print(f"\n⚖️  Class Balance Analysis:")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 10:
            print("   ⚠️  WARNING: Severe class imbalance detected!")
            print("      Recommendation: Use class weights or oversampling")
        elif imbalance_ratio > 3:
            print("   ⚠️  Moderate class imbalance")
        else:
            print("   ✅ Classes are relatively balanced")
    else:
        print(f"\n⚠️  Could not auto-detect label column")
        print(f"   Available columns: {list(df.columns[:10])}")
        n_classes = 2  # Default assumption
        imbalance_ratio = None
    
    # Missing values
    print(f"\n🔍 Missing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"   Columns with missing values: {len(missing_df)}")
        print(f"   Top 10 columns with most missing values:")
        for idx, row in missing_df.head(10).iterrows():
            print(f"      {row['Column']}: {row['Missing Count']:,} ({row['Percentage']:.2f}%)")
    else:
        print("   ✅ No missing values detected")
    
    # Feature statistics (if numeric features exist)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Feature Statistics (Numeric Columns):")
        print(f"   Number of numeric features: {len(numeric_cols)}")
        print(f"   Mean values range: [{df[numeric_cols].mean().min():.2f}, {df[numeric_cols].mean().max():.2f}]")
        print(f"   Std values range: [{df[numeric_cols].std().min():.2f}, {df[numeric_cols].std().max():.2f}]")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("TRAINING PARAMETER RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Calculate recommendations based on dataset size and characteristics
    total_rows = row_count
    
    # Recommendation 1: n_samples
    print(f"\n1️⃣  RECOMMENDED n_samples:")
    
    if total_rows < 1000:
        recommended_samples = total_rows
        print(f"   Dataset is small ({total_rows:,} rows)")
        print(f"   ✅ Use ALL data: n_samples={recommended_samples:,} (or None)")
    elif total_rows < 10000:
        recommended_samples = total_rows
        print(f"   Dataset is medium ({total_rows:,} rows)")
        print(f"   ✅ Use ALL data: n_samples={recommended_samples:,} (or None)")
    elif total_rows < 100000:
        # For medium-large datasets, use 80-100% of data
        recommended_samples = int(total_rows * 0.9)
        print(f"   Dataset is medium-large ({total_rows:,} rows)")
        print(f"   ✅ Recommended: n_samples={recommended_samples:,} (90% of data)")
        print(f"   Alternative: Use all data with n_samples=None")
    elif total_rows < 1000000:
        # For large datasets, use 50-80% for initial training
        recommended_samples = int(total_rows * 0.7)
        print(f"   Dataset is large ({total_rows:,} rows)")
        print(f"   ✅ Recommended: n_samples={recommended_samples:,} (70% of data)")
        print(f"   For full training: n_samples=None (use all {total_rows:,} rows)")
    else:
        # For very large datasets, start with a subset
        recommended_samples = 500000
        print(f"   Dataset is very large ({total_rows:,} rows)")
        print(f"   ✅ Recommended for initial training: n_samples={recommended_samples:,}")
        print(f"   For full training: n_samples=None (use all {total_rows:,} rows)")
        print(f"   Note: Full dataset training may take significant time")
    
    # Recommendation 2: Epochs
    print(f"\n2️⃣  RECOMMENDED n_epochs:")
    
    # Epochs depend on dataset size and complexity
    if total_rows < 10000:
        recommended_epochs = (30, 50)
        print(f"   Small dataset ({total_rows:,} rows)")
        print(f"   ✅ Recommended range: {recommended_epochs[0]}-{recommended_epochs[1]} epochs")
        print(f"   Reason: Small datasets need more epochs to learn patterns")
    elif total_rows < 100000:
        recommended_epochs = (20, 40)
        print(f"   Medium dataset ({total_rows:,} rows)")
        print(f"   ✅ Recommended range: {recommended_epochs[0]}-{recommended_epochs[1]} epochs")
        print(f"   Reason: Medium datasets have enough data for moderate training")
    elif total_rows < 1000000:
        recommended_epochs = (15, 30)
        print(f"   Large dataset ({total_rows:,} rows)")
        print(f"   ✅ Recommended range: {recommended_epochs[0]}-{recommended_epochs[1]} epochs")
        print(f"   Reason: Large datasets provide more examples per epoch")
    else:
        recommended_epochs = (10, 25)
        print(f"   Very large dataset ({total_rows:,} rows)")
        print(f"   ✅ Recommended range: {recommended_epochs[0]}-{recommended_epochs[1]} epochs")
        print(f"   Reason: Very large datasets converge faster")
    
    # Additional recommendations
    print(f"\n3️⃣  ADDITIONAL RECOMMENDATIONS:")
    
    # Batch size recommendation
    if total_rows < 10000:
        batch_size = 32
    elif total_rows < 100000:
        batch_size = 64
    else:
        batch_size = 128
    
    print(f"   Batch size: {batch_size} (adjust based on GPU memory)")
    
    # Early stopping
    patience = max(5, recommended_epochs[0] // 5)
    print(f"   Early stopping patience: {patience} epochs")
    
    # Learning rate
    print(f"   Learning rate: 1e-4 to 1e-3 (use AutoML to find optimal)")
    
    # Optuna trials
    if total_rows < 50000:
        n_trials = 20
    elif total_rows < 500000:
        n_trials = 30
    else:
        n_trials = 50
    
    print(f"   Optuna trials: {n_trials} (for hyperparameter optimization)")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total dataset size: {total_rows:,} rows")
    if total_rows < 100000:
        print(f"Recommended n_samples: {recommended_samples:,} (or None for all data)")
    else:
        print(f"Recommended n_samples: {recommended_samples:,}")
    print(f"Recommended epochs: {recommended_epochs[0]}-{recommended_epochs[1]}")
    print(f"Recommended batch size: {batch_size}")
    print(f"Recommended Optuna trials: {n_trials}")
    
    # Create visualization if label column exists
    if label_col:
        try:
            os.makedirs('results', exist_ok=True)
            
            # Class distribution plot
            plt.figure(figsize=(12, 6))
            class_counts.plot(kind='bar')
            plt.title('Class Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/class_distribution.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualization saved: results/class_distribution.png")
            plt.close()
        except Exception as e:
            print(f"\n⚠️  Could not create visualization: {e}")
    
    # Save recommendations to file
    try:
        recommendations = {
            'total_rows': total_rows,
            'recommended_n_samples': recommended_samples if total_rows >= 100000 else None,
            'recommended_epochs_min': recommended_epochs[0],
            'recommended_epochs_max': recommended_epochs[1],
            'recommended_batch_size': batch_size,
            'recommended_n_trials': n_trials,
            'n_classes': n_classes,
            'class_imbalance_ratio': imbalance_ratio if label_col else None
        }
        
        import json
        with open('results/eda_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"💾 Recommendations saved: results/eda_recommendations.json")
    except Exception as e:
        print(f"⚠️  Could not save recommendations: {e}")
    
    return recommendations

if __name__ == "__main__":
    recommendations = analyze_dataset('data/data.csv')
    
    if recommendations:
        print(f"\n✅ EDA Complete!")
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the recommendations above")
        print(f"   2. Update main.py with recommended parameters")
        print(f"   3. Run training with: python main.py")
    else:
        print(f"\n❌ EDA Failed - Please check the data file path")

