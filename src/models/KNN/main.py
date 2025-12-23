"""
Main script to run KNN experiment
Usage: python main.py (from project root)
       or python -m src.models.KNN.main
"""

import sys
from pathlib import Path

# Add project root to sys.path so imports work correctly
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.KNN.experiment import run_knn_train_test


if __name__ == "__main__":
    # Load data
    print("=" * 60)
    print("CHẠY KNN EXPERIMENT")
    print("=" * 60)
    
    df = pd.read_csv('data/IRIS_cleaned.csv')
    print(f"\n Đã tải dữ liệu: {len(df)} mẫu")
    print(f"  Columns: {list(df.columns)}")
    
    # Run experiment
    print()
    train_results, best_k, best_train_acc, test_acc, model = run_knn_train_test(
        df, test_size=0.2, random_state=42
    )
    
    print("\n" + "=" * 60)
    print("Hoàn tất")
    print("=" * 60)
