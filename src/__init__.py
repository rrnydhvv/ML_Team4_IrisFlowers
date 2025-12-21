import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like 'src.models' work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.KNN import run_knn_train_test, predict_single


data_path = project_root / "data" / "IRIS_cleaned.csv"
df = pd.read_csv(data_path)

if __name__ == "__main__":
    train_results, best_k, best_train_acc, test_acc, model = run_knn_train_test(df, test_size=0.2, random_state=42)

    # Demo prediction: predict the first row's features
    feature_cols = model["feature_cols"]
    sample_row = df.iloc[0][feature_cols]
    pred = predict_single(model, sample_row)
    print("\n=== DEMO PREDICTION ===")
    print("Sample features:")
    print(sample_row.to_dict())
    print("Predicted species:", pred)