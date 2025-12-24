import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like 'src.models' work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from models.KNN import knn_predict_weighted, evaluate_knn_kfold, run_knn_train_test, save_model_KNN, load_model_KNN

from src.models.Naive_Bayes import GaussianNaiveBayes, accuracy_score


data_path = project_root / "data" / "IRIS_cleaned.csv"
df = pd.read_csv(data_path)

if __name__ == "__main__":
    train_results, best_k, best_train_acc, test_acc, model = run_knn_train_test(df, test_size=0.2, random_state=42)

    # Demo prediction: predict the first row's features
    feature_cols = model["feature_cols"]
    sample_row = df.iloc[0][feature_cols]

    # Demo Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    X = df[feature_cols].values
    y = df["species"].values
    gnb.fit(X, y)
    y_pred = gnb.predict(X)
    acc = accuracy_score(y, y_pred)
    print("\n=== GAUSSIAN NAIVE BAYES ===")
    print(f"Accuracy on entire dataset: {acc:.4f}")
    