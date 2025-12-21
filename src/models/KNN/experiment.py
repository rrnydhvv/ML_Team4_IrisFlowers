"""
Weighted KNN training/testing experiment.
"""

import pandas as pd
from .knn_core import knn_predict_weighted
from .knn_cv import evaluate_knn_kfold


def run_knn_train_test(df, test_size=0.2, random_state=42):
    """
    Huấn luyện Weighted KNN: chọn k tối ưu bằng 5-Fold CV trên train, đánh giá trên test.
    Chọn k NHỎ NHẤT trong số các k đạt accuracy cao nhất (xử lý floating-point precision).
    
    Note: df phải là dữ liệu đã chuẩn hóa (IRIS_cleaned.csv).
    """
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_col = "species"

    # Chia train/test (dữ liệu đã chuẩn hóa)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_size))
    train_norm = df_shuffled.iloc[:-n_test].reset_index(drop=True)
    test_norm = df_shuffled.iloc[-n_test:].reset_index(drop=True)

    # Chọn k bằng 5-Fold CV
    print("=== CHỌN K TỐI ƯU BẰNG 5-FOLD CV (Weighted KNN) ===")
    print(f"{'k':>3} | {'Accuracy (CV)':>15}")
    print("-" * 25)
    
    train_results = []
    for k in range(1, 31):
        acc = evaluate_knn_kfold(train_norm, feature_cols, label_col, k, n_folds=5, random_state=42)
        train_results.append((k, acc))
        print(f"{k:>3} | {acc:>15.4f}")

    # Tìm accuracy cao nhất
    best_train_acc = max(acc for k, acc in train_results)
    
    # Dùng tolerance để tìm tất cả k gần như đạt max (fix floating-point issue)
    tolerance = 1e-6
    best_ks = [k for k, acc in train_results if abs(acc - best_train_acc) < tolerance]
    
    # Chọn k nhỏ nhất
    best_k = min(best_ks)

    print(f"\nCác k đạt accuracy cao nhất (~{best_train_acc:.4f}): {sorted(best_ks)}")
    print(f"→ Chọn k nhỏ nhất: {best_k} (ưu tiên mô hình đơn giản hơn)")

    # Đánh giá trên test
    correct = 0
    for i in range(len(test_norm)):
        x_new = test_norm.iloc[i][feature_cols].values
        y_true = test_norm.iloc[i][label_col]
        y_pred = knn_predict_weighted(train_norm, x_new, feature_cols, best_k)
        if y_pred == y_true:
            correct += 1

    test_acc = correct / len(test_norm)

    print("\n=== KẾT LUẬN (Weighted KNN) ===")
    print(f"Best k (nhỏ nhất trong các k tốt nhất): {best_k}")
    print(f"Accuracy trung bình trên train (5-Fold CV): {best_train_acc:.4f}")
    print(f"Accuracy trên tập test:                   {test_acc:.4f}")

    # Đóng gói model (không lưu normalization params vì dữ liệu đã chuẩn hóa)
    model = {
        "train_norm": train_norm,
        "feature_cols": feature_cols,
        "best_k": best_k,
    }

    return train_results, best_k, best_train_acc, test_acc, model
