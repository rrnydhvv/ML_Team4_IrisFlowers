import pandas as pd
import numpy as np
import pickle

def knn_predict_weighted(train_df, x_new, feature_cols, k):
    dist = np.zeros(len(train_df))
    for i, col in enumerate(feature_cols):
        dist += (train_df[col].values - x_new[i]) ** 2
    dist = np.sqrt(dist)

    df_tmp = train_df.copy()
    df_tmp["dist"] = dist

    neighbors = df_tmp.sort_values("dist").head(k)
    epsilon = 1e-10
    neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)

    class_weights = neighbors.groupby("species")["weight"].sum()
    return class_weights.idxmax()


def evaluate_knn_kfold(df, feature_cols, label_col, k, n_folds=5, random_state=42):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    fold_size = n // n_folds
    accuracies = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = n if fold == n_folds - 1 else start_idx + fold_size

        test_indices = list(range(start_idx, end_idx))
        train_indices = [i for i in range(n) if i not in test_indices]

        train_fold = df_shuffled.iloc[train_indices].reset_index(drop=True)
        test_fold = df_shuffled.iloc[test_indices].reset_index(drop=True)

        correct = 0
        for i in range(len(test_fold)):
            x_new = test_fold.iloc[i][feature_cols].values
            y_true = test_fold.iloc[i][label_col]
            y_pred = knn_predict_weighted(train_fold, x_new, feature_cols, k)
            if y_pred == y_true:
                correct += 1

        accuracies.append(correct / len(test_fold))

    return np.mean(accuracies)


def run_knn_train_test(df, test_size=0.2, random_state=42):
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_col = "species"

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_size))
    train_norm = df_shuffled.iloc[:-n_test].reset_index(drop=True)
    test_norm = df_shuffled.iloc[-n_test:].reset_index(drop=True)

    print("Chọn K bằng 5-Fold Cross-Validation trên tập train:")
    print(f"{'k':>3} | {'Accuracy (CV)':>15}")
    print("-" * 25)

    train_results = []
    for k in range(1, 31):
        acc = evaluate_knn_kfold(train_norm, feature_cols, label_col, k, n_folds=5, random_state=42)
        train_results.append((k, acc))
        print(f"{k:>3} | {acc:>15.4f}")

    best_train_acc = max(acc for k, acc in train_results)
    tolerance = 1e-6
    best_ks = [k for k, acc in train_results if abs(acc - best_train_acc) < tolerance]
    best_k = min(best_ks)

    print(f"\nCác k đạt accuracy cao nhất (~{best_train_acc:.4f}): {sorted(best_ks)}")
    print(f"→ Chọn k nhỏ nhất: {best_k}")

    correct = 0
    for i in range(len(test_norm)):
        x_new = test_norm.iloc[i][feature_cols].values
        y_true = test_norm.iloc[i][label_col]
        y_pred = knn_predict_weighted(train_norm, x_new, feature_cols, best_k)
        if y_pred == y_true:
            correct += 1

    test_acc = correct / len(test_norm)

    print("\nKết luận (KNN Weighted):")
    print(f"Best k (nhỏ nhất trong các k tốt nhất): {best_k}")
    print(f"Accuracy trung bình trên train (5-Fold CV): {best_train_acc:.4f}")
    print(f"Accuracy trên tập test:                   {test_acc:.4f}")

    model = {
        "train_norm": train_norm,
        "feature_cols": feature_cols,
        "best_k": best_k,
    }

    return train_results, best_k, best_train_acc, test_acc, model

def save_model_KNN(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model_KNN(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model