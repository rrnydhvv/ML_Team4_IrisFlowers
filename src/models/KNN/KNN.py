# import pandas as pd
# import numpy as np

# from src.preprocess import normalize_dataframe, apply_normalize


# def knn_predict_weighted(train_df, x_new, feature_cols, k):
#     """
#     Dự đoán class mẫu mới với trọng số nghịch đảo khoảng cách (Weighted KNN)
#     """
#     dist = np.zeros(len(train_df))

#     for i, col in enumerate(feature_cols):
#         dist += (train_df[col].values - x_new[i]) ** 2

#     dist = np.sqrt(dist)
#     train_df = train_df.copy()
#     train_df["dist"] = dist

#     neighbors = train_df.sort_values("dist").head(k)
    
#     epsilon = 1e-10
#     neighbors = neighbors.copy()
#     neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)
    
#     class_weights = neighbors.groupby("species")["weight"].sum()
#     pred_class = class_weights.idxmax()

#     return pred_class


# def evaluate_knn_kfold(df, feature_cols, label_col, k, n_folds=5, random_state=42):
#     """
#     Đánh giá Weighted KNN bằng K-Fold cross-validation.
#     """
#     df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     n = len(df_shuffled)
#     fold_size = n // n_folds
    
#     accuracies = []
    
#     for fold in range(n_folds):
#         start_idx = fold * fold_size
#         end_idx = n if fold == n_folds - 1 else start_idx + fold_size
        
#         test_indices = list(range(start_idx, end_idx))
#         train_indices = [i for i in range(n) if i not in test_indices]
        
#         train_fold = df_shuffled.iloc[train_indices].reset_index(drop=True)
#         test_fold = df_shuffled.iloc[test_indices].reset_index(drop=True)
        
#         correct = 0
#         for i in range(len(test_fold)):
#             x_new = test_fold.iloc[i][feature_cols].values
#             y_true = test_fold.iloc[i][label_col]
#             y_pred = knn_predict_weighted(train_fold, x_new, feature_cols, k)
#             if y_pred == y_true:
#                 correct += 1
        
#         fold_acc = correct / len(test_fold)
#         accuracies.append(fold_acc)
    
#     return np.mean(accuracies)


# def run_knn_train_test(df, test_size=0.2, random_state=42):
#     """
#     Huấn luyện Weighted KNN: chọn k tối ưu bằng 5-Fold CV trên train, đánh giá trên test.
#     Chọn k NHỎ NHẤT trong số các k đạt accuracy cao nhất (xử lý floating-point precision).
#     """
#     feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
#     label_col = "species"

#     # Chia train/test
#     df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     n_test = max(1, int(len(df) * test_size))
#     train_df = df_shuffled.iloc[:-n_test].reset_index(drop=True)
#     test_df = df_shuffled.iloc[-n_test:].reset_index(drop=True)

#     # Chuẩn hóa
#     train_norm, params = normalize_dataframe(train_df, feature_cols)
#     test_norm = apply_normalize(test_df, feature_cols, params)

#     # Chọn k bằng 5-Fold CV
#     print("=== CHỌN K TỐI ƯU BẰNG 5-FOLD CV (Weighted KNN) ===")
#     print(f"{'k':>3} | {'Accuracy (CV)':>15}")
#     print("-" * 25)
    
#     train_results = []
#     for k in range(1, 31):
#         acc = evaluate_knn_kfold(train_norm, feature_cols, label_col, k, n_folds=5, random_state=42)
#         train_results.append((k, acc))
#         print(f"{k:>3} | {acc:>15.4f}")

#     # Tìm accuracy cao nhất
#     best_train_acc = max(acc for k, acc in train_results)
    
#     # Dùng tolerance để tìm tất cả k gần như đạt max (fix floating-point issue)
#     tolerance = 1e-6
#     best_ks = [k for k, acc in train_results if abs(acc - best_train_acc) < tolerance]
    
#     # Chọn k nhỏ nhất
#     best_k = min(best_ks)

#     print(f"\nCác k đạt accuracy cao nhất (~{best_train_acc:.4f}): {sorted(best_ks)}")
#     print(f"→ Chọn k nhỏ nhất: {best_k} (ưu tiên mô hình đơn giản hơn)")

#     # Đánh giá trên test
#     correct = 0
#     for i in range(len(test_norm)):
#         x_new = test_norm.iloc[i][feature_cols].values
#         y_true = test_norm.iloc[i][label_col]
#         y_pred = knn_predict_weighted(train_norm, x_new, feature_cols, best_k)
#         if y_pred == y_true:
#             correct += 1

#     test_acc = correct / len(test_norm)

#     print("\n=== KẾT LUẬN (Weighted KNN) ===")
#     print(f"Best k (nhỏ nhất trong các k tốt nhất): {best_k}")
#     print(f"Accuracy trung bình trên train (5-Fold CV): {best_train_acc:.4f}")
#     print(f"Accuracy trên tập test:                   {test_acc:.4f}")

#     # Đóng gói model
#     model = {
#         "train_norm": train_norm,
#         "params": params,
#         "feature_cols": feature_cols,
#         "best_k": best_k,
#     }

#     return train_results, best_k, best_train_acc, test_acc, model


# def fit_knn(train_df, k_min=1, k_max=30, n_folds=5):
#     """
#     Huấn luyện Weighted KNN và chọn k tốt nhất bằng K-Fold CV.
#     """
#     feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
#     label_col = "species"

#     train_norm, params = normalize_dataframe(train_df, feature_cols)

#     best_k = k_min
#     best_acc = -1.0
#     for k in range(k_min, k_max + 1):
#         acc = evaluate_knn_kfold(train_norm, feature_cols, label_col, k, n_folds=n_folds)
#         if acc > best_acc:
#             best_acc = acc
#             best_k = k

#     model = {
#         "train_norm": train_norm,
#         "params": params,
#         "feature_cols": feature_cols,
#         "best_k": best_k,
#     }

#     return model


# def predict_single(model, row):
#     """Dự đoán một mẫu mới (Weighted KNN)"""
#     feature_cols = model["feature_cols"]
    
#     if isinstance(row, pd.Series):
#         df_row = row.to_frame().T
#     elif isinstance(row, dict):
#         df_row = pd.DataFrame([row])
#     else:
#         df_row = pd.DataFrame([row], columns=feature_cols)

#     df_norm = apply_normalize(df_row, feature_cols, model["params"])
#     x_new = df_norm.iloc[0][feature_cols].values
    
#     return knn_predict_weighted(model["train_norm"], x_new, feature_cols, model["best_k"])


# def predict_batch(model, df):
#     """Dự đoán nhiều mẫu (Weighted KNN)"""
#     df_norm = apply_normalize(df, model["feature_cols"], model["params"])
#     preds = []
#     for i in range(len(df_norm)):
#         x_new = df_norm.iloc[i][model["feature_cols"]].values
#         preds.append(knn_predict_weighted(model["train_norm"], x_new, model["feature_cols"], model["best_k"]))
#     return preds