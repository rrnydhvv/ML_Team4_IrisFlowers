"""
KNN model training and prediction functions.
"""

import pandas as pd
from .knn_core import knn_predict_weighted
from .knn_cv import evaluate_knn_kfold


def fit_knn(train_df, k_min=1, k_max=30, n_folds=5):
    """
    Huấn luyện Weighted KNN và chọn k tốt nhất bằng K-Fold CV.
    
    """
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_col = "species"

    best_k = k_min
    best_acc = -1.0
    for k in range(k_min, k_max + 1):
        acc = evaluate_knn_kfold(train_df, feature_cols, label_col, k, n_folds=n_folds)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    model = {
        "train_norm": train_df,
        "feature_cols": feature_cols,
        "best_k": best_k,
    }

    return model


def predict_single(model, row):
    """
    Dự đoán một mẫu mới (Weighted KNN).
    """
    feature_cols = model["feature_cols"]
    
    if isinstance(row, pd.Series):
        df_row = row.to_frame().T
    elif isinstance(row, dict):
        df_row = pd.DataFrame([row])
    else:
        df_row = pd.DataFrame([row], columns=feature_cols)

    x_new = df_row.iloc[0][feature_cols].values
    return knn_predict_weighted(model["train_norm"], x_new, feature_cols, model["best_k"])


def predict_batch(model, df):
    """
    Dự đoán nhiều mẫu (Weighted KNN).
    """
    preds = []
    for i in range(len(df)):
        x_new = df.iloc[i][model["feature_cols"]].values
        preds.append(knn_predict_weighted(model["train_norm"], x_new, model["feature_cols"], model["best_k"]))
    return preds
