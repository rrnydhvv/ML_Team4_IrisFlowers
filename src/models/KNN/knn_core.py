"""
Core KNN prediction function with distance weighting.
"""

import pandas as pd
import numpy as np


def knn_predict_weighted(train_df, x_new, feature_cols, k):
    """
    Dự đoán class mẫu mới với trọng số nghịch đảo khoảng cách (Weighted KNN)
    """
    dist = np.zeros(len(train_df))

    for i, col in enumerate(feature_cols):
        dist += (train_df[col].values - x_new[i]) ** 2

    dist = np.sqrt(dist)
    train_df = train_df.copy()
    train_df["dist"] = dist

    neighbors = train_df.sort_values("dist").head(k)
    
    epsilon = 1e-10
    neighbors = neighbors.copy()
    neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)
    
    class_weights = neighbors.groupby("species")["weight"].sum()
    pred_class = class_weights.idxmax()

    return pred_class
