

import numpy as np
from .knn_core import knn_predict_weighted


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
        
        fold_acc = correct / len(test_fold)
        accuracies.append(fold_acc)
    
    return np.mean(accuracies)
