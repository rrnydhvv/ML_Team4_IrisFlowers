import pandas as pd
import numpy as np

from src.preprocess import normalize_dataframe, apply_normalize


def knn_predict(train_df, x_new, feature_cols, k):
    dist = np.zeros(len(train_df))

    for i, col in enumerate(feature_cols):
        dist += (train_df[col].values - x_new[i]) ** 2

    dist = np.sqrt(dist)
    train_df = train_df.copy()
    train_df["dist"] = dist

    neighbors = train_df.sort_values("dist").head(k)
    pred_class = neighbors["species"].mode()[0]

    return pred_class


def evaluate_knn(df, feature_cols, label_col, k):
    correct = 0

    for i in range(len(df)):
        test_row = df.iloc[i]
        train_df = df.drop(index=df.index[i])

        x_new = test_row[feature_cols].values
        y_true = test_row[label_col]

        y_pred = knn_predict(train_df, x_new, feature_cols, k)

        if y_pred == y_true:
            correct += 1

    return correct / len(df)


def run_knn_train_test(df, test_size=0.2, random_state=42):
    """Split `df` into train/test, normalize based on train, then evaluate KNN on test set.

    Returns: (results_list, best_k, best_acc)
    where results_list is [(k, accuracy_on_test), ...]
    """
    feature_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    label_col = "species"

    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_size))
    train_df = df_shuffled.iloc[:-n_test].reset_index(drop=True)
    test_df = df_shuffled.iloc[-n_test:].reset_index(drop=True)

    # Normalize using train params
    train_norm, params = normalize_dataframe(train_df, feature_cols)
    test_norm = apply_normalize(test_df, feature_cols, params)

    # First: choose best k using only the training set (leave-one-out on train)
    train_results = []
    print("=== CHON K TREN TAP TRAIN (LOO) ===")
    for k in range(1, 31):
        acc_train = evaluate_knn(train_norm, feature_cols, label_col, k)
        train_results.append((k, acc_train))
        print(f"k = {k:2d} | Accuracy (train, LOO) = {acc_train:.4f}")

    best_k, best_train_acc = max(train_results, key=lambda x: x[1])

    # Build model object to use for prediction
    model = {
        "train_norm": train_norm,
        "params": params,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "best_k": best_k,
    }

    # Then: evaluate that best_k on the held-out test set
    correct = 0
    for i in range(len(test_norm)):
        x_new = test_norm.iloc[i][feature_cols].values
        y_true = test_norm.iloc[i][label_col]
        y_pred = knn_predict(train_norm, x_new, feature_cols, best_k)
        if y_pred == y_true:
            correct += 1

    test_acc = correct / len(test_norm)

    print("\n=== KET LUAN ===")
    print("Best k (selected on train):", best_k)
    print("Train accuracy (LOO) for best k:", best_train_acc)
    print("Test accuracy for best k:", test_acc)

    return train_results, best_k, best_train_acc, test_acc, model


def fit_knn(train_df, k_min=1, k_max=30):
    """Fit KNN on training dataframe and select best k using LOO.

    Returns a model dict containing normalized train, params, feature/label cols and best_k.
    """
    feature_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    label_col = "species"

    train_norm, params = normalize_dataframe(train_df, feature_cols)

    best_k = k_min
    best_acc = -1.0
    for k in range(k_min, k_max + 1):
        acc = evaluate_knn(train_norm, feature_cols, label_col, k)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    model = {
        "train_norm": train_norm,
        "params": params,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "best_k": best_k,
    }

    return model


def predict_single(model, row):
    """Predict label for a single sample.

    `row` can be a pandas Series, dict, or list/array of feature values (in feature_cols order).
    """
    import pandas as pd

    feature_cols = model["feature_cols"]

    if isinstance(row, pd.Series):
        df_row = row.to_frame().T
    elif isinstance(row, dict):
        df_row = pd.DataFrame([row])
    else:
        # assume array-like in correct order
        df_row = pd.DataFrame([row], columns=feature_cols)

    df_norm = apply_normalize(df_row, feature_cols, model["params"])
    x_new = df_norm.iloc[0][feature_cols].values
    return knn_predict(model["train_norm"], x_new, feature_cols, model["best_k"])


def predict_batch(model, df):
    """Predict labels for a dataframe of samples (returns list of predictions)."""
    df_norm = apply_normalize(df, model["feature_cols"], model["params"])
    preds = []
    for i in range(len(df_norm)):
        x_new = df_norm.iloc[i][model["feature_cols"]].values
        preds.append(knn_predict(model["train_norm"], x_new, model["feature_cols"], model["best_k"]))
    return preds
