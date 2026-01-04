import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'knn_model.pkl'
FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABEL_COL = 'species'

def load_data(train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Load dữ liệu từ file CSV train và test riêng biệt"""
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        # Xử lý tên cột có khoảng trắng
        df_train.columns = df_train.columns.str.strip()
        df_test.columns = df_test.columns.str.strip()
        
        print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
        return df_train, df_test
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        return None, None

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


def run_knn_train_test(train_df, test_df, random_state=42):
    """Chạy toàn bộ pipeline train và test KNN với train/test dataframes riêng"""
    # Xử lý tên cột có khoảng trắng
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    feature_cols = FEATURE_COLS
    label_col = LABEL_COL

    train_norm = train_df.reset_index(drop=True)
    test_norm = test_df.reset_index(drop=True)

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

    y_true_list = []
    y_pred_list = []

    correct = 0
    for i in range(len(test_norm)):
        x_new = test_norm.iloc[i][feature_cols].values
        y_true = test_norm.iloc[i][label_col]
        y_pred = knn_predict_weighted(train_norm, x_new, feature_cols, best_k)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

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
        "y_true_test": y_true_list,
        "y_pred_test": y_pred_list
    }

    return (train_results, best_k, best_train_acc, test_acc, model, y_true_list, y_pred_list)

def save_model_KNN(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model_KNN(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def plot_accuracy_vs_k(train_results, best_k):
    ks = [k for k, acc in train_results]
    accs = [acc for k, acc in train_results]

    plt.figure()
    plt.plot(ks, accs, marker='o')
    plt.scatter(best_k, accs[best_k - 1])
    plt.axvline(best_k)

    plt.xlabel("K")
    plt.ylabel("Accuracy (5-Fold CV)")
    plt.title("Accuracy theo K (KNN Weighted)")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels):
    n = len(class_labels)
    cm = np.zeros((n, n), dtype=int)

    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    for yt, yp in zip(y_true, y_pred):
        i = label_to_index[yt]
        j = label_to_index[yp]
        cm[i, j] += 1

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Số lượng mẫu")

    plt.xticks(range(n), class_labels, rotation=45)
    plt.yticks(range(n), class_labels)

    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.title("Ma trận nhầm lẫn (KNN có trọng số)")

    for i in range(n):
        for j in range(n):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    
    if train_df is not None and test_df is not None:
        train_results, best_k, best_train_acc, test_acc, model, y_true_list, y_pred_list = run_knn_train_test(train_df, test_df)
        save_model_KNN(model, MODEL_FILE)
        print(f"\n[OK] Đã lưu model tại: {MODEL_FILE}")

