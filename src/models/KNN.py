import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


class KNNClassifier:
    """K-Nearest Neighbors Classifier với Weighted Distance"""
    
    # --- CẤU HÌNH MẶC ĐỊNH ---
    FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    LABEL_COL = 'species'
    
    def __init__(self, k=None, random_state=42):
        """
        Khởi tạo KNN Classifier
        
        Args:
            k: Số láng giềng (nếu None sẽ tự động chọn bằng cross-validation)
            random_state: Random seed cho reproducibility
        """
        self.k = k
        self.random_state = random_state
        self.best_k = None
        self.train_data = None
        self.feature_cols = self.FEATURE_COLS
        self.label_col = self.LABEL_COL
        self.is_fitted = False
        self.train_results = []
        self.y_true_test = []
        self.y_pred_test = []
    
    def load_data(self, train_file, test_file):
        """
        Load dữ liệu từ file CSV train và test riêng biệt
        
        Args:
            train_file: Đường dẫn file train
            test_file: Đường dẫn file test
            
        Returns:
            train_df, test_df
        """
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
    
    def _predict_single(self, x_new):
        """
        Dự đoán class cho một sample sử dụng weighted KNN
        
        Args:
            x_new: Feature vector của sample cần dự đoán
            
        Returns:
            Class được dự đoán
        """
        dist = np.zeros(len(self.train_data))
        for i, col in enumerate(self.feature_cols):
            dist += (self.train_data[col].values - x_new[i]) ** 2
        dist = np.sqrt(dist)

        df_tmp = self.train_data.copy()
        df_tmp["dist"] = dist

        neighbors = df_tmp.sort_values("dist").head(self.best_k)
        epsilon = 1e-10
        neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)

        class_weights = neighbors.groupby(self.label_col)["weight"].sum()
        return class_weights.idxmax()
    
    def _evaluate_kfold(self, df, k, n_folds=5):
        """
        Đánh giá accuracy với K-Fold Cross-Validation
        
        Args:
            df: DataFrame dữ liệu
            k: Số láng giềng
            n_folds: Số fold
            
        Returns:
            Accuracy trung bình
        """
        df_shuffled = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
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
                x_new = test_fold.iloc[i][self.feature_cols].values
                y_true = test_fold.iloc[i][self.label_col]
                
                # Tính khoảng cách và dự đoán
                dist = np.zeros(len(train_fold))
                for j, col in enumerate(self.feature_cols):
                    dist += (train_fold[col].values - x_new[j]) ** 2
                dist = np.sqrt(dist)

                df_tmp = train_fold.copy()
                df_tmp["dist"] = dist
                neighbors = df_tmp.sort_values("dist").head(k)
                epsilon = 1e-10
                neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)
                class_weights = neighbors.groupby(self.label_col)["weight"].sum()
                y_pred = class_weights.idxmax()
                
                if y_pred == y_true:
                    correct += 1

            accuracies.append(correct / len(test_fold))

        return np.mean(accuracies)
    
    def fit(self, train_df, k_range=range(1, 31), n_folds=5):
        """
        Huấn luyện model KNN
        
        Args:
            train_df: DataFrame dữ liệu train
            k_range: Phạm vi k để tìm kiếm (mặc định 1-30)
            n_folds: Số fold cho cross-validation
            
        Returns:
            self
        """
        # Xử lý tên cột
        train_df.columns = train_df.columns.str.strip()
        self.train_data = train_df.reset_index(drop=True)
        
        if self.k is not None:
            # Nếu đã chỉ định k, sử dụng luôn
            self.best_k = self.k
            print(f"Sử dụng k = {self.best_k} (đã chỉ định)")
        else:
            # Tìm k tốt nhất bằng cross-validation
            print("Chọn K bằng 5-Fold Cross-Validation trên tập train:")
            print(f"{'k':>3} | {'Accuracy (CV)':>15}")
            print("-" * 25)

            self.train_results = []
            for k in k_range:
                acc = self._evaluate_kfold(self.train_data, k, n_folds)
                self.train_results.append((k, acc))
                print(f"{k:>3} | {acc:>15.4f}")

            best_train_acc = max(acc for k, acc in self.train_results)
            tolerance = 1e-6
            best_ks = [k for k, acc in self.train_results if abs(acc - best_train_acc) < tolerance]
            self.best_k = min(best_ks)

            print(f"\nCác k đạt accuracy cao nhất (~{best_train_acc:.4f}): {sorted(best_ks)}")
            print(f"→ Chọn k nhỏ nhất: {self.best_k}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Dự đoán class cho nhiều samples
        
        Args:
            X: DataFrame hoặc array các features
            
        Returns:
            List các class được dự đoán
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        predictions = []
        
        if isinstance(X, pd.DataFrame):
            for i in range(len(X)):
                x_new = X.iloc[i][self.feature_cols].values
                pred = self._predict_single(x_new)
                predictions.append(pred)
        else:
            for x_new in X:
                pred = self._predict_single(x_new)
                predictions.append(pred)
        
        return predictions
    
    def score(self, test_df):
        """
        Tính accuracy trên tập test
        
        Args:
            test_df: DataFrame dữ liệu test
            
        Returns:
            Accuracy (0-1)
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        test_df.columns = test_df.columns.str.strip()
        test_norm = test_df.reset_index(drop=True)
        
        y_pred = self.predict(test_norm)
        y_true = test_norm[self.label_col].tolist()
        
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        return correct / len(test_norm)
    
    def evaluate(self, test_df):
        """
        Đánh giá model và trả về accuracy, predictions, true labels
        
        Args:
            test_df: DataFrame dữ liệu test
            
        Returns:
            accuracy, y_pred, y_true
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        test_df.columns = test_df.columns.str.strip()
        test_norm = test_df.reset_index(drop=True)
        
        self.y_pred_test = self.predict(test_norm)
        self.y_true_test = test_norm[self.label_col].tolist()
        
        correct = sum(1 for yt, yp in zip(self.y_true_test, self.y_pred_test) if yt == yp)
        acc = correct / len(test_norm)
        
        print("\nKết luận (KNN Weighted):")
        print(f"Best k: {self.best_k}")
        print(f"Accuracy trên tập test: {acc:.4f}")
        
        return acc, self.y_pred_test, self.y_true_test
    
    def save_model(self, filename):
        """
        Lưu model ra file
        
        Args:
            filename: Tên file để lưu
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        model_data = {
            "train_data": self.train_data,
            "feature_cols": self.feature_cols,
            "best_k": self.best_k,
            "train_results": self.train_results,
            "y_true_test": self.y_true_test,
            "y_pred_test": self.y_pred_test
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n[OK] Đã xuất model tại: {filename}")
    
    def load_model(self, filename):
        """
        Load model từ file
        
        Args:
            filename: Tên file chứa model
            
        Returns:
            self
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.train_data = model_data["train_data"]
        self.feature_cols = model_data["feature_cols"]
        self.best_k = model_data["best_k"]
        self.train_results = model_data.get("train_results", [])
        self.y_true_test = model_data.get("y_true_test", [])
        self.y_pred_test = model_data.get("y_pred_test", [])
        self.is_fitted = True
        
        print(f"[OK] Đã load model từ: {filename}")
        return self
    
    def plot_accuracy_vs_k(self):
        """Vẽ biểu đồ accuracy theo k"""
        if not self.train_results:
            print("Chưa có dữ liệu train_results để vẽ.")
            return
        
        ks = [k for k, acc in self.train_results]
        accs = [acc for k, acc in self.train_results]

        plt.figure(figsize=(10, 6))
        plt.plot(ks, accs, marker='o')
        plt.scatter(self.best_k, accs[self.best_k - 1], color='red', s=100, zorder=5)
        plt.axvline(self.best_k, color='red', linestyle='--', alpha=0.5)

        plt.xlabel("K")
        plt.ylabel("Accuracy (5-Fold CV)")
        plt.title("Accuracy theo K (KNN Weighted)")
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(self, class_labels=None):
        """
        Vẽ confusion matrix
        
        Args:
            class_labels: Danh sách các class labels
        """
        if not self.y_true_test or not self.y_pred_test:
            print("Chưa có dữ liệu để vẽ confusion matrix. Hãy gọi evaluate() trước.")
            return
        
        if class_labels is None:
            class_labels = sorted(list(set(self.y_true_test)))
        
        n = len(class_labels)
        cm = np.zeros((n, n), dtype=int)

        label_to_index = {label: idx for idx, label in enumerate(class_labels)}

        for yt, yp in zip(self.y_true_test, self.y_pred_test):
            i = label_to_index[yt]
            j = label_to_index[yp]
            cm[i, j] += 1

        plt.figure(figsize=(8, 6))
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


# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'knn_model.pkl'


if __name__ == "__main__":
    # Khởi tạo classifier
    clf = KNNClassifier(random_state=42)
    
    # Load dữ liệu
    train_df, test_df = clf.load_data(TRAIN_FILE, TEST_FILE)
    
    if train_df is not None and test_df is not None:
        # Huấn luyện model (tự động tìm k tốt nhất)
        clf.fit(train_df)
        
        # Đánh giá trên tập test
        acc, y_pred, y_true = clf.evaluate(test_df)
        
        # Lưu model
        clf.save_model(MODEL_FILE)

