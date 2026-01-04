import numpy as np
import pandas as pd
import pickle
import os


class DecisionTreeClassifier:
    """Decision Tree Classifier với Entropy/Gini"""
    
    # --- CẤU HÌNH MẶC ĐỊNH ---
    FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    LABEL_COL = 'species'
    
    def __init__(self, max_depth=None, criterion="entropy"):
        """
        Khởi tạo Decision Tree Classifier
        
        Args:
            max_depth: Độ sâu tối đa của cây (None = không giới hạn)
            criterion: Tiêu chí phân chia ('entropy' hoặc 'gini')
        """
        if criterion not in ("entropy", "gini"):
            raise ValueError("criterion must be 'entropy' or 'gini'")
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None
        self.feature_cols = self.FEATURE_COLS
        self.label_col = self.LABEL_COL
        self.is_fitted = False

    def load_data(self, train_file, test_file):
        """
        Load dữ liệu từ file CSV train và test riêng biệt
        
        Args:
            train_file: Đường dẫn file train
            test_file: Đường dẫn file test
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            
            print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
            
            X_train = df_train[self.feature_cols].values
            y_train = df_train[self.label_col].values
            X_test = df_test[self.feature_cols].values
            y_test = df_test[self.label_col].values
            
            return X_train, y_train, X_test, y_test
            
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file - {e}")
            return None, None, None, None

    # ================= PUBLIC API =================
    def fit(self, X, y):
        """
        Huấn luyện model Decision Tree
        
        Args:
            X: Features (m samples, n features)
            y: Labels
            
        Returns:
            self
        """
        print(f"Training Decision Tree (max_depth={self.max_depth}, criterion={self.criterion})...")
        self.tree = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        
        # Tính accuracy trên tập train
        y_pred_train = self.predict(X)
        train_acc = np.mean(y_pred_train == y)
        print(f"Accuracy trên tập train: {train_acc:.4f}")
        
        return self

    def predict(self, X):
        """
        Dự đoán class cho nhiều samples
        
        Args:
            X: Features
            
        Returns:
            Array các class được dự đoán
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def score(self, X, y):
        """
        Tính độ chính xác trên tập dữ liệu
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Accuracy (0-1)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def evaluate(self, X, y):
        """
        Đánh giá model và trả về accuracy, predictions, true labels
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            accuracy, y_pred, y_true
        """
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        print(f"Accuracy trên tập test: {acc:.4f}")
        return acc, y_pred, y
    
    def save_model(self, filename):
        """
        Lưu model ra file
        
        Args:
            filename: Tên file để lưu
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        model_data = {
            "tree": self.tree,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "feature_cols": self.feature_cols,
            "info": "Decision Tree Classifier from scratch"
        }
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, "wb") as f:
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
        with open(filename, "rb") as f:
            model_data = pickle.load(f)
        
        self.tree = model_data["tree"]
        self.criterion = model_data["criterion"]
        self.max_depth = model_data["max_depth"]
        self.feature_cols = model_data.get("feature_cols", self.FEATURE_COLS)
        self.is_fitted = True
        
        print(f"[OK] Đã load model từ: {filename}")
        return self

    # ================= TREE BUILD =================
    def _build_tree(self, X, y, depth):
        if self._is_pure(y):
            return {"value": y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return {"value": self._majority_class(y)}

        if len(y) == 0:
            return {"value": None}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"value": self._majority_class(y)}

        LX, Ly, RX, Ry = self._split(X, y, feature, threshold)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(LX, Ly, depth + 1),
            "right": self._build_tree(RX, Ry, depth + 1)
        }

    # ================= SPLIT LOGIC =================
    def _best_split(self, X, y):
        best_gain, best_f, best_t = 0, None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                gain = self._information_gain(X, y, f, t)
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t
        return best_f, best_t

    def _information_gain(self, X, y, f, t):
        impurity = self._entropy if self.criterion == "entropy" else self._gini
        parent = impurity(y)
        LX, Ly, RX, Ry = self._split(X, y, f, t)
        if len(Ly) == 0 or len(Ry) == 0:
            return 0
        n = len(y)
        child = (len(Ly)/n)*impurity(Ly) + (len(Ry)/n)*impurity(Ry)
        return parent - child

    # ================= METRICS =================
    def _entropy(self, y):
        _, cnt = np.unique(y, return_counts=True)
        p = cnt / len(y)
        return -np.sum(p * np.log2(p))

    def _gini(self, y):
        _, cnt = np.unique(y, return_counts=True)
        p = cnt / len(y)
        return 1 - np.sum(p**2)

    # ================= UTILS =================
    def _split(self, X, y, f, t):
        mask = X[:, f] <= t
        return X[mask], y[mask], X[~mask], y[~mask]

    def _is_pure(self, y):
        return len(np.unique(y)) == 1

    def _majority_class(self, y):
        v, c = np.unique(y, return_counts=True)
        return v[np.argmax(c)]

    # ================= PREDICT =================
    def _predict_one(self, x, node):
        if "value" in node:
            return node["value"]
        return self._predict_one(
            x,
            node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        )


# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'decision_tree_model.pkl'


if __name__ == "__main__":
    # Khởi tạo classifier
    clf = DecisionTreeClassifier(max_depth=None, criterion="entropy")
    
    # Load dữ liệu
    X_train, y_train, X_test, y_test = clf.load_data(TRAIN_FILE, TEST_FILE)
    
    if X_train is not None:
        # Huấn luyện model
        clf.fit(X_train, y_train)
        
        # Đánh giá trên tập test
        acc, y_pred, y_true = clf.evaluate(X_test, y_test)
        
        # Lưu model
        clf.save_model(MODEL_FILE)