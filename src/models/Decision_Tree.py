import numpy as np
import pandas as pd
import pickle
import os

# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'decision_tree_model.pkl'
FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABEL_COL = 'species'

def load_data(train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Load dữ liệu từ file CSV train và test riêng biệt"""
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
        
        X_train = df_train[FEATURE_COLS].values
        y_train = df_train[LABEL_COL].values
        X_test = df_test[FEATURE_COLS].values
        y_test = df_test[LABEL_COL].values
        
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        return None, None, None, None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, criterion="entropy"):
        if criterion not in ("entropy", "gini"):
            raise ValueError("criterion must be 'entropy' or 'gini'")
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    # ================= PUBLIC API =================
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

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

# ================= EXPORT / LOAD  =================
def export_model(model, filename):
    model_data = {
        "model": model,
        "criterion": model.criterion,
        "max_depth": model.max_depth,
        "info": "Decision Tree Classifier from scratch"
    }
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[OK] Đã xuất Decision Tree model tại: {filename}")


def load_model(filename):
    with open(filename, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"]
def train(X_train, y_train, max_depth=None, criterion="entropy"):
    """Train Decision Tree model"""
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    model.fit(X_train, y_train)
    
    # Tính accuracy trên tập train
    y_pred_train = model.predict(X_train)
    train_acc = np.mean(y_pred_train == y_train)
    print(f"Accuracy trên tập train: {train_acc:.4f}")
    
    return model

def test(model, X_test, y_test):
    """Test Decision Tree model"""
    y_pred = model.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"Accuracy trên tập test: {test_acc:.4f}")
    return test_acc, y_pred

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE)
    
    if X_train is not None:
        model = train(X_train, y_train, max_depth=None, criterion="entropy")
        test_acc, y_pred = test(model, X_test, y_test)
        export_model(model, MODEL_FILE)