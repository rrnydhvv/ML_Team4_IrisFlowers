import numpy as np
import pandas as pd
import pickle


class GaussianNaiveBayesClassifier:
    """Gaussian Naive Bayes Classifier"""
    
    # --- CẤU HÌNH MẶC ĐỊNH ---
    FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    LABEL_COL = 'species'
    
    def __init__(self, feature_cols=None):
        """Khởi tạo Gaussian Naive Bayes Classifier
        
        Args:
            feature_cols: Danh sách features để sử dụng (mặc định: tất cả)
        """
        self.classes = None  # C
        self.means = {}  # {class: [mean_feature1, mean_feature2, ...]}
        self.vars = {}  # {class: [var_feature1, var_feature2, ...]} phương sai
        self.priors = {}  # {class: prior_probability} tiên nghiệm
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.label_col = self.LABEL_COL
        self.is_fitted = False

    def set_feature_cols(self, feature_cols):
        """
        Đặt lại danh sách features (hữu ích cho ablation study)
        
        Args:
            feature_cols: Danh sách tên features
            
        Returns:
            self
        """
        self.feature_cols = feature_cols
        print(f"Features updated: {self.feature_cols}")
        return self

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
            print(f"Using features: {self.feature_cols}")
            
            X_train = df_train[self.feature_cols].values
            y_train = df_train[self.label_col].values
            X_test = df_test[self.feature_cols].values
            y_test = df_test[self.label_col].values
            
            return X_train, y_train, X_test, y_test
            
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file - {e}")
            return None, None, None, None

    def fit(self, X, y):
        """
        Huấn luyện model Gaussian Naive Bayes
        
        Args:
            X: Features (m samples, n features)
            y: Labels
            
        Returns:
            self
        """
        print("Training Gaussian Naive Bayes...")
        
        self.classes = np.unique(y)
        X = np.array(X)
        y = np.array(y)

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
        
        self.is_fitted = True
        
        # Tính accuracy trên tập train
        y_pred_train = self.predict(X)
        train_acc = np.mean(np.array(y_pred_train) == y)
        print(f"Accuracy trên tập train: {train_acc:.4f}")
        
        return self

    def _gaussian_likelihood(self, x, mean, var):
        """Tính Gaussian likelihood"""
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        """
        Dự đoán class cho nhiều samples
        
        Args:
            X: Features
            
        Returns:
            List các class được dự đoán
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        X = np.array(X)
        y_pred = []

        for x in X:
            class_probs = {}

            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._gaussian_likelihood(x, self.means[c], self.vars[c])))
                class_probs[c] = prior + likelihood

            best_class = max(class_probs, key=class_probs.get)
            y_pred.append(best_class)

        return y_pred

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
        return np.mean(np.array(y_pred) == np.array(y))

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
        acc = np.mean(np.array(y_pred) == np.array(y))
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
            "classes": self.classes.tolist() if self.classes is not None else None,
            "means": {k: v.tolist() for k, v in self.means.items()},
            "vars": {k: v.tolist() for k, v in self.vars.items()},
            "priors": self.priors,
            "feature_cols": self.feature_cols,
            "info": "Gaussian Naive Bayes Classifier from scratch"
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
        
        self.classes = np.array(model_data["classes"])
        self.means = {k: np.array(v) for k, v in model_data["means"].items()}
        self.vars = {k: np.array(v) for k, v in model_data["vars"].items()}
        self.priors = model_data["priors"]
        self.feature_cols = model_data.get("feature_cols", self.FEATURE_COLS)
        self.is_fitted = True
        
        print(f"[OK] Đã load model từ: {filename}")
        return self


# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'naive_bayes_model.pkl'


if __name__ == "__main__":
    # Khởi tạo classifier
    clf = GaussianNaiveBayesClassifier()
    
    # Load dữ liệu
    X_train, y_train, X_test, y_test = clf.load_data(TRAIN_FILE, TEST_FILE)
    
    if X_train is not None:
        # Huấn luyện model
        clf.fit(X_train, y_train)
        
        # Đánh giá trên tập test
        acc, y_pred, y_true = clf.evaluate(X_test, y_test)
        
        # Lưu model
        clf.save_model(MODEL_FILE)