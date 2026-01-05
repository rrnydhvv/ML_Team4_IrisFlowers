import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


class SoftMaxClassifier:
    """Softmax Regression Classifier với Mini-Batch Gradient Descent"""
    
    # --- CẤU HÌNH MẶC ĐỊNH ---
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_EPOCHS = 3000
    DEFAULT_BATCH_SIZE = 10
    CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    LABEL_COL = 'species'
    
    def __init__(self, learning_rate=None, epochs=None, batch_size=None, feature_cols=None):
        """
        Khởi tạo SoftMax Classifier
        
        Args:
            learning_rate: Tốc độ học (mặc định: 0.1)
            epochs: Số lần lặp (mặc định: 200)
            batch_size: Kích thước batch (mặc định: 10)
            feature_cols: Danh sách features để sử dụng (mặc định: tất cả)
        """
        self.learning_rate = learning_rate or self.DEFAULT_LEARNING_RATE
        self.epochs = epochs or self.DEFAULT_EPOCHS
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.feature_cols = feature_cols or self.FEATURE_COLS
        
        # Weights và bias sẽ được khởi tạo khi fit
        self.W = None
        self.b = None
        self.losses = []
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
    
    @staticmethod
    def _softmax(z):
        """Hàm softmax để tính xác suất"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _to_onehot(self, y_labels):
        """Chuyển đổi labels sang one-hot encoding"""
        y_onehot = np.zeros((len(y_labels), len(self.CLASS_ORDER)))
        for i, label in enumerate(y_labels):
            if label in self.CLASS_ORDER:
                y_onehot[i, self.CLASS_ORDER.index(label)] = 1.0
        return y_onehot
    
    def load_data(self, train_file, test_file):
        """
        Load dữ liệu từ file CSV train và test riêng biệt
        
        Args:
            train_file: Đường dẫn file train
            test_file: Đường dẫn file test
            
        Returns:
            x_train, y_train, x_test, y_test
        """
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            
            print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
            print(f"Using features: {self.feature_cols}")
            
            x_train = df_train[self.feature_cols].values
            y_train = self._to_onehot(df_train[self.LABEL_COL].values)
            
            x_test = df_test[self.feature_cols].values
            y_test = self._to_onehot(df_test[self.LABEL_COL].values)
            
            return x_train, y_train, x_test, y_test

        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file - {e}")
            return None, None, None, None
    
    def fit(self, X, y):
        """
        Huấn luyện model với Mini-Batch Gradient Descent
        
        Args:
            X: Features (m samples, n features)
            y: Labels one-hot encoded (m samples, k classes)
            
        Returns:
            self
        """
        m, n = X.shape
        k = y.shape[1]
        
        # Khởi tạo weights và bias
        self.W = np.zeros((n, k))
        self.b = np.zeros((1, k))
        self.losses = []
        
        print(f"Training Mini-Batch (Batch Size: {self.batch_size}, Epochs: {self.epochs})...")
        
        for epoch in range(self.epochs):
            # Shuffle dữ liệu
            indices = np.random.permutation(m)
            x_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0

            for i in range(0, m, self.batch_size):
                x_batch = x_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                
                current_batch_size = x_batch.shape[0]
                
                # --- Forward ---
                z = np.dot(x_batch, self.W) + self.b
                y_hat = self._softmax(z)
                
                # --- Loss ---
                batch_loss = -np.mean(np.sum(y_batch * np.log(y_hat + 1e-15), axis=1))
                epoch_loss += batch_loss * current_batch_size
                
                # --- Gradient ---
                error = y_hat - y_batch
                dW = (1 / current_batch_size) * np.dot(x_batch.T, error)
                db = (1 / current_batch_size) * np.sum(error, axis=0)
                
                # --- Update ---
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
                
            # Lưu loss trung bình của cả epoch
            self.losses.append(epoch_loss / m)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {self.losses[-1]:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho từng class
        
        Args:
            X: Features
            
        Returns:
            Xác suất cho mỗi class
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        z = np.dot(X, self.W) + self.b
        return self._softmax(z)
    
    def predict(self, X):
        """
        Dự đoán class cho mỗi sample
        
        Args:
            X: Features
            
        Returns:
            Index của class được dự đoán
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def score(self, X, y):
        """
        Tính độ chính xác trên tập dữ liệu
        
        Args:
            X: Features
            y: Labels one-hot encoded
            
        Returns:
            Accuracy (%)
        """
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true) * 100
    
    def evaluate(self, X, y):
        """
        Đánh giá model và trả về accuracy, predictions, true labels
        
        Args:
            X: Features
            y: Labels one-hot encoded
            
        Returns:
            accuracy, y_pred, y_true
        """
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        acc = np.mean(y_pred == y_true) * 100
        return acc, y_pred, y_true
    
    def save_model(self, filename):
        """
        Lưu model ra file
        
        Args:
            filename: Tên file để lưu
        """
        if not self.is_fitted:
            raise Exception("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        model_data = {
            "weights": self.W,
            "bias": self.b,
            "classes": self.CLASS_ORDER,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "info": f"Softmax Mini-Batch (Size={self.batch_size})"
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
        
        self.W = model_data["weights"]
        self.b = model_data["bias"]
        self.is_fitted = True
        print(f"[OK] Đã load model từ: {filename}")
        return self
    
    def plot_loss(self):
        """Vẽ biểu đồ loss theo epoch"""
        if not self.losses:
            print("Chưa có dữ liệu loss để vẽ.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


# --- CẤU HÌNH ---
base_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(base_dir, '../../data/IRIS_train.csv')
TEST_FILE  = os.path.join(base_dir, '../../data/IRIS_test.csv')
MODEL_FILE = os.path.join(base_dir, 'softmax_model.pkl')


if __name__ == "__main__":
    # Khởi tạo classifier
    clf = SoftMaxClassifier(learning_rate=0.1, epochs=3000, batch_size=10)
    
    # Load dữ liệu
    x_train, y_train, x_test, y_test = clf.load_data(TRAIN_FILE, TEST_FILE)
    
    if x_train is not None:
        # Huấn luyện model
        clf.fit(x_train, y_train)
        
        # Đánh giá trên tập test
        acc, y_pred, y_true = clf.evaluate(x_test, y_test)
        print(f"Độ chính xác trên tập Test: {acc:.2f}%")
        
        # Lưu model
        clf.save_model(MODEL_FILE)