import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'softmax_model.pkl'
LEARNING_RATE = 0.1
EPOCHS = 200
BATCH_SIZE = 10
CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABEL_COL = 'species'

def load_data(train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Load dữ liệu từ file CSV train và test riêng biệt"""
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
        
        def to_numpy(df_sub):
            x = df_sub[FEATURE_COLS].values
            y_labels = df_sub[LABEL_COL].values
            y_onehot = np.zeros((len(y_labels), len(CLASS_ORDER)))
            for i, label in enumerate(y_labels):
                if label in CLASS_ORDER:
                    y_onehot[i, CLASS_ORDER.index(label)] = 1.0
            return x, y_onehot

        x_train, y_train = to_numpy(df_train)
        x_test, y_test = to_numpy(df_test)
        
        return x_train, y_train, x_test, y_test

    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        return None, None, None, None

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- TRAIN ---
def train(x, y, lr, epochs, batch_size):
    m, n = x.shape
    k = y.shape[1]
    
    W = np.zeros((n, k))
    b = np.zeros((1, k))
    losses = []
    
    print(f"Training Mini-Batch (Batch Size: {batch_size}, Epochs: {epochs})...")
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0

        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            
            current_batch_size = x_batch.shape[0]
            
            # --- Forward ---
            z = np.dot(x_batch, W) + b
            y_hat = softmax(z)
            
            # --- Loss ---
            batch_loss = -np.mean(np.sum(y_batch * np.log(y_hat + 1e-15), axis=1))
            epoch_loss += batch_loss * current_batch_size
            
            # --- Gradient (Tính trên 10 mẫu) ---
            error = y_hat - y_batch
            
            # Tính gradient trung bình của batch này
            dW = (1/current_batch_size) * np.dot(x_batch.T, error)
            db = (1/current_batch_size) * np.sum(error, axis=0)
            
            # --- Update ---
            W -= lr * dW
            b -= lr * db
            
        # Lưu loss trung bình của cả epoch
        losses.append(epoch_loss / m)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")
            
    return W, b, losses

def export_model(W, b, filename):
    model_data = {
        "weights": W,
        "bias": b,
        "classes": CLASS_ORDER,
        "info": f"Softmax Mini-Batch (Size={BATCH_SIZE})"
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[OK] Đã xuất model tại: {filename}")

def test(x_test, y_test, W, b):
    """Đánh giá model trên tập test"""
    z_test = np.dot(x_test, W) + b
    y_pred = np.argmax(softmax(z_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = np.mean(y_pred == y_true) * 100
    return acc, y_pred, y_true

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data(TRAIN_FILE, TEST_FILE)
    
    if x_train is not None:
        W, b, losses = train(x_train, y_train, LEARNING_RATE, EPOCHS, BATCH_SIZE)
        
        acc, y_pred, y_true = test(x_test, y_test, W, b)
        print(f"Độ chính xác trên tập Test: {acc:.2f}%")
        
        export_model(W, b, MODEL_FILE)
        
    