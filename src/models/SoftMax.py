import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
INPUT_FILE = './data/IRIS_cleaned.csv'
MODEL_FILE = './src/models/softmax_model.pkl'
LEARNING_RATE = 0.1
EPOCHS = 200
BATCH_SIZE = 10
CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def load_and_split_data(filename, train_ratio=0.8):
    try:
        df = pd.read_csv(filename)
        # Shuffle dữ liệu ngay từ đầu
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        print(f"Dataset: {len(df)} dòng. Train: {len(df_train)} | Test: {len(df_test)}")
        
        def to_numpy(df_sub):
            x = df_sub.iloc[:, :4].values
            y_labels = df_sub.iloc[:, -1].values
            y_onehot = np.zeros((len(y_labels), len(CLASS_ORDER)))
            for i, label in enumerate(y_labels):
                if label in CLASS_ORDER:
                    y_onehot[i, CLASS_ORDER.index(label)] = 1.0
            return x, y_onehot

        x_train, y_train = to_numpy(df_train)
        x_test, y_test = to_numpy(df_test)
        
        return x_train, y_train, x_test, y_test

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
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

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_split_data(INPUT_FILE)
    
    if x_train is not None:
        W, b, losses = train(x_train, y_train, LEARNING_RATE, EPOCHS, BATCH_SIZE)
        
        z_test = np.dot(x_test, W) + b
        y_pred = np.argmax(softmax(z_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        acc = np.mean(y_pred == y_true) * 100
        print(f"Độ chính xác trên tập Test: {acc:.2f}%")
        
        export_model(W, b, MODEL_FILE)