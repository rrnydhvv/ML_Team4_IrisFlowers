import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

INPUT_FILE = './data/IRIS_cleaned.csv'
MODEL_FILE = './src/models/iris_model.pkl'
LEARNING_RATE = 0.2
EPOCHS = 5000
CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def load_and_split_data(filename, train_ratio=0.8):
    try:
        df = pd.read_csv(filename)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        print(f"Dataset: {len(df)} dòng. Train: {len(df_train)} | Test: {len(df_test)}")
        
        def to_numpy(df_sub):
            x = df_sub.iloc[:, :4].values
            y_labels = df_sub.iloc[:, -1].values
            
            # One-Hot Encoding
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

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def train(x, y, lr, epochs):
    m, n = x.shape
    k = y.shape[1]
    
    W = np.zeros((n, k))
    b = np.zeros((1, k))
    losses = []
    
    print(f"Đang training ({epochs} epochs)...")
    for epoch in range(epochs):
        # Forward
        z = np.dot(x, W) + b
        y_hat = softmax(z)
        
        # Loss
        loss = -np.mean(np.sum(y * np.log(y_hat + 1e-15), axis=1))
        losses.append(loss)
        
        # Gradient & Update
        error = y_hat - y
        W -= lr * (1/m) * np.dot(x.T, error)
        b -= lr * (1/m) * np.sum(error, axis=0)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
    return W, b, losses

# Export model
def export_model(W, b, filename):
    model_data = {
        "weights": W,
        "bias": b,
        "classes": CLASS_ORDER,
        "info": "Softmax Regression from scratch"
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[OK] Đã xuất model tại: {filename}")

if __name__ == "__main__":
    # Load Data
    x_train, y_train, x_test, y_test = load_and_split_data(INPUT_FILE)
    
    if x_train is not None:
        # Train
        W, b, losses = train(x_train, y_train, LEARNING_RATE, EPOCHS)
        
        # Đánh giá nhanh
        z_test = np.dot(x_test, W) + b
        y_pred = np.argmax(softmax(z_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        acc = np.mean(y_pred == y_true) * 100
        print(f"Độ chính xác trên tập Test: {acc:.2f}%")
        
        # Export Model
        export_model(W, b, MODEL_FILE)