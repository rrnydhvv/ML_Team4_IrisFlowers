import numpy as np
import csv
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
LEARNING_RATE = 0.01
EPOCHS = 3000
INPUT_FILE_TRAIN = './data/train.csv'
INPUT_FILE_TEST = './data/test.csv'

# Setosa -> [1, 0, 0]
# Versicolor -> [0, 1, 0]
# Virginica -> [0, 0, 1]
CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def to_one_hot(label_str):
    y_vec = np.zeros(len(CLASS_ORDER))
    if label_str in CLASS_ORDER:
        idx = CLASS_ORDER.index(label_str)
        y_vec[idx] = 1.0
    return y_vec

def load_data(filename):
    data_x = []
    data_y = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            
            for row in reader:
                if not row: continue
                
                features = [float(val) for val in row[:-1]]
                data_x.append(features)
                
                label_str = row[-1]
                y_vec = to_one_hot(label_str)
                data_y.append(y_vec)
                
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return None, None

    return np.array(data_x), np.array(data_y)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def fit(x, y, lr=0.1, epochs=1000):
    """
    Huấn luyện mô hình Softmax Regression
    Input:
      - x: Ma trận đặc trưng (Features Matrix)
      - y: Ma trận One-Hot (Labels Matrix)
    """
    m_samples = x.shape[0]
    n_features = x.shape[1]
    k_classes = y.shape[1]
    
    W = np.zeros((n_features, k_classes)) 
    b = np.zeros((1, k_classes))
    
    losses = []
    
    print(f"Đang huấn luyện trên {m_samples} mẫu với {n_features} thuộc tính...")
    
    for epoch in range(epochs):
        # 1. Tính Z (Linear Score)
        # Z = x.W + b
        z = np.dot(x, W) + b
        
        # 2. Tính y_hat (Predicted Probability) qua hàm Softmax
        y_hat = softmax(z)
        
        # 3. Tính Loss (Categorical Cross-Entropy)
        # L = - sum(y * log(y_hat))
        # Thêm 1e-15 để tránh lỗi log(0)
        loss = -np.mean(np.sum(y * np.log(y_hat + 1e-15), axis=1))
        losses.append(loss)
        
        # 4. Tính Gradient (Đạo hàm)
        # Error = y_hat - y (Sự sai lệch giữa dự đoán và thực tế one-hot)
        error = y_hat - y
        
        # dW = (1/m) * x.T . error
        dW = (1/m_samples) * np.dot(x.T, error)
        
        # db = (1/m) * sum(error)
        db = (1/m_samples) * np.sum(error, axis=0)
        
        # 5. Cập nhật tham số (Gradient Descent)
        W -= lr * dW
        b -= lr * db
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.5f}")
            
    return W, b, losses

def predict(x, W, b):
    """Dự đoán lớp dựa trên x và trọng số đã học"""
    z = np.dot(x, W) + b
    y_hat = softmax(z)
    # Trả về vị trí có xác suất cao nhất (0, 1 hoặc 2)
    return np.argmax(y_hat, axis=1)

# --- PHẦN 3: CHẠY CHƯƠNG TRÌNH ---

def main():
    # 1. Load dữ liệu
    print("--- ĐỌC DỮ LIỆU ---")
    x_train, y_train = load_data(INPUT_FILE_TRAIN)
    x_test, y_test = load_data(INPUT_FILE_TEST)
    
    if x_train is None: return

    # In kiểm tra kích thước
    print(f"Kích thước x_train (Features): {x_train.shape}")
    print(f"Kích thước y_train (One-Hot) : {y_train.shape}")
    print(f"Mẫu y_train đầu tiên (One-Hot): {y_train[0]}") # Ví dụ: [1. 0. 0.]
    
    # 2. Huấn luyện
    print("\n--- BẮT ĐẦU TRAINING ---")
    W, b, losses = fit(x_train, y_train, lr=LEARNING_RATE, epochs=EPOCHS)
    
    # 3. Kiểm thử (Testing)
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST ---")
    
    # Chuyển y_test (one-hot) về dạng chỉ số (0, 1, 2) để so sánh
    y_test_indices = np.argmax(y_test, axis=1)
    
    # Dự đoán
    y_pred_indices = predict(x_test, W, b)
    
    # Tính độ chính xác
    accuracy = np.mean(y_test_indices == y_pred_indices) * 100
    print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
    
    # In một vài kết quả chi tiết
    print("\nChi tiết 5 mẫu đầu tiên:")
    print(f"{'x (Features)':<30} | {'y (One-Hot Real)':<20} | {'y_pred (Label)':<10}")
    print("-" * 70)
    for i in range(min(5, len(x_test))):
        print(f"{str(x_test[i]):<30} | {str(y_test[i]):<20} | {y_pred_indices[i]}")

    # Vẽ biểu đồ Loss
    plt.plot(losses)
    plt.title("Hàm mất mát (Loss) theo thời gian")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.show()

if __name__ == "__main__":
    main()