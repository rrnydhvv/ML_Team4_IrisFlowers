import numpy as np
import pickle

MODEL_FILE = 'iris_model.pkl'

# Tái sử dụng hàm softmax (hoặc import từ file khác nếu muốn, nhưng viết lại cho gọn)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class IrisPredictor:
    def __init__(self, model_path):
        self.W = None
        self.b = None
        self.classes = []
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.W = data['weights']
                self.b = data['bias']
                self.classes = data['classes']
            print(f"Đã load model thành công từ: {model_path}")
            print(f"Các lớp hỗ trợ: {self.classes}")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file model '{model_path}'. Hãy chạy train.py trước!")
            exit()

    def predict(self, input_features):
        """
        input_features: List hoặc Numpy array 2D chứa các đặc trưng
        Ví dụ: [[5.1, 3.5, 1.4, 0.2]]
        """
        x = np.array(input_features)
        
        # Tính toán Forward pass: Z = X.W + b
        z = np.dot(x, self.W) + self.b
        
        # Tính xác suất
        probs = softmax(z)
        
        # Lấy chỉ số có xác suất cao nhất
        pred_indices = np.argmax(probs, axis=1)
        
        # Map sang tên lớp
        results = []
        for i, idx in enumerate(pred_indices):
            class_name = self.classes[idx]
            confidence = probs[i][idx] * 100
            results.append((class_name, confidence))
            
        return results

# --- MAIN: TEST DỰ ĐOÁN ---
if __name__ == "__main__":
    # Khởi tạo bộ dự đoán
    predictor = IrisPredictor(MODEL_FILE)
    
    print("\n--- NHẬP DỮ LIỆU CẦN DỰ ĐOÁN ---")
    
    # Giả lập dữ liệu mới (Ví dụ: Lấy từ người dùng nhập vào)
    # Mẫu 1: Nhỏ (Setosa), Mẫu 2: Lớn (Virginica)
    new_data = [
        [0.22, 0.62, 0.06, 0.04], 
        [0.70, 0.30, 0.80, 0.90]
    ]
    
    predictions = predictor.predict(new_data)
    
    print(f"\nKết quả dự đoán cho {len(new_data)} mẫu:")
    for i, (name, conf) in enumerate(predictions):
        print(f"Mẫu {i+1}: Hoa {name} (Độ tin cậy: {conf:.2f}%)")