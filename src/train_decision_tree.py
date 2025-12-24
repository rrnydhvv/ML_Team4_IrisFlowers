import numpy as np
import os
from src.models.Decision_Tree import DecisionTreeClassifier, load_iris_data

def train_and_evaluate_decision_tree(data_path='data/IRIS_cleaned.csv', criterion='entropy', max_depth=5, test_size=0.2):
    """
    Hàm huấn luyện và đánh giá Decision Tree.
    
    Args:
        data_path (str): Đường dẫn đến file dữ liệu.
        criterion (str): Tiêu chí ('entropy' hoặc 'gini').
        max_depth (int): Độ sâu tối đa của cây.
        test_size (float): Tỷ lệ test (0-1).
    
    Returns:
        float: Độ chính xác trên tập test.
    """
    # Đọc dữ liệu
    X, y = load_iris_data(data_path)

    # Xáo trộn dữ liệu để đảm bảo phân chia ngẫu nhiên
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Chia train/test
    n_train = int((1 - test_size) * X.shape[0])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Huấn luyện mô hình
    print(f"Sử dụng criterion: {criterion}")
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf.fit(X_train, y_train)

    # Dự đoán trên tập test
    predictions = clf.predict(X_test)

    # In kết quả theo định dạng yêu cầu
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")
    print()

    # In kết quả dự đoán từng sample
    for i, (true, pred) in enumerate(zip(y_test, predictions), 1):
        status = "✓" if true == pred else "✗"
        print(f"Sample {i}: True = {true} | Predicted = {pred} {status}")

    # In các mẫu bị dự đoán sai
    print("\nMisclassified samples:")
    for i, (true, pred, x) in enumerate(zip(y_test, predictions, X_test), 1):
        if true != pred:
            print(f"Sample {i}: Features = {x}, True = {true}, Pred = {pred}")


    # Tính độ chính xác
    correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
    accuracy = correct / len(y_test)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    return accuracy

if __name__ == "__main__":
    # Chạy với entropy
    print("=== ENTROPY ===")
    acc_entropy = train_and_evaluate_decision_tree(criterion='entropy')
    
    # Chạy với gini
    print("\n=== GINI ===")
    acc_gini = train_and_evaluate_decision_tree(criterion='gini')
    
    # So sánh
    print("\n=== SO SÁNH ===")
    print(f"Entropy Accuracy: {acc_entropy:.2%}")
    print(f"Gini Accuracy: {acc_gini:.2%}")
    if acc_entropy > acc_gini:
        print("Entropy tốt hơn.")
    elif acc_gini > acc_entropy:
        print("Gini tốt hơn.")
    else:
        print("Cả hai tương đương.")