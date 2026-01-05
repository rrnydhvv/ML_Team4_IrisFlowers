# SoftMax Regression (SoftmaxClassifier)

Tài liệu này mô tả cách sử dụng và cơ chế hoạt động của **SoftMax Regression** trong dự án.

- Source: `src/models/SoftMax.py`
- Class chính: `SoftMaxClassifier`
- Kiểu bài toán: phân loại đa lớp (3 lớp Iris)

---

## 1) Tổng quan

Mô hình Softmax Regression (còn gọi là Multinomial Logistic Regression) học một ánh xạ tuyến tính $z = XW + b$, sau đó áp dụng **softmax** để thu được xác suất theo lớp.

- Input: vector đặc trưng 4 chiều (sepal/petal)
- Output: xác suất 3 lớp và nhãn dự đoán (dạng index)

Trong code, mô hình được train bằng **Mini-Batch Gradient Descent**.

---

## 2) Quy ước dữ liệu

### 2.1 Feature columns

Mặc định mô hình dùng 4 đặc trưng:

```text
['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

Bạn có thể đổi tập feature (phục vụ ablation study) bằng `set_feature_cols()`.

### 2.2 Class order

Thứ tự lớp cố định:

```text
['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
```

Quan trọng: `predict()` trả về **index** theo đúng thứ tự này.

---

## 3) API chính

### 3.1 Khởi tạo

```python
from src.models.SoftMax import SoftMaxClassifier

clf = SoftMaxClassifier(
    learning_rate=0.1,
    epochs=3000,
    batch_size=10,
    feature_cols=['sepal_length','sepal_width','petal_length','petal_width']
)
```

### 3.2 Load dữ liệu (helper)

Hàm `load_data(train_file, test_file)` đọc CSV và trả về:

- `x_train`: `(n_train, n_features)`
- `y_train`: one-hot `(n_train, 3)`
- `x_test`: `(n_test, n_features)`
- `y_test`: one-hot `(n_test, 3)`

```python
x_train, y_train, x_test, y_test = clf.load_data('data/IRIS_train.csv', 'data/IRIS_test.csv')
```

### 3.3 Train

```python
clf.fit(x_train, y_train)
```

- `X`: numpy array `(m, n)`
- `y`: one-hot `(m, 3)`

Kết quả train sẽ tạo:
- `W`: `(n_features, 3)`
- `b`: `(1, 3)`
- `losses`: list loss theo epoch

### 3.4 Predict

- `predict_proba(X)` → trả xác suất `(n_samples, 3)`
- `predict(X)` → trả index lớp `(n_samples,)`

```python
proba = clf.predict_proba(x_test)
pred_idx = clf.predict(x_test)
```

Nếu cần label string, map theo `CLASS_ORDER`:

```python
labels = [clf.CLASS_ORDER[i] for i in pred_idx]
```

### 3.5 Evaluate / Score

- `score(X, y)` trả accuracy theo % (0–100)
- `evaluate(X, y)` trả `(acc, y_pred, y_true)`

---

## 4) Lưu & tải model (pickle)

### 4.1 Save

```python
clf.save_model('src/models/softmax_model.pkl')
```

File `.pkl` chứa:
- `weights`, `bias`
- `classes`, `learning_rate`, `epochs`, `batch_size`

### 4.2 Load

```python
clf2 = SoftMaxClassifier()
clf2.load_model('src/models/softmax_model.pkl')
```

---

## 5) Tích hợp với Streamlit UI

Trong `main.py`, khi dùng SoftMax Regression:
- UI gọi `model.predict(X_input)` và nhận **index**
- UI map index → species name bằng `species_mapping`

Lưu ý: nếu bạn thay `CLASS_ORDER` hoặc thứ tự lớp, mapping ở UI cũng cần đồng bộ.

---

## 6) Lỗi thường gặp

1) **Quên one-hot labels khi train**
- `fit()` yêu cầu `y` là one-hot.

2) **Predict ra số (index) nhưng hiển thị nhãn sai**
- Hãy map theo đúng `CLASS_ORDER`.

3) **Sai thứ tự feature**
- Dữ liệu đầu vào phải theo đúng `feature_cols` đã train.
