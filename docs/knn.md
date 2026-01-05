# K-Nearest Neighbors (KNNClassifier)

Tài liệu này mô tả cách sử dụng và cơ chế hoạt động của **KNN Weighted Distance** trong dự án.

- Source: `src/models/KNN.py`
- Class chính: `KNNClassifier`

---

## 1) Tổng quan

KNN là thuật toán “lazy learning”: không học tham số như linear model, mà lưu dữ liệu train và dự đoán bằng cách:

1. Tính khoảng cách từ sample mới đến tất cả sample train.
2. Chọn `k` láng giềng gần nhất.
3. Vote nhãn theo **trọng số nghịch đảo khoảng cách** (weighted distance).

Trong dự án này:
- Nếu `k=None`, model sẽ tự chọn `best_k` bằng **K-Fold Cross-Validation** (mặc định 5-fold).

---

## 2) Điểm khác biệt quan trọng: KNN dùng DataFrame để train

`KNNClassifier.fit()` nhận **DataFrame** `train_df` (không phải numpy array):

- `train_df` phải có các cột feature và cột label `species`.
- Model sẽ lưu `train_data` (DataFrame) để dùng cho predict.

---

## 3) API chính

### 3.1 Khởi tạo

```python
from src.models.KNN import KNNClassifier

knn = KNNClassifier(k=None)  # tự chọn k
# hoặc
knn = KNNClassifier(k=7)     # cố định k
```

Tuỳ chọn:
- `random_state`: phục vụ shuffle trong CV
- `feature_cols`: chọn subset features

### 3.2 Load dữ liệu (helper)

```python
train_df, test_df = knn.load_data('data/IRIS_train.csv', 'data/IRIS_test.csv')
```

### 3.3 Train

```python
knn.fit(train_df, k_range=range(1, 31), n_folds=5)
```

Kết quả:
- `knn.best_k`: k tốt nhất
- `knn.train_results`: list (k, acc_cv)
- `knn.train_data`: DataFrame train đã reset index

### 3.4 Predict

`predict(X)` hỗ trợ:
- `X` là DataFrame (khuyến nghị)
- hoặc `X` là array/list `(n_samples, n_features)`

Ví dụ:

```python
y_pred = knn.predict(test_df)  # list label strings
```

### 3.5 Evaluate / Score

- `score(test_df)` → accuracy (0–1)
- `evaluate(test_df)` → in kết luận và trả `(acc, y_pred, y_true)`

---

## 4) Lưu & tải model (pickle)

### 4.1 Save

```python
knn.save_model('src/models/knn_model.pkl')
```

File `.pkl` chứa:
- `train_data` (DataFrame)
- `feature_cols`, `best_k`
- một số metadata như `train_results`, `y_true_test`, `y_pred_test`

### 4.2 Load

```python
knn2 = KNNClassifier()
knn2.load_model('src/models/knn_model.pkl')
```

---

## 5) Lưu ý & lỗi thường gặp

1) **Thiếu cột label khi predict bằng DataFrame**
- Khi gọi `predict(DataFrame)`, code chỉ lấy feature_cols nên label có thể không bắt buộc.
- Tuy nhiên một số chỗ khác (ví dụ ensemble tạo dummy label) vẫn gán cột `species` để đồng bộ.

2) **Sai feature_cols giữa train và predict**
- `feature_cols` được dùng để lấy cột từ DataFrame; nếu thiếu cột sẽ lỗi.

3) **Khoảng cách và scale**
- KNN nhạy với scale. Dataset Iris thường cùng thang đo cm nên ổn; nếu dữ liệu khác, cân nhắc chuẩn hoá.
