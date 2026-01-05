## Tài liệu chi tiết: Stacking Ensemble

Phần này mô tả chi tiết cách **Stacking Ensemble** được implement trong dự án (meta-classifier + base models), nhằm giúp bạn:
- Hiểu rõ luồng dữ liệu (base → meta → dự đoán)
- Biết cách train / save / load đúng cách
- Tránh các lỗi thường gặp khi tích hợp vào UI (Streamlit)

### 1) Stacking Ensemble là gì?

**Stacking** là một kỹ thuật ensemble gồm 2 tầng:

1. **Base models (level-0)**: nhiều mô hình “gốc” dự đoán trên cùng một input $X$.
2. **Meta-model (level-1)**: học cách **kết hợp** đầu ra của các base models để ra dự đoán cuối cùng.

Trong dự án này, stacking được cài đặt trong `src/models/Ensemble.py` với lớp `StackingClassifier`.

### 2) Các thành phần trong implementation

#### 2.1 Base models được dùng

`StackingClassifier` kế thừa `BaseEnsemble`. Khi khởi tạo, nó **tự động load các model đã train sẵn** từ các file `.pkl` (mặc định nằm cùng thư mục với `Ensemble.py`, tức `src/models/`).

Các base models (level-0):
- SoftMax Regression (`softmax_model.pkl`)
- KNN (`knn_model.pkl`)
- Decision Tree (`decision_tree_model.pkl`)
- Naive Bayes (`naive_bayes_model.pkl`)

Lưu ý: ensemble **không train lại base models**; nó chỉ “load và sử dụng” chúng.

#### 2.2 Meta-model được dùng

Meta-model (level-1) là một **Softmax classifier** đơn giản (Logistic Regression đa lớp) được train bằng Gradient Descent trên “stacked features”.

Trong code, meta-model được biểu diễn bởi:
- `meta_weights`: ma trận trọng số $W$ (shape: `[n_meta_features, n_classes]`)
- `meta_bias`: bias $b$ (shape: `[1, n_classes]`)

### 3) Quy ước dữ liệu (rất quan trọng)

#### 3.1 Feature order

Toàn bộ ensemble dùng thứ tự feature cố định:

```text
['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

Tương ứng với `FEATURE_COLS` trong `BaseEnsemble`.

Input `X` khi gọi `predict` / `predict_proba` phải là numpy array shape:

```text
X.shape == (n_samples, 4)
```

#### 3.2 Class order

Thứ tự lớp cố định:

```text
['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
```

Tương ứng `CLASS_ORDER` trong `BaseEnsemble`. Meta-model sẽ học và dự đoán theo đúng thứ tự này.

### 4) Cách tạo “meta-features” (stacked features)

Đây là phần cốt lõi của stacking: biến đầu ra của base models thành một vector feature mới để feed vào meta-model.

Hàm thực hiện: `StackingClassifier._create_meta_features(X)`.

Giả sử:
- Số base models $M = 4$
- Số lớp $C = 3$

Thì số chiều meta-features sẽ là:

$$n\_meta\_features = M \times C = 12$$

#### 4.1 Chế độ `use_probas=True` (mặc định)

Với `use_probas=True`, mỗi base model đóng góp một vector độ dài 3 cho mỗi sample:

- Softmax: dùng `predict_proba(X)` → xác suất thật
- KNN: tự tính xác suất theo **weighted neighbors** (từ khoảng cách)
- Naive Bayes: tự tính xác suất từ log-likelihood (chuẩn hoá)
- Decision Tree: hiện tại dùng “pseudo-proba” theo kiểu one-hot từ nhãn dự đoán (không phải probability thực)

Sau đó các vector xác suất được ghép ngang (concatenate) theo thứ tự các estimators đã load.

#### 4.2 Chế độ `use_probas=False`

Với `use_probas=False`, mỗi base model tạo **one-hot** từ nhãn dự đoán (ví dụ dự đoán Versicolor → `[0,1,0]`).

### 5) Train meta-classifier

Train meta-model được thực hiện bằng:

```python
from src.models.Ensemble import StackingClassifier

stacking = StackingClassifier(use_probas=True)
stacking.fit_meta(X_train, y_train, learning_rate=0.1, epochs=200)
```

Trong `fit_meta`:
1. Tạo `meta_features = _create_meta_features(X_train)`
2. One-hot hoá `y_train` theo `CLASS_ORDER`
3. Chạy Gradient Descent để tối ưu Softmax Cross-Entropy

Gợi ý tinh chỉnh:
- Nếu loss/accuracy dao động mạnh: giảm `learning_rate`
- Nếu underfit: tăng `epochs`

### 6) Dự đoán

Sau khi đã có `meta_weights` và `meta_bias`:

- `predict_proba(X)`:
   - Tạo `meta_features`
   - Tính $\text{softmax}(XW+b)$ → trả về shape `(n_samples, 3)`

- `predict(X)`:
   - Lấy `argmax` trên xác suất → index lớp
   - Map index → label theo `CLASS_ORDER`

### 7) Save / Load Stacking model

#### 7.1 Save

`StackingClassifier.save_model(filepath)` sẽ lưu:
- `type` (để biết đây là StackingClassifier)
- `model_dir` (nơi chứa base models)
- `use_probas`
- `meta_weights`, `meta_bias`

Lưu ý quan trọng:
- File stacking `.pkl` **không chứa base models**. Khi load lại, hệ thống vẫn cần các file base models `.pkl` trong `model_dir`.

#### 7.2 Load

Dùng:

```python
from src.models.Ensemble import BaseEnsemble

stacking = BaseEnsemble.load_model('stacking_ensemble.pkl')
```

`load_model` sẽ:
1. Tạo `StackingClassifier(model_dir=...)` → tự load base models
2. Gán lại `meta_weights` / `meta_bias` từ file

### 8) Tích hợp với Streamlit UI (main.py)

UI trong `main.py` mặc định sẽ tìm file model tại:

```text
src/models/stacking_ensemble.pkl
```

Vì vậy, sau khi train & save stacking model, bạn nên:
- Lưu trực tiếp vào `src/models/`, hoặc
- Di chuyển file `stacking_ensemble.pkl` vào `src/models/`

### 9) Lỗi thường gặp & cách xử lý

1) **Không tìm thấy base model `.pkl`**
- Triệu chứng: `BaseEnsemble` báo warning/exception khi load.
- Cách xử lý: đảm bảo 4 file base models tồn tại trong `src/models/` (hoặc truyền `model_dir` đúng).

2) **Chưa train meta-classifier nhưng gọi `predict`/`save_model`**
- Triệu chứng: exception “Chưa train meta-classifier…”.
- Cách xử lý: gọi `fit_meta(X_train, y_train)` trước.

3) **Sai thứ tự cột feature**
- Triệu chứng: dự đoán sai bất thường.
- Cách xử lý: đảm bảo input luôn theo đúng thứ tự `['sepal_length','sepal_width','petal_length','petal_width']`.

4) **Load stacking model ở máy/thư mục khác**
- Vì stacking model không chứa base models, bạn cần mang theo cả 4 base model `.pkl` và giữ đúng `model_dir`.