# Hard Voting Ensemble (HardVotingClassifier)

Tài liệu này mô tả cách hoạt động và cách dùng **Hard Voting Ensemble** trong dự án.

- Source: `src/models/Ensemble.py`
- Class chính: `HardVotingClassifier` (kế thừa `BaseEnsemble`)

---

## 1) Tổng quan

**Hard Voting** là ensemble đơn giản:
- Mỗi base model dự đoán **1 nhãn class**.
- Nhãn nào được vote nhiều nhất sẽ là dự đoán cuối.

Trong dự án, Hard Voting sử dụng các base models đã train sẵn:
- SoftMax Regression
- KNN
- Decision Tree
- Naive Bayes

---

## 2) Cơ chế load pre-trained models

Khi khởi tạo `HardVotingClassifier()`, lớp cha `BaseEnsemble` sẽ tự load các file `.pkl` trong `model_dir` (mặc định là `src/models/`).

Các file cần có:

```text
src/models/softmax_model.pkl
src/models/knn_model.pkl
src/models/decision_tree_model.pkl
src/models/naive_bayes_model.pkl
```

Lưu ý:
- Nếu thiếu bất kỳ file nào, ensemble có thể load không đủ estimators hoặc fail (tuỳ số file còn lại).

---

## 3) Quy ước dữ liệu

### 3.1 Feature order

Hard Voting nhận input `X` dạng numpy array:

```text
X.shape == (n_samples, 4)
```

với thứ tự feature cố định:

```text
['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

### 3.2 Output

`predict(X)` trả về:
- `List[str]` nhãn dự đoán, ví dụ `"Iris-setosa"`

---

## 4) Dự đoán bằng Hard Voting

Cách `HardVotingClassifier.predict(X)` hoạt động:

1. Tạo một DataFrame từ `X` để phục vụ KNN (`KNNClassifier.predict` hỗ trợ DataFrame).
2. Gọi predict trên từng base model:
   - KNN: dự đoán nhãn trực tiếp (string)
   - Softmax: dự đoán index → map về nhãn theo `CLASS_ORDER`
   - Decision Tree, Naive Bayes: dự đoán nhãn (string)
3. Với mỗi sample: đếm vote theo nhãn và lấy nhãn có vote cao nhất.

---

## 5) Usage

### 5.1 Dùng trực tiếp

```python
from src.models.Ensemble import HardVotingClassifier

hv = HardVotingClassifier(model_dir='src/models')
y_pred = hv.predict(X_test)
```

### 5.2 Evaluate

`BaseEnsemble.evaluate(X, y)` hỗ trợ tính accuracy và in ra kết quả:

```python
acc, y_pred, y_true = hv.evaluate(X_test, y_test)
```

---

## 6) Lưu & tải Hard Voting model

### 6.1 Save

Hard Voting lưu **config** (không lưu lại base models):

```python
hv.save_model('src/models/hard_voting_ensemble.pkl')
```

File `.pkl` chứa:
- `type`: HardVotingClassifier
- `model_dir`
- `estimator_names`

### 6.2 Load

```python
from src.models.Ensemble import BaseEnsemble

hv2 = BaseEnsemble.load_model('src/models/hard_voting_ensemble.pkl')
```

Khi load:
- Ensemble sẽ dựa vào `model_dir` để load lại các base models.

---

## 7) Lỗi thường gặp

1) **Thiếu base model `.pkl`**
- Triệu chứng: cảnh báo/exception khi khởi tạo.
- Cách xử lý: đảm bảo đủ 4 file model trong `src/models/` hoặc truyền `model_dir` đúng.

2) **Sai input shape**
- `X` phải là `(n_samples, 4)` theo đúng thứ tự feature.

3) **KNN cần DataFrame feature_cols**
- Implementation đã tự tạo DataFrame từ `X`; nếu bạn tự gọi KNN riêng, hãy đảm bảo cột đúng.
