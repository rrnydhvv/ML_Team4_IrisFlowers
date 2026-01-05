# Gaussian Naive Bayes (GaussianNaiveBayesClassifier)

Tài liệu này mô tả cách sử dụng và cơ chế hoạt động của **Gaussian Naive Bayes** trong dự án.

- Source: `src/models/Naive_Bayes.py`
- Class chính: `GaussianNaiveBayesClassifier`

---

## 1) Tổng quan

Gaussian Naive Bayes giả định:
- Mỗi feature tuân theo phân phối Gaussian (chuẩn) theo từng class.
- Các feature **độc lập có điều kiện** khi biết class.

Mô hình học các tham số theo class:
- Mean vector: $\mu_c$
- Variance vector: $\sigma_c^2$
- Prior: $P(c)$

Khi dự đoán, mô hình chọn class $c$ tối đa hoá:

$$\log P(c) + \sum_j \log P(x_j | c)$$

---

## 2) Quy ước dữ liệu

### 2.1 Feature columns

Mặc định:

```text
['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

Có thể thay bằng `set_feature_cols()`.

### 2.2 Label column

Mặc định nhãn nằm ở cột:

```text
species
```

---

## 3) API chính

### 3.1 Khởi tạo

```python
from src.models.Naive_Bayes import GaussianNaiveBayesClassifier

clf = GaussianNaiveBayesClassifier()
# hoặc
clf = GaussianNaiveBayesClassifier(feature_cols=['sepal_length','sepal_width'])
```

### 3.2 Load dữ liệu (helper)

```python
X_train, y_train, X_test, y_test = clf.load_data('data/IRIS_train.csv', 'data/IRIS_test.csv')
```

- `X_*`: numpy array `(n_samples, n_features)`
- `y_*`: array/list label string

### 3.3 Train

```python
clf.fit(X_train, y_train)
```

Sau train, mô hình lưu:
- `classes`: list các class
- `means[class]`: mean vector `(n_features,)`
- `vars[class]`: variance vector `(n_features,)`
- `priors[class]`: xác suất tiên nghiệm

### 3.4 Predict

```python
y_pred = clf.predict(X_test)  # list label strings
```

### 3.5 Score / Evaluate

- `score(X, y)` → accuracy (0–1)
- `evaluate(X, y)` → in accuracy và trả `(acc, y_pred, y_true)`

---

## 4) Lưu & tải model (pickle)

### 4.1 Save

```python
clf.save_model('src/models/naive_bayes_model.pkl')
```

File `.pkl` chứa:
- `classes`, `means`, `vars`, `priors`
- `feature_cols`

### 4.2 Load

```python
clf2 = GaussianNaiveBayesClassifier()
clf2.load_model('src/models/naive_bayes_model.pkl')
```

---

## 5) Lỗi thường gặp

1) **Variance bằng 0 (hiếm nhưng có thể xảy ra)**
- Nếu một feature có phương sai ~0 trong một class, likelihood có thể gây lỗi chia cho 0. Trong dự án hiện tại, dữ liệu Iris thường không gặp.

2) **Sai thứ tự/thiếu feature**
- Input phải đúng thứ tự theo `feature_cols` đã train.

3) **Dữ liệu không phải numpy array**
- `fit/predict` ép về numpy được, nhưng tốt nhất bạn truyền numpy array `(n, d)`.
