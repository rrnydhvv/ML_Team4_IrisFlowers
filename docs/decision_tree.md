# Decision Tree (DecisionTreeClassifier)

Tài liệu này mô tả cách sử dụng và cơ chế hoạt động của **Decision Tree** trong dự án.

- Source: `src/models/Decision_Tree.py`
- Class chính: `DecisionTreeClassifier`

---

## 1) Tổng quan

Decision Tree học cách chia không gian feature bằng các rule dạng:

```text
if feature_i <= threshold: đi trái
else: đi phải
```

Cây được xây bằng greedy split theo tiêu chí:
- `entropy` (Information Gain)
- hoặc `gini` (Gini impurity)

---

## 2) Hyperparameters

- `criterion`: `'entropy'` hoặc `'gini'`
- `max_depth`: giới hạn độ sâu của cây (None = không giới hạn)
- `feature_cols`: danh sách feature dùng để train/predict

Ví dụ:

```python
from src.models.Decision_Tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
```

---

## 3) Quy ước dữ liệu

### 3.1 Feature columns

Mặc định:

```text
['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

Có thể đổi bằng:

```python
clf.set_feature_cols(['petal_length', 'petal_width'])
```

### 3.2 Input/Output

- `fit(X, y)`:
  - `X`: numpy array `(n_samples, n_features)`
  - `y`: array/list label strings

- `predict(X)`:
  - trả về numpy array label strings `(n_samples,)`

---

## 4) API chính

### 4.1 Load dữ liệu (helper)

```python
X_train, y_train, X_test, y_test = clf.load_data('data/IRIS_train.csv', 'data/IRIS_test.csv')
```

### 4.2 Train

```python
clf.fit(X_train, y_train)
```

Sau khi train:
- `clf.tree` là một cấu trúc dict lồng nhau.
- Node lá có dạng: `{"value": <label>}`
- Node split có dạng:
  - `feature`: index feature (0..d-1)
  - `threshold`: giá trị ngưỡng
  - `left`, `right`: subtree

### 4.3 Evaluate

```python
acc, y_pred, y_true = clf.evaluate(X_test, y_test)
```

- `score(X, y)` trả accuracy (0–1)

---

## 5) Lưu & tải model (pickle)

### 5.1 Save

```python
clf.save_model('src/models/decision_tree_model.pkl')
```

File `.pkl` chứa:
- `tree`, `criterion`, `max_depth`, `feature_cols`

### 5.2 Load

```python
clf2 = DecisionTreeClassifier()
clf2.load_model('src/models/decision_tree_model.pkl')
```

---

## 6) Lưu ý & lỗi thường gặp

1) **Overfitting khi `max_depth=None`**
- Cây có thể fit rất sát train set.
- Cách xử lý: đặt `max_depth` nhỏ hơn (ví dụ 3–6).

2) **Split threshold chạy trên tất cả giá trị unique**
- Với dataset lớn, cách này tốn thời gian. Iris nhỏ nên ổn.

3) **Sai thứ tự feature**
- Vì tree lưu index feature theo cột của `X`, input predict phải có cùng thứ tự như khi train.
