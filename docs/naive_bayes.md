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

## 1.1) Vì sao chọn Gaussian Naive Bayes cho bài toán Iris?

Bài toán Iris trong dự án là **phân loại 3 loài hoa** dựa trên 4 đặc trưng **liên tục (continuous)**:

- `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (đều là số thực đo chiều dài/chiều rộng)

Vì vậy **Gaussian Naive Bayes (GNB)** là lựa chọn “đúng kiểu dữ liệu” trong họ Naive Bayes:

1. **Phù hợp với feature liên tục**

- GNB mô hình hoá $P(x_j|c)$ bằng phân phối chuẩn với 2 tham số $(\mu_{c,j}, \sigma^2_{c,j})$.
- Không cần “ép” dữ liệu sang dạng đếm/tần suất hoặc nhị phân.

2. **Không cần discretize/quantize**

- Nếu dùng NB dạng “đếm” (Multinomial), thường phải biến feature liên tục thành các bins (rời rạc hoá) → có thể mất thông tin.
- GNB dùng trực tiếp số đo gốc nên giữ được thông tin biên độ.

3. **Nhanh, ít tham số, hợp với dataset nhỏ**

- Iris có ít mẫu, ít feature → GNB train rất nhanh (chỉ tính mean/variance theo từng class).
- Ít nguy cơ overfit hơn các mô hình phức tạp khi dữ liệu nhỏ.

4. **Baseline mạnh + dễ giải thích**

- Dễ trình bày: mỗi class “học” trung bình và độ phân tán của từng feature.
- Dễ debug: xem mean/variance theo class để hiểu đặc trưng nào phân biệt loài.

5. **Thực tế dữ liệu Iris thường gần Gaussian theo từng class (tương đối)**

- Với các measurement vật lý như chiều dài/chiều rộng, phân phối theo từng nhóm thường “gần chuẩn” ở mức vừa đủ.
- Dù không hoàn hảo, GNB vẫn có thể cho kết quả tốt nhờ tính đơn giản và log-likelihood.

Lưu ý: giả định “độc lập có điều kiện” giữa các feature không luôn đúng (petal_length và petal_width thường tương quan), nhưng Naive Bayes vẫn thường hoạt động tốt trong thực tế.

---

## 1.2) Gaussian NB khác gì so với các biến thể Naive Bayes khác?

Điểm khác nhau chủ yếu nằm ở cách mô hình hoá **likelihood** $P(x_j|c)$ theo kiểu dữ liệu.

### a) Gaussian Naive Bayes (GNB)

- **Dữ liệu phù hợp**: feature **liên tục** (real-valued).
- **Mô hình**: $x_j|c \sim \mathcal{N}(\mu_{c,j}, \sigma^2_{c,j})$.
- **Ưu**: không cần rời rạc hoá, nhanh, dễ triển khai.
- **Nhược**: nhạy với giả định Gaussian; cần xử lý khi $\sigma^2$ quá nhỏ (tránh chia cho 0).

### b) Multinomial Naive Bayes

- **Dữ liệu phù hợp**: **đếm/tần suất** (word counts, bag-of-words), thường dùng cho text.
- **Mô hình**: $x_j$ là số lần xuất hiện; thường kết hợp Laplace smoothing.
- **Không phù hợp trực tiếp cho Iris** vì feature là số đo liên tục, không phải count.

### c) Bernoulli Naive Bayes

- **Dữ liệu phù hợp**: feature **nhị phân** (0/1: có/không).
- **Mô hình**: $x_j|c \sim \mathrm{Bernoulli}(p_{c,j})$.
- Muốn dùng cho Iris phải binarize (vd: $x_j > t$) → thường mất thông tin và phụ thuộc ngưỡng.

### d) Categorical Naive Bayes

- **Dữ liệu phù hợp**: feature **rời rạc nhiều giá trị** (mã hoá danh mục).
- **Mô hình**: phân phối categorical theo từng feature.
- Iris không phải dữ liệu dạng category.

### e) Complement Naive Bayes (thường là biến thể của Multinomial)

- **Dữ liệu phù hợp**: tương tự Multinomial (text), đặc biệt khi **class imbalance**.
- Iris khá cân bằng nên không phải lợi thế chính.

Tóm lại: với Iris (4 feature liên tục), **Gaussian NB là biến thể tự nhiên nhất** trong họ Naive Bayes.

---

## 1.3) Điểm nổi bật (để đưa vào báo cáo/slide)

- **Đúng kiểu dữ liệu**: continuous features → dùng Gaussian likelihood.
- **Train cực nhanh**: chỉ cần tính mean/variance/prior theo class.
- **Ít tham số**: mỗi class có 2 vector tham số $(\mu, \sigma^2)$ cho 4 feature.
- **Diễn giải tốt**: đọc được feature nào “kéo” xác suất về class nào thông qua log-likelihood.
- **Baseline đáng tin**: thường cho kết quả ổn trên bài toán nhỏ và là mốc so sánh công bằng.

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

1. **Variance bằng 0 (hiếm nhưng có thể xảy ra)**

- Nếu một feature có phương sai ~0 trong một class, likelihood có thể gây lỗi chia cho 0. Trong dự án hiện tại, dữ liệu Iris thường không gặp.

2. **Sai thứ tự/thiếu feature**

- Input phải đúng thứ tự theo `feature_cols` đã train.

3. **Dữ liệu không phải numpy array**

- `fit/predict` ép về numpy được, nhưng tốt nhất bạn truyền numpy array `(n, d)`.
