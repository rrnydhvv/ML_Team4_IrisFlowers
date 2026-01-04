# Iris Flower Classification Project

Dự án Machine Learning phân loại hoa Iris sử dụng nhiều thuật toán khác nhau và giao diện web tương tác.

## Mô tả dự án

Dự án này xây dựng và so sánh hiệu suất của nhiều mô hình Machine Learning để phân loại 3 loài hoa Iris:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

### Các thuật toán được implement:

1. **Decision Tree** - Cây quyết định với Entropy/Gini
2. **K-Nearest Neighbors (KNN)** - Phân loại dựa trên láng giềng gần nhất
3. **Naive Bayes** - Phân loại Bayes với giả thiết độc lập
4. **SoftMax Regression** - Hồi quy đa lớp
5. **Hard Voting Ensemble** - Kết hợp bằng bỏ phiếu đa số
6. **Stacking Ensemble** - Kết hợp với meta-classifier

### Đặc điểm nổi bật:

- **Tự implement các thuật toán từ đầu** (không dùng scikit-learn)  
- **Giao diện web tương tác** với Streamlit  
- **Model persistence** với pickle  
- **Ablation study** để phân tích features  
- **Visualization** và đánh giá model chi tiết

## Cấu trúc dự án

```
ML_Team4_IrisFlowers/
├── main.py                      # Streamlit UI cho prediction
├── retrain_stacking.py          # Script train lại Stacking model
├── README.md
├── data/
│   ├── IRIS.csv                 # Dữ liệu gốc
│   ├── IRIS_cleaned.csv         # Dữ liệu đã làm sạch
│   ├── IRIS_train.csv           # Tập train (80%)
│   └── IRIS_test.csv            # Tập test (20%)
├── src/
│   ├── preprocess.py            # Script xử lý dữ liệu
│   ├── models/
│   │   ├── Decision_Tree.py     # Implementation Decision Tree
│   │   ├── KNN.py               # Implementation KNN
│   │   ├── Naive_Bayes.py       # Implementation Naive Bayes
│   │   ├── SoftMax.py           # Implementation SoftMax
│   │   ├── Ensemble.py          # Implementation Ensemble methods
│   │   ├── *.pkl                # Trained models
│   └── notebooks/
│       ├── test_models.ipynb    # Test và so sánh models
│       ├── test_ablation.ipynb  # Ablation study
│       ├── train_*.ipynb        # Training notebooks cho từng model
```

## Cài đặt và Chạy

### 1. Cài đặt thư viện

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

Hoặc nếu có file requirements:

```bash
pip install -r requirements.txt
```

### 2. Chạy giao diện web

```bash
streamlit run main.py
```

Mở browser và truy cập: `http://localhost:8501`

### 3. Sử dụng UI

1. **Chọn model** từ sidebar (6 models có sẵn)
2. **Nhập thông số** hoa Iris:
   - Sepal Length (chiều dài đài hoa)
   - Sepal Width (chiều rộng đài hoa)
   - Petal Length (chiều dài cánh hoa)
   - Petal Width (chiều rộng cánh hoa)
3. **Click "Predict"** để xem kết quả phân loại

### 4. Train lại models (nếu cần)

**Train từng model riêng:**
```bash
jupyter notebook src/notebooks/train_decision_tree.ipynb
```

## Dataset

**Iris Dataset** - Bộ dữ liệu cổ điển trong ML
- **Số mẫu**: 150 (50 mẫu/loài)
- **Features**: 4 đặc trưng số
- **Classes**: 3 loài hoa
- **Split**: 80% train (120 mẫu) / 20% test (30 mẫu)

### Thống kê features:

| Feature | Min | Max | Mean |
|---------|-----|-----|------|
| Sepal Length | 4.3 | 7.9 | 5.84 |
| Sepal Width | 2.0 | 4.4 | 3.05 |
| Petal Length | 1.0 | 6.9 | 3.76 |
| Petal Width | 0.1 | 2.5 | 1.20 |

## Kết quả Models

| Model | Accuracy | Điểm mạnh |
|-------|----------|-----------|
| Decision Tree | ~96% | Dễ hiểu, visualization tốt |
| KNN | ~97% | Đơn giản, hiệu quả |
| Naive Bayes | ~94% | Nhanh, ít tham số |
| SoftMax | ~95% | Xác suất rõ ràng |
| Hard Voting | ~97% | Ổn định, robust |
| Stacking | ~97% | Hiệu suất cao nhất |

## Chi tiết kỹ thuật

### Decision Tree
- Criterion: Entropy hoặc Gini
- Max depth: Configurable
- Pruning: Optional

### KNN
- K selection: Cross-validation tự động
- Distance: Euclidean
- Weighting: Inverse distance

### Naive Bayes
- Type: Gaussian
- Features: Độc lập điều kiện
- Smoothing: Laplace (nếu cần)

### SoftMax
- Optimizer: Gradient Descent
- Learning rate: 0.01
- Regularization: L2 (optional)

### Ensemble
- Hard Voting: Majority vote
- Stacking: Meta-classifier (Softmax)
- Base models: All 4 basic models

## Notebooks

Các Jupyter notebooks trong `src/notebooks/`:

1. **train_*.ipynb** - Training notebooks cho từng model
2. **test_models.ipynb** - So sánh và đánh giá tất cả models
3. **test_ablation.ipynb** - Phân tích ảnh hưởng của từng feature

## Troubleshooting

### Lỗi: "Model doesn't have predict method"
- **Nguyên nhân**: Cache cũ hoặc model chưa train
- **Giải pháp**: Clear cache Streamlit (nhấn `C`) hoặc train lại model

### Lỗi: "Meta-classifier not trained" (Stacking)
- **Nguyên nhân**: Model chưa train meta-classifier
- **Giải pháp**: Chạy `python retrain_stacking.py`

### Lỗi: "Module not found"
- **Nguyên nhân**: Thiếu thư viện
- **Giải pháp**: `pip install -r requirements.txt`

## Team 4

Dự án Machine Learning - Phân loại Iris Flowers

## License

Educational project - Free to use for learning purposes

## Features nổi bật

- Giao diện đẹp với Streamlit
- 6 models ML khác nhau
- Visualization và đánh giá chi tiết
- Model persistence với pickle
- Responsive UI
- Dark/Light theme support
- Model caching thông minh

---

**Để bắt đầu:** `streamlit run main.py`