import sys
import os
import pandas as pd
import numpy as np

# 1. Thêm src vào sys.path để import KNN module
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.KNN import knn_predict_weighted, load_model_KNN

# 2. Load model đã train
model_path = os.path.join(os.path.dirname(__file__), "src", "knn_model.pkl")
model_loaded = load_model_KNN(model_path)

# 3. Dự đoán mẫu mới
x_new = np.array([0.2, 0.5, 0.1, 0.05])  # theo thứ tự feature_cols
y_pred = knn_predict_weighted(model_loaded["train_norm"], x_new, model_loaded["feature_cols"], model_loaded["best_k"])
print("Dự đoán mẫu mới:", y_pred)

# 4. Dự đoán batch nhiều mẫu
df_new = pd.DataFrame([
    [0.2, 0.5, 0.1, 0.05],
    [0.8, 0.3, 0.7, 0.2]
], columns=model_loaded["feature_cols"])

preds = []
for i in range(len(df_new)):
    x_row = df_new.iloc[i].values
    y_pred = knn_predict_weighted(model_loaded["train_norm"], x_row, model_loaded["feature_cols"], model_loaded["best_k"])
    preds.append(y_pred)

df_new["predicted"] = preds
print("\nDự đoán batch:")
print(df_new)
