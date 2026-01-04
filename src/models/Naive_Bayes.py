import numpy as np
import pandas as pd
import pickle

# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'
MODEL_FILE = 'naive_bayes_model.pkl'
FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABEL_COL = 'species'

def load_data(train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Load dữ liệu từ file CSV train và test riêng biệt"""
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")
        
        X_train = df_train[FEATURE_COLS].values
        y_train = df_train[LABEL_COL].values
        X_test = df_test[FEATURE_COLS].values
        y_test = df_test[LABEL_COL].values
        
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        return None, None, None, None

class GaussianNaiveBayes:
	def __init__(self):
		self.classes = None  # C
		self.means = {}  # {class: [mean_feature1, mean_feature2, ...]}
		self.vars = {} # {class: [var_feature1, var_feature2, ...]} phương sai
		self.priors = {}  # {class: prior_probability} tiên nghiệm

	def fit(self, X, y):
		self.classes = np.unique(y)

		X = np.array(X)
		y = np.array(y)

		for c in self.classes:
			X_c = X[y == c]

			self.means[c] = X_c.mean(axis=0)
			self.vars[c] = X_c.var(axis=0)
			self.priors[c] = X_c.shape[0] / X.shape[0]

	def gaussian_likelihood(self, x, mean, var):
		numerator  = np.exp(-((x - mean) ** 2) / (2 * var)) # Phần tử số
		denominator  = np.sqrt(2 * np.pi * var) # Phần mẫu số
		return numerator / denominator
	
	def predict(self, X):
		X = np.array(X)
		y_pred = []

		for x in X:
			class_probs = {}

			for c in self.classes:
				prior = np.log(self.priors[c])
				likelihood = np.sum(np.log(self.gaussian_likelihood(x, self.means[c], self.vars[c])))
				class_probs[c] = prior + likelihood

			best_class = max(class_probs, key=class_probs.get)
			y_pred.append(best_class)

		return y_pred
	
def accuracy_score(y_true, y_pred):
	correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
	return correct / len(y_true) if len(y_true) > 0 else 0.0
def train(X_train, y_train):
    """Train Naive Bayes model"""
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    # Tính accuracy trên tập train
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy trên tập train: {train_acc:.4f}")
    
    return model

def test(model, X_test, y_test):
    """Test Naive Bayes model"""
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy trên tập test: {test_acc:.4f}")
    return test_acc, y_pred

def export_model(model, filename):
    """Lưu model vào file"""
    model_data = {
        "model": model,
        "classes": model.classes.tolist() if model.classes is not None else None,
        "info": "Gaussian Naive Bayes Classifier from scratch"
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[OK] Đã xuất Naive Bayes model tại: {filename}")

def load_model(filename):
    """Load model từ file"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data["model"]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE)
    
    if X_train is not None:
        model = train(X_train, y_train)
        test_acc, y_pred = test(model, X_test, y_test)
        export_model(model, MODEL_FILE)