import numpy as np
import pandas as pd
import pickle
import sys
import os

# Thêm path để import các modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các base classifiers
from SoftMax import SoftMaxClassifier
from KNN import KNNClassifier
from Decision_Tree import DecisionTreeClassifier
from Naive_Bayes import GaussianNaiveBayesClassifier


class BaseEnsemble:
    """Base class cho các Ensemble methods - Load pre-trained models"""
    
    CLASS_ORDER = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    LABEL_COL = 'species'
    
    def __init__(self, model_dir=None):
        """
        Khởi tạo Ensemble và load các model đã train sẵn
        
        Args:
            model_dir: Thư mục chứa các file model (mặc định: cùng thư mục)
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = model_dir
        self.estimators = []
        self.is_fitted = False
        
        # Tự động load pre-trained models
        self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Load các model đã train sẵn từ file .pkl"""
        model_files = {
            'softmax': os.path.join(self.model_dir, 'softmax_model.pkl'),
            'knn': os.path.join(self.model_dir, 'knn_model.pkl'),
            'decision_tree': os.path.join(self.model_dir, 'decision_tree_model.pkl'),
            'naive_bayes': os.path.join(self.model_dir, 'naive_bayes_model.pkl')
        }
        
        loaded_estimators = []
        
        for name, filepath in model_files.items():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Tạo estimator và load weights
                if name == 'softmax':
                    estimator = SoftMaxClassifier()
                    estimator.W = model_data['weights']
                    estimator.b = model_data['bias']
                    estimator.is_fitted = True
                    
                elif name == 'knn':
                    estimator = KNNClassifier()
                    estimator.train_data = model_data['train_data']
                    estimator.feature_cols = model_data['feature_cols']
                    estimator.best_k = model_data['best_k']
                    estimator.is_fitted = True
                    
                elif name == 'decision_tree':
                    estimator = DecisionTreeClassifier()
                    estimator.tree = model_data['tree']
                    estimator.criterion = model_data['criterion']
                    estimator.max_depth = model_data['max_depth']
                    estimator.is_fitted = True
                    
                elif name == 'naive_bayes':
                    estimator = GaussianNaiveBayesClassifier()
                    estimator.classes = np.array(model_data['classes'])
                    estimator.means = {k: np.array(v) for k, v in model_data['means'].items()}
                    estimator.vars = {k: np.array(v) for k, v in model_data['vars'].items()}
                    estimator.priors = model_data['priors']
                    estimator.is_fitted = True
                
                loaded_estimators.append((name, estimator))
                print(f"[OK] Loaded {name}")
                
            except FileNotFoundError:
                print(f"[WARNING] Không tìm thấy {filepath}")
            except Exception as e:
                print(f"[ERROR] Lỗi khi load {name}: {e}")
        
        if loaded_estimators:
            self.estimators = loaded_estimators
            self.is_fitted = True
            print(f"\n=== Đã load {len(loaded_estimators)} pre-trained models ===\n")
        else:
            raise Exception("Không load được model nào! Hãy train các model trước.")
    
    def _label_to_index(self, label):
        """Chuyển label sang index"""
        return self.CLASS_ORDER.index(label)
    
    def _index_to_label(self, idx):
        """Chuyển index sang label"""
        return self.CLASS_ORDER[idx]
    
    def load_data(self, train_file, test_file):
        """Load dữ liệu từ file CSV"""
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            
            print(f"Train: {len(df_train)} samples | Test: {len(df_test)} samples")
            
            X_train = df_train[self.FEATURE_COLS].values
            y_train = df_train[self.LABEL_COL].values
            X_test = df_test[self.FEATURE_COLS].values
            y_test = df_test[self.LABEL_COL].values
            
            return X_train, y_train, X_test, y_test
            
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file - {e}")
            return None, None, None, None
    
    def score(self, X, y):
        """Tính accuracy"""
        y_pred = self.predict(X)
        return np.mean(np.array(y_pred) == np.array(y))
    
    def evaluate(self, X, y):
        """Đánh giá model"""
        y_pred = self.predict(X)
        acc = np.mean(np.array(y_pred) == np.array(y))
        print(f"Accuracy: {acc*100:.2f}%")
        return acc, y_pred, y


class HardVotingClassifier(BaseEnsemble):
    """
    Hard Voting Ensemble - Majority Voting
    Mỗi model vote 1 class, class nào nhiều vote nhất thắng
    """
    
    def __init__(self, model_dir=None):
        self.name = "Hard Voting"
        super().__init__(model_dir)
    
    def predict(self, X):
        """
        Dự đoán bằng Hard Voting (Majority Voting)
        
        Args:
            X: Features array
            
        Returns:
            List các labels dự đoán
        """
        if not self.is_fitted:
            raise Exception("Model chưa được load.")
        
        n_samples = len(X)
        predictions = []
        
        # Chuẩn bị DataFrame cho KNN
        X_test_df = pd.DataFrame(X, columns=self.FEATURE_COLS)
        X_test_df[self.LABEL_COL] = self.CLASS_ORDER[0]  # Dummy label
        
        # Lấy predictions từ mỗi model
        for name, estimator in self.estimators:
            if name == 'knn':
                preds = estimator.predict(X_test_df)
            elif name == 'softmax':
                pred_indices = estimator.predict(X)
                preds = [self._index_to_label(idx) for idx in pred_indices]
            else:
                preds = list(estimator.predict(X))
            
            predictions.append(preds)
        
        # Hard voting: chọn class được vote nhiều nhất
        final_predictions = []
        for i in range(n_samples):
            votes = [predictions[j][i] for j in range(len(self.estimators))]
            
            # Đếm votes
            vote_counts = {}
            for v in votes:
                vote_counts[v] = vote_counts.get(v, 0) + 1
            
            # Chọn class có nhiều votes nhất
            winner = max(vote_counts, key=vote_counts.get)
            final_predictions.append(winner)
        
        return final_predictions


class SoftVotingClassifier(BaseEnsemble):
    """
    Soft Voting Ensemble - Probability Averaging
    Lấy trung bình xác suất từ các model, chọn class có xác suất cao nhất
    """
    
    def __init__(self, model_dir=None, weights=None):
        """
        Args:
            model_dir: Thư mục chứa các model
            weights: Trọng số cho mỗi model (nếu None, dùng trọng số bằng nhau)
        """
        self.name = "Soft Voting"
        self.weights = weights
        super().__init__(model_dir)
        
        # Nếu không có weights, dùng trọng số bằng nhau
        if self.weights is None:
            self.weights = [1.0 / len(self.estimators)] * len(self.estimators)
    
    def _get_probabilities(self, X):
        """
        Lấy xác suất từ mỗi model
        
        Returns:
            List of probability arrays, mỗi array shape (n_samples, n_classes)
        """
        n_samples = len(X)
        n_classes = len(self.CLASS_ORDER)
        
        X_test_df = pd.DataFrame(X, columns=self.FEATURE_COLS)
        X_test_df[self.LABEL_COL] = self.CLASS_ORDER[0]
        
        all_probas = []
        
        for name, estimator in self.estimators:
            if name == 'softmax':
                probas = estimator.predict_proba(X)
            
            elif name == 'knn':
                probas = self._knn_probabilities(estimator, X_test_df)
            
            elif name == 'naive_bayes':
                probas = self._naive_bayes_probabilities(estimator, X)
            
            elif name == 'decision_tree':
                preds = estimator.predict(X)
                probas = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(preds):
                    probas[i, self._label_to_index(pred)] = 1.0
            
            else:
                preds = estimator.predict(X)
                probas = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(preds):
                    probas[i, self._label_to_index(pred)] = 1.0
            
            all_probas.append(probas)
        
        return all_probas
    
    def _knn_probabilities(self, estimator, X_test_df):
        """Tính xác suất cho KNN dựa trên weighted voting"""
        n_samples = len(X_test_df)
        n_classes = len(self.CLASS_ORDER)
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            x_new = X_test_df.iloc[i][estimator.feature_cols].values
            
            dist = np.zeros(len(estimator.train_data))
            for j, col in enumerate(estimator.feature_cols):
                dist += (estimator.train_data[col].values - x_new[j]) ** 2
            dist = np.sqrt(dist)
            
            df_tmp = estimator.train_data.copy()
            df_tmp["dist"] = dist
            neighbors = df_tmp.sort_values("dist").head(estimator.best_k)
            
            epsilon = 1e-10
            neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)
            
            total_weight = neighbors["weight"].sum()
            for c in self.CLASS_ORDER:
                class_weight = neighbors[neighbors[estimator.label_col] == c]["weight"].sum()
                probas[i, self._label_to_index(c)] = class_weight / total_weight
        
        return probas
    
    def _naive_bayes_probabilities(self, estimator, X):
        """Tính xác suất cho Naive Bayes"""
        n_samples = len(X)
        n_classes = len(self.CLASS_ORDER)
        probas = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            log_probs = {}
            for c in estimator.classes:
                log_prior = np.log(estimator.priors[c])
                log_likelihood = np.sum(np.log(
                    estimator._gaussian_likelihood(x, estimator.means[c], estimator.vars[c])
                ))
                log_probs[c] = log_prior + log_likelihood
            
            max_log = max(log_probs.values())
            probs = {c: np.exp(lp - max_log) for c, lp in log_probs.items()}
            total = sum(probs.values())
            
            for c, p in probs.items():
                probas[i, self._label_to_index(c)] = p / total
        
        return probas
    
    def predict_proba(self, X):
        """Tính xác suất trung bình có trọng số"""
        if not self.is_fitted:
            raise Exception("Model chưa được load.")
        
        all_probas = self._get_probabilities(X)
        
        n_samples = len(X)
        n_classes = len(self.CLASS_ORDER)
        avg_probas = np.zeros((n_samples, n_classes))
        
        for probas, weight in zip(all_probas, self.weights):
            avg_probas += weight * probas
        
        avg_probas = avg_probas / avg_probas.sum(axis=1, keepdims=True)
        
        return avg_probas
    
    def predict(self, X):
        """Dự đoán class có xác suất cao nhất"""
        probas = self.predict_proba(X)
        pred_indices = np.argmax(probas, axis=1)
        return [self._index_to_label(idx) for idx in pred_indices]


class StackingClassifier(BaseEnsemble):
    """
    Stacking Ensemble
    Sử dụng predictions của các base models làm features cho meta-model (Logistic Regression)
    """
    
    def __init__(self, model_dir=None, use_probas=True):
        """
        Args:
            model_dir: Thư mục chứa các model
            use_probas: Dùng probabilities thay vì predictions làm features
        """
        self.name = "Stacking"
        self.use_probas = use_probas
        self.meta_weights = None
        self.meta_bias = None
        super().__init__(model_dir)
    
    def fit_meta(self, X_train, y_train, learning_rate=0.1, epochs=200):
        """
        Train meta-classifier (Softmax) trên tập train
        
        Args:
            X_train: Features array
            y_train: Labels array
        """
        print("Training meta-classifier on stacked features...")
        
        # Tạo meta-features từ base models
        meta_features = self._create_meta_features(X_train)
        
        # One-hot encode labels
        n_samples = len(y_train)
        n_classes = len(self.CLASS_ORDER)
        y_onehot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y_train):
            y_onehot[i, self._label_to_index(label)] = 1.0
        
        # Train simple softmax on meta-features
        n_features = meta_features.shape[1]
        self.meta_weights = np.zeros((n_features, n_classes))
        self.meta_bias = np.zeros((1, n_classes))
        
        for epoch in range(epochs):
            # Forward
            z = np.dot(meta_features, self.meta_weights) + self.meta_bias
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Gradient
            error = y_hat - y_onehot
            dW = (1/n_samples) * np.dot(meta_features.T, error)
            db = (1/n_samples) * np.sum(error, axis=0)
            
            # Update
            self.meta_weights -= learning_rate * dW
            self.meta_bias -= learning_rate * db
        
        print(f"Meta-classifier trained with {n_features} stacked features.")
        return self
    
    def _create_meta_features(self, X):
        """Tạo meta-features từ base models"""
        n_samples = len(X)
        n_classes = len(self.CLASS_ORDER)
        
        X_test_df = pd.DataFrame(X, columns=self.FEATURE_COLS)
        X_test_df[self.LABEL_COL] = self.CLASS_ORDER[0]
        
        meta_features_list = []
        
        for name, estimator in self.estimators:
            if self.use_probas:
                if name == 'softmax':
                    probas = estimator.predict_proba(X)
                elif name == 'knn':
                    probas = self._knn_probabilities(estimator, X_test_df)
                elif name == 'naive_bayes':
                    probas = self._naive_bayes_probabilities(estimator, X)
                else:
                    preds = estimator.predict(X)
                    probas = np.zeros((n_samples, n_classes))
                    for i, pred in enumerate(preds):
                        probas[i, self._label_to_index(pred)] = 1.0
                
                meta_features_list.append(probas)
            else:
                if name == 'knn':
                    preds = estimator.predict(X_test_df)
                elif name == 'softmax':
                    pred_indices = estimator.predict(X)
                    preds = [self._index_to_label(idx) for idx in pred_indices]
                else:
                    preds = list(estimator.predict(X))
                
                one_hot = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(preds):
                    one_hot[i, self._label_to_index(pred)] = 1.0
                
                meta_features_list.append(one_hot)
        
        return np.hstack(meta_features_list)
    
    def _knn_probabilities(self, estimator, X_test_df):
        """Tính xác suất cho KNN"""
        n_samples = len(X_test_df)
        n_classes = len(self.CLASS_ORDER)
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            x_new = X_test_df.iloc[i][estimator.feature_cols].values
            
            dist = np.zeros(len(estimator.train_data))
            for j, col in enumerate(estimator.feature_cols):
                dist += (estimator.train_data[col].values - x_new[j]) ** 2
            dist = np.sqrt(dist)
            
            df_tmp = estimator.train_data.copy()
            df_tmp["dist"] = dist
            neighbors = df_tmp.sort_values("dist").head(estimator.best_k)
            
            epsilon = 1e-10
            neighbors["weight"] = 1.0 / (neighbors["dist"] + epsilon)
            
            total_weight = neighbors["weight"].sum()
            for c in self.CLASS_ORDER:
                class_weight = neighbors[neighbors[estimator.label_col] == c]["weight"].sum()
                probas[i, self._label_to_index(c)] = class_weight / total_weight
        
        return probas
    
    def _naive_bayes_probabilities(self, estimator, X):
        """Tính xác suất cho Naive Bayes"""
        n_samples = len(X)
        n_classes = len(self.CLASS_ORDER)
        probas = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            log_probs = {}
            for c in estimator.classes:
                log_prior = np.log(estimator.priors[c])
                log_likelihood = np.sum(np.log(
                    estimator._gaussian_likelihood(x, estimator.means[c], estimator.vars[c])
                ))
                log_probs[c] = log_prior + log_likelihood
            
            max_log = max(log_probs.values())
            probs = {c: np.exp(lp - max_log) for c, lp in log_probs.items()}
            total = sum(probs.values())
            
            for c, p in probs.items():
                probas[i, self._label_to_index(c)] = p / total
        
        return probas
    
    def predict_proba(self, X):
        """Tính xác suất từ meta-classifier"""
        if self.meta_weights is None:
            raise Exception("Chưa train meta-classifier. Hãy gọi fit_meta() trước.")
        
        meta_features = self._create_meta_features(X)
        z = np.dot(meta_features, self.meta_weights) + self.meta_bias
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, X):
        """Dự đoán class"""
        probas = self.predict_proba(X)
        pred_indices = np.argmax(probas, axis=1)
        return [self._index_to_label(idx) for idx in pred_indices]


# --- CẤU HÌNH ---
TRAIN_FILE = '../../data/IRIS_train.csv'
TEST_FILE = '../../data/IRIS_test.csv'


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("   ENSEMBLE METHODS (Using Pre-trained Models)")
    print("=" * 60)
    
    # Test Hard Voting
    print("\n>>> HARD VOTING <<<")
    hard_voting = HardVotingClassifier()
    X_train, y_train, X_test, y_test = hard_voting.load_data(TRAIN_FILE, TEST_FILE)
    
    if X_test is not None:
        hv_acc, _, _ = hard_voting.evaluate(X_test, y_test)
    
    # Test Soft Voting
    print("\n>>> SOFT VOTING <<<")
    soft_voting = SoftVotingClassifier()
    sv_acc, _, _ = soft_voting.evaluate(X_test, y_test)
    
    # Test Stacking (cần fit meta-classifier trên train set)
    print("\n>>> STACKING <<<")
    stacking = StackingClassifier(use_probas=True)
    stacking.fit_meta(X_train, y_train)  # Chỉ train meta-classifier
    st_acc, _, _ = stacking.evaluate(X_test, y_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} | {'Accuracy':>10}")
    print("-" * 35)
    print(f"{'Hard Voting':<20} | {hv_acc*100:>9.2f}%")
    print(f"{'Soft Voting':<20} | {sv_acc*100:>9.2f}%")
    print(f"{'Stacking':<20} | {st_acc*100:>9.2f}%")
    print("=" * 60)
