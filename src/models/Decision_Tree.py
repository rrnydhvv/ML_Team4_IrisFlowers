import math
import csv
import numpy as np
import os

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, criterion='entropy'):
        self.max_depth = max_depth
        self.criterion = criterion  # 'entropy' hoặc 'gini'
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, self.tree))
        return predictions

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return {'value': y[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': self._most_common_label(y)}
        if X.shape[0] == 0:
            return {'value': self._most_common_label(y)}

        best_feature, best_threshold = self._choose_best_feature(X, y)
        if best_feature is None:
            return {'value': self._most_common_label(y)}

        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)

        left_tree = self._build_tree(left_X, left_y, depth + 1)
        right_tree = self._build_tree(right_X, right_y, depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _choose_best_feature(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        if self.criterion == 'entropy':
            parent_impurity = self._entropy(y)
            impurity_func = self._entropy
        elif self.criterion == 'gini':
            parent_impurity = self._gini(y)
            impurity_func = self._gini
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

        left_X, left_y, right_X, right_y = self._split_data(X, y, feature, threshold)

        if left_y.shape[0] == 0 or right_y.shape[0] == 0:
            return 0

        n = y.shape[0]
        n_left = left_y.shape[0]
        n_right = right_y.shape[0]

        weighted_impurity = (n_left / n) * impurity_func(left_y) + (n_right / n) * impurity_func(right_y)
        return parent_impurity - weighted_impurity

    def _entropy(self, y):
        if y.shape[0] == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.shape[0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _gini(self, y):
        if y.shape[0] == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.shape[0]
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _split_data(self, X, y, feature, threshold):
        mask = X[:, feature] <= threshold
        left_X = X[mask]
        left_y = y[mask]
        right_X = X[~mask]
        right_y = y[~mask]
        return left_X, left_y, right_X, right_y

    def _most_common_label(self, y):
        if y.shape[0] == 0:
            return None
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_sample(self, sample, tree):
        if 'value' in tree:
            return tree['value']
        feature = tree['feature']
        threshold = tree['threshold']
        if sample[feature] <= threshold:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

# Hàm để đọc dữ liệu từ CSV
def load_iris_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', dtype=str, skip_header=1)
    X = data[:, :-1].astype(float)
    y = data[:, -1]
    return X, y