import numpy as np

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
