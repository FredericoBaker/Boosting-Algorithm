import numpy as np
import pandas as pd

class DecisionStump:
    def __init__(self, decision_limits=None):
        if decision_limits is None:
            decision_limits = [-1.5, -0.5, 0.5, 1.5]
        self.decision_limits = decision_limits
        self.feature_idx = None
        self.decision_limit = None
        self.greater_than_limit = None
        self.error = float('inf')

    def fit(self, X, y, weights):
        n, n_features = X.shape

        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx]

            for decision_limit in self.decision_limits:
                greater_than_limit = True
                predictions = np.ones(n)

                # Values greater than limit are predicted as +1, otherwise -1
                predictions[X_feature <= decision_limit] = -1

                error = np.sum(weights * (predictions != y)) / np.sum(weights)

                if error > 0.5:
                    # It means that would be better to predict values greater than limit as -1
                    error = 1 - error
                    greater_than_limit = not greater_than_limit

                if error < self.error:
                    self.feature_idx = feature_idx
                    self.decision_limit = decision_limit
                    self.greater_than_limit = greater_than_limit
                    self.error = error

    def predict(self, X):
        n = X.shape[0]
        X_feature = X[:, self.feature_idx]
        predictions = np.ones(n)

        if self.greater_than_limit:
            predictions[X_feature <= self.decision_limit] = -1
        else:
            predictions[X_feature > self.decision_limit] = -1

        return predictions

class AdaBoost:
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_iterations):
            stump = DecisionStump()

            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            error = np.sum(weights * (predictions != y)) / np.sum(weights)

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stumps_preds = np.array([alpha * stump.predict(X) for stump, alpha in zip(self.models, self.alphas)])
        return np.sign(np.sum(stumps_preds, axis=0))
