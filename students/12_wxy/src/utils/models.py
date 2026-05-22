import numpy as np

class CustomOLS:
    def __init__(self, fit_intercept=True, alpha=0.01):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        n, p = X.shape
        if self.fit_intercept:
            X = np.hstack([np.ones((n,1)), X])
        L = self.alpha * np.eye(X.shape[1])
        beta = np.linalg.inv(X.T @ X + L) @ X.T @ y
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

    def predict(self, X):
        if self.fit_intercept:
            return self.intercept_ + X @ self.coef_
        return X @ self.coef_
