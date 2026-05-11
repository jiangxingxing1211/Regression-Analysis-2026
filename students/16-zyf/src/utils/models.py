import numpy as np


class AnalyticalOLS:
    """
    Analytical OLS using the closed-form solution:
        beta = (X^T X)^(-1) X^T y
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        xtx = X.T @ X
        self.coef_ = np.linalg.pinv(xtx) @ X.T @ y

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        if np.isclose(sst, 0):
            return 1.0

        return float(1 - sse / sst)


class GradientDescentOLS:
    """
    OLS solved by Gradient Descent.

    Supports:
    - full_batch gradient descent
    - mini_batch gradient descent
    """

    def __init__(
        self,
        learning_rate=0.01,
        tol=1e-5,
        max_iter=1000,
        gd_type="full_batch",
        batch_fraction=0.2,
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.random_state = random_state

        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        self.coef_ = np.zeros(p)
        self.loss_history_ = []

        for iteration in range(self.max_iter):
            if self.gd_type == "full_batch":
                X_batch = X
                y_batch = y

            elif self.gd_type == "mini_batch":
                batch_size = max(1, int(n * self.batch_fraction))
                batch_idx = rng.choice(n, size=batch_size, replace=False)
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

            else:
                raise ValueError("gd_type must be 'full_batch' or 'mini_batch'.")

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch

            gradient = (2 / len(y_batch)) * (X_batch.T @ error_batch)
            self.coef_ -= self.learning_rate * gradient

            full_error = X @ self.coef_ - y
            loss = np.mean(full_error ** 2)
            self.loss_history_.append(loss)

            if len(self.loss_history_) > 1:
                loss_change = abs(self.loss_history_[-2] - self.loss_history_[-1])
                if loss_change < self.tol:
                    break

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)

        if np.isclose(sst, 0):
            return 1.0

        return float(1 - sse / sst)