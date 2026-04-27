import numpy as np

class AnalyticalOLS:
    """解析解OLS（来自Week06）"""
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.n = None
        self.k = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n, self.k = X.shape
        xtx = X.T @ X
        xtx += np.eye(self.k) * 1e-6
        xtx_inv = np.linalg.inv(xtx)
        beta_hat = xtx_inv @ X.T @ y

        residuals = y - X @ beta_hat
        self.sigma2_ = (residuals @ residuals) / (self.n - self.k)
        self.cov_matrix_ = self.sigma2_ * xtx_inv
        self.df_resid_ = self.n - self.k
        self.coef_ = beta_hat
        return self

    def predict(self, X: np.ndarray):
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst)

class GradientDescentOLS:
    """Week07 梯度下降优化引擎"""
    def __init__(
        self,
        learning_rate=0.01,
        tol=1e-5,
        max_iter=1000,
        gd_type="full_batch",
        batch_fraction=0.2
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X, y, seed=42):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, batch_size, replace=False)
                Xb, yb = X[indices], y[indices]
            else:
                Xb, yb = X, y

            y_pred = Xb @ self.coef_
            error = y_pred - yb
            grad = (2 / len(Xb)) * (Xb.T @ error)

            self.coef_ -= self.learning_rate * grad

            mse = np.mean((X @ self.coef_ - y) ** 2)
            self.loss_history_.append(mse)

            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst