import time
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from solvers import AnalyticalSolver, GradientDescentSolver


def generate_data(n_samples, n_features):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features + 1)
    y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n_samples) * 0.5
    return X, y


def benchmark(name, model, X, y):
    start = time.time()
    model.fit(X, y)
    fit_time = time.time() - start
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"[{name}] 耗时: {fit_time:.4f}s | MSE: {mse:.4f}")
    return fit_time, mse


if __name__ == "__main__":
    N = 10000
    LOW_DIM = 10
    HIGH_DIM = 2000

    print("=" * 50)
    print("实验A：低维数据（10特征）")
    X_low, y_low = generate_data(N, LOW_DIM)
    benchmark("自定义解析解", AnalyticalSolver(), X_low, y_low)
    benchmark("自定义梯度下降", GradientDescentSolver(), X_low, y_low)

    ols_low = sm.OLS(y_low, sm.add_constant(X_low)).fit()
    start = time.time()
    y_pred = ols_low.predict(sm.add_constant(X_low))
    fit_time = time.time() - start
    mse = mean_squared_error(y_low, y_pred)
    print(f"[statsmodels OLS] 耗时: {fit_time:.4f}s | MSE: {mse:.4f}")

    benchmark("sklearn LR", LinearRegression(), X_low, y_low)
    benchmark("sklearn SGD", SGDRegressor(random_state=42), X_low, y_low)

    print("\n" + "=" * 50)
    print("实验B：高维数据（2000特征)")
    X_high, y_high = generate_data(N, HIGH_DIM)
    benchmark("自定义解析解", AnalyticalSolver(), X_high, y_high)
    benchmark("自定义梯度下降", GradientDescentSolver(), X_high, y_high)

    ols_high = sm.OLS(y_high, sm.add_constant(X_high)).fit()
    start = time.time()
    y_pred = ols_high.predict(sm.add_constant(X_high))
    fit_time = time.time() - start
    mse = mean_squared_error(y_high, y_pred)
    print(f"[statsmodels OLS] 耗时: {fit_time:.4f}s | MSE: {mse:.4f}")

    benchmark("sklearn LR", LinearRegression(), X_high, y_high)
    benchmark("sklearn SGD", SGDRegressor(random_state=42), X_high, y_high)
