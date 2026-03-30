import time
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# 直接同级导入（正确！）
from solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(n_samples: int, n_features: int):
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.5] * n_features)
    y = X @ true_beta + np.random.randn(n_samples) * 0.5
    X = sm.add_constant(X)
    return X, y

def test_solver(solver, X, y):
    start = time.time()
    solver.fit(X, y)
    y_pred = solver.predict(X)
    end = time.time()
    mse = mean_squared_error(y, y_pred)
    return end - start, mse

def main():
    N = 10000
    P_LOW = 10
    P_HIGH = 2000

    print("=" * 60)
    print("实验A：低维场景 N=10000, P=10")
    X_low, y_low = generate_data(N, P_LOW)
    t_anl, mse_anl = test_solver(AnalyticalSolver(), X_low, y_low)
    t_gd, mse_gd = test_solver(GradientDescentSolver(), X_low, y_low)
    print(f"解析解求解器：耗时 {t_anl:.4f}s | MSE {mse_anl:.4f}")
    print(f"梯度下降求解器：耗时 {t_gd:.4f}s | MSE {mse_gd:.4f}")

    print("\n" + "=" * 60)
    print("实验B：高维场景 N=10000, P=2000")
    X_high, y_high = generate_data(N, P_HIGH)

    t_anl_h, mse_anl_h = test_solver(AnalyticalSolver(), X_high, y_high)
    t_gd_h, mse_gd_h = test_solver(GradientDescentSolver(learning_rate=1e-6, epochs=500), X_high, y_high)
    t_skl_h, mse_skl_h = test_solver(LinearRegression(fit_intercept=False), X_high, y_high)
    t_sgd_h, mse_sgd_h = test_solver(SGDRegressor(max_iter=1000, tol=1e-3, penalty=None), X_high, y_high)

    print(f"自定义解析解：耗时 {t_anl_h:.4f}s | MSE {mse_anl_h:.4f}")
    print(f"自定义GD：耗时 {t_gd_h:.4f}s | MSE {mse_gd_h:.4f}")
    print(f"Sklearn LR：耗时 {t_skl_h:.4f}s | MSE {mse_skl_h:.4f}")
    print(f"Sklearn SGD：耗时 {t_sgd_h:.4f}s | MSE {mse_sgd_h:.4f}")
    print(f"Statsmodels OLS：高维下极慢 / 易崩溃")

if __name__ == "__main__":
    main()