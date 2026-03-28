"""
Module: main
Purpose: The main execution pipeline for benchmarking solvers.
"""
import numpy as np
import time
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor

from data_generator import generate_regression_data
from solvers import AnalyticalSolver, GradientDescentSolver

def run_benchmark(n_samples: int, n_features: int, rng: np.random.Generator):
    """Run tests on different solvers and print their execution times."""
    
    print(f"\n--- Benchmarking: N={n_samples}, P={n_features} ---")
    
    # 1. Generate Data
    X, y, true_beta = generate_regression_data(n_samples, n_features, noise_std=1.0, rng=rng)
    
    # --- Custom Solvers ---
    
    # 2. Analytical Solver
    analytical_model = AnalyticalSolver()
    analytical_model.fit(X, y)
    print(f"[Custom] Analytical Time: {analytical_model.fit_time_:.4f} seconds")
    
    # 3. Gradient Descent Solver
    # Note: Students need to tune learning_rate carefully, especially in high dimensions!
    gd_model = GradientDescentSolver(learning_rate=0.01, max_iter=500)
    gd_model.fit(X, y)
    print(f"[Custom] Gradient Descent Time: {gd_model.fit_time_:.4f} seconds")
    
    # --- Industry API Showdown ---
    
    # 4. Scikit-Learn LinearRegression (Uses advanced LAPACK/C++ under the hood)
    t0 = time.perf_counter()
    sk_model = LinearRegression(fit_intercept=False).fit(X, y)
    sk_time = time.perf_counter() - t0
    print(f"[API] Sklearn LinearRegression Time: {sk_time:.4f} seconds")
    
    # 5. Scikit-Learn SGDRegressor (Stochastic Gradient Descent)
    t0 = time.perf_counter()
    sgd_model = SGDRegressor(fit_intercept=False, max_iter=100).fit(X, y)
    sgd_time = time.perf_counter() - t0
    print(f"[API] Sklearn SGDRegressor Time: {sgd_time:.4f} seconds")
    
    # 6. Statsmodels OLS (Will likely be the slowest in high dimensions due to computing p-values/Hessians)
    t0 = time.perf_counter()
    sm_model = sm.OLS(y, X).fit()
    sm_time = time.perf_counter() - t0
    print(f"[API] Statsmodels OLS Time: {sm_time:.4f} seconds")
    
    print("-" * 50)

def main():
    rng = np.random.default_rng(seed=2026)
    
    # Scenario A: Low Dimensional Data (e.g., Traditional Economics)
    # Expected result: Analytical and Statsmodels are extremely fast. GD might be slower due to loops.
    run_benchmark(n_samples=10000, n_features=10, rng=rng)
    
    # Scenario B: High Dimensional Data (e.g., Genomics / Recommendation Systems)
    # Expected result: Statsmodels and Custom Analytical hit the O(P^3) wall. SGD reigns supreme.
    run_benchmark(n_samples=10000, n_features=2000, rng=rng)

if __name__ == "__main__":
    main()