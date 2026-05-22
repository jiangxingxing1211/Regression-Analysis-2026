"""
Benchmarking linear regression solvers across different dimensionality scenarios.
"""

import numpy as np
import time
import warnings
from typing import Dict, Any

# Suppress warnings from statsmodels
warnings.filterwarnings('ignore')

# Import custom solvers
from solvers import AnalyticalSolver, GradientDescentSolver

# Import industrial APIs
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def generate_synthetic_data(n_samples: int, n_features: int, 
                            noise_std: float = 0.1, 
                            random_seed: int = 42) -> tuple:
    """
    Generate synthetic linear regression data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise_std : float
        Standard deviation of Gaussian noise
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    true_coef : ndarray of shape (n_features,)
        True coefficients used to generate data
    """
    np.random.seed(random_seed)
    
    # Generate feature matrix with some correlation
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients (sparse for high-dim)
    if n_features > 100:
        # Sparse coefficients for high-dimensional case
        true_coef = np.random.randn(n_features) * 0.5
        true_coef[np.random.rand(n_features) > 0.1] = 0  # 90% sparsity
    else:
        true_coef = np.random.randn(n_features)
    
    # Generate target with noise
    y = X @ true_coef + noise_std * np.random.randn(n_samples)
    
    return X, y, true_coef


def benchmark_solver(solver, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     name: str) -> Dict[str, Any]:
    """
    Benchmark a single solver.
    
    Returns
    -------
    dict containing:
        - name: solver name
        - fit_time: training time in seconds
        - mse: mean squared error on test set
        - r2: R^2 score on test set
    """
    try:
        # For SGDRegressor, we need to scale features
        if name == "SGDRegressor":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            fit_start = time.time()
            solver.fit(X_train_scaled, y_train)
            fit_time = time.time() - fit_start
            y_pred = solver.predict(X_test_scaled)
        else:
            fit_start = time.time()
            solver.fit(X_train, y_train)
            fit_time = time.time() - fit_start
            y_pred = solver.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = solver.score(X_test, y_test) if hasattr(solver, 'score') else None
        
        return {
            'name': name,
            'fit_time': fit_time,
            'mse': mse,
            'r2': r2
        }
    except Exception as e:
        return {
            'name': name,
            'fit_time': float('inf'),
            'mse': float('nan'),
            'r2': float('nan'),
            'error': str(e)
        }


def run_experiment(n_samples: int, n_features: int, 
                   scenario_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Run complete benchmarking experiment.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {scenario_name}")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"{'='*60}")
    
    # Generate data
    X, y, true_coef = generate_synthetic_data(n_samples, n_features)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    results = {}
    
    # 1. Custom Analytical Solver
    print("\n>> Running Custom Analytical Solver...")
    solver = AnalyticalSolver()
    results['Custom Analytical'] = benchmark_solver(
        solver, X_train, y_train, X_test, y_test, "AnalyticalSolver"
    )
    print(f"   Time: {results['Custom Analytical']['fit_time']:.4f}s, "
          f"MSE: {results['Custom Analytical']['mse']:.6f}")
    
    # 2. Custom Gradient Descent Solver
    print("\n>> Running Custom Gradient Descent Solver...")
    # Adjust learning rate based on feature dimension
    lr = 0.01 if n_features <= 10 else 0.001 / (n_features / 100)
    epochs = 500 if n_features <= 10 else 2000
    
    solver = GradientDescentSolver(
        learning_rate=lr,
        epochs=epochs,
        tolerance=1e-6,
        verbose=False
    )
    results['Custom GD'] = benchmark_solver(
        solver, X_train, y_train, X_test, y_test, "GradientDescentSolver"
    )
    print(f"   Time: {results['Custom GD']['fit_time']:.4f}s, "
          f"MSE: {results['Custom GD']['mse']:.6f}")
    
    # 3. Statsmodels OLS
    print("\n>> Running Statsmodels OLS...")
    try:
        # Add constant for intercept
        X_train_sm = sm.add_constant(X_train)
        fit_start = time.time()
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        fit_time = time.time() - fit_start
        X_test_sm = sm.add_constant(X_test)
        y_pred = sm_model.predict(X_test_sm)
        mse = mean_squared_error(y_test, y_pred)
        results['Statsmodels OLS'] = {
            'name': 'Statsmodels OLS',
            'fit_time': fit_time,
            'mse': mse,
            'r2': sm_model.rsquared
        }
        print(f"   Time: {fit_time:.4f}s, MSE: {mse:.6f}")
    except Exception as e:
        print(f"   FAILED: {str(e)}")
        results['Statsmodels OLS'] = {
            'name': 'Statsmodels OLS',
            'fit_time': float('inf'),
            'mse': float('nan'),
            'error': str(e)
        }
    
    # 4. Scikit-learn LinearRegression
    print("\n>> Running Sklearn LinearRegression...")
    solver = LinearRegression()
    results['Sklearn Linear'] = benchmark_solver(
        solver, X_train, y_train, X_test, y_test, "LinearRegression"
    )
    print(f"   Time: {results['Sklearn Linear']['fit_time']:.4f}s, "
          f"MSE: {results['Sklearn Linear']['mse']:.6f}")
    
    # 5. Scikit-learn SGDRegressor
    print("\n>> Running Sklearn SGDRegressor...")
    solver = SGDRegressor(
        max_iter=1000,
        tol=1e-4,
        learning_rate='optimal',
        eta0=0.001,
        random_state=42,
        verbose=0
    )
    results['Sklearn SGD'] = benchmark_solver(
        solver, X_train, y_train, X_test, y_test, "SGDRegressor"
    )
    print(f"   Time: {results['Sklearn SGD']['fit_time']:.4f}s, "
          f"MSE: {results['Sklearn SGD']['mse']:.6f}")
    
    return results


def main():
    """Main execution function."""
    print("Linear Regression Solver Benchmark")
    print("=" * 60)
    
    all_results = {}
    
    # Experiment A: Low-dimensional scenario
    results_low = run_experiment(
        n_samples=10000,
        n_features=10,
        scenario_name="LOW-DIMENSION (N=10000, P=10)"
    )
    all_results['low_dim'] = results_low
    
    # Experiment B: High-dimensional scenario
    results_high = run_experiment(
        n_samples=10000,
        n_features=2000,
        scenario_name="HIGH-DIMENSION (N=10000, P=2000)"
    )
    all_results['high_dim'] = results_high
    
    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY - Time Comparison (seconds)")
    print("=" * 80)
    
    print("\n| Solver | Low-Dim (P=10) | High-Dim (P=2000) |")
    print("|--------|----------------|-------------------|")
    
    solvers = ['Custom Analytical', 'Custom GD', 'Statsmodels OLS', 
               'Sklearn Linear', 'Sklearn SGD']
    
    for solver in solvers:
        low_time = all_results['low_dim'].get(solver, {}).get('fit_time', float('nan'))
        high_time = all_results['high_dim'].get(solver, {}).get('fit_time', float('nan'))
        
        low_str = f"{low_time:.4f}" if not np.isnan(low_time) and low_time != float('inf') else "FAILED"
        high_str = f"{high_time:.4f}" if not np.isnan(high_time) and high_time != float('inf') else "FAILED"
        
        print(f"| {solver} | {low_str}s | {high_str}s |")
    
    print("\n" + "=" * 80)
    print("SUMMARY - MSE Comparison")
    print("=" * 80)
    
    print("\n| Solver | Low-Dim (P=10) | High-Dim (P=2000) |")
    print("|--------|----------------|-------------------|")
    
    for solver in solvers:
        low_mse = all_results['low_dim'].get(solver, {}).get('mse', float('nan'))
        high_mse = all_results['high_dim'].get(solver, {}).get('mse', float('nan'))
        
        low_str = f"{low_mse:.6f}" if not np.isnan(low_mse) else "FAILED"
        high_str = f"{high_mse:.6f}" if not np.isnan(high_mse) else "FAILED"
        
        print(f"| {solver} | {low_str} | {high_str} |")
    
    return all_results


if __name__ == "__main__":
    results = main()