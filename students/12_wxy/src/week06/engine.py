"""
Custom Ordinary Least Squares (OLS) Regression Engine
Implemented from scratch using NumPy
"""

import numpy as np
import scipy.stats as stats


class CustomOLS:
    """
    Custom OLS Regression Model built with NumPy.
    
    Attributes:
        coef_ (np.ndarray): Estimated coefficients (beta_hat)
        cov_matrix_ (np.ndarray): Covariance matrix of coefficients
        sigma2_ (float): Estimated error variance
        df_resid_ (int): Residual degrees of freedom
        n_ (int): Number of observations
        k_ (int): Number of parameters (including intercept)
        X_ (np.ndarray): Design matrix used in fit
        residuals_ (np.ndarray): Residuals from fit
    """
    
    def __init__(self, add_intercept: bool = True):
        """
        Initialize CustomOLS model.
        
        Args:
            add_intercept (bool): Whether to automatically add an intercept column.
                                 Default True for user convenience.
        """
        self.add_intercept = add_intercept
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.n_ = None
        self.k_ = None
        self.X_ = None
        self.y_ = None
        self.residuals_ = None
        
    def _add_intercept_column(self, X: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to the design matrix for intercept term.
        
        Args:
            X (np.ndarray): Original feature matrix
            
        Returns:
            np.ndarray: Design matrix with intercept column as first column
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack([ones, X])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomOLS':
        """
        Fit the OLS model.
        
        Formula: beta_hat = (X^T X)^-1 X^T y
                 sigma2 = (y - Xβ)^T (y - Xβ) / (n - k)
                 cov_matrix = sigma2 * (X^T X)^-1
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target vector of shape (n_samples,)
            
        Returns:
            self: Returns self for method chaining
        """
        # Store original X for reference
        X_original = X
        
        # Add intercept if requested
        if self.add_intercept:
            X = self._add_intercept_column(X)
        
        self.n_, self.k_ = X.shape
        self.X_ = X
        self.y_ = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Check for full column rank
        rank = np.linalg.matrix_rank(X)
        if rank < self.k_:
            raise ValueError(f"Design matrix is rank-deficient (rank={rank}, expected={self.k_}). "
                           "Check for multicollinearity.")
        
        # Calculate beta_hat = (X^T X)^-1 X^T y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ self.y_
        self.coef_ = (XtX_inv @ XtY).flatten()
        
        # Calculate residuals
        y_pred = X @ self.coef_
        self.residuals_ = (self.y_.flatten() - y_pred)
        
        # Calculate sigma2 (unbiased estimate of error variance)
        SSE = np.sum(self.residuals_ ** 2)
        self.df_resid_ = self.n_ - self.k_
        self.sigma2_ = SSE / self.df_resid_
        
        # Calculate covariance matrix of coefficients
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        
        # Add intercept if model was fit with it
        if self.add_intercept:
            X = self._add_intercept_column(X)
        
        return X @ self.coef_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        R^2 = 1 - SSE/SST
        where SSE = sum((y - y_pred)^2), SST = sum((y - y_mean)^2)
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            float: R-squared value
        """
        y_pred = self.predict(X)
        y = y.flatten() if y.ndim > 1 else y
        
        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - SSE / SST
    
    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        Perform General Linear Hypothesis Test: C * beta = d
        
        F statistic formula:
        F = (SSE_restricted - SSE_unrestricted) / q
            ---------------------------------------
            SSE_unrestricted / (n - k)
        
        Or equivalently using Wald statistic:
        F = (Cβ - d)^T [C (X^T X)^-1 C^T]^-1 (Cβ - d) / q
        
        Args:
            C (np.ndarray): Constraint matrix of shape (q, k)
            d (np.ndarray): Constraint vector of shape (q,)
            
        Returns:
            dict: {'f_stat': float, 'p_value': float, 'q': int, 'reject_null': bool}
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before performing F-test.")
        
        # Ensure d is 2D column vector
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        
        q = C.shape[0]  # Number of constraints
        
        # Calculate (C * beta_hat - d)
        beta_vector = self.coef_.reshape(-1, 1)
        C_beta_minus_d = C @ beta_vector - d
        
        # Calculate the covariance matrix part: C (X^T X)^-1 C^T
        # Recall: cov_matrix = sigma2 * (X^T X)^-1, so (X^T X)^-1 = cov_matrix / sigma2
        XtX_inv = self.cov_matrix_ / self.sigma2_
        C_XtX_inv_CT = C @ XtX_inv @ C.T
        
        # Calculate Wald statistic numerator
        # Using pseudo-inverse for numerical stability
        try:
            C_XtX_inv_CT_inv = np.linalg.inv(C_XtX_inv_CT)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            C_XtX_inv_CT_inv = np.linalg.pinv(C_XtX_inv_CT)
        
        numerator = C_beta_minus_d.T @ C_XtX_inv_CT_inv @ C_beta_minus_d
        numerator = numerator.item() / q
        
        # F statistic
        f_stat = numerator / self.sigma2_
        
        # P-value from F-distribution
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        
        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'q': q,
            'df_resid': self.df_resid_,
            'reject_null': p_value < 0.05
        }
    
    def get_coefficients(self) -> dict:
        """
        Get coefficient estimates with standard errors, t-stats, and p-values.
        
        Returns:
            dict: Dictionary containing coefficient statistics
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fit first.")
        
        se = np.sqrt(np.diag(self.cov_matrix_))
        t_stats = self.coef_ / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_resid_))
        
        # Create coefficient names
        if self.add_intercept:
            names = ['Intercept'] + [f'X{i+1}' for i in range(len(self.coef_) - 1)]
        else:
            names = [f'X{i+1}' for i in range(len(self.coef_))]
        
        return {
            'coefficient': self.coef_,
            'std_error': se,
            't_statistic': t_stats,
            'p_value': p_values,
            'names': names
        }
    
    def summary(self) -> str:
        """
        Generate a summary table of regression results.
        
        Returns:
            str: Formatted summary string
        """
        if self.coef_ is None:
            return "Model not fitted yet."
        
        coef_stats = self.get_coefficients()
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("Custom OLS Regression Results")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Dep. Variable:     y")
        summary_lines.append(f"Model:             OLS")
        summary_lines.append(f"No. Observations:  {self.n_}")
        summary_lines.append(f"Df Residuals:      {self.df_resid_}")
        summary_lines.append(f"Df Model:          {self.k_ - 1 if self.add_intercept else self.k_}")
        summary_lines.append(f"Sigma^2:           {self.sigma2_:.6f}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'':<12} {'coef':>12} {'std err':>12} {'t':>12} {'P>|t|':>12}")
        summary_lines.append("-" * 80)
        
        for i, name in enumerate(coef_stats['names']):
            summary_lines.append(
                f"{name:<12} {coef_stats['coefficient'][i]:>12.6f} "
                f"{coef_stats['std_error'][i]:>12.6f} {coef_stats['t_statistic'][i]:>12.6f} "
                f"{coef_stats['p_value'][i]:>12.6f}"
            )
        
        summary_lines.append("-" * 80)
        
        return "\n".join(summary_lines)
    