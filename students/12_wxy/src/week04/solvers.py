"""
Custom linear regression solvers implemented with pure NumPy.
"""

import numpy as np
import time
from typing import Tuple, Optional


class AnalyticalSolver:
    """
    Analytical solver using normal equations.
    Uses np.linalg.solve for numerical stability instead of np.linalg.inv.
    """
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.fit_time_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AnalyticalSolver':
        """
        Fit linear model using normal equations.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : returns an instance of self
        """
        start_time = time.time()
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equations: (X^T X) beta = X^T y
        # Use np.linalg.solve for better numerical stability
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y
        
        # Solve linear system instead of computing inverse
        beta = np.linalg.solve(XTX, XTy)
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        self.fit_time_ = time.time() - start_time
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values
        """
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R^2 score.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples
        y : ndarray of shape (n_samples,)
            True target values
            
        Returns
        -------
        r2 : float
            R^2 coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)


class GradientDescentSolver:
    """
    Linear regression solver using Batch Gradient Descent.
    
    Gradient of MSE loss: ∇L(β) = (2/n) * X^T (Xβ - y)
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 tolerance: float = 1e-6, verbose: bool = False):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for gradient descent
        epochs : int
            Maximum number of iterations
        tolerance : float
            Stopping criterion - if change in loss is less than tolerance, stop
        verbose : bool
            Print loss every 100 epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = None
        self.fit_time_ = None
        self.loss_history_ = []
        
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, 
                          beta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss.
        
        Loss: L(β) = (1/n) * Σ(y_i - X_iβ)^2
        Gradient: ∇L(β) = (-2/n) * X^T (y - Xβ)
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (including intercept column)
        y : ndarray of shape (n_samples,)
            Target values
        beta : ndarray of shape (n_features,)
            Current parameters
            
        Returns
        -------
        gradient : ndarray of shape (n_features,)
        """
        n = X.shape[0]
        predictions = X @ beta
        errors = predictions - y
        gradient = (2 / n) * X.T @ errors
        return gradient
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray, 
                      beta: np.ndarray) -> float:
        """Compute MSE loss."""
        n = X.shape[0]
        predictions = X @ beta
        return np.mean((predictions - y) ** 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentSolver':
        """
        Fit linear model using batch gradient descent.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : returns an instance of self
        """
        start_time = time.time()
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_with_intercept.shape[1]
        
        # Initialize parameters (small random values)
        np.random.seed(42)
        beta = np.random.randn(n_features) * 0.01
        
        # Gradient descent iterations
        self.loss_history_ = []
        
        for epoch in range(self.epochs):
            # Compute gradient
            gradient = self._compute_gradient(X_with_intercept, y, beta)
            
            # Update parameters
            beta_new = beta - self.learning_rate * gradient
            
            # Compute loss for monitoring
            loss = self._compute_loss(X_with_intercept, y, beta_new)
            self.loss_history_.append(loss)
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < self.tolerance:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                beta = beta_new
                break
            
            beta = beta_new
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.6f}")
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        self.fit_time_ = time.time() - start_time
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R^2 score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)