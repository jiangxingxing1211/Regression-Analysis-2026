"""
Module: utils.diagnostics
Purpose: Statistical diagnostics for regression models.
"""
import numpy as np
from .models import AnalyticalOLS


def calculate_vif(X: np.ndarray) -> list:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in X.

    VIF_j = 1 / (1 - R_j^2), where R_j^2 is the R-squared from regressing
    feature j on all other features.

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        List of VIF values for each feature
    """
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        # Features excluding j
        X_others = np.delete(X, j, axis=1)
        y_j = X[:, j]

        # Fit OLS on other features to predict j
        ols = AnalyticalOLS()
        try:
            ols.fit(X_others, y_j)
            # R-squared
            r_squared = ols.score(X_others, y_j)
            # VIF
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf
        vif_values.append(vif)

    return vif_values