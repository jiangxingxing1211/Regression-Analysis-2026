"""
Module: src.evaluation
Purpose: Cross-validation, hyperparameter tuning, and metrics.
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cross_validation_ols(X, y, n_splits=5, random_state=42):
    """对AnalyticalOLS做5折交叉验证"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    rmse_scores = []

    print("\n--- Task 2: 5-Fold CV on AnalyticalOLS ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        preds = model.predict(X_val)

        fold_r2 = r2_score(y_val, preds)
        fold_rmse = rmse(y_val, preds)

        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"Fold {fold}: R2={fold_r2:.4f}, RMSE={fold_rmse:.4f}")

    print(f"Average CV R2: {np.mean(r2_scores):.4f}")
    print(f"Average CV RMSE: {np.mean(rmse_scores):.4f}")
    return r2_scores, rmse_scores


def tune_learning_rate(X_train, y_train, X_val, y_val):
    """为GradientDescentOLS调学习率"""
    print("\n--- Task 3: Tuning Learning Rate for GD ---")
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_score = -np.inf

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_rmse = rmse(y_val, val_preds)

        print(f"LR={lr:<8} | Val R2={val_r2:.4f} | Val RMSE={val_rmse:.4f}")

        if val_r2 > best_score:
            best_score = val_r2
            best_lr = lr

    print(f"Selected best learning rate: {best_lr}")
    return best_lr