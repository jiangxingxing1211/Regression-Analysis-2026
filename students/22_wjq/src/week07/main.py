"""
Module: week07.main
Executes:
- Task 2: 5‑fold CV for AnalyticalOLS
- Task 3: Hyperparameter tuning for GradientDescentOLS (learning rate)
- Task 4: Feature scaling (no leakage) + learning curve plot
- Final comparison on test set
"""

import sys
from pathlib import Path

# Add parent directory to path so that `utils` can be imported
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def task2_cross_validation(X, y):
    """5‑fold CV for AnalyticalOLS (assuming X already includes intercept)."""
    print("\n" + "=" * 60)
    print("Task 2: 5‑Fold Cross‑Validation on AnalyticalOLS")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        y_pred = model.predict(X_val)

        fold_r2 = r2_score(y_val, y_pred)
        fold_rmse = rmse(y_val, y_pred)

        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"Fold {fold}: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")

    print(f"\nAverage CV R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}\n")


def task3_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Grid search over learning rates for GradientDescentOLS."""
    print("\n" + "=" * 60)
    print("Task 3: Tuning Learning Rate for Gradient Descent")
    print("=" * 60)

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_val_r2 = -np.inf
    best_val_rmse = None

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = rmse(y_val, y_val_pred)

        print(f"LR = {lr:<8} | Val R² = {val_r2:.4f} | Val RMSE = {val_rmse:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_rmse = val_rmse
            best_lr = lr

    print(f"\n✅ Best learning rate: {best_lr} (Val R² = {best_val_r2:.4f}, RMSE = {best_val_rmse:.4f})")
    return best_lr


def task4_plot_learning_curve(X_train, y_train, save_path: Path):
    """Compare full‑batch vs mini‑batch loss trajectories."""
    print("\n" + "=" * 60)
    print("Task 4: Learning Curve (Full Batch vs Mini Batch)")
    print("=" * 60)

    # Full batch GD
    model_full = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="full_batch",
        max_iter=300,
        tol=1e-8,
    ).fit(X_train, y_train)

    # Mini‑batch GD
    model_mini = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
        tol=1e-8,
    ).fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    epochs_full = range(1, len(model_full.loss_history_) + 1)
    epochs_mini = range(1, len(model_mini.loss_history_) + 1)
    plt.plot(epochs_full, model_full.loss_history_, label="Full Batch GD", linewidth=2, color="steelblue")
    plt.plot(epochs_mini, model_mini.loss_history_, label="Mini-Batch GD", linewidth=2, color="darkorange", alpha=0.8)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Learning Curve: Full Batch vs Mini‑Batch (lr=0.01)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Learning curve saved to {save_path}")


def main():
    # -------------------- Setup --------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # -------------------- Load data --------------------
    # Adjust path to your CSV file if necessary
    data_path = Path("homework/week06/data/q3_marketing.csv")
    if not data_path.exists():
        # Fallback: try relative path from current working directory
        data_path = Path("q3_marketing.csv")
    df = pd.read_csv(data_path)

    # Specify feature columns and target (modify if needed)
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    target_col = "Sales"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    # -------------------- Task 2: CV for AnalyticalOLS --------------------
    # Note: CV is done on raw features (without scaling) because OLS is scale-invariant.
    # We must add an intercept column manually.
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    task2_cross_validation(X_with_intercept, y)

    # -------------------- Task 3 & 4: Train/Val/Test split --------------------
    # First split: 60% train, 40% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    # Second split: 20% validation, 20% test (from the 40%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # -------------------- Feature Scaling (NO LEAKAGE) --------------------
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Add intercept column AFTER scaling (do not scale the intercept)
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled   = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled  = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # -------------------- Hyperparameter Tuning --------------------
    best_lr = task3_hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val)

    # -------------------- Final Test: GD vs Analytical OLS --------------------
    print("\n" + "=" * 60)
    print("Final Test Set Comparison")
    print("=" * 60)

    # Train final GD model with best lr
    gd_final = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train)

    # Analytical OLS model
    ols_final = AnalyticalOLS().fit(X_train_scaled, y_train)

    # Predictions on test set
    gd_pred = gd_final.predict(X_test_scaled)
    ols_pred = ols_final.predict(X_test_scaled)

    gd_r2 = r2_score(y_test, gd_pred)
    gd_rmse = rmse(y_test, gd_pred)
    ols_r2 = r2_score(y_test, ols_pred)
    ols_rmse = rmse(y_test, ols_pred)

    print(f"GradientDescentOLS  | Test R² = {gd_r2:.4f} | Test RMSE = {gd_rmse:.4f}")
    print(f"AnalyticalOLS       | Test R² = {ols_r2:.4f} | Test RMSE = {ols_rmse:.4f}")

    # -------------------- Task 4: Learning Curve Plot --------------------
    # Use training data (scaled, with intercept) for the learning curve comparison
    task4_plot_learning_curve(X_train_scaled, y_train, results_dir / "learning_curve_full_vs_mini.png")

    # -------------------- Generate Markdown Report --------------------
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Week 7 Assignment Report\n\n")
        f.write("## 1. Gradient Descent Implementation\n")
        f.write("The `GradientDescentOLS` class supports both full‑batch and mini‑batch gradient descent. ")
        f.write("It updates coefficients using the gradient of the MSE loss. Early stopping is triggered when the change in loss falls below `tol`.\n\n")

        f.write("## 2. Best Learning Rate\n")
        f.write(f"After tuning on the validation set, the best learning rate is **{best_lr}**.\n\n")

        f.write("## 3. Test Set Performance\n")
        f.write(f"- **GradientDescentOLS**: R² = {gd_r2:.4f}, RMSE = {gd_rmse:.4f}\n")
        f.write(f"- **AnalyticalOLS**: R² = {ols_r2:.4f}, RMSE = {ols_rmse:.4f}\n\n")

        f.write("## 4. Feature Scaling and Data Leakage Prevention\n")
        f.write("StandardScaler was fitted **only on the training set**. The same scaler (with training mean/std) was used to transform validation and test sets. ")
        f.write("This prevents information from validation/test leaking into training. The intercept column was added **after** scaling, so it remains 1 and is not scaled.\n\n")


    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()