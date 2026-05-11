"""Module: week07.main
Purpose: Cross-validation, hyperparameter tuning, and generalization analysis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS, GradientDescentOLS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STUDENT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = STUDENT_ROOT.parent.parent
DATA_PATH = PROJECT_ROOT / "homework" / "week06" / "data" / "q3_marketing.csv"
RESULTS_DIR = STUDENT_ROOT / "results"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ===== Task 2: 5-Fold Cross-Validation for AnalyticalOLS ===================
def task_cross_validation(X, y):
    print("=" * 60)
    print("Task 2: 5-Fold Cross-Validation — AnalyticalOLS")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, rmse_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        preds = model.predict(X_val)

        fold_r2 = r2_score(y_val, preds)
        fold_rmse = rmse(y_val, preds)
        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"  Fold {fold}: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")

    print(f"\n  Average R²   : {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"  Average RMSE : {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    return r2_scores, rmse_scores


# ===== Task 3: Hyperparameter Tuning for GradientDescentOLS ================
def task_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("Task 3: Learning Rate Tuning — GradientDescentOLS (mini_batch)")
    print("=" * 60)

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr, best_r2 = None, -np.inf
    results = []

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
        results.append((lr, val_r2, val_rmse))

        marker = ""
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_lr = lr
            marker = " ← best"

        print(f"  LR = {lr:<10} | Val R² = {val_r2:+.4f} | Val RMSE = {val_rmse:.4f}{marker}")

    print(f"\n  Best learning rate: {best_lr}")
    return best_lr, results


# ===== Task 4-2: Learning Curve (full_batch vs mini_batch) ==================
def task_plot_learning_curve(X_train, y_train, results_dir: Path):
    print("\n" + "=" * 60)
    print("Task 4: Learning Curve — Full-Batch vs Mini-Batch")
    print("=" * 60)

    model_full = GradientDescentOLS(
        learning_rate=0.01, gd_type="full_batch", max_iter=300
    ).fit(X_train, y_train)

    model_mini = GradientDescentOLS(
        learning_rate=0.01, gd_type="mini_batch", batch_fraction=0.1, max_iter=300
    ).fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model_full.loss_history_, label="Full-Batch GD", color="steelblue")
    ax.plot(model_mini.loss_history_, label="Mini-Batch GD", color="darkorange", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Learning Curve: Full-Batch vs Mini-Batch Gradient Descent")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = results_dir / "learning_curve_full_vs_mini.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ===== Main =================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load data ----
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    target_col = "Sales"

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # ---- Task 2: CV on AnalyticalOLS (raw features + intercept column) ----
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    cv_r2, cv_rmse = task_cross_validation(X_with_intercept, y)

    # ---- Task 3 & 4: Train / Val / Test split ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"\nSplit → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Task 4-1: Feature scaling (fit on Train only) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Add intercept column AFTER scaling
    X_train_sc = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_sc = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_sc = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # ---- Task 3: Hyperparameter tuning ----
    best_lr, tuning_results = task_hyperparameter_tuning(
        X_train_sc, y_train, X_val_sc, y_val
    )

    # ---- Final Test: GD (best lr) vs AnalyticalOLS ----
    print("\n" + "=" * 60)
    print("Final Comparison on Test Set")
    print("=" * 60)

    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_sc, y_train)

    ols_model = AnalyticalOLS().fit(X_train_sc, y_train)

    gd_preds = gd_model.predict(X_test_sc)
    ols_preds = ols_model.predict(X_test_sc)

    gd_r2 = r2_score(y_test, gd_preds)
    gd_rmse = rmse(y_test, gd_preds)
    ols_r2 = r2_score(y_test, ols_preds)
    ols_rmse = rmse(y_test, ols_preds)

    print(f"  GradientDescentOLS (lr={best_lr}) → Test R² = {gd_r2:.4f}, RMSE = {gd_rmse:.4f}")
    print(f"  AnalyticalOLS                    → Test R² = {ols_r2:.4f}, RMSE = {ols_rmse:.4f}")

    # ---- Task 4-2: Learning curve ----
    task_plot_learning_curve(X_train_sc, y_train, RESULTS_DIR)

    # ---- Write summary report ----
    report_lines = [
        "# Week 7 — Summary Report",
        "",
        "## 1. GradientDescentOLS Implementation",
        "",
        "- Uses MSE as the loss function.",
        "- Supports `full_batch` and `mini_batch` modes.",
        "- `mini_batch` randomly samples `batch_fraction` of training data each epoch.",
        "- Early stopping triggers when the loss change between consecutive epochs falls below `tol`.",
        "- Intercept is handled by prepending a column of ones to X (external to the model).",
        "",
        "## 2. 5-Fold Cross-Validation (AnalyticalOLS)",
        "",
        f"- Average R²  : {np.mean(cv_r2):.4f} (±{np.std(cv_r2):.4f})",
        f"- Average RMSE: {np.mean(cv_rmse):.4f} (±{np.std(cv_rmse):.4f})",
        "",
        "## 3. Best Learning Rate",
        "",
        f"- Selected: **{best_lr}** (highest Validation R²)",
        "",
        "| Learning Rate | Val R² | Val RMSE |",
        "|---|---|---|",
    ]
    for lr, val_r2, val_rmse in tuning_results:
        report_lines.append(f"| {lr} | {val_r2:.4f} | {val_rmse:.4f} |")

    report_lines += [
        "",
        "## 4. Test Set Comparison",
        "",
        "| Model | Test R² | Test RMSE |",
        "|---|---|---|",
        f"| GradientDescentOLS (lr={best_lr}) | {gd_r2:.4f} | {gd_rmse:.4f} |",
        f"| AnalyticalOLS | {ols_r2:.4f} | {ols_rmse:.4f} |",
        "",
        "## 5. Feature Scaling & Data Leakage Prevention",
        "",
        "- `StandardScaler` is fit **only on the training set**; the same mean/std are applied to validation and test sets.",
        "- This prevents information from validation/test sets from leaking into the training process.",
        "- If scaling were done on the full dataset, the model would indirectly see validation/test statistics,",
        "  leading to over-optimistic generalization estimates.",
        "- The intercept column (all ones) is appended **after** scaling so it is not affected by standardization.",
        "",
        "## 6. Learning Curve",
        "",
        "See `results/learning_curve_full_vs_mini.png`.",
        "",
        "- Full-batch GD converges smoothly but each epoch is more expensive.",
        "- Mini-batch GD shows noisier descent but can escape shallow local minima and is cheaper per epoch.",
    ]

    report_path = RESULTS_DIR / "summary_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
