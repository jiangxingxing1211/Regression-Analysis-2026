import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.utils.models import AnalyticalOLS, GradientDescentOLS


def add_intercept(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def setup_results_dir():
    results_dir = Path("results")

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_marketing_data():
    possible_paths = [
        Path("src/week07/data/q3_marketing.csv"),
        Path("src/week06/data/q3_marketing.csv"),
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            "Cannot find q3_marketing.csv. "
            "Please place it in src/week07/data/ or src/week06/data/."
        )

    df = pd.read_csv(data_path, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df["Region"] = df["Region"].astype(str).str.strip().str.upper()

    feature_cols = [
        "TV_Budget",
        "Radio_Budget",
        "SocialMedia_Budget",
        "Is_Holiday",
    ]

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y = df["Sales"].to_numpy(dtype=float)

    return X_raw, y, data_path


def standardize_train_val_test(X_train_raw, X_val_raw, X_test_raw):
    """
    防止数据泄露：
    只能用 Train 集的均值和标准差。
    Validation 和 Test 只能使用 Train 的 scaler。
    """
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)

    std[std == 0] = 1.0

    X_train_scaled = (X_train_raw - mean) / std
    X_val_scaled = (X_val_raw - mean) / std
    X_test_scaled = (X_test_raw - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled


def task2_cross_validation(X_raw, y):
    """
    Task 2:
    AnalyticalOLS 5-Fold Cross-Validation
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    r2_scores = []
    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_raw), start=1):
        X_train_raw = X_raw[train_idx]
        X_val_raw = X_raw[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        X_train = add_intercept(X_train_raw)
        X_val = add_intercept(X_val_raw)

        model = AnalyticalOLS()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        r2 = model.score(X_val, y_val)
        fold_rmse = rmse(y_val, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(fold_rmse)

        rows.append({
            "fold": fold_idx,
            "r2": r2,
            "rmse": fold_rmse,
        })

    summary = {
        "mean_r2": float(np.mean(r2_scores)),
        "mean_rmse": float(np.mean(rmse_scores)),
    }

    return rows, summary


def task3_hyperparameter_tuning(X_raw, y):
    """
    Task 3:
    Train / Validation / Test split.
    Use Validation set to select learning_rate.
    Use Test set only once for final comparison.
    """
    X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val_raw,
        y_train_val,
        test_size=0.25,
        random_state=42,
        shuffle=True,
    )

    X_train_scaled_raw, X_val_scaled_raw, X_test_scaled_raw = standardize_train_val_test(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    X_train = add_intercept(X_train_scaled_raw)
    X_val = add_intercept(X_val_scaled_raw)
    X_test = add_intercept(X_test_scaled_raw)

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]

    tuning_rows = []

    for lr in learning_rates:
        gd_model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
            random_state=42,
        )

        gd_model.fit(X_train, y_train)
        y_val_pred = gd_model.predict(X_val)

        val_r2 = gd_model.score(X_val, y_val)
        val_rmse = rmse(y_val, y_val_pred)

        tuning_rows.append({
            "learning_rate": lr,
            "val_r2": val_r2,
            "val_rmse": val_rmse,
            "n_iter": len(gd_model.loss_history_),
        })

    best_row = max(tuning_rows, key=lambda row: row["val_r2"])
    best_lr = best_row["learning_rate"]

    final_gd = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
        random_state=42,
    )

    final_gd.fit(X_train, y_train)

    gd_test_pred = final_gd.predict(X_test)
    gd_test_r2 = final_gd.score(X_test, y_test)
    gd_test_rmse = rmse(y_test, gd_test_pred)

    analytical = AnalyticalOLS()
    analytical.fit(X_train, y_train)

    analytical_test_pred = analytical.predict(X_test)
    analytical_test_r2 = analytical.score(X_test, y_test)
    analytical_test_rmse = rmse(y_test, analytical_test_pred)

    test_comparison = {
        "best_learning_rate": best_lr,
        "gd_test_r2": gd_test_r2,
        "gd_test_rmse": gd_test_rmse,
        "analytical_test_r2": analytical_test_r2,
        "analytical_test_rmse": analytical_test_rmse,
    }

    return tuning_rows, best_row, test_comparison, X_train, y_train


def task4_learning_curve(X_train, y_train, results_dir):
    """
    Task 4:
    Compare full_batch and mini_batch loss curves.
    """
    full_batch_model = GradientDescentOLS(
        learning_rate=0.01,
        tol=1e-5,
        max_iter=1000,
        gd_type="full_batch",
        batch_fraction=0.2,
        random_state=42,
    )

    mini_batch_model = GradientDescentOLS(
        learning_rate=0.01,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
        random_state=42,
    )

    full_batch_model.fit(X_train, y_train)
    mini_batch_model.fit(X_train, y_train)

    plt.figure(figsize=(8, 5))
    plt.plot(full_batch_model.loss_history_, label="Full Batch")
    plt.plot(mini_batch_model.loss_history_, label="Mini Batch")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini Batch")
    plt.legend()
    plt.tight_layout()

    output_path = results_dir / "learning_curve_full_vs_mini.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def write_summary_report(
    results_dir,
    data_path,
    cv_rows,
    cv_summary,
    tuning_rows,
    best_row,
    test_comparison,
    learning_curve_path,
):
    report = []

    report.append("# Week 7 Summary Report\n\n")

    report.append("## Dataset\n\n")
    report.append(f"- Data source: `{data_path}`\n")
    report.append("- Target variable: `Sales`\n")
    report.append("- Features: `TV_Budget`, `Radio_Budget`, `SocialMedia_Budget`, `Is_Holiday`\n\n")

    report.append("## Task 1: Models Implemented\n\n")
    report.append("Two OLS models were implemented in `src/utils/models.py`:\n\n")
    report.append("- `AnalyticalOLS`: uses the closed-form OLS solution.\n")
    report.append("- `GradientDescentOLS`: uses iterative gradient descent optimization.\n\n")

    report.append("## Task 2: 5-Fold Cross-Validation for AnalyticalOLS\n\n")
    report.append("| Fold | R2 | RMSE |\n")
    report.append("|---:|---:|---:|\n")

    for row in cv_rows:
        report.append(f"| {row['fold']} | {row['r2']:.4f} | {row['rmse']:.4f} |\n")

    report.append(f"\n- Mean R2: **{cv_summary['mean_r2']:.4f}**\n")
    report.append(f"- Mean RMSE: **{cv_summary['mean_rmse']:.4f}**\n\n")

    report.append("## Task 3: Learning Rate Tuning for GradientDescentOLS\n\n")
    report.append("| Learning Rate | Validation R2 | Validation RMSE | Iterations |\n")
    report.append("|---:|---:|---:|---:|\n")

    for row in tuning_rows:
        report.append(
            f"| {row['learning_rate']} | {row['val_r2']:.4f} | "
            f"{row['val_rmse']:.4f} | {row['n_iter']} |\n"
        )

    report.append(f"\nBest learning rate: **{best_row['learning_rate']}**\n\n")

    report.append("## Final Test Set Comparison\n\n")
    report.append("| Model | Test R2 | Test RMSE |\n")
    report.append("|---|---:|---:|\n")
    report.append(
        f"| AnalyticalOLS | {test_comparison['analytical_test_r2']:.4f} | "
        f"{test_comparison['analytical_test_rmse']:.4f} |\n"
    )
    report.append(
        f"| GradientDescentOLS | {test_comparison['gd_test_r2']:.4f} | "
        f"{test_comparison['gd_test_rmse']:.4f} |\n"
    )

    report.append("\n## Task 4: Feature Scaling and Data Leakage Prevention\n\n")
    report.append(
        "For GradientDescentOLS, features were standardized because gradient descent is "
        "sensitive to feature scale. The scaler was fitted only on the Train set, "
        "using Train-set means and standard deviations. The same scaler was then used "
        "to transform the Validation and Test sets. This avoids data leakage because "
        "information from Validation or Test was not used during preprocessing.\n\n"
    )

    report.append("## Learning Curve\n\n")
    report.append(f"Learning curve saved to: `{learning_curve_path}`\n\n")

    report_path = results_dir / "summary_report.md"
    report_path.write_text("".join(report), encoding="utf-8")

    return report_path


def main():
    results_dir = setup_results_dir()

    X_raw, y, data_path = load_marketing_data()

    cv_rows, cv_summary = task2_cross_validation(X_raw, y)

    tuning_rows, best_row, test_comparison, X_train, y_train = task3_hyperparameter_tuning(
        X_raw,
        y,
    )

    learning_curve_path = task4_learning_curve(X_train, y_train, results_dir)

    report_path = write_summary_report(
        results_dir=results_dir,
        data_path=data_path,
        cv_rows=cv_rows,
        cv_summary=cv_summary,
        tuning_rows=tuning_rows,
        best_row=best_row,
        test_comparison=test_comparison,
        learning_curve_path=learning_curve_path,
    )

    print("✅ Week 7 homework finished.")
    print(f"📄 Summary report saved to: {report_path}")
    print(f"📈 Learning curve saved to: {learning_curve_path}")


if __name__ == "__main__":
    main()