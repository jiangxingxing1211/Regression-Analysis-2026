"""
Week 06 Milestone Project - Main Entry Point
Tasks 2, 3, 4: evaluate_model / scenario_A / scenario_B / automated I/O

Run with:  uv run main.py   (from the week06-hw directory)
"""

import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---- local import -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from models import CustomOLS

# Workspace root  (Regression-Analysis-2026/)
_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Path to the shared marketing dataset
_DATA_PATH = _ROOT / "homework" / "week06" / "data" / "q3_marketing.csv"


# =========================================================================
# Task 4 helper: results/ folder management
# =========================================================================

def setup_results_dir() -> Path:
    """Auto-manage results/ at the template project root. Wipe & recreate."""
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# =========================================================================
# Task 2: Universal evaluator (Duck Typing / "面向接口编程")
# =========================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    Works with *any* model that exposes .fit(), .predict(), and .score().
    Returns one Markdown table row.
    """
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    r2 = model.score(X_test, y_test)
    return f"| {model_name:<35} | {fit_time:.5f} sec | {r2:.6f} |\n"


# =========================================================================
# Task 3 – Scenario A: Synthetic White-Box Test
# =========================================================================

def scenario_A_synthetic(results_dir: Path) -> CustomOLS:
    """
    Generate synthetic data with a known DGP, fit both CustomOLS and sklearn,
    assert R² is sane, and write synthetic_report.md.
    """
    print("▶  Scenario A: Synthetic data …")
    rng = np.random.default_rng(seed=2026)

    # --- DGP ------------------------------------------------------------
    n = 1000
    X_raw = rng.standard_normal((n, 3))
    X = np.column_stack([np.ones(n), X_raw])   # design matrix w/ intercept
    true_beta = np.array([5.0, 2.0, -1.5, 3.0])
    y = X @ true_beta + rng.normal(0, 1.0, n)

    # --- train / test split (80 / 20) ------------------------------------
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # --- compare models --------------------------------------------------
    custom_model = CustomOLS()
    sklearn_model = LinearRegression(fit_intercept=False)

    header = (
        "| Model                               | Fit Time      | R² (test)  |\n"
        "|-------------------------------------|---------------|------------|\n"
    )
    row_custom = evaluate_model(custom_model, X_tr, y_tr, X_te, y_te, "CustomOLS (NumPy)")
    row_sklearn = evaluate_model(sklearn_model, X_tr, y_tr, X_te, y_te, "sklearn LinearRegression")

    # --- assertions ------------------------------------------------------
    r2 = custom_model.score(X_te, y_te)
    assert r2 > 0.9, f"R² too low: {r2:.4f}"
    assert np.allclose(custom_model.coef_, true_beta, atol=0.3), (
        f"Beta mismatch: got {custom_model.coef_}, expected {true_beta}"
    )

    # --- report ----------------------------------------------------------
    report_lines = [
        "# Scenario A — Synthetic Data Baseline Report\n\n",
        "## Data Generating Process\n\n",
        f"- n = {n}  (800 train / 200 test)\n",
        f"- True β = {true_beta}\n",
        f"- Estimated β = {np.round(custom_model.coef_, 4)}\n\n",
        "## Model Comparison\n\n",
        header,
        row_custom,
        row_sklearn,
        f"\n✅ Assertions passed — R²(CustomOLS, test) = {r2:.4f}\n",
    ]
    (results_dir / "synthetic_report.md").write_text("".join(report_lines))
    print(f"   R²(test) = {r2:.4f}   ✅")
    return custom_model


# =========================================================================
# Task 3 – Scenario B: Real-World Data + Multiple OLS Instances
# =========================================================================

def scenario_B_real_world(results_dir: Path):
    """
    Load q3_marketing.csv, split by Region, fit two independent CustomOLS
    instances, run F-tests, produce plots, and write real_world_report.md.
    """
    print("▶  Scenario B: Real-world marketing data …")

    # keep_default_na=False prevents pandas from silently converting "NA" → NaN
    df = pd.read_csv(_DATA_PATH, keep_default_na=False)

    # --- helper: build design matrix + response --------------------------
    def prepare(df_market: pd.DataFrame):
        X = np.column_stack([
            np.ones(len(df_market)),
            df_market["TV_Budget"].values,
            df_market["Radio_Budget"].values,
            df_market["SocialMedia_Budget"].values,
            df_market["Is_Holiday"].values,
        ])
        y = df_market["Sales"].values
        return X, y

    df_na = df[df["Region"] == "NA"].copy()
    df_eu = df[df["Region"] == "EU"].copy()

    X_na, y_na = prepare(df_na)
    X_eu, y_eu = prepare(df_eu)

    # --- OOP power: two completely independent model instances -----------
    model_na = CustomOLS()
    model_eu = CustomOLS()
    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)

    # --- F-Test: H0 — TV, Radio, SocialMedia all have zero effect --------
    # coef order: [intercept, TV, Radio, SocialMedia, Holiday]
    C = np.array([
        [0, 1, 0, 0, 0],   # TV
        [0, 0, 1, 0, 0],   # Radio
        [0, 0, 0, 1, 0],   # SocialMedia
    ])
    d = np.zeros(3)

    na_f = model_na.f_test(C, d)
    eu_f = model_eu.f_test(C, d)

    # --- Residual plots --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Residual Plots by Market", fontsize=14)
    for ax, model, X, y, name in [
        (axes[0], model_na, X_na, y_na, "North America (NA)"),
        (axes[1], model_eu, X_eu, y_eu, "Europe (EU)"),
    ]:
        resid = y - model.predict(X)
        ax.scatter(model.predict(X), resid, alpha=0.35, s=12)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")

    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png", dpi=150)
    plt.close()

    # --- Report ----------------------------------------------------------
    def fmt_coefs(model):
        names = ["Intercept", "TV", "Radio", "SocialMedia", "Holiday"]
        return "\n".join(f"  - {n}: {v:+.4f}" for n, v in zip(names, model.coef_))

    alpha = 0.05

    def interpret(region, f_result):
        emoji = "✅" if f_result["p_value"] < alpha else "❌"
        verdict = "EFFECTIVE" if f_result["p_value"] < alpha else "NOT effective"
        return (
            f"{emoji} {region}: advertising channels are **{verdict}**  "
            f"(F = {f_result['f_stat']:.2f}, p = {f_result['p_value']:.2e})\n"
        )

    report = (
        "# Scenario B — Real-World Market Analysis\n\n"
        f"Dataset: `{_DATA_PATH.name}`  ·  "
        f"NA rows: {len(df_na)}  ·  EU rows: {len(df_eu)}\n\n"
        "---\n\n"
        "## North America (NA)\n\n"
        f"R² = **{model_na.score(X_na, y_na):.4f}**\n\n"
        "Coefficients:\n" + fmt_coefs(model_na) + "\n\n"
        "F-Test — H₀: β_TV = β_Radio = β_SocialMedia = 0\n\n"
        f"- F-stat = {na_f['f_stat']:.4f},  p-value = {na_f['p_value']:.2e}\n\n"
        "---\n\n"
        "## Europe (EU)\n\n"
        f"R² = **{model_eu.score(X_eu, y_eu):.4f}**\n\n"
        "Coefficients:\n" + fmt_coefs(model_eu) + "\n\n"
        "F-Test — H₀: β_TV = β_Radio = β_SocialMedia = 0\n\n"
        f"- F-stat = {eu_f['f_stat']:.4f},  p-value = {eu_f['p_value']:.2e}\n\n"
        "---\n\n"
        "## Conclusions\n\n"
        + interpret("NA", na_f)
        + interpret("EU", eu_f)
        + "\n> Residual plots saved to `market_comparison.png`\n"
    )
    (results_dir / "real_world_report.md").write_text(report)
    print(
        f"   NA: F={na_f['f_stat']:.2f}, p={na_f['p_value']:.2e}"
        f"   |   EU: F={eu_f['f_stat']:.2f}, p={eu_f['p_value']:.2e}   ✅"
    )


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    results_dir = setup_results_dir()
    print(f"📁 results/ → {results_dir}\n")

    scenario_A_synthetic(results_dir)
    scenario_B_real_world(results_dir)

    print("\n🎉  All tasks complete!  Check results/ for reports and plots.")
