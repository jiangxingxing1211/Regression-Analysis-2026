from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from models import CustomOLS


TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DATA_CANDIDATES = [
    WORKSPACE_ROOT / "homework" / "week06" / "data" / "q3_marketing.csv",
    TEMPLATE_ROOT / "src" / "week06" / "data" / "q3_marketing.csv",
]


def resolve_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate q3_marketing.csv in expected homework or template paths.")


def setup_results_dir() -> Path:
    results_dir = TEMPLATE_ROOT / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict[str, float | str]:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    r2_value = float(model.score(X_test, y_test))
    return {
        "name": model_name,
        "fit_time": fit_time,
        "r2": r2_value,
        "row": f"| {model_name} | {fit_time:.5f} sec | {r2_value:.4f} |\n",
    }


def scenario_A_synthetic(results_dir: Path) -> dict[str, float | np.ndarray]:
    rng = np.random.default_rng(2026)
    sample_size = 1000
    X_raw = rng.normal(size=(sample_size, 3))
    X = np.column_stack([np.ones(sample_size), X_raw])
    true_beta = np.array([10.0, 2.5, -1.2, 3.8])
    noise_sigma = 1.0
    y_signal = X @ true_beta
    y = y_signal + rng.normal(scale=noise_sigma, size=sample_size)

    split_index = int(sample_size * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    signal_test = y_signal[split_index:]

    custom_eval = evaluate_model(CustomOLS(), X_train, y_train, X_test, y_test, "CustomOLS")
    sklearn_eval = evaluate_model(
        LinearRegression(fit_intercept=False),
        X_train,
        y_train,
        X_test,
        y_test,
        "sklearn LinearRegression",
    )

    custom_model = CustomOLS().fit(X_train, y_train)
    actual_r2 = float(custom_model.score(X_test, y_test))
    expected_r2 = float(np.var(signal_test) / (np.var(signal_test) + noise_sigma**2))

    assert abs(actual_r2 - expected_r2) < 0.08, (
        f"Synthetic R^2 deviates too much from the DGP expectation: {actual_r2:.4f} vs {expected_r2:.4f}"
    )
    assert np.allclose(custom_model.coef_, true_beta, atol=0.2), "Recovered coefficients are not close to the DGP."

    report = (
        "# Scenario A: Synthetic Baseline\n\n"
        "## DGP\n"
        f"- Sample size: {sample_size}\n"
        f"- True beta: {np.round(true_beta, 4).tolist()}\n"
        f"- Noise sigma: {noise_sigma:.1f}\n"
        f"- Expected test R^2 from DGP: {expected_r2:.4f}\n"
        f"- Actual test R^2 from CustomOLS: {actual_r2:.4f}\n\n"
        "## Model Comparison\n\n"
        "| Model | Fit Time | R^2 |\n"
        "| --- | --- | --- |\n"
        f"{custom_eval['row']}"
        f"{sklearn_eval['row']}"
        "\n"
        "## Conclusion\n"
        "CustomOLS recovers the synthetic DGP and stays close to sklearn on the same split.\n"
    )
    (results_dir / "synthetic_report.md").write_text(report, encoding="utf-8")
    return {
        "actual_r2": actual_r2,
        "expected_r2": expected_r2,
        "coefficients": custom_model.coef_,
    }


def build_design_matrix(df_market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([
        np.ones(len(df_market)),
        df_market["TV_Budget"].to_numpy(dtype=float),
        df_market["Radio_Budget"].to_numpy(dtype=float),
        df_market["SocialMedia_Budget"].to_numpy(dtype=float),
        df_market["Is_Holiday"].to_numpy(dtype=float),
    ])
    y = df_market["Sales"].to_numpy(dtype=float)
    return X, y


def describe_market(model: CustomOLS, X: np.ndarray, y: np.ndarray, f_result: dict[str, float], region: str) -> str:
    channel_effective = "显著" if f_result["p_value"] < 0.05 else "不显著"
    coef_names = ["Intercept", "TV", "Radio", "SocialMedia", "Holiday"]
    coef_lines = "\n".join(
        f"- {name}: {value:+.4f}" for name, value in zip(coef_names, model.coef_)
    )
    return (
        f"## {region}\n\n"
        f"- In-sample R^2: {model.score(X, y):.4f}\n"
        f"- Joint F statistic: {f_result['f_stat']:.4f}\n"
        f"- Joint F p-value: {f_result['p_value']:.4e}\n"
        f"- Advertising strategy verdict: {channel_effective}\n\n"
        "Coefficients\n\n"
        f"{coef_lines}\n\n"
    )


def scenario_B_real_world(results_dir: Path) -> dict[str, dict[str, float]]:
    data_path = resolve_data_path()
    df = pd.read_csv(data_path, keep_default_na=False)

    df_na = df.loc[df["Region"] == "NA"].copy()
    df_eu = df.loc[df["Region"] == "EU"].copy()
    X_na, y_na = build_design_matrix(df_na)
    X_eu, y_eu = build_design_matrix(df_eu)

    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    C = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    d = np.zeros(3)
    na_f = model_na.f_test(C, d)
    eu_f = model_eu.f_test(C, d)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, region, model, X_market, y_market in [
        (axes[0], "North America", model_na, X_na, y_na),
        (axes[1], "Europe", model_eu, X_eu, y_eu),
    ]:
        fitted = model.predict(X_market)
        residuals = y_market - fitted
        axis.scatter(fitted, residuals, alpha=0.45, s=14)
        axis.axhline(0.0, color="crimson", linestyle="--", linewidth=1)
        axis.set_title(region)
        axis.set_xlabel("Fitted Sales")
        axis.set_ylabel("Residual")
    figure.tight_layout()
    figure.savefig(results_dir / "market_comparison.png", dpi=160)
    plt.close(figure)

    report = (
        "# Scenario B: Real-World Market Analysis\n\n"
        f"- Data source: {data_path}\n"
        f"- NA rows: {len(df_na)}\n"
        f"- EU rows: {len(df_eu)}\n\n"
        "The null hypothesis for both markets is $H_0: \\beta_{TV}=\\beta_{Radio}=\\beta_{SocialMedia}=0$.\n\n"
        f"{describe_market(model_na, X_na, y_na, na_f, 'North America')}"
        f"{describe_market(model_eu, X_eu, y_eu, eu_f, 'Europe')}"
        "Residual scatter plots are saved to market_comparison.png.\n"
    )
    (results_dir / "real_world_report.md").write_text(report, encoding="utf-8")
    return {"NA": na_f, "EU": eu_f}


def write_summary_report(results_dir: Path, synthetic_result: dict, real_result: dict) -> None:
    summary = (
        "# Week06 Demo Summary\n\n"
        "Generated artifacts:\n\n"
        "- synthetic_report.md\n"
        "- real_world_report.md\n"
        "- market_comparison.png\n\n"
        "## Synthetic Scenario\n\n"
        f"- Expected R^2: {synthetic_result['expected_r2']:.4f}\n"
        f"- Actual R^2: {synthetic_result['actual_r2']:.4f}\n\n"
        "## Real-World Scenario\n\n"
        f"- NA F-test p-value: {real_result['NA']['p_value']:.4e}\n"
        f"- EU F-test p-value: {real_result['EU']['p_value']:.4e}\n"
    )
    (results_dir / "summary_report.md").write_text(summary, encoding="utf-8")


def main() -> None:
    results_dir = setup_results_dir()
    synthetic_result = scenario_A_synthetic(results_dir)
    real_result = scenario_B_real_world(results_dir)
    write_summary_report(results_dir, synthetic_result, real_result)
    print(f"Results regenerated under: {results_dir}")


if __name__ == "__main__":
    main()