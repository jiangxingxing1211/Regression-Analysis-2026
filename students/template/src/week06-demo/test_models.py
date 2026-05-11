from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from main import scenario_A_synthetic, setup_results_dir
from models import CustomOLS


def test_custom_ols_recovers_exact_coefficients():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([np.ones_like(x), x])
    y = 1.5 + 2.0 * x

    model = CustomOLS().fit(X, y)

    assert np.allclose(model.coef_, np.array([1.5, 2.0]), atol=1e-10)
    assert np.allclose(model.predict(X), y, atol=1e-10)


def test_f_test_rejects_false_null():
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    X = np.column_stack([np.ones_like(x), x])
    y = 3.0 + 4.5 * x + rng.normal(scale=0.5, size=x.shape[0])

    model = CustomOLS().fit(X, y)
    result = model.f_test(np.array([[0.0, 1.0]]), np.array([0.0]))

    assert result["p_value"] < 1e-12


def test_scenario_a_generates_report_file():
    results_dir = setup_results_dir()
    try:
        result = scenario_A_synthetic(results_dir)
        report_path = results_dir / "synthetic_report.md"

        assert report_path.exists()
        assert result["actual_r2"] > 0.8
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)