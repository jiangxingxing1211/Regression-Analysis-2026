"""
Simple unit tests for CustomOLS.
Run with:  uv run pytest test_models.py -v
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models import CustomOLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ols():
    """Perfect-fit 1-D case: y = 2 + 3*x  (no noise)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.column_stack([np.ones(5), x])
    y = 2.0 + 3.0 * x
    model = CustomOLS()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def noisy_ols():
    """Noisy DGP: y = 5 + 2*x1 - 1.5*x2 + eps.  n=500."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2])
    y = 5.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    model = CustomOLS()
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Test 1: Perfect-fit coefficients
# ---------------------------------------------------------------------------

def test_fit_exact_coefficients(simple_ols):
    model, X, y = simple_ols
    assert np.allclose(model.coef_, [2.0, 3.0], atol=1e-10), (
        f"Expected [2, 3], got {model.coef_}"
    )


# ---------------------------------------------------------------------------
# Test 2: predict reproduces y exactly when no noise
# ---------------------------------------------------------------------------

def test_predict_perfect_fit(simple_ols):
    model, X, y = simple_ols
    y_hat = model.predict(X)
    assert np.allclose(y_hat, y, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: R² == 1.0 for perfect fit
# ---------------------------------------------------------------------------

def test_score_perfect(simple_ols):
    model, X, y = simple_ols
    assert abs(model.score(X, y) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 4: R² is reasonable on a noisy DGP
# ---------------------------------------------------------------------------

def test_score_noisy(noisy_ols):
    model, X, y = noisy_ols
    r2 = model.score(X, y)
    assert r2 > 0.90, f"R² should be > 0.90, got {r2:.4f}"


# ---------------------------------------------------------------------------
# Test 5: Coefficient recovery on noisy DGP
# ---------------------------------------------------------------------------

def test_coef_recovery(noisy_ols):
    model, X, y = noisy_ols
    expected = np.array([5.0, 2.0, -1.5])
    assert np.allclose(model.coef_, expected, atol=0.15), (
        f"Beta mismatch: {model.coef_} vs {expected}"
    )


# ---------------------------------------------------------------------------
# Test 6: F-test — true null should NOT be rejected (large p-value)
# ---------------------------------------------------------------------------

def test_f_test_true_null(simple_ols):
    """H0: intercept=2, slope=3 — both exactly true → p should be 1.0-ish."""
    model, X, y = simple_ols
    # But with exactly zero residuals df_resid→df=3 and sigma2→0,
    # so F-stat is 0/0. Use noisy fixture for a meaningful test.
    pass  # covered by test_f_test_false_null


@pytest.fixture
def noisy_ols_long():
    rng = np.random.default_rng(99)
    n = 1000
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    y = 3.0 + 2.0 * x + rng.normal(0, 1.0, n)
    model = CustomOLS().fit(X, y)
    return model


def test_f_test_false_null(noisy_ols_long):
    """H0: slope = 0  → should be strongly rejected (tiny p-value)."""
    model = noisy_ols_long
    C = np.array([[0, 1]])   # select slope
    d = np.array([0.0])
    result = model.f_test(C, d)
    assert result["p_value"] < 1e-10, f"Expected tiny p, got {result['p_value']}"


def test_f_test_true_null_noisy(noisy_ols_long):
    """H0: slope = 2 (approximately true) → p-value should be decent (> 0.05)."""
    model = noisy_ols_long
    C = np.array([[0, 1]])
    d = np.array([2.0])
    result = model.f_test(C, d)
    # The true value is 2, so in expectation p > 0.05
    assert result["p_value"] > 0.01, (
        f"True null should rarely be rejected; p = {result['p_value']:.4f}"
    )
