"""Tests for EC50, relative potency, and BMD analysis."""

import numpy as np
import pytest

from pystatsbio.doseresponse import ll4, fit_drm, ec50, relative_potency, bmd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_ll4():
    """A fitted LL.4 model on clean data."""
    np.random.seed(42)
    dose = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    response = ll4(dose, 10, 90, 1.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    return fit_drm(dose, response, model="LL.4")


@pytest.fixture
def two_fitted_curves():
    """Two fitted LL.4 curves with different EC50s."""
    np.random.seed(42)
    dose = np.array([0, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    r1 = ll4(dose, 10, 90, 1.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    r2 = ll4(dose, 10, 90, 5.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    fit1 = fit_drm(dose, r1)
    fit2 = fit_drm(dose, r2)
    return fit1, fit2


# ---------------------------------------------------------------------------
# EC50
# ---------------------------------------------------------------------------

class TestEC50:
    """EC50 extraction with delta method CI."""

    def test_ec50_positive(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.estimate > 0

    def test_ci_contains_estimate(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.ci_lower < r.estimate < r.ci_upper

    def test_ci_contains_true_value(self, fitted_ll4):
        """CI should contain the true EC50=1.0 (most of the time)."""
        r = ec50(fitted_ll4)
        # With seed 42, this should hold
        assert r.ci_lower < 1.5  # generous check
        assert r.ci_upper > 0.5

    def test_se_positive(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.se > 0

    def test_conf_level_95(self, fitted_ll4):
        r = ec50(fitted_ll4, conf_level=0.95)
        assert r.conf_level == 0.95

    def test_wider_ci_at_99(self, fitted_ll4):
        r95 = ec50(fitted_ll4, conf_level=0.95)
        r99 = ec50(fitted_ll4, conf_level=0.99)
        assert (r99.ci_upper - r99.ci_lower) > (r95.ci_upper - r95.ci_lower)

    def test_invalid_conf_level(self, fitted_ll4):
        with pytest.raises(ValueError, match="conf_level"):
            ec50(fitted_ll4, conf_level=1.5)

    def test_invalid_method(self, fitted_ll4):
        with pytest.raises(ValueError, match="method"):
            ec50(fitted_ll4, method="profile")


# ---------------------------------------------------------------------------
# Relative potency
# ---------------------------------------------------------------------------

class TestRelativePotency:
    """Relative potency with Fieller's CI."""

    def test_ratio_positive(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ratio > 0

    def test_ratio_approximately_5(self, two_fitted_curves):
        """EC50_2/EC50_1 should be approximately 5.0."""
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ratio == pytest.approx(5.0, rel=0.5)

    def test_ci_contains_ratio(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ci_lower < r.ratio < r.ci_upper

    def test_method_is_fieller(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.method == "fieller"


# ---------------------------------------------------------------------------
# BMD
# ---------------------------------------------------------------------------

class TestBMD:
    """Benchmark dose analysis."""

    def test_bmd_positive(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmd > 0

    def test_bmdl_less_than_bmd(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmdl < r.bmd

    def test_bmdu_greater_than_bmd(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmdu > r.bmd

    def test_bmd_10_less_than_ec50(self, fitted_ll4):
        """BMD10 should be less than EC50 (10% effect < 50% effect)."""
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmd < fitted_ll4.params.ec50

    def test_higher_bmr_higher_bmd(self, fitted_ll4):
        """Higher BMR → higher BMD (need more dose for more effect)."""
        r10 = bmd(fitted_ll4, bmr=0.10)
        r25 = bmd(fitted_ll4, bmr=0.25)
        assert r25.bmd > r10.bmd

    def test_invalid_bmr(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=0.0)

    def test_invalid_bmr_type(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr_type"):
            bmd(fitted_ll4, bmr_type="invalid")

    def test_additional_risk(self, fitted_ll4):
        """Additional risk type should also work."""
        r = bmd(fitted_ll4, bmr=0.10, bmr_type="additional")
        assert r.bmd > 0
