"""Benchmark dose (BMD) analysis for toxicology.

Computes the dose at which a specified benchmark response (BMR) is
reached, with lower/upper confidence limits (BMDL/BMDU) via the
delta method.

Validates against: EPA BMDS software, R BMDL packages
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from pystatsbio.doseresponse._common import CurveParams, DoseResponseResult
from pystatsbio.doseresponse._models import _MODEL_MAP


@dataclass(frozen=True)
class BMDResult:
    """Benchmark dose result."""

    bmd: float  # benchmark dose (point estimate)
    bmdl: float  # lower confidence limit
    bmdu: float  # upper confidence limit
    bmr: float  # benchmark response level
    conf_level: float
    method: str  # 'delta'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bmd_ll4_analytical(params: CurveParams, target: float) -> float:
    """Analytical BMD for LL.4: dose = ec50 * ((top-target)/(target-bottom))^(1/hill)."""
    c, d, e, h = params.bottom, params.top, params.ec50, params.hill
    numer = d - target
    denom = target - c
    if denom == 0 or numer == 0:
        return float("nan")
    ratio = numer / denom
    if ratio <= 0:
        return float("nan")
    return e * ratio ** (1.0 / h)


def _bmd_numerical(params: CurveParams, target: float) -> float:
    """Numerical BMD via root-finding in log-dose space."""

    def f(log_dose: float) -> float:
        dose_arr = np.array([np.exp(log_dose)])
        return float(params.predict(dose_arr)[0]) - target

    try:
        log_bmd = brentq(f, -50, 50, xtol=1e-12)
        return float(np.exp(log_bmd))
    except ValueError:
        return float("nan")


def _bmd_from_params_array(
    params_arr: NDArray,
    model: str,
    target: float,
) -> float:
    """Compute BMD for a given parameter array (used for numerical gradient)."""
    cp = CurveParams.from_array(params_arr, model)
    if model == "LL.4":
        return _bmd_ll4_analytical(cp, target)
    return _bmd_numerical(cp, target)


def _bmd_delta_ci(
    fit_result: DoseResponseResult,
    bmd_val: float,
    target: float,
    conf_level: float,
) -> tuple[float, float]:
    """BMDL/BMDU via delta method with numerical gradient."""
    if np.isnan(bmd_val):
        return float("nan"), float("nan")

    params_arr = fit_result.params.to_array()
    n_params = len(params_arr)
    model = fit_result.model

    # Numerical gradient of BMD w.r.t. parameters (central differences)
    eps = 1e-6
    grad = np.zeros(n_params)
    for i in range(n_params):
        p_plus = params_arr.copy()
        p_plus[i] += eps
        p_minus = params_arr.copy()
        p_minus[i] -= eps
        bmd_plus = _bmd_from_params_array(p_plus, model, target)
        bmd_minus = _bmd_from_params_array(p_minus, model, target)
        grad[i] = (bmd_plus - bmd_minus) / (2 * eps)

    # Parameter covariance matrix
    jac = fit_result.jac
    n_obs = fit_result.n_obs
    s2 = fit_result.rss / (n_obs - n_params)

    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    var_bmd = float(grad @ cov @ grad)
    if var_bmd < 0:
        return float("nan"), float("nan")
    se_bmd = np.sqrt(var_bmd)

    z = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    # CI on log scale (BMD is positive)
    if bmd_val > 0 and se_bmd > 0:
        se_log = se_bmd / bmd_val
        log_bmd = np.log(bmd_val)
        bmdl = float(np.exp(log_bmd - z * se_log))
        bmdu = float(np.exp(log_bmd + z * se_log))
    else:
        bmdl = float("nan")
        bmdu = float("nan")

    return bmdl, bmdu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bmd(
    fit_result: DoseResponseResult,
    *,
    bmr: float = 0.10,
    bmr_type: str = "extra",
    conf_level: float = 0.95,
    method: str = "delta",
) -> BMDResult:
    """Compute benchmark dose (BMD) with BMDL/BMDU.

    Parameters
    ----------
    fit_result : DoseResponseResult
        A fitted dose-response model.
    bmr : float
        Benchmark response level (default 10 % = 0.10).
    bmr_type : str
        ``'extra'`` (extra risk) or ``'additional'`` (additional risk).
    conf_level : float
        Confidence level (default 0.95).
    method : str
        ``'delta'`` (delta method).

    Returns
    -------
    BMDResult

    Notes
    -----
    For **extra risk**: the target response is
    ``top - bmr * (top - bottom)`` (i.e. a ``bmr`` fraction of the
    full range from the upper asymptote).

    For **additional risk**: the target is
    ``top - bmr * |top - bottom|``.

    Validates against: EPA BMDS software, R BMDL packages
    """
    if not (0.0 < bmr < 1.0):
        raise ValueError(f"bmr must be in (0, 1), got {bmr}")
    if bmr_type not in ("extra", "additional"):
        raise ValueError(f"bmr_type must be 'extra' or 'additional', got {bmr_type!r}")
    if not (0.0 < conf_level < 1.0):
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    if method != "delta":
        raise ValueError(f"method must be 'delta', got {method!r}")

    p = fit_result.params
    c, d = p.bottom, p.top

    # Compute target response level
    if bmr_type == "extra":
        target = d - bmr * (d - c)
    else:  # additional
        target = d - bmr * abs(d - c)

    # Point estimate
    if fit_result.model == "LL.4":
        bmd_val = _bmd_ll4_analytical(p, target)
    else:
        bmd_val = _bmd_numerical(p, target)

    # Confidence limits
    bmdl, bmdu = _bmd_delta_ci(fit_result, bmd_val, target, conf_level)

    return BMDResult(
        bmd=bmd_val,
        bmdl=bmdl,
        bmdu=bmdu,
        bmr=bmr,
        conf_level=conf_level,
        method=method,
    )
