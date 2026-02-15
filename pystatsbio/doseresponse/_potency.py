"""EC50/IC50 estimation and relative potency analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatsbio.doseresponse._common import DoseResponseResult


@dataclass(frozen=True)
class EC50Result:
    """EC50 (or IC50) with confidence interval."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    conf_level: float
    method: str  # 'delta' or 'profile'


@dataclass(frozen=True)
class RelativePotencyResult:
    """Relative potency (ratio of EC50s) with Fieller's CI."""

    ratio: float
    ci_lower: float
    ci_upper: float
    conf_level: float
    method: str  # 'fieller'


def ec50(
    fit_result: DoseResponseResult,
    *,
    conf_level: float = 0.95,
    method: str = "delta",
) -> EC50Result:
    """Extract EC50 with confidence interval from a fitted model.

    Parameters
    ----------
    fit_result : DoseResponseResult
        A fitted dose-response model.
    conf_level : float
        Confidence level.
    method : str
        'delta' (delta method) or 'profile' (profile likelihood).

    Returns
    -------
    EC50Result

    Validates against: R drc::ED()
    """
    raise NotImplementedError("ec50 not yet implemented")


def relative_potency(
    fit1: DoseResponseResult,
    fit2: DoseResponseResult,
    *,
    conf_level: float = 0.95,
) -> RelativePotencyResult:
    """Relative potency: ratio of EC50s between two curves with Fieller's CI.

    Parameters
    ----------
    fit1 : DoseResponseResult
        First fitted model (reference).
    fit2 : DoseResponseResult
        Second fitted model (test).
    conf_level : float
        Confidence level.

    Returns
    -------
    RelativePotencyResult

    Validates against: R drc::compParm(), drc::EDcomp()
    """
    raise NotImplementedError("relative_potency not yet implemented")
