"""Benchmark dose (BMD) analysis for toxicology."""

from __future__ import annotations

from dataclasses import dataclass

from pystatsbio.doseresponse._common import DoseResponseResult


@dataclass(frozen=True)
class BMDResult:
    """Benchmark dose result."""

    bmd: float  # benchmark dose (point estimate)
    bmdl: float  # lower confidence limit
    bmdu: float  # upper confidence limit
    bmr: float  # benchmark response level
    conf_level: float
    method: str  # 'delta' or 'profile'


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
        Benchmark response level (default 10% = 0.10).
    bmr_type : str
        'extra' (extra risk) or 'additional' (additional risk).
    conf_level : float
        Confidence level.
    method : str
        'delta' (delta method) or 'profile' (profile likelihood).

    Returns
    -------
    BMDResult

    Validates against: EPA BMDS software, R BMDL packages
    """
    raise NotImplementedError("bmd not yet implemented")
