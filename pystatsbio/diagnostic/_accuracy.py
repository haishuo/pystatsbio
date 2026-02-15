"""Sensitivity, specificity, predictive values, and likelihood ratios."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatsbio.diagnostic._common import DiagnosticResult


def diagnostic_accuracy(
    response: NDArray[np.integer],
    predictor: NDArray[np.floating],
    *,
    cutoff: float,
    direction: str = "<",
    prevalence: float | None = None,
    conf_level: float = 0.95,
    ci_method: str = "clopper-pearson",
) -> DiagnosticResult:
    """Compute diagnostic accuracy metrics at a fixed cutoff.

    Parameters
    ----------
    response : array of int
        Binary outcome (0/1).
    predictor : array of float
        Continuous predictor.
    cutoff : float
        Classification threshold.
    direction : str
        '<' means predictor >= cutoff is classified positive.
        '>' means predictor <= cutoff is classified positive.
    prevalence : float or None
        Disease prevalence for PPV/NPV adjustment. If None, uses sample prevalence.
    conf_level : float
        Confidence level.
    ci_method : str
        'clopper-pearson' (exact) or 'wilson'.

    Returns
    -------
    DiagnosticResult

    Validates against: R epiR::epi.tests()
    """
    raise NotImplementedError("diagnostic_accuracy not yet implemented")
