"""Optimal cutoff selection for diagnostic tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatsbio.diagnostic._common import ROCResult


@dataclass(frozen=True)
class CutoffResult:
    """Result of optimal cutoff selection."""

    cutoff: float
    sensitivity: float
    specificity: float
    method: str  # 'youden', 'closest_topleft', 'cost'
    criterion_value: float  # value of the optimization criterion


def optimal_cutoff(
    roc_result: ROCResult,
    *,
    method: str = "youden",
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    prevalence: float | None = None,
) -> CutoffResult:
    """Find optimal classification cutoff from an ROC curve.

    Parameters
    ----------
    roc_result : ROCResult
        A computed ROC curve.
    method : str
        'youden' (maximize sensitivity + specificity - 1),
        'closest_topleft' (minimize distance to (0,1)),
        'cost' (minimize weighted misclassification cost).
    cost_fp, cost_fn : float
        Costs of false positives and false negatives (for method='cost').
    prevalence : float or None
        Disease prevalence (for method='cost'). Uses sample prevalence if None.

    Returns
    -------
    CutoffResult

    Validates against: R OptimalCutpoints::optimal.cutpoints()
    """
    raise NotImplementedError("optimal_cutoff not yet implemented")
