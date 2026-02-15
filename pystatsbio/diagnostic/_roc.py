"""ROC curve analysis with DeLong confidence intervals and comparison test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatsbio.diagnostic._common import ROCResult


@dataclass(frozen=True)
class ROCTestResult:
    """Result of comparing two correlated ROC curves (DeLong test)."""

    statistic: float
    p_value: float
    auc1: float
    auc2: float
    auc_diff: float
    method: str  # 'delong'

    def summary(self) -> str:
        raise NotImplementedError


def roc(
    response: NDArray[np.integer],
    predictor: NDArray[np.floating],
    *,
    direction: str = "auto",
    conf_level: float = 0.95,
) -> ROCResult:
    """Compute empirical ROC curve with DeLong AUC confidence interval.

    Parameters
    ----------
    response : array of int
        Binary outcome (0/1).
    predictor : array of float
        Continuous predictor (biomarker value).
    direction : str
        '<' (higher predictor -> positive) or '>' (lower -> positive)
        or 'auto' (choose direction that gives AUC >= 0.5).
    conf_level : float
        Confidence level for AUC CI.

    Returns
    -------
    ROCResult

    Validates against: R pROC::roc(), pROC::ci.auc()
    """
    raise NotImplementedError("roc not yet implemented")


def roc_test(
    roc1: ROCResult,
    roc2: ROCResult,
    *,
    predictor1: NDArray[np.floating] | None = None,
    predictor2: NDArray[np.floating] | None = None,
    response: NDArray[np.integer] | None = None,
    method: str = "delong",
) -> ROCTestResult:
    """Compare two correlated ROC curves using DeLong's test.

    Parameters
    ----------
    roc1, roc2 : ROCResult
        Two ROC curves computed on the same subjects.
    predictor1, predictor2 : array of float
        Original predictor values (needed to compute DeLong covariance).
    response : array of int
        Shared binary outcome.
    method : str
        'delong' (only supported method).

    Returns
    -------
    ROCTestResult

    Validates against: R pROC::roc.test()
    """
    raise NotImplementedError("roc_test not yet implemented")
