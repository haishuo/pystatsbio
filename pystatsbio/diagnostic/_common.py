"""Shared result types for diagnostic accuracy analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ROCResult:
    """Result of ROC analysis."""

    thresholds: NDArray[np.floating]
    tpr: NDArray[np.floating]  # sensitivity / true positive rate
    fpr: NDArray[np.floating]  # 1 - specificity / false positive rate
    auc: float
    auc_se: float  # DeLong standard error
    auc_ci_lower: float
    auc_ci_upper: float
    conf_level: float
    n_positive: int
    n_negative: int
    direction: str  # '<' or '>'

    def summary(self) -> str:
        """Human-readable summary."""
        raise NotImplementedError


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of diagnostic accuracy evaluation at a fixed cutoff."""

    cutoff: float
    sensitivity: float
    sensitivity_ci: tuple[float, float]
    specificity: float
    specificity_ci: tuple[float, float]
    ppv: float
    npv: float
    lr_positive: float
    lr_negative: float
    dor: float  # diagnostic odds ratio
    dor_ci: tuple[float, float]
    prevalence: float
    conf_level: float
    method: str  # CI method, e.g. 'clopper-pearson'

    def summary(self) -> str:
        """Human-readable summary."""
        raise NotImplementedError
