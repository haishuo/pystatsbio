"""Shared result types for dose-response modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CurveParams:
    """Parameters of a fitted dose-response curve.

    For 4PL: bottom + (top - bottom) / (1 + (ec50/x)^hill)
    """

    bottom: float
    top: float
    ec50: float
    hill: float
    asymmetry: float | None = None  # 5PL only

    def predict(self, dose: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict response at given dose levels."""
        raise NotImplementedError


@dataclass(frozen=True)
class DoseResponseResult:
    """Result of fitting a single dose-response curve."""

    params: CurveParams
    se: NDArray[np.floating]  # standard errors of parameters
    residuals: NDArray[np.floating]
    rss: float
    aic: float
    bic: float
    converged: bool
    n_iter: int
    model: str  # e.g., "LL.4", "LL.5", "W1.4"

    def summary(self) -> str:
        """Human-readable summary, similar to R drc::summary()."""
        raise NotImplementedError


@dataclass(frozen=True)
class BatchDoseResponseResult:
    """Result of batch-fitting dose-response curves (HTS).

    Each array has length n_compounds.
    """

    ec50: NDArray[np.floating]
    hill: NDArray[np.floating]
    top: NDArray[np.floating]
    bottom: NDArray[np.floating]
    converged: NDArray[np.bool_]
    rss: NDArray[np.floating]
    n_compounds: int
