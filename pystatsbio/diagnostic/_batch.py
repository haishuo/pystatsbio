"""Batch AUC computation for high-throughput biomarker panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BatchAUCResult:
    """Result of batch AUC computation across multiple biomarkers."""

    auc: NDArray[np.floating]  # shape (n_markers,)
    se: NDArray[np.floating]  # DeLong SE for each
    n_markers: int


def batch_auc(
    response: NDArray[np.integer],
    predictors: NDArray[np.floating],
    *,
    backend: str = "auto",
) -> BatchAUCResult:
    """Compute AUC for many biomarker candidates simultaneously.

    Parameters
    ----------
    response : array of int, shape (n_samples,)
        Shared binary outcome.
    predictors : array of float, shape (n_samples, n_markers)
        Matrix of biomarker values (one column per candidate marker).
    backend : str
        'cpu', 'gpu', or 'auto'.

    Returns
    -------
    BatchAUCResult

    Notes
    -----
    GPU backend is beneficial when n_markers > 100. Uses rank-based
    AUC computation which is embarrassingly parallel across markers.
    """
    raise NotImplementedError("batch_auc not yet implemented")
