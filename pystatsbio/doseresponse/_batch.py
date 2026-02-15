"""Batch dose-response fitting for high-throughput screening (HTS).

This is the primary GPU showcase: fit thousands of 4PL curves simultaneously.
Each compound's curve fit is independent — perfect for GPU batching.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatsbio.doseresponse._common import BatchDoseResponseResult


def fit_drm_batch(
    dose_matrix: NDArray[np.floating],
    response_matrix: NDArray[np.floating],
    *,
    model: str = "LL.4",
    backend: str = "auto",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> BatchDoseResponseResult:
    """Batch-fit dose-response curves across many compounds.

    Parameters
    ----------
    dose_matrix : array, shape (n_compounds, n_doses)
        Dose values for each compound.
    response_matrix : array, shape (n_compounds, n_doses)
        Response values for each compound.
    model : str
        Model name (currently only 'LL.4' for batch fitting).
    backend : str
        'cpu', 'gpu', or 'auto'. GPU uses batched Levenberg-Marquardt
        via PyTorch for massive parallelism.
    max_iter : int
        Maximum iterations per curve.
    tol : float
        Convergence tolerance.

    Returns
    -------
    BatchDoseResponseResult

    Notes
    -----
    GPU backend requires ``pip install pystatsbio[gpu]`` (PyTorch).
    On CPU, curves are fit sequentially using scipy.optimize.
    On GPU, all curves are fit simultaneously using batched Jacobian
    computation and batched normal equations.
    """
    raise NotImplementedError("fit_drm_batch not yet implemented")
