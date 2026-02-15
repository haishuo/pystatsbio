"""Non-compartmental pharmacokinetic analysis (NCA)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatsbio.pk._common import NCAResult


def nca(
    time: NDArray[np.floating],
    concentration: NDArray[np.floating],
    *,
    dose: float | None = None,
    route: str = "ev",
    auc_method: str = "linear-up/log-down",
    lambda_z_n_points: int | None = None,
) -> NCAResult:
    """Non-compartmental pharmacokinetic analysis.

    Parameters
    ----------
    time : array
        Time points.
    concentration : array
        Plasma concentration values.
    dose : float or None
        Administered dose (needed for CL and Vz).
    route : str
        'iv' (intravenous bolus) or 'ev' (extravascular / oral).
    auc_method : str
        'linear' (linear trapezoidal),
        'log-linear' (log-linear trapezoidal),
        'linear-up/log-down' (linear up, log-linear down — the default,
        recommended by FDA guidance).
    lambda_z_n_points : int or None
        Number of terminal points for half-life estimation.
        If None, automatically selects the best terminal phase
        (maximum adjusted r-squared with >= 3 points).

    Returns
    -------
    NCAResult

    Notes
    -----
    CPU-only. PK data is always small (typically 10-20 time points per subject).

    Validates against: R PKNCA::pk.nca(), NonCompart::sNCA()
    """
    raise NotImplementedError("nca not yet implemented")
