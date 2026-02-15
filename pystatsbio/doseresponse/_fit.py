"""Single dose-response curve fitting via nonlinear least squares."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatsbio.doseresponse._common import DoseResponseResult


def fit_drm(
    dose: NDArray[np.floating],
    response: NDArray[np.floating],
    *,
    model: str = "LL.4",
    weights: NDArray[np.floating] | None = None,
    start: dict[str, float] | None = None,
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
) -> DoseResponseResult:
    """Fit a dose-response model to a single curve.

    Uses Levenberg-Marquardt nonlinear least squares.

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    response : array
        Response values.
    model : str
        Model name: 'LL.4', 'LL.5', 'W1.4', 'W2.4', 'BC.5'.
    weights : array or None
        Optional observation weights.
    start : dict or None
        Starting values for parameters. If None, uses self-starting estimates.
    lower, upper : dict or None
        Box constraints on parameters.

    Returns
    -------
    DoseResponseResult

    Validates against: R drc::drm()
    """
    raise NotImplementedError("fit_drm not yet implemented")
