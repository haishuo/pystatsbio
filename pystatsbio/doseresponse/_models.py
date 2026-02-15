"""Dose-response model functions.

Each function computes the mean response at given dose levels for a
specific parametric model. These are the building blocks for curve fitting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ll4(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """4-parameter log-logistic (LL.4) model.

    y = bottom + (top - bottom) / (1 + (dose/ec50)^hill)

    This is the standard sigmoidal dose-response curve used across
    pharmacology, toxicology, and bioassay analysis.

    Validates against: R drc::LL.4()
    """
    raise NotImplementedError


def ll5(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
    asymmetry: float,
) -> NDArray[np.floating]:
    """5-parameter log-logistic (LL.5) model.

    Asymmetric version of LL.4 with an extra shape parameter.

    Validates against: R drc::LL.5()
    """
    raise NotImplementedError


def weibull1(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """Weibull type 1 (W1.4) model.

    Asymmetric dose-response, left-skewed.

    Validates against: R drc::W1.4()
    """
    raise NotImplementedError


def weibull2(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """Weibull type 2 (W2.4) model.

    Asymmetric dose-response, right-skewed.

    Validates against: R drc::W2.4()
    """
    raise NotImplementedError


def brain_cousens(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
    hormesis: float,
) -> NDArray[np.floating]:
    """Brain-Cousens hormesis model.

    Biphasic dose-response with low-dose stimulation.

    Validates against: R drc::BC.5()
    """
    raise NotImplementedError
