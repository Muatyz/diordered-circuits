"""Gaussian equilibrium predictions from paper Eqs. 11--13."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from prospective.common.geometry import normalized_gaussian


def equilibrium_widths(beta: float, sigma_r: float) -> tuple[float, float]:
    """Return theoretical feedforward-weight and membrane-bump widths."""

    if not 0 < beta < 2:
        raise ValueError("equilibrium widths require 0 < beta < 2")
    if sigma_r <= 0:
        raise ValueError("sigma_r must be positive")
    sigma_j = np.sqrt(3.0 * beta / (2.0 - beta)) * sigma_r
    sigma_u = np.sqrt((2.0 * beta + 2.0) / (2.0 - beta)) * sigma_r
    return float(sigma_j), float(sigma_u)


def equilibrium_weight_amplitude(beta: float, sigma_j: float, integrated_drive: float, alpha: float) -> float:
    """Return the integrated Gaussian weight amplitude in paper Eq. 12."""

    if not 0 < beta < 2 or sigma_j <= 0 or integrated_drive <= 0 or alpha <= 0:
        raise ValueError("valid beta and positive sigma_j, drive, alpha are required")
    c_beta = np.sqrt(np.power(2.0 * np.pi * sigma_j**2, 1.0 - beta) / beta)
    return float(np.power(integrated_drive / (alpha * c_beta), 1.0 / beta))


def theoretical_weights(
    input_positions: NDArray[np.float64],
    competitive_positions: NDArray[np.float64],
    *,
    beta: float,
    sigma_r: float,
    integrated_drive: float,
    alpha: float,
    length: float,
    periodic: bool,
) -> NDArray[np.float64]:
    """Construct the analytic Gaussian `J[post, pre]` used as the M1 oracle."""

    sigma_j, _ = equilibrium_widths(beta, sigma_r)
    amplitude = equilibrium_weight_amplitude(beta, sigma_j, integrated_drive, alpha)
    rows = [
        amplitude
        * normalized_gaussian(
            np.asarray(input_positions),
            center,
            sigma_j,
            length=length,
            periodic=periodic,
        )
        for center in np.asarray(competitive_positions)
    ]
    return np.asarray(rows, dtype=float)

