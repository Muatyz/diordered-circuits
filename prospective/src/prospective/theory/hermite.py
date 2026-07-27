"""First-Hermite prospective-shift prediction (paper Eqs. 20--22)."""

from __future__ import annotations

import numpy as np


def prospective_gamma(
    *,
    speed: float,
    adaptation_strength: float,
    tau_u: float,
    tau_v: float,
    sigma_u: float,
) -> tuple[float, float]:
    """Return theoretical `(gamma, adaptation_lag)` for a moving bump.

    The expression is a small-asymmetry Gaussian/Hermite approximation. At
    zero speed symmetry requires zero shift, while the lag formula is singular;
    this function returns `(0, 0)` for that limiting control.
    """

    if tau_u <= 0 or tau_v <= 0 or sigma_u <= 0 or adaptation_strength < 0:
        raise ValueError("positive time/width parameters and nonnegative adaptation are required")
    u = tau_u * speed / (np.sqrt(2.0) * sigma_u)
    if abs(u) < 1e-12:
        return 0.0, 0.0
    gamma_ratio = tau_v / tau_u
    numerator = adaptation_strength + 1.0 - gamma_ratio * u**2
    denominator = 2.0 * gamma_ratio * u
    square_denominator = (gamma_ratio * u**2 - (adaptation_strength + 1.0)) ** 2
    radical = np.sqrt(1.0 + 4.0 * u**2 * gamma_ratio * (gamma_ratio + 1.0) / square_denominator)
    y = numerator / denominator * (radical - 1.0)
    gamma = adaptation_strength * y / (gamma_ratio * y * u + 1.0) - u
    lag = np.sqrt(2.0) * sigma_u * y
    return float(gamma), float(lag)
