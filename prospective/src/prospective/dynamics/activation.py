"""Quadratic rectification with global divisive inhibition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def divisive_quadratic_rate(
    membrane: NDArray[np.float64], inhibition_strength: float
) -> NDArray[np.float64]:
    """Evaluate paper Eq. 5 after uniform-grid discretization.

    The continuous factor `rho_c * integral dx` becomes a direct sum because
    `rho_c * delta_x = 1` on the uniform competitive grid.
    """

    if inhibition_strength < 0:
        raise ValueError("inhibition_strength must be nonnegative")
    rectified_squared = np.maximum(np.asarray(membrane, dtype=float), 0.0) ** 2
    denominator = 1.0 + inhibition_strength * float(rectified_squared.sum())
    return rectified_squared / denominator

