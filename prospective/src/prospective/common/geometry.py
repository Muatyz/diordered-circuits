"""One-dimensional spatial geometry and Gaussian profiles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def uniform_positions(n: int, length: float) -> FloatArray:
    """Return `n` uniformly spaced preferred positions in `[0, length)`."""

    if n < 2 or length <= 0:
        raise ValueError("n must be at least two and length positive")
    return np.linspace(0.0, length, n, endpoint=False, dtype=float)


def signed_circular_difference(a: FloatArray | float, b: FloatArray | float, length: float) -> FloatArray:
    """Return the signed shortest displacement `a - b` on a periodic segment."""

    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 0.5 * length) % length - 0.5 * length


def normalized_gaussian(
    positions: FloatArray,
    center: FloatArray | float,
    sigma: float,
    *,
    length: float | None = None,
    periodic: bool = False,
) -> FloatArray:
    """Evaluate a unit-integral Gaussian using linear or shortest-ring distance.

    The normalization is the real-line convention used by Emina and Kropff.
    A periodic profile uses the nearest-image approximation, which is accurate
    for `length` much larger than `sigma` and is only a declared control.
    """

    positions = np.asarray(positions, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if periodic:
        if length is None or length <= 0:
            raise ValueError("positive length is required for a periodic Gaussian")
        displacement = signed_circular_difference(positions, center, length)
    else:
        displacement = positions - np.asarray(center, dtype=float)
    return np.exp(-0.5 * (displacement / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def tutor_position(initial_position: float, speed: float, time: float, length: float) -> float:
    """Return the paper-reset tutor position for either velocity sign."""

    return float((initial_position + speed * time) % length)

