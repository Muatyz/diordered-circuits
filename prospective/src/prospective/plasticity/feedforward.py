"""Emina-Kropff feedforward Hebbian plasticity (paper Eq. 6)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def feedforward_update_rate(
    weights: NDArray[np.float64],
    presynaptic_rate: NDArray[np.float64],
    postsynaptic_rate: NDArray[np.float64],
    *,
    eta: float,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """Return `dJ/dt` using only pre-rate, post-rate, and current weight.

    Matrix convention is `weights[post, pre]`. The exponent is applied
    elementwise, never as a matrix power.
    """

    weights = np.asarray(weights, dtype=float)
    pre = np.asarray(presynaptic_rate, dtype=float)
    post = np.asarray(postsynaptic_rate, dtype=float)
    if weights.shape != (post.size, pre.size):
        raise ValueError("weights must have shape (n_post, n_pre)")
    if eta < 0 or alpha <= 0 or not 0 < beta < 2:
        raise ValueError("eta >= 0, alpha > 0, and 0 < beta < 2 are required")
    return eta * post[:, None] * (pre[None, :] - alpha * np.power(weights, beta))


def feedforward_update(
    weights: NDArray[np.float64],
    presynaptic_rate: NDArray[np.float64],
    postsynaptic_rate: NDArray[np.float64],
    *,
    dt: float,
    eta: float,
    alpha: float,
    beta: float,
    nonnegative_clip: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Apply one local plasticity step and report raw increment and clipping.

    The returned increment is the pre-clipping `delta_J`, so depression that
    reaches Dale's-law boundary remains visible to diagnostics and animation.
    """

    delta = dt * feedforward_update_rate(
        weights,
        presynaptic_rate,
        postsynaptic_rate,
        eta=eta,
        alpha=alpha,
        beta=beta,
    )
    candidate = np.asarray(weights, dtype=float) + delta
    clipped = candidate < 0.0
    if nonnegative_clip:
        candidate = np.maximum(candidate, 0.0)
    return candidate, delta, clipped

