"""Visual teacher input for HD cells."""

from __future__ import annotations

import numpy as np

from learning.common.angles import circular_difference


def make_i_vis_to_hd(
    *,
    theta_hd_pref: np.ndarray,
    theta_true: float,
    amplitude: float,
    kappa: float,
    baseline: float,
    normalize_peak: bool,
) -> np.ndarray:
    """Return a von-Mises-like visual bump centered on theta_true.

    Vafidis' release code uses ``exp(-sin(delta/2)^2 / (2*sigma^2))`` with
    ``sigma=0.15``.  Near the peak, the matching normalized von-Mises width is
    approximately ``kappa = 1 / (4*sigma^2)``.
    """
    angle_error = circular_difference(theta_hd_pref, theta_true)
    if normalize_peak:
        i_vis_to_hd = amplitude * np.exp(kappa * (np.cos(angle_error) - 1.0))
    else:
        i_vis_to_hd = amplitude * np.exp(kappa * np.cos(angle_error))
    return i_vis_to_hd - baseline


def make_zero_i_vis_to_hd(n_theta: int) -> np.ndarray:
    return np.zeros(n_theta, dtype=float)
