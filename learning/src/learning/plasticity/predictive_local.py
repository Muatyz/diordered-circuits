"""Vafidis-style predictive local plasticity."""

from __future__ import annotations

import numpy as np

from learning.connectivity.constraints import constrain_w_hd_to_hd, constrain_w_hr_to_hd


def compute_e_hd(*, r_hd: np.ndarray, r_hd_distal_prediction: np.ndarray) -> np.ndarray:
    """Postsynaptic local prediction error for HD neurons."""
    return r_hd - r_hd_distal_prediction


def update_predictive_local_weights(
    *,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    delta_w_hd_to_hd: np.ndarray,
    delta_w_hr_to_hd: np.ndarray,
    e_hd: np.ndarray,
    p_hd: np.ndarray,
    p_hr: np.ndarray,
    dt: float,
    tau_delta: float,
    eta_hd_to_hd: float,
    eta_hr_to_hd: float,
    w_hd_to_hd_min: float,
    w_hd_to_hd_max: float,
    w_hr_to_hd_min: float,
    w_hr_to_hd_max: float,
    hd_to_hd_symmetry_mode: str = "none",
    hd_to_hd_balance_mode: str = "none",
    hr_to_hd_balance_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply local predictive updates to HD-to-HD and HR-to-HD weights.

    The update uses only postsynaptic HD error and presynaptic PSP traces.
    The plasticity-induction variable is low-pass filtered before changing
    weights, matching Vafidis et al. Eq. 12-16.
    """
    pi_hd_to_hd = np.outer(e_hd, p_hd)
    pi_hr_to_hd = np.outer(e_hd, p_hr)
    next_delta_w_hd_to_hd = delta_w_hd_to_hd + (dt / tau_delta) * (
        -delta_w_hd_to_hd + pi_hd_to_hd
    )
    next_delta_w_hr_to_hd = delta_w_hr_to_hd + (dt / tau_delta) * (
        -delta_w_hr_to_hd + pi_hr_to_hd
    )
    d_w_hd_to_hd = eta_hd_to_hd * next_delta_w_hd_to_hd
    d_w_hr_to_hd = eta_hr_to_hd * next_delta_w_hr_to_hd
    next_w_hd_to_hd = w_hd_to_hd + dt * d_w_hd_to_hd
    next_w_hr_to_hd = w_hr_to_hd + dt * d_w_hr_to_hd
    next_w_hd_to_hd = constrain_w_hd_to_hd(
        next_w_hd_to_hd,
        lower_bound=w_hd_to_hd_min,
        upper_bound=w_hd_to_hd_max,
        symmetry_mode=hd_to_hd_symmetry_mode,
        balance_mode=hd_to_hd_balance_mode,
    )
    next_w_hr_to_hd = constrain_w_hr_to_hd(
        next_w_hr_to_hd,
        lower_bound=w_hr_to_hd_min,
        upper_bound=w_hr_to_hd_max,
        balance_mode=hr_to_hd_balance_mode,
    )
    return next_w_hd_to_hd, next_w_hr_to_hd, next_delta_w_hd_to_hd, next_delta_w_hr_to_hd
