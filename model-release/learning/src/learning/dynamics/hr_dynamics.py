"""Head-rotation population dynamics."""

from __future__ import annotations

import numpy as np


def euler_update_r_hd_to_hr_lp(
    *,
    r_hd_to_hr_lp: np.ndarray,
    r_hd: np.ndarray,
    dt: float,
    tau_s: float,
) -> np.ndarray:
    """Euler update for the low-pass HD rate that drives HR cells."""
    return r_hd_to_hr_lp + (dt / tau_s) * (-r_hd_to_hr_lp + r_hd)


def compute_i_hr(
    *,
    w_hd_to_hr: np.ndarray,
    r_hd_to_hr_lp: np.ndarray,
    i_vel_to_hr: np.ndarray,
    b_hr: float,
) -> np.ndarray:
    """Return HR voltage/current with additive instantaneous velocity input.

    This follows Vafidis et al. Eq. 8-10: the HD drive is low-pass filtered,
    while the angular-velocity current is added directly to the HR voltage.
    """
    hd_drive_to_hr = w_hd_to_hr @ r_hd_to_hr_lp
    return hd_drive_to_hr + i_vel_to_hr - b_hr
