"""Head-direction population distal/proximal dynamics."""

from __future__ import annotations

import numpy as np


def euler_update_i_hd_distal(
    *,
    i_hd_distal: np.ndarray,
    w_hd_to_hd: np.ndarray,
    r_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    r_hr: np.ndarray,
    b_hd: float,
    dt: float,
    tau_s: float,
) -> np.ndarray:
    """Euler update for the HD distal input current."""
    drive_hd_distal = w_hd_to_hd @ r_hd + w_hr_to_hd @ r_hr - b_hd
    return i_hd_distal + (dt / tau_s) * (-i_hd_distal + drive_hd_distal)


def euler_update_v_hd_distal(
    *,
    v_hd_distal: np.ndarray,
    i_hd_distal: np.ndarray,
    dt: float,
    tau_l_hd: float,
) -> np.ndarray:
    """Euler update for the HD axon-distal voltage.

    This implements Vafidis et al. Eq. 3: tau_l dVd/dt = -Vd + Id.
    """
    return v_hd_distal + (dt / tau_l_hd) * (-v_hd_distal + i_hd_distal)


def compute_hd_compartments(
    *,
    v_hd_distal: np.ndarray,
    i_vis_to_hd: np.ndarray,
    p_distal_to_proximal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distal voltage, distal prediction, and proximal voltage."""
    v_hd_ss = p_distal_to_proximal * v_hd_distal
    v_hd_proximal = v_hd_ss + i_vis_to_hd
    return v_hd_distal, v_hd_ss, v_hd_proximal
