"""Head-direction population distal/proximal dynamics."""

from __future__ import annotations

import numpy as np


HD_DISTAL_NORMALIZATION_RAW_SUM = "raw_sum"
HD_DISTAL_NORMALIZATION_PRESYNAPTIC_MEAN = "presynaptic_population_mean"
HD_DISTAL_NORMALIZATION_MODES = {
    HD_DISTAL_NORMALIZATION_RAW_SUM,
    HD_DISTAL_NORMALIZATION_PRESYNAPTIC_MEAN,
}


def effective_hd_distal_weight_matrices(
    *,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the matrices that actually act on HD and HR firing rates.

    In population-mean mode the HD recurrent population and the two HR wings
    are separate dense pathways. Each is divided by its own presynaptic count.
    The fixed sparse HD-to-HR projection is intentionally outside this helper.
    """
    w_hd_to_hd = np.asarray(w_hd_to_hd, dtype=float)
    w_hr_to_hd = np.asarray(w_hr_to_hd, dtype=float)
    if w_hd_to_hd.ndim != 2 or w_hd_to_hd.shape[0] != w_hd_to_hd.shape[1]:
        raise ValueError("w_hd_to_hd must be square")
    n_hd = w_hd_to_hd.shape[1]
    if w_hr_to_hd.ndim != 2 or w_hr_to_hd.shape[0] != n_hd:
        raise ValueError("w_hr_to_hd must have one row per HD neuron")
    n_hr = w_hr_to_hd.shape[1]
    if n_hr <= 0 or n_hr % 2 != 0:
        raise ValueError("w_hr_to_hd must contain equal non-empty LHR/RHR wings")
    normalization = str(normalization).lower()
    if normalization not in HD_DISTAL_NORMALIZATION_MODES:
        raise ValueError(f"Unknown HD distal normalization: {normalization}")
    if normalization == HD_DISTAL_NORMALIZATION_RAW_SUM:
        return w_hd_to_hd, w_hr_to_hd
    n_hr_per_wing = n_hr // 2
    return w_hd_to_hd / n_hd, w_hr_to_hd / n_hr_per_wing


def compute_hd_distal_pathway_drives(
    *,
    w_hd_to_hd: np.ndarray,
    r_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    r_hr: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the HD, LHR, and RHR contributions to HD distal current."""
    r_hd = np.asarray(r_hd, dtype=float)
    r_hr = np.asarray(r_hr, dtype=float)
    effective_w_hd_to_hd, effective_w_hr_to_hd = effective_hd_distal_weight_matrices(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        normalization=normalization,
    )
    if r_hd.shape != (effective_w_hd_to_hd.shape[1],):
        raise ValueError("r_hd does not match w_hd_to_hd")
    if r_hr.shape != (effective_w_hr_to_hd.shape[1],):
        raise ValueError("r_hr does not match w_hr_to_hd")
    n_hr_per_wing = r_hr.size // 2
    i_hd_from_hd = effective_w_hd_to_hd @ r_hd
    i_hd_from_lhr = (
        effective_w_hr_to_hd[:, :n_hr_per_wing] @ r_hr[:n_hr_per_wing]
    )
    i_hd_from_rhr = (
        effective_w_hr_to_hd[:, n_hr_per_wing:] @ r_hr[n_hr_per_wing:]
    )
    return i_hd_from_hd, i_hd_from_lhr, i_hd_from_rhr


def euler_update_i_hd_distal_from_pathway_drives(
    *,
    i_hd_distal: np.ndarray,
    i_hd_from_hd: np.ndarray,
    i_hd_from_lhr: np.ndarray,
    i_hd_from_rhr: np.ndarray,
    b_hd: float,
    dt: float,
    tau_s: float,
) -> np.ndarray:
    """Euler update using already computed dense-pathway contributions."""
    drive_hd_distal = i_hd_from_hd + i_hd_from_lhr + i_hd_from_rhr - b_hd
    return i_hd_distal + (dt / tau_s) * (-i_hd_distal + drive_hd_distal)


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
    normalization: str = HD_DISTAL_NORMALIZATION_RAW_SUM,
) -> np.ndarray:
    """Euler update for the HD distal input current."""
    i_hd_from_hd, i_hd_from_lhr, i_hd_from_rhr = compute_hd_distal_pathway_drives(
        w_hd_to_hd=w_hd_to_hd,
        r_hd=r_hd,
        w_hr_to_hd=w_hr_to_hd,
        r_hr=r_hr,
        normalization=normalization,
    )
    return euler_update_i_hd_distal_from_pathway_drives(
        i_hd_distal=i_hd_distal,
        i_hd_from_hd=i_hd_from_hd,
        i_hd_from_lhr=i_hd_from_lhr,
        i_hd_from_rhr=i_hd_from_rhr,
        b_hd=b_hd,
        dt=dt,
        tau_s=tau_s,
    )


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
