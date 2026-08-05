"""Head-direction population distal/proximal dynamics."""

from __future__ import annotations

import numpy as np


HD_DISTAL_NORMALIZATION_RAW_SUM = "raw_sum"
HD_DISTAL_NORMALIZATION_PRESYNAPTIC_MEAN = "presynaptic_population_mean"
HD_DISTAL_NORMALIZATION_MODES = {
    HD_DISTAL_NORMALIZATION_RAW_SUM,
    HD_DISTAL_NORMALIZATION_PRESYNAPTIC_MEAN,
}

PROXIMAL_INTEGRATION_FORWARD_EULER = "forward_euler"
PROXIMAL_INTEGRATION_EXACT_LINEAR = "exact_linear"
PROXIMAL_INTEGRATION_METHODS = {
    PROXIMAL_INTEGRATION_FORWARD_EULER,
    PROXIMAL_INTEGRATION_EXACT_LINEAR,
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


def euler_update_v_hd_proximal(
    *,
    v_hd_proximal: np.ndarray,
    v_hd_distal: np.ndarray,
    i_vis_to_hd: np.ndarray,
    dt: float,
    c_hd_proximal: float,
    g_l_hd_proximal: float,
    g_d_hd_to_proximal: float,
) -> np.ndarray:
    """Euler update for the HD axon-proximal voltage (paper Eq. 4).

    ``i_vis_to_hd`` is the complete current injected at the proximal
    compartment during this step: visual current, light-only excitation, and
    any explicitly requested proximal current noise.
    """
    if c_hd_proximal <= 0.0:
        raise ValueError("c_hd_proximal must be positive")
    proximal_current = (
        -g_l_hd_proximal * v_hd_proximal
        - g_d_hd_to_proximal * (v_hd_proximal - v_hd_distal)
        + i_vis_to_hd
    )
    return v_hd_proximal + (dt / c_hd_proximal) * proximal_current


def proximal_voltage_update_coefficients(
    *,
    dt: float,
    c_hd_proximal: float,
    g_l_hd_proximal: float,
    g_d_hd_to_proximal: float,
    integration_method: str,
) -> tuple[float, float, float]:
    """Return coefficients for one ordered proximal-voltage substep.

    The returned values ``(retention, distal_gain, input_gain)`` implement

    ``Va_next = retention * Va + distal_gain * Vd + input_gain * Iprox``.

    ``exact_linear`` is the analytic solution of paper Eq. (4) over one
    timestep while the already-updated distal voltage and current input are
    held piecewise constant. It is exact for this linear subproblem, not for
    the complete coupled network step.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if c_hd_proximal <= 0.0:
        raise ValueError("c_hd_proximal must be positive")
    if g_l_hd_proximal < 0.0 or g_d_hd_to_proximal < 0.0:
        raise ValueError("HD proximal conductances must be non-negative")
    total_conductance = g_l_hd_proximal + g_d_hd_to_proximal
    if total_conductance <= 0.0:
        raise ValueError("HD proximal conductances must have a positive sum")

    integration_method = str(integration_method).lower()
    if integration_method not in PROXIMAL_INTEGRATION_METHODS:
        raise ValueError(
            "integration_method must be one of "
            f"{sorted(PROXIMAL_INTEGRATION_METHODS)}"
        )

    scaled_step = dt * total_conductance / c_hd_proximal
    if integration_method == PROXIMAL_INTEGRATION_FORWARD_EULER:
        return (
            1.0 - scaled_step,
            dt * g_d_hd_to_proximal / c_hd_proximal,
            dt / c_hd_proximal,
        )

    # expm1 retains relative accuracy when dt is much smaller than the
    # effective proximal time constant C / (gL + gD).
    response_fraction = float(-np.expm1(-scaled_step))
    retention = 1.0 - response_fraction
    return (
        retention,
        response_fraction * g_d_hd_to_proximal / total_conductance,
        response_fraction / total_conductance,
    )


def exact_linear_update_v_hd_proximal(
    *,
    v_hd_proximal: np.ndarray,
    v_hd_distal: np.ndarray,
    i_vis_to_hd: np.ndarray,
    dt: float,
    c_hd_proximal: float,
    g_l_hd_proximal: float,
    g_d_hd_to_proximal: float,
) -> np.ndarray:
    """Advance Eq. (4) exactly for piecewise-constant ordered inputs."""
    retention, distal_gain, input_gain = proximal_voltage_update_coefficients(
        dt=dt,
        c_hd_proximal=c_hd_proximal,
        g_l_hd_proximal=g_l_hd_proximal,
        g_d_hd_to_proximal=g_d_hd_to_proximal,
        integration_method=PROXIMAL_INTEGRATION_EXACT_LINEAR,
    )
    return (
        retention * np.asarray(v_hd_proximal, dtype=float)
        + distal_gain * np.asarray(v_hd_distal, dtype=float)
        + input_gain * np.asarray(i_vis_to_hd, dtype=float)
    )


def update_v_hd_proximal(
    *,
    v_hd_proximal: np.ndarray,
    v_hd_distal: np.ndarray,
    i_vis_to_hd: np.ndarray,
    dt: float,
    c_hd_proximal: float,
    g_l_hd_proximal: float,
    g_d_hd_to_proximal: float,
    integration_method: str,
) -> np.ndarray:
    """Dispatch the configured paper Eq. (4) integration method."""
    integration_method = str(integration_method).lower()
    if integration_method == PROXIMAL_INTEGRATION_FORWARD_EULER:
        return euler_update_v_hd_proximal(
            v_hd_proximal=v_hd_proximal,
            v_hd_distal=v_hd_distal,
            i_vis_to_hd=i_vis_to_hd,
            dt=dt,
            c_hd_proximal=c_hd_proximal,
            g_l_hd_proximal=g_l_hd_proximal,
            g_d_hd_to_proximal=g_d_hd_to_proximal,
        )
    if integration_method == PROXIMAL_INTEGRATION_EXACT_LINEAR:
        return exact_linear_update_v_hd_proximal(
            v_hd_proximal=v_hd_proximal,
            v_hd_distal=v_hd_distal,
            i_vis_to_hd=i_vis_to_hd,
            dt=dt,
            c_hd_proximal=c_hd_proximal,
            g_l_hd_proximal=g_l_hd_proximal,
            g_d_hd_to_proximal=g_d_hd_to_proximal,
        )
    raise ValueError(
        "integration_method must be one of "
        f"{sorted(PROXIMAL_INTEGRATION_METHODS)}"
    )


def compute_v_hd_proximal_steady_state(
    *,
    v_hd_distal: np.ndarray,
    i_vis_to_hd: np.ndarray,
    g_l_hd_proximal: float,
    g_d_hd_to_proximal: float,
) -> np.ndarray:
    """Return the Eq. 4 fixed point, used only to initialize a consistent state."""
    total_conductance = g_l_hd_proximal + g_d_hd_to_proximal
    if total_conductance <= 0.0:
        raise ValueError("proximal conductances must have a positive sum")
    return (
        g_d_hd_to_proximal * np.asarray(v_hd_distal, dtype=float)
        + np.asarray(i_vis_to_hd, dtype=float)
    ) / total_conductance


def compute_hd_compartments(
    *,
    v_hd_distal: np.ndarray,
    v_hd_proximal: np.ndarray,
    p_distal_to_proximal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two dynamic voltages and the distal steady-state prediction."""
    v_hd_ss = p_distal_to_proximal * v_hd_distal
    return v_hd_distal, v_hd_ss, v_hd_proximal
