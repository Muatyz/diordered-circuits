"""Weight initialization for the Vafidis toy model."""

from __future__ import annotations

import numpy as np

from learning.common.angles import circular_difference, make_vafidis_paired_theta_hd_pref
from learning.connectivity.constraints import constrain_w_hd_to_hd, constrain_w_hr_to_hd


def make_local_circular_kernel(
    *,
    theta_target_pref: np.ndarray,
    theta_source_pref: np.ndarray,
    scale: float,
    sigma: float,
) -> np.ndarray:
    angle_error_matrix = circular_difference(theta_target_pref[:, None], theta_source_pref[None, :])
    return scale * np.exp(-0.5 * (angle_error_matrix / sigma) ** 2)


def initialize_w_hd_to_hd(
    *,
    n_theta: int,
    mode: str,
    scale: float,
    local_sigma: float,
    random_jitter: float,
    rng: np.random.Generator,
    lower_bound: float,
    upper_bound: float,
    symmetry_mode: str = "none",
    balance_mode: str = "none",
) -> np.ndarray:
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(n_theta)
    if mode == "zeros":
        w_hd_to_hd = np.zeros((n_theta, n_theta), dtype=float)
    elif mode == "local_kernel":
        w_hd_to_hd = make_local_circular_kernel(
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_hd_pref,
            scale=scale,
            sigma=local_sigma,
        )
    elif mode == "random_uniform":
        w_hd_to_hd = rng.uniform(0.0, scale, size=(n_theta, n_theta))
    elif mode == "random_normal":
        w_hd_to_hd = rng.normal(0.0, scale, size=(n_theta, n_theta))
    else:
        raise ValueError(f"Unknown w_hd_to_hd init mode: {mode}")
    if random_jitter > 0.0:
        w_hd_to_hd = w_hd_to_hd + rng.normal(0.0, random_jitter, size=w_hd_to_hd.shape)
    return constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        symmetry_mode=symmetry_mode,
        balance_mode=balance_mode,
    )


def initialize_w_hr_to_hd(
    *,
    n_theta: int,
    n_hr: int,
    mode: str,
    scale: float,
    local_sigma: float,
    random_jitter: float,
    rng: np.random.Generator,
    lower_bound: float,
    upper_bound: float,
    balance_mode: str = "none",
) -> np.ndarray:
    if n_hr != n_theta:
        raise ValueError("The Vafidis-style toy model expects n_hr == n_theta")
    if n_hr % 2 != 0:
        raise ValueError("n_hr must be even so left/right HR wings have equal size")
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(n_theta)
    n_hr_per_wing = n_hr // 2
    theta_lhr_pref = theta_hd_pref[0::2]
    theta_rhr_pref = theta_hd_pref[1::2]
    if theta_lhr_pref.size != n_hr_per_wing or theta_rhr_pref.size != n_hr_per_wing:
        raise ValueError("n_theta must be even for odd/even HD-to-HR wing mapping")
    if mode == "zeros":
        w_lhr_to_hd = np.zeros((n_theta, n_hr_per_wing), dtype=float)
        w_rhr_to_hd = np.zeros((n_theta, n_hr_per_wing), dtype=float)
    elif mode == "local_kernel":
        w_lhr_to_hd = make_local_circular_kernel(
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_lhr_pref,
            scale=scale,
            sigma=local_sigma,
        )
        w_rhr_to_hd = make_local_circular_kernel(
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_rhr_pref,
            scale=scale,
            sigma=local_sigma,
        )
    elif mode == "random_uniform":
        w_lhr_to_hd = rng.uniform(0.0, scale, size=(n_theta, n_hr_per_wing))
        w_rhr_to_hd = rng.uniform(0.0, scale, size=(n_theta, n_hr_per_wing))
    elif mode == "random_normal":
        w_lhr_to_hd = rng.normal(0.0, scale, size=(n_theta, n_hr_per_wing))
        w_rhr_to_hd = rng.normal(0.0, scale, size=(n_theta, n_hr_per_wing))
    else:
        raise ValueError(f"Unknown w_hr_to_hd init mode: {mode}")
    w_hr_to_hd = np.concatenate([w_lhr_to_hd, w_rhr_to_hd], axis=1)
    if random_jitter > 0.0:
        w_hr_to_hd = w_hr_to_hd + rng.normal(0.0, random_jitter, size=w_hr_to_hd.shape)
    return constrain_w_hr_to_hd(
        w_hr_to_hd,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        balance_mode=balance_mode,
    )


def initialize_w_hd_to_hr(*, n_theta: int, n_hr: int, strength: float) -> np.ndarray:
    """Return the fixed Vafidis HD-to-HR projection.

    In zero-based indexing, even HD cells project to the first HR wing and odd
    HD cells project to the second HR wing.  This is the release-code version
    of the paper's one-based statement: odd HD j projects to L-HR, even HD j
    projects to R-HR.
    """
    if n_hr != n_theta:
        raise ValueError("The Vafidis-style toy model expects n_hr == n_theta")
    if n_hr % 2 != 0:
        raise ValueError("n_hr must be even so left/right HR wings have equal size")
    n_hr_per_wing = n_hr // 2
    w_hd_to_hr = np.zeros((n_hr, n_theta), dtype=float)
    for wing_index in range(n_hr_per_wing):
        w_hd_to_hr[wing_index, 2 * wing_index] = strength
        w_hd_to_hr[n_hr_per_wing + wing_index, 2 * wing_index + 1] = strength
    return w_hd_to_hr
