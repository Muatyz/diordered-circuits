"""Weight constraints for the toy model."""

from __future__ import annotations

import numpy as np

from learning.common.arrays import assert_finite


def remove_diagonal(weight_matrix: np.ndarray) -> np.ndarray:
    constrained_weight_matrix = weight_matrix.copy()
    np.fill_diagonal(constrained_weight_matrix, 0.0)
    return constrained_weight_matrix


def constrain_w_hd_to_hd(
    w_hd_to_hd: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    symmetry_mode: str = "none",
    balance_mode: str = "none",
) -> np.ndarray:
    constrained_w_hd_to_hd = np.clip(w_hd_to_hd, lower_bound, upper_bound)
    constrained_w_hd_to_hd = symmetrize_w_hd_to_hd(
        constrained_w_hd_to_hd,
        symmetry_mode=symmetry_mode,
    )
    constrained_w_hd_to_hd = remove_diagonal(constrained_w_hd_to_hd)
    constrained_w_hd_to_hd = balance_w_hd_to_hd_common_mode(
        constrained_w_hd_to_hd,
        balance_mode=balance_mode,
    )
    constrained_w_hd_to_hd = remove_diagonal(constrained_w_hd_to_hd)
    assert_finite(constrained_w_hd_to_hd, "w_hd_to_hd")
    return constrained_w_hd_to_hd


def symmetrize_w_hd_to_hd(
    w_hd_to_hd: np.ndarray,
    *,
    symmetry_mode: str,
) -> np.ndarray:
    """Optionally remove antisymmetric recurrent drive around the HD ring."""
    if symmetry_mode == "none":
        return w_hd_to_hd
    if symmetry_mode != "symmetric":
        raise ValueError(f"Unknown hd_to_hd symmetry mode: {symmetry_mode}")
    return 0.5 * (w_hd_to_hd + w_hd_to_hd.T)


def balance_w_hd_to_hd_common_mode(
    w_hd_to_hd: np.ndarray,
    *,
    balance_mode: str,
) -> np.ndarray:
    """Optionally remove recurrent common-mode drive around the HD ring."""
    if balance_mode == "none":
        return w_hd_to_hd
    if balance_mode != "zero_sum":
        raise ValueError(f"Unknown hd_to_hd balance mode: {balance_mode}")
    n_target = w_hd_to_hd.shape[0]
    if n_target <= 2:
        return remove_diagonal(w_hd_to_hd - np.mean(w_hd_to_hd))
    off_diagonal_row_sum = np.sum(w_hd_to_hd, axis=1)
    total_off_diagonal_sum = float(np.sum(off_diagonal_row_sum))
    common_sum = total_off_diagonal_sum / (2.0 * (n_target - 1))
    row_common_mode = (off_diagonal_row_sum - common_sum) / (n_target - 2)
    centered_w_hd_to_hd = (
        w_hd_to_hd
        - row_common_mode[:, None]
        - row_common_mode[None, :]
    )
    centered_w_hd_to_hd = remove_diagonal(centered_w_hd_to_hd)
    return centered_w_hd_to_hd


def constrain_w_hr_to_hd(
    w_hr_to_hd: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    balance_mode: str = "none",
) -> np.ndarray:
    constrained_w_hr_to_hd = np.clip(w_hr_to_hd, lower_bound, upper_bound)
    constrained_w_hr_to_hd = balance_w_hr_to_hd_common_mode(
        constrained_w_hr_to_hd,
        balance_mode=balance_mode,
    )
    assert_finite(constrained_w_hr_to_hd, "w_hr_to_hd")
    return constrained_w_hr_to_hd


def balance_w_hr_to_hd_common_mode(
    w_hr_to_hd: np.ndarray,
    *,
    balance_mode: str,
) -> np.ndarray:
    """Optionally remove HR left/right common-mode drive.

    In zero velocity, left and right HR populations receive the same HD drive.
    If the learned LHR/RHR projections contain an imbalanced common mode, they
    can push the HD bump around the ring even when angular velocity is zero.
    The antisymmetric mode preserves the differential velocity pathway while
    cancelling that zero-velocity HR drive.
    """
    if balance_mode == "none":
        return w_hr_to_hd
    if balance_mode != "antisymmetric_wings":
        raise ValueError(f"Unknown hr_to_hd balance mode: {balance_mode}")
    if w_hr_to_hd.shape[1] % 2 != 0:
        raise ValueError("w_hr_to_hd must contain concatenated left/right HR wings")
    n_hr_per_wing = w_hr_to_hd.shape[1] // 2
    w_lhr_to_hd = w_hr_to_hd[:, :n_hr_per_wing]
    w_rhr_to_hd = w_hr_to_hd[:, n_hr_per_wing:]
    differential_lhr_to_hd = 0.5 * (w_lhr_to_hd - w_rhr_to_hd)
    differential_rhr_to_hd = -differential_lhr_to_hd
    return np.concatenate([differential_lhr_to_hd, differential_rhr_to_hd], axis=1)
