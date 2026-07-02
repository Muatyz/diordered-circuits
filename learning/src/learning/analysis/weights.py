"""Weight-structure diagnostics."""

from __future__ import annotations

import numpy as np

from learning.common.angles import (
    circular_difference,
    make_theta_hd_pref,
    make_vafidis_paired_theta_hd_pref,
)
from learning.common.arrays import l2_norm


def mean_source_offset(
    *,
    weight_target_by_source: np.ndarray,
    theta_target_pref: np.ndarray,
    theta_source_pref: np.ndarray,
    use_absolute_weight: bool = False,
) -> float:
    """Return circular mean of source-target offsets, weighted by matrix entries."""
    metric_weight = np.abs(weight_target_by_source) if use_absolute_weight else np.maximum(weight_target_by_source, 0.0)
    if np.sum(metric_weight) <= 1e-12:
        return float("nan")
    offset_matrix = circular_difference(theta_source_pref[None, :], theta_target_pref[:, None])
    complex_offset = np.sum(metric_weight * np.exp(1j * offset_matrix))
    return float(np.angle(complex_offset))


def mean_negative_source_offset(
    *,
    weight_target_by_source: np.ndarray,
    theta_target_pref: np.ndarray,
    theta_source_pref: np.ndarray,
) -> float:
    """Return circular mean of source-target offsets, weighted by inhibitory magnitude."""
    metric_weight = np.maximum(-weight_target_by_source, 0.0)
    if np.sum(metric_weight) <= 1e-12:
        return float("nan")
    offset_matrix = circular_difference(theta_source_pref[None, :], theta_target_pref[:, None])
    complex_offset = np.sum(metric_weight * np.exp(1j * offset_matrix))
    return float(np.angle(complex_offset))


def local_symmetry_score(w_hd_to_hd: np.ndarray) -> float:
    """Return one minus normalized antisymmetric energy for HD recurrent weights."""
    symmetric_part = 0.5 * (w_hd_to_hd + w_hd_to_hd.T)
    antisymmetric_part = 0.5 * (w_hd_to_hd - w_hd_to_hd.T)
    denominator = l2_norm(symmetric_part) + l2_norm(antisymmetric_part) + 1e-12
    return float(1.0 - l2_norm(antisymmetric_part) / denominator)


def summarize_weight_structure(w_hd_to_hd: np.ndarray, w_hr_to_hd: np.ndarray) -> dict[str, float]:
    n_theta = w_hd_to_hd.shape[0]
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(n_theta)
    if w_hr_to_hd.shape[1] % 2 != 0:
        raise ValueError("w_hr_to_hd must contain equal left/right HR wings")
    n_hr_per_wing = w_hr_to_hd.shape[1] // 2
    if w_hr_to_hd.shape[1] == n_theta:
        theta_lhr_pref = theta_hd_pref[0::2]
        theta_rhr_pref = theta_hd_pref[1::2]
    else:
        theta_lhr_pref = make_theta_hd_pref(n_hr_per_wing)
        theta_rhr_pref = make_theta_hd_pref(n_hr_per_wing)
    w_lhr_to_hd = w_hr_to_hd[:, :n_hr_per_wing]
    w_rhr_to_hd = w_hr_to_hd[:, n_hr_per_wing:]
    return {
        "hd_to_hd_local_symmetry_score": local_symmetry_score(w_hd_to_hd),
        "hd_to_hd_mean_source_offset": mean_source_offset(
            weight_target_by_source=w_hd_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_hd_pref,
            use_absolute_weight=False,
        ),
        "lhr_to_hd_mean_source_offset_abs": mean_source_offset(
            weight_target_by_source=w_lhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_lhr_pref,
            use_absolute_weight=True,
        ),
        "rhr_to_hd_mean_source_offset_abs": mean_source_offset(
            weight_target_by_source=w_rhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_rhr_pref,
            use_absolute_weight=True,
        ),
        "lhr_to_hd_excitatory_source_offset": mean_source_offset(
            weight_target_by_source=w_lhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_lhr_pref,
            use_absolute_weight=False,
        ),
        "rhr_to_hd_excitatory_source_offset": mean_source_offset(
            weight_target_by_source=w_rhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_rhr_pref,
            use_absolute_weight=False,
        ),
        "lhr_to_hd_inhibitory_source_offset": mean_negative_source_offset(
            weight_target_by_source=w_lhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_lhr_pref,
        ),
        "rhr_to_hd_inhibitory_source_offset": mean_negative_source_offset(
            weight_target_by_source=w_rhr_to_hd,
            theta_target_pref=theta_hd_pref,
            theta_source_pref=theta_rhr_pref,
        ),
    }
