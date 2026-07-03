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


def compute_weight_eigenvalues(weight_matrix: np.ndarray) -> np.ndarray:
    """Return eigenvalues for a square weight matrix."""
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("Eigenvalue diagnostics require a square matrix")
    return np.linalg.eigvals(weight_matrix)


def _constant_mode_rayleigh_value(weight_matrix: np.ndarray) -> complex:
    constant_vector = np.ones(weight_matrix.shape[0], dtype=float)
    return complex(constant_vector @ weight_matrix @ constant_vector / (constant_vector @ constant_vector))


def summarize_eigenvalue_pair_degeneracy(
    *,
    weight_matrix: np.ndarray,
    pair_tolerance_fraction: float = 0.02,
) -> dict[str, float]:
    """Summarize approximate nonconstant-mode double degeneracy.

    A circularly symmetric ring operator has one constant mode and paired
    sine/cosine eigenmodes for each nonzero Fourier frequency.  This diagnostic
    removes the eigenvalue closest to the constant-mode Rayleigh quotient and
    then measures gaps between adjacent sorted real eigenvalues.
    """
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    eigenvalues = compute_weight_eigenvalues(weight_matrix)
    real_eigenvalues = np.real(eigenvalues)
    sorted_real = np.sort(real_eigenvalues)[::-1]
    if sorted_real.size == 0:
        return {
            "constant_mode_eigenvalue_real": float("nan"),
            "spectral_radius": float("nan"),
            "imag_abs_max": float("nan"),
            "nonconstant_pair_count": 0.0,
            "first_nonconstant_pair_gap_norm": float("nan"),
            "median_nonconstant_pair_gap_norm": float("nan"),
            "max_nonconstant_pair_gap_norm": float("nan"),
            "nonconstant_pair_fraction_le_2pct": float("nan"),
        }

    constant_mode_value = _constant_mode_rayleigh_value(weight_matrix)
    constant_index = int(np.argmin(np.abs(sorted_real - float(np.real(constant_mode_value)))))
    nonconstant_real = np.delete(sorted_real, constant_index)
    pair_count = nonconstant_real.size // 2
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else float("nan")
    value_scale = max(float(np.ptp(sorted_real)), spectral_radius, 1e-12)
    if pair_count == 0:
        first_gap = float("nan")
        median_gap = float("nan")
        max_gap = float("nan")
        pair_fraction = float("nan")
    else:
        paired_values = nonconstant_real[: 2 * pair_count].reshape(pair_count, 2)
        normalized_pair_gaps = np.abs(paired_values[:, 0] - paired_values[:, 1]) / value_scale
        first_gap = float(normalized_pair_gaps[0])
        median_gap = float(np.median(normalized_pair_gaps))
        max_gap = float(np.max(normalized_pair_gaps))
        pair_fraction = float(np.mean(normalized_pair_gaps <= pair_tolerance_fraction))

    return {
        "constant_mode_eigenvalue_real": float(np.real(constant_mode_value)),
        "spectral_radius": spectral_radius,
        "imag_abs_max": float(np.max(np.abs(np.imag(eigenvalues)))) if eigenvalues.size else float("nan"),
        "nonconstant_pair_count": float(pair_count),
        "first_nonconstant_pair_gap_norm": first_gap,
        "median_nonconstant_pair_gap_norm": median_gap,
        "max_nonconstant_pair_gap_norm": max_gap,
        "nonconstant_pair_fraction_le_2pct": pair_fraction,
    }


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
