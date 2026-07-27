"""Circular-angle helpers for ring-attractor simulations."""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def make_theta_hd_pref(n_theta: int) -> np.ndarray:
    """Return uniformly spaced HD preferred directions in [-pi, pi)."""
    if n_theta <= 0:
        raise ValueError("n_theta must be positive")
    return np.linspace(-np.pi, np.pi, n_theta, endpoint=False, dtype=float)


def make_vafidis_paired_theta_hd_pref(n_theta: int) -> np.ndarray:
    """Return Vafidis-style paired HD preferred directions.

    The released LearnPI code uses 60 HD cells but 30 angular positions: odd
    and even HD cells form left/right partners with the same preferred
    direction.  This helper preserves that representation for the toy model
    while keeping ``make_theta_hd_pref`` as the generic uniformly spaced ring.
    """
    if n_theta <= 0:
        raise ValueError("n_theta must be positive")
    if n_theta % 2 != 0:
        raise ValueError("Vafidis paired HD preferences require an even n_theta")
    theta_pair_pref = np.linspace(-np.pi, np.pi, n_theta // 2, endpoint=False, dtype=float)
    return np.repeat(theta_pair_pref, 2)


def wrap_angle(theta_value: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(theta_value) + np.pi) % TWO_PI - np.pi


def circular_difference(theta_a: float | np.ndarray, theta_b: float | np.ndarray) -> float | np.ndarray:
    """Return theta_a - theta_b wrapped to (-pi, pi]."""
    return (np.asarray(theta_a) - np.asarray(theta_b) + np.pi) % TWO_PI - np.pi


def pva_decode(theta_hd_pref: np.ndarray, r_hd: np.ndarray) -> float:
    """Decode HD activity using the population vector average."""
    weighted_complex_vector = np.sum(r_hd * np.exp(1j * theta_hd_pref))
    if np.abs(weighted_complex_vector) < 1e-12:
        return float("nan")
    return float(wrap_angle(np.angle(weighted_complex_vector)))


def collapse_activity_by_theta(theta_hd_pref: np.ndarray, r_hd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average rates for cells with identical preferred directions.

    The Vafidis release-code geometry has odd/even HD partners at the same
    angular position.  Collapsing those partners gives plotting and peak-based
    decoding functions a true angular grid instead of duplicated x values.
    """
    theta_hd_pref = np.asarray(theta_hd_pref, dtype=float)
    r_hd = np.asarray(r_hd, dtype=float)
    if theta_hd_pref.ndim != 1 or r_hd.ndim != 1:
        raise ValueError("theta_hd_pref and r_hd must be 1D arrays")
    if theta_hd_pref.size != r_hd.size:
        raise ValueError("theta_hd_pref and r_hd must have the same size")
    if theta_hd_pref.size == 0:
        return theta_hd_pref.copy(), r_hd.copy()

    sort_index = np.argsort(theta_hd_pref)
    sorted_theta = theta_hd_pref[sort_index]
    sorted_rates = r_hd[sort_index]
    unique_theta, inverse_index = np.unique(sorted_theta, return_inverse=True)
    collapsed_rates = np.zeros(unique_theta.size, dtype=float)
    counts = np.zeros(unique_theta.size, dtype=float)
    np.add.at(collapsed_rates, inverse_index, sorted_rates)
    np.add.at(counts, inverse_index, 1.0)
    return unique_theta, collapsed_rates / counts


def _circular_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Return contiguous true runs on a circular boolean grid."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    if mask.size == 0 or not np.any(mask):
        return []

    runs: list[np.ndarray] = []
    run_start: int | None = None
    for index, value in enumerate(mask):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            runs.append(np.arange(run_start, index, dtype=int))
            run_start = None
    if run_start is not None:
        runs.append(np.arange(run_start, mask.size, dtype=int))

    if len(runs) > 1 and mask[0] and mask[-1]:
        merged_run = np.concatenate([runs[-1], runs[0]])
        runs = [merged_run, *runs[1:-1]]
    return runs


def _select_peak_component(peak_mask: np.ndarray, collapsed_rates: np.ndarray) -> np.ndarray | None:
    """Select one contiguous near-peak component, or None if it is ambiguous."""
    peak_runs = _circular_true_runs(peak_mask)
    if not peak_runs:
        return None
    if len(peak_runs) == 1:
        return peak_runs[0]

    run_scores = np.asarray([np.sum(collapsed_rates[run]) for run in peak_runs], dtype=float)
    best_score = float(np.max(run_scores))
    tied_best = np.flatnonzero(np.isclose(run_scores, best_score, rtol=1e-9, atol=1e-12))
    if tied_best.size != 1:
        return None
    return peak_runs[int(tied_best[0])]


def peak_decode(
    theta_hd_pref: np.ndarray,
    r_hd: np.ndarray,
    *,
    relative_tolerance: float = 5e-3,
    absolute_tolerance: float = 1e-9,
) -> float:
    """Decode HD activity from the strongest angular peak.

    If several adjacent angular bins share the maximum because the bump has a
    saturated plateau, the decoder returns the circular center of that highest
    plateau.  This makes it a useful diagnostic companion to PVA/COM decoding:
    disagreement between the two points to asymmetric tails, while agreement
    on a wrong position points to an attractor or teacher-alignment problem.
    """
    unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    finite_mask = np.isfinite(unique_theta) & np.isfinite(collapsed_rates)
    if np.count_nonzero(finite_mask) == 0:
        return float("nan")
    unique_theta = unique_theta[finite_mask]
    collapsed_rates = collapsed_rates[finite_mask]
    max_rate = float(np.max(collapsed_rates))
    min_rate = float(np.min(collapsed_rates))
    if max_rate <= 1e-12 and max_rate - min_rate <= 1e-12:
        return float("nan")
    tolerance = max(
        absolute_tolerance,
        relative_tolerance * max(abs(max_rate), max_rate - min_rate, 1.0),
    )
    peak_mask = collapsed_rates >= max_rate - tolerance
    peak_component = _select_peak_component(peak_mask, collapsed_rates)
    if peak_component is None:
        return float("nan")
    peak_vector = np.sum(np.exp(1j * unique_theta[peak_component]))
    if np.abs(peak_vector) < 1e-12:
        return float("nan")
    return float(wrap_angle(np.angle(peak_vector)))


def pva_vector_strength(theta_hd_pref: np.ndarray, r_hd: np.ndarray) -> float:
    """Return normalized PVA vector length as a bump-confidence measure."""
    rate_mass = float(np.sum(r_hd))
    if rate_mass <= 1e-12:
        return float("nan")
    weighted_complex_vector = np.sum(r_hd * np.exp(1j * theta_hd_pref))
    return float(np.abs(weighted_complex_vector) / rate_mass)


def unwrap_heading_trace(theta_trace: np.ndarray) -> np.ndarray:
    """Unwrap a heading trace while preserving NaNs."""
    theta_trace = np.asarray(theta_trace, dtype=float)
    finite_mask = np.isfinite(theta_trace)
    unwrapped_trace = np.full_like(theta_trace, np.nan, dtype=float)
    if np.any(finite_mask):
        unwrapped_trace[finite_mask] = np.unwrap(theta_trace[finite_mask])
    return unwrapped_trace
