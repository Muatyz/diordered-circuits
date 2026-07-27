"""Translation-invariant weight alignment and Gaussian diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from prospective.common.geometry import signed_circular_difference
from prospective.theory.equilibrium import equilibrium_widths


@dataclass(frozen=True)
class WeightDiagnostics:
    """Summary of learned feedforward organization."""

    gaussian_correlation: float
    learned_width: float
    theoretical_width: float
    width_relative_error: float
    translation_invariance_error: float
    learned_amplitude: float


def align_weight_rows(weights: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Roll each row so its peak lies at the central displacement bin."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must be a matrix")
    center = weights.shape[1] // 2
    peaks = np.argmax(weights, axis=1)
    aligned = np.asarray([np.roll(row, center - int(peak)) for row, peak in zip(weights, peaks)], dtype=float)
    return aligned, peaks.astype(np.int64)


def relative_displacements(n_input: int, length: float) -> NDArray[np.float64]:
    """Return centered displacement coordinates matching aligned row columns."""

    spacing = length / n_input
    return (np.arange(n_input, dtype=float) - n_input // 2) * spacing


def _gaussian_profile(displacement: NDArray[np.float64], amplitude: float, sigma: float) -> NDArray[np.float64]:
    return amplitude * np.exp(-0.5 * (displacement / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def diagnose_weights(
    weights: NDArray[np.float64], *, beta: float, sigma_r: float, length: float, periodic: bool = False
) -> tuple[WeightDiagnostics, dict[str, NDArray[np.float64]]]:
    """Compare aligned learned rows with the no-fit theoretical width law.

    The paper-reset theory assumes `L >> sigma_R` and replaces finite-domain
    integrals by real-line integrals. Rows whose learned peaks lie within
    `2.5 * sigma_R` of a reset boundary are therefore excluded from the primary
    profile diagnostics. Periodic controls retain every row.
    """

    aligned_all, peaks = align_weight_rows(weights)
    peak_positions = peaks * (length / weights.shape[1])
    eligible = np.ones(len(peaks), dtype=bool) if periodic else (
        (peak_positions >= 2.5 * sigma_r) & (peak_positions <= length - 2.5 * sigma_r)
    )
    if np.count_nonzero(eligible) < 3:
        eligible = np.ones(len(peaks), dtype=bool)
    aligned = aligned_all[eligible]
    displacement = relative_displacements(weights.shape[1], length)
    mean_profile = aligned.mean(axis=0)
    sigma_theory, _ = equilibrium_widths(beta, sigma_r)
    initial_amplitude = max(float(np.trapezoid(mean_profile, displacement)), np.finfo(float).eps)
    try:
        params, _ = curve_fit(
            _gaussian_profile,
            displacement,
            mean_profile,
            p0=(initial_amplitude, sigma_theory),
            bounds=([0.0, length / weights.shape[1] / 4.0], [np.inf, length]),
            maxfev=10000,
        )
        amplitude_fit, sigma_fit = map(float, params)
    except (RuntimeError, ValueError):
        amplitude_fit, sigma_fit = float("nan"), float("nan")
    theory_shape = _gaussian_profile(displacement, 1.0, sigma_theory)
    if np.std(mean_profile) > 0 and np.std(theory_shape) > 0:
        correlation = float(np.corrcoef(mean_profile, theory_shape)[0, 1])
    else:
        correlation = float("nan")
    row_norms = np.linalg.norm(aligned, axis=1)
    normalized = aligned / np.maximum(row_norms[:, None], np.finfo(float).eps)
    translation_error = float(np.mean(np.std(normalized, axis=0)) / max(np.mean(np.abs(normalized)), 1e-12))
    width_error = abs(sigma_fit - sigma_theory) / sigma_theory if np.isfinite(sigma_fit) else float("nan")
    diagnostics = WeightDiagnostics(
        gaussian_correlation=correlation,
        learned_width=sigma_fit,
        theoretical_width=sigma_theory,
        width_relative_error=float(width_error),
        translation_invariance_error=translation_error,
        learned_amplitude=amplitude_fit,
    )
    details = {
        "aligned_rows": aligned,
        "peak_indices": peaks,
        "eligible_rows": eligible,
        "relative_displacement": displacement,
        "mean_profile": mean_profile,
        "theory_unit_profile": theory_shape,
    }
    return diagnostics, details


def signed_offset(decoded: float, tutor: float, length: float, periodic: bool) -> float:
    """Return decoded-minus-tutor displacement with declared boundary semantics."""

    if not np.isfinite(decoded):
        return float("nan")
    if periodic:
        return float(signed_circular_difference(decoded, tutor, length))
    return float(decoded - tutor)
