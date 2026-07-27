"""Serializable scalar metrics for training and prospective probes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prospective.analysis.decoding import vector_strength
from prospective.analysis.weights import diagnose_weights
from prospective.config.schema import ExperimentConfig


def summarize_training(
    *,
    weights: NDArray[np.float64],
    rate: NDArray[np.float64],
    learned_positions: NDArray[np.float64],
    update_norm: float,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Return the joint structural, convergence, and bump-validity summary."""

    diagnostics, _ = diagnose_weights(
        weights,
        beta=config.feedforward_learning.beta,
        sigma_r=config.tutor.sigma,
        length=config.geometry.length,
        periodic=config.geometry.boundary_mode == "periodic_ring",
    )
    row_max = np.max(weights, axis=1)
    row_min = np.min(weights, axis=1)
    row_contrast = (row_max - row_min) / np.maximum(row_max, np.finfo(float).eps)
    selective = (row_contrast >= 0.5) & (row_max >= 2.0 * config.feedforward_learning.initial_weight_scale)
    peak_bins = np.argmax(weights, axis=1)
    result = asdict(diagnostics)
    result.update(
        {
            "weight_update_norm": float(update_norm),
            "weight_norm": float(np.linalg.norm(weights)),
            "weight_max": float(np.max(weights)),
            "rate_sum": float(np.sum(rate)),
            "rate_peak_to_baseline": float(np.max(rate) - np.min(rate)),
            "rate_vector_strength": vector_strength(rate, learned_positions, config.geometry.length),
            "row_contrast_median": float(np.median(row_contrast)),
            "selective_row_fraction": float(np.mean(selective)),
            "learned_position_coverage_fraction": float(
                np.unique(peak_bins[selective]).size / config.geometry.n_input
            ) if np.any(selective) else 0.0,
        }
    )
    return result
