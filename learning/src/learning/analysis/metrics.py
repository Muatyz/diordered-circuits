"""Behavioral metrics for Vafidis toy-model tests."""

from __future__ import annotations

import numpy as np

from learning.common.angles import circular_difference, unwrap_heading_trace


def linear_fit_slope_intercept(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return least-squares slope and intercept for a 1D line fit."""
    x_mean = float(np.mean(x_values))
    y_mean = float(np.mean(y_values))
    centered_x = x_values - x_mean
    denominator = float(np.sum(centered_x**2))
    if denominator <= 1e-12:
        return float("nan"), float("nan")
    slope = float(np.sum(centered_x * (y_values - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def circular_error_trace(theta_decoded: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    return circular_difference(theta_decoded, theta_true)


def rms_circular_error(theta_decoded: np.ndarray, theta_true: np.ndarray) -> float:
    error_trace = circular_error_trace(theta_decoded, theta_true)
    finite_error = error_trace[np.isfinite(error_trace)]
    if finite_error.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite_error**2)))


def final_abs_circular_error(theta_decoded: np.ndarray, theta_reference: float) -> float:
    finite_decoded = theta_decoded[np.isfinite(theta_decoded)]
    if finite_decoded.size == 0:
        return float("nan")
    return float(abs(circular_difference(finite_decoded[-1], theta_reference)))


def estimate_decoded_velocity(
    *,
    time: np.ndarray,
    theta_decoded: np.ndarray,
    start_fraction: float = 0.25,
) -> float:
    finite_mask = np.isfinite(time) & np.isfinite(theta_decoded)
    if np.count_nonzero(finite_mask) < 3:
        return float("nan")
    finite_time = time[finite_mask]
    finite_theta_decoded = unwrap_heading_trace(theta_decoded[finite_mask])
    start_index = int(np.floor(start_fraction * finite_time.size))
    selected_time = finite_time[start_index:]
    selected_theta_decoded = finite_theta_decoded[start_index:]
    if selected_time.size < 3:
        return float("nan")
    slope, _intercept = linear_fit_slope_intercept(selected_time, selected_theta_decoded)
    return float(slope)


def summarize_velocity_gain(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
) -> dict[str, float]:
    finite_mask = np.isfinite(commanded_velocity) & np.isfinite(decoded_velocity)
    if np.count_nonzero(finite_mask) < 2:
        return {"gain": float("nan"), "intercept": float("nan")}
    slope, intercept = linear_fit_slope_intercept(
        commanded_velocity[finite_mask],
        decoded_velocity[finite_mask],
    )
    return {"gain": float(slope), "intercept": float(intercept)}
