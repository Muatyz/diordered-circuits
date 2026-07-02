"""Array validation helpers."""

from __future__ import annotations

import numpy as np


def assert_shape(array: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {expected_shape}")


def assert_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def assert_nonnegative(array: np.ndarray, name: str, tolerance: float = 1e-12) -> None:
    if np.min(array) < -tolerance:
        raise ValueError(f"{name} contains negative values")


def clip_array(array: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be <= upper_bound")
    return np.clip(array, lower_bound, upper_bound)


def l2_norm(array: np.ndarray) -> float:
    """Return L2 norm without depending on numpy.linalg."""
    return float(np.sqrt(np.sum(np.asarray(array, dtype=float) ** 2)))


def check_model_arrays(
    *,
    r_hd: np.ndarray,
    r_hr: np.ndarray,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    w_hd_to_hr: np.ndarray,
    n_theta: int,
    n_hr: int,
) -> None:
    """Validate the shape and basic numerical health required by the toy model."""
    assert_shape(r_hd, (n_theta,), "r_hd")
    assert_shape(r_hr, (n_hr,), "r_hr")
    assert_shape(w_hd_to_hd, (n_theta, n_theta), "w_hd_to_hd")
    assert_shape(w_hr_to_hd, (n_theta, n_hr), "w_hr_to_hd")
    assert_shape(w_hd_to_hr, (n_hr, n_theta), "w_hd_to_hr")
    for array_name, array_value in {
        "r_hd": r_hd,
        "r_hr": r_hr,
        "w_hd_to_hd": w_hd_to_hd,
        "w_hr_to_hd": w_hr_to_hd,
        "w_hd_to_hr": w_hd_to_hr,
    }.items():
        assert_finite(array_value, array_name)
    assert_nonnegative(r_hd, "r_hd")
    assert_nonnegative(r_hr, "r_hr")
