"""Vafidis-style predictive local plasticity."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from learning.connectivity.constraints import constrain_w_hd_to_hd, constrain_w_hr_to_hd


@lru_cache(maxsize=32)
def _plasticity_block_coefficients(
    n_steps: int,
    dt: float,
    tau_delta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Cache coefficients shared by every equal-sized training block."""

    step_fraction = dt / tau_delta
    retention = 1.0 - step_fraction
    steps_remaining = np.arange(n_steps, 0, -1, dtype=float)
    delta_source_weights = step_fraction * np.power(
        retention,
        steps_remaining - 1.0,
    )
    if step_fraction == 0.0:
        weight_source_weights = np.zeros(n_steps, dtype=float)
    else:
        weight_source_weights = (
            step_fraction
            * (1.0 - np.power(retention, steps_remaining))
            / (1.0 - retention)
        )
    delta_initial_coefficient = float(retention**n_steps)
    weight_initial_coefficient = float(
        np.sum(np.power(retention, np.arange(1, n_steps + 1, dtype=float)))
    )
    return (
        delta_source_weights,
        weight_source_weights,
        delta_initial_coefficient,
        weight_initial_coefficient,
    )


def compute_e_hd(*, r_hd: np.ndarray, r_hd_distal_prediction: np.ndarray) -> np.ndarray:
    """Postsynaptic local prediction error for HD neurons."""
    return r_hd - r_hd_distal_prediction


def update_predictive_local_weights(
    *,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    delta_w_hd_to_hd: np.ndarray,
    delta_w_hr_to_hd: np.ndarray,
    e_hd: np.ndarray,
    p_hd: np.ndarray,
    p_hr: np.ndarray,
    dt: float,
    tau_delta: float,
    eta_hd_to_hd: float,
    eta_hr_to_hd: float,
    w_hd_to_hd_min: float | None,
    w_hd_to_hd_max: float | None,
    w_hr_to_hd_min: float | None,
    w_hr_to_hd_max: float | None,
    hd_to_hd_symmetry_mode: str = "none",
    hd_to_hd_balance_mode: str = "none",
    hr_to_hd_balance_mode: str = "none",
    zero_hd_to_hd_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply local predictive updates to HD-to-HD and HR-to-HD weights.

    The update uses only postsynaptic HD error and presynaptic PSP traces.
    The plasticity-induction variable is low-pass filtered before changing
    weights, matching Vafidis et al. Eq. 12-16.
    """
    pi_hd_to_hd = np.outer(e_hd, p_hd)
    pi_hr_to_hd = np.outer(e_hd, p_hr)
    next_delta_w_hd_to_hd = delta_w_hd_to_hd + (dt / tau_delta) * (
        -delta_w_hd_to_hd + pi_hd_to_hd
    )
    next_delta_w_hr_to_hd = delta_w_hr_to_hd + (dt / tau_delta) * (
        -delta_w_hr_to_hd + pi_hr_to_hd
    )
    d_w_hd_to_hd = eta_hd_to_hd * next_delta_w_hd_to_hd
    d_w_hr_to_hd = eta_hr_to_hd * next_delta_w_hr_to_hd
    next_w_hd_to_hd = w_hd_to_hd + dt * d_w_hd_to_hd
    next_w_hr_to_hd = w_hr_to_hd + dt * d_w_hr_to_hd
    next_w_hd_to_hd = constrain_w_hd_to_hd(
        next_w_hd_to_hd,
        lower_bound=w_hd_to_hd_min,
        upper_bound=w_hd_to_hd_max,
        symmetry_mode=hd_to_hd_symmetry_mode,
        balance_mode=hd_to_hd_balance_mode,
        zero_diagonal=zero_hd_to_hd_diagonal,
    )
    next_w_hr_to_hd = constrain_w_hr_to_hd(
        next_w_hr_to_hd,
        lower_bound=w_hr_to_hd_min,
        upper_bound=w_hr_to_hd_max,
        balance_mode=hr_to_hd_balance_mode,
    )
    return next_w_hd_to_hd, next_w_hr_to_hd, next_delta_w_hd_to_hd, next_delta_w_hr_to_hd


def update_predictive_local_weights_block(
    *,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    delta_w_hd_to_hd: np.ndarray,
    delta_w_hr_to_hd: np.ndarray,
    e_hd_history: np.ndarray,
    p_hd_history: np.ndarray,
    p_hr_history: np.ndarray,
    dt: float,
    tau_delta: float,
    eta_hd_to_hd: float,
    eta_hr_to_hd: float,
    w_hd_to_hd_min: float | None,
    w_hd_to_hd_max: float | None,
    w_hr_to_hd_min: float | None,
    w_hr_to_hd_max: float | None,
    hd_to_hd_symmetry_mode: str = "none",
    hd_to_hd_balance_mode: str = "none",
    hr_to_hd_balance_mode: str = "none",
    zero_hd_to_hd_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate several local-plasticity samples as one multirate block.

    For the supplied sequence of ``E P^T`` samples, this function evaluates
    the same Euler recurrences used by :func:`update_predictive_local_weights`
    algebraically:

    ``delta[n+1] = a * delta[n] + (1-a) * E[n] P[n]^T``
    ``W[n+1] = W[n] + dt * eta * delta[n+1]``.

    The weighted outer-product sums are computed as matrix products. Network
    dynamics may hold ``W`` fixed while collecting the samples; that slow-fast
    splitting is the only multirate approximation. Optional constraints are
    applied once at the block boundary, so constrained runs need a dedicated
    convergence comparison against the single-clock baseline.
    """

    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if tau_delta <= 0.0:
        raise ValueError("tau_delta must be positive")

    e_hd_history = np.asarray(e_hd_history, dtype=float)
    p_hd_history = np.asarray(p_hd_history, dtype=float)
    p_hr_history = np.asarray(p_hr_history, dtype=float)
    if e_hd_history.ndim != 2:
        raise ValueError("e_hd_history must have shape (n_steps, n_hd)")
    n_steps, n_hd = e_hd_history.shape
    if n_steps <= 0:
        raise ValueError("plasticity block must contain at least one step")
    if p_hd_history.shape != (n_steps, n_hd):
        raise ValueError("p_hd_history must match e_hd_history")
    if p_hr_history.ndim != 2 or p_hr_history.shape[0] != n_steps:
        raise ValueError("p_hr_history must have shape (n_steps, n_hr)")
    n_hr = p_hr_history.shape[1]
    expected_hd_shape = (n_hd, n_hd)
    expected_hr_shape = (n_hd, n_hr)
    for array_name, array_value, expected_shape in (
        ("w_hd_to_hd", w_hd_to_hd, expected_hd_shape),
        ("delta_w_hd_to_hd", delta_w_hd_to_hd, expected_hd_shape),
        ("w_hr_to_hd", w_hr_to_hd, expected_hr_shape),
        ("delta_w_hr_to_hd", delta_w_hr_to_hd, expected_hr_shape),
    ):
        if np.asarray(array_value).shape != expected_shape:
            raise ValueError(f"{array_name} must have shape {expected_shape}")

    (
        delta_source_weights,
        weight_source_weights,
        delta_initial_coefficient,
        weight_initial_coefficient,
    ) = _plasticity_block_coefficients(n_steps, float(dt), float(tau_delta))

    def integrate_pathway(
        *,
        weight: np.ndarray,
        delta_weight: np.ndarray,
        p_history: np.ndarray,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta_source = e_hd_history.T @ (
            delta_source_weights[:, None] * p_history
        )
        integrated_delta_source = e_hd_history.T @ (
            weight_source_weights[:, None] * p_history
        )
        next_delta_weight = (
            delta_initial_coefficient * np.asarray(delta_weight, dtype=float)
            + delta_source
        )
        next_weight = np.asarray(weight, dtype=float) + dt * eta * (
            weight_initial_coefficient * np.asarray(delta_weight, dtype=float)
            + integrated_delta_source
        )
        return next_weight, next_delta_weight

    next_w_hd_to_hd, next_delta_w_hd_to_hd = integrate_pathway(
        weight=w_hd_to_hd,
        delta_weight=delta_w_hd_to_hd,
        p_history=p_hd_history,
        eta=eta_hd_to_hd,
    )
    next_w_hr_to_hd, next_delta_w_hr_to_hd = integrate_pathway(
        weight=w_hr_to_hd,
        delta_weight=delta_w_hr_to_hd,
        p_history=p_hr_history,
        eta=eta_hr_to_hd,
    )
    next_w_hd_to_hd = constrain_w_hd_to_hd(
        next_w_hd_to_hd,
        lower_bound=w_hd_to_hd_min,
        upper_bound=w_hd_to_hd_max,
        symmetry_mode=hd_to_hd_symmetry_mode,
        balance_mode=hd_to_hd_balance_mode,
        zero_diagonal=zero_hd_to_hd_diagonal,
    )
    next_w_hr_to_hd = constrain_w_hr_to_hd(
        next_w_hr_to_hd,
        lower_bound=w_hr_to_hd_min,
        upper_bound=w_hr_to_hd_max,
        balance_mode=hr_to_hd_balance_mode,
    )
    return (
        next_w_hd_to_hd,
        next_w_hr_to_hd,
        next_delta_w_hd_to_hd,
        next_delta_w_hr_to_hd,
    )
