"""Frozen path-integration performance across saved training weights.

Vafidis et al. time-average the absolute local learning error in Eq. (19).
This diagnostic borrows the time-window idea, but deliberately measures a
different quantity: circular accumulated PI error while each saved weight
snapshot is frozen. Constant velocities make snapshots directly comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from learning.analysis.metrics import linear_fit_slope_intercept
from learning.common.angles import (
    circular_difference,
    pva_vector_strength,
    wrap_angle,
)
from learning.config.schema import ExperimentConfig
from learning.diagnostics.protocols import initialize_frozen_protocol_state
from learning.dynamics.hd_dynamics import effective_hd_distal_weight_matrices
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    step_vafidis_toy,
)


def _validated_probe_settings(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    settings = config.tests
    velocities = np.asarray(settings.weight_snapshot_pi_velocities, dtype=float)
    if velocities.ndim != 1 or velocities.size == 0:
        raise ValueError(
            "tests.weight_snapshot_pi_velocities must be a non-empty 1D list"
        )
    if not np.isfinite(velocities).all():
        raise ValueError("tests.weight_snapshot_pi_velocities must be finite")
    if np.unique(velocities).size != velocities.size:
        raise ValueError(
            "tests.weight_snapshot_pi_velocities must not contain duplicates"
        )

    configured_headings = np.asarray(
        settings.weight_snapshot_pi_initial_headings,
        dtype=float,
    )
    initial_headings = (
        configured_headings
        if configured_headings.size
        else np.asarray([settings.weight_snapshot_pi_initial_heading], dtype=float)
    )
    if initial_headings.ndim != 1 or initial_headings.size == 0:
        raise ValueError(
            "tests.weight_snapshot_pi_initial_headings must be a 1D list"
        )
    if not np.isfinite(initial_headings).all():
        raise ValueError(
            "tests.weight_snapshot_pi_initial_headings must be finite"
        )
    wrapped_headings = np.asarray(wrap_angle(initial_headings), dtype=float)
    pairwise_separation = np.abs(
        circular_difference(
            wrapped_headings[:, None],
            wrapped_headings[None, :],
        )
    )
    duplicate_mask = np.triu(pairwise_separation <= 1e-12, k=1)
    if np.any(duplicate_mask):
        raise ValueError(
            "tests.weight_snapshot_pi_initial_headings must be unique modulo 2*pi"
        )

    dt = float(config.simulation.dt)
    cue_duration = float(settings.weight_snapshot_pi_cue_duration)
    probe_duration = float(settings.weight_snapshot_pi_duration)
    average_start_time = float(settings.weight_snapshot_pi_average_start_time)
    if not np.isfinite(cue_duration) or cue_duration < 0.0:
        raise ValueError(
            "tests.weight_snapshot_pi_cue_duration must be finite and non-negative"
        )
    if not np.isfinite(probe_duration) or probe_duration <= 0.0:
        raise ValueError(
            "tests.weight_snapshot_pi_duration must be finite and positive"
        )
    if (
        not np.isfinite(average_start_time)
        or average_start_time < 0.0
        or average_start_time >= probe_duration
    ):
        raise ValueError(
            "tests.weight_snapshot_pi_average_start_time must lie in "
            "[0, weight_snapshot_pi_duration)"
        )

    cue_steps = int(round(cue_duration / dt))
    probe_steps = max(1, int(round(probe_duration / dt)))
    average_start_step = int(round(average_start_time / dt))
    if probe_steps - average_start_step < 1:
        raise ValueError("weight snapshot PI averaging window contains no timesteps")
    return (
        velocities,
        wrapped_headings,
        cue_steps,
        probe_steps,
        average_start_step,
    )


def _coerce_weight_snapshots(
    *,
    trained_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None,
    interval_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight_history = weight_history or {}
    snapshot_time = np.asarray(weight_history.get("time", np.empty(0)), dtype=float)
    w_hd_to_hd = np.asarray(
        weight_history.get("w_hd_to_hd", np.empty((0, 0, 0))),
        dtype=float,
    )
    w_hr_to_hd = np.asarray(
        weight_history.get("w_hr_to_hd", np.empty((0, 0, 0))),
        dtype=float,
    )
    if snapshot_time.size == 0:
        return (
            np.asarray([float(trained_state.time)], dtype=float),
            trained_state.w_hd_to_hd[None, ...].copy(),
            trained_state.w_hr_to_hd[None, ...].copy(),
            np.asarray([0], dtype=int),
        )
    if snapshot_time.ndim != 1 or not np.isfinite(snapshot_time).all():
        raise ValueError("weight_history time must be a finite 1D array")
    if np.any(np.diff(snapshot_time) < 0.0):
        raise ValueError("weight_history time must be chronological")
    expected_hd_shape = (
        snapshot_time.size,
        *trained_state.w_hd_to_hd.shape,
    )
    expected_hr_shape = (
        snapshot_time.size,
        *trained_state.w_hr_to_hd.shape,
    )
    if w_hd_to_hd.shape != expected_hd_shape:
        raise ValueError(
            f"weight_history w_hd_to_hd must have shape {expected_hd_shape}"
        )
    if w_hr_to_hd.shape != expected_hr_shape:
        raise ValueError(
            f"weight_history w_hr_to_hd must have shape {expected_hr_shape}"
        )
    if not np.isfinite(w_hd_to_hd).all() or not np.isfinite(w_hr_to_hd).all():
        raise ValueError("weight snapshots must be finite")
    selected_indices = select_weight_snapshot_indices(
        snapshot_time=snapshot_time,
        interval_fraction=interval_fraction,
    )
    return (
        snapshot_time[selected_indices],
        w_hd_to_hd[selected_indices],
        w_hr_to_hd[selected_indices],
        selected_indices,
    )


def select_weight_snapshot_indices(
    *,
    snapshot_time: np.ndarray,
    interval_fraction: float,
) -> np.ndarray:
    """Select saved states nearest to regular fractions of training time."""

    snapshot_time = np.asarray(snapshot_time, dtype=float)
    interval_fraction = float(interval_fraction)
    if (
        not np.isfinite(interval_fraction)
        or interval_fraction <= 0.0
        or interval_fraction > 1.0
    ):
        raise ValueError(
            "tests.weight_snapshot_pi_interval_fraction must lie in (0, 1]"
        )
    if snapshot_time.ndim != 1 or snapshot_time.size == 0:
        return np.empty(0, dtype=int)
    if snapshot_time.size == 1 or np.isclose(snapshot_time[-1], snapshot_time[0]):
        return np.asarray([0], dtype=int)
    fraction_targets = np.arange(
        0.0,
        1.0 + 0.5 * interval_fraction,
        interval_fraction,
        dtype=float,
    )
    fraction_targets = fraction_targets[fraction_targets <= 1.0 + 1e-12]
    if fraction_targets.size == 0 or not np.isclose(fraction_targets[-1], 1.0):
        fraction_targets = np.append(fraction_targets, 1.0)
    target_times = snapshot_time[0] + fraction_targets * (
        snapshot_time[-1] - snapshot_time[0]
    )
    nearest_indices = np.asarray(
        [
            int(np.argmin(np.abs(snapshot_time - target_time)))
            for target_time in target_times
        ],
        dtype=int,
    )
    # np.unique sorts indices, which is also chronological because the saved
    # history must itself be chronological.
    return np.unique(nearest_indices)


def _run_one_frozen_probe(
    *,
    config: ExperimentConfig,
    snapshot_state: VafidisToyState,
    theta_initial: float,
    angular_velocity: float,
    cue_steps: int,
    probe_steps: int,
    average_start_step: int,
    progress: Any | None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    params = VafidisToyParams.from_config(config)
    state = initialize_frozen_protocol_state(
        config=config,
        trained_state=snapshot_state,
        theta_true=float(theta_initial),
    )
    for _step_index in range(cue_steps):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=0.0,
            visual_teacher=True,
            training=False,
            visual_noise=None,
        )
        if progress is not None:
            progress.update(1)

    previous_decoded = float(state.theta_hd_decoded)
    decoded_displacement = 0.0
    true_displacement = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    sample_count = 0
    pva_strength_sum = 0.0
    minimum_pva_strength = float("inf")
    bump_contrast_sum = 0.0
    final_abs_error = float("nan")
    wrapped_absolute_error_sum = 0.0
    wrapped_squared_error_sum = 0.0
    final_wrapped_abs_error = float("nan")
    fit_time: list[float] = []
    fit_decoded_displacement: list[float] = []
    for probe_step in range(probe_steps):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=float(angular_velocity),
            visual_teacher=False,
            training=False,
            visual_noise=None,
        )
        current_decoded = float(state.theta_hd_decoded)
        decoded_displacement += float(
            circular_difference(current_decoded, previous_decoded)
        )
        previous_decoded = current_decoded
        true_displacement += float(angular_velocity) * params.dt
        accumulated_error = decoded_displacement - true_displacement
        wrapped_accumulated_error = float(wrap_angle(accumulated_error))
        final_abs_error = abs(accumulated_error)
        final_wrapped_abs_error = abs(wrapped_accumulated_error)
        if probe_step >= average_start_step:
            absolute_error_sum += final_abs_error
            squared_error_sum += accumulated_error**2
            wrapped_absolute_error_sum += final_wrapped_abs_error
            wrapped_squared_error_sum += wrapped_accumulated_error**2
            fit_time.append(float((probe_step + 1) * params.dt))
            fit_decoded_displacement.append(float(decoded_displacement))
            strength = float(pva_vector_strength(state.theta_hd_pref, state.r_hd))
            contrast = float(np.max(state.r_hd) - np.min(state.r_hd))
            pva_strength_sum += strength
            minimum_pva_strength = min(minimum_pva_strength, strength)
            bump_contrast_sum += contrast
            sample_count += 1
        if progress is not None:
            progress.update(1)

    denominator = max(sample_count, 1)
    decoded_velocity, _intercept = (
        linear_fit_slope_intercept(
            np.asarray(fit_time, dtype=float),
            np.asarray(fit_decoded_displacement, dtype=float),
        )
        if len(fit_time) >= 2
        else (float("nan"), float("nan"))
    )
    return (
        absolute_error_sum / denominator,
        float(np.sqrt(squared_error_sum / denominator)),
        final_abs_error,
        wrapped_absolute_error_sum / denominator,
        float(np.sqrt(wrapped_squared_error_sum / denominator)),
        final_wrapped_abs_error,
        float(decoded_velocity - angular_velocity),
        pva_strength_sum / denominator,
        minimum_pva_strength,
        bump_contrast_sum / denominator,
    )


def run_weight_snapshot_pi_development_diagnostic(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Measure frozen PI performance for every saved training snapshot."""

    velocities, initial_headings, cue_steps, probe_steps, average_start_step = (
        _validated_probe_settings(config)
    )
    (
        snapshot_time,
        w_hd_history,
        w_hr_history,
        snapshot_source_index,
    ) = _coerce_weight_snapshots(
        trained_state=trained_state,
        weight_history=weight_history,
        interval_fraction=config.tests.weight_snapshot_pi_interval_fraction,
    )
    detailed_shape = (
        snapshot_time.size,
        initial_headings.size,
        velocities.size,
    )
    detailed_time_averaged_abs_error = np.empty(detailed_shape, dtype=float)
    detailed_rms_error = np.empty(detailed_shape, dtype=float)
    detailed_final_abs_error = np.empty(detailed_shape, dtype=float)
    detailed_wrapped_time_averaged_abs_error = np.empty(
        detailed_shape, dtype=float
    )
    detailed_wrapped_rms_error = np.empty(detailed_shape, dtype=float)
    detailed_wrapped_final_abs_error = np.empty(detailed_shape, dtype=float)
    detailed_velocity_bias = np.empty(detailed_shape, dtype=float)
    detailed_mean_pva_strength = np.empty(detailed_shape, dtype=float)
    detailed_minimum_pva_strength = np.empty(detailed_shape, dtype=float)
    detailed_mean_bump_contrast = np.empty(detailed_shape, dtype=float)

    snapshot_state = trained_state.copy()
    for snapshot_index, snapshot_training_time in enumerate(snapshot_time):
        snapshot_state.w_hd_to_hd = w_hd_history[snapshot_index].copy()
        snapshot_state.w_hr_to_hd = w_hr_history[snapshot_index].copy()
        if progress is not None:
            progress.set_postfix(
                job=(
                    "snapshot PI "
                    f"{snapshot_index + 1}/{snapshot_time.size} "
                    f"at t={snapshot_training_time:.3g} s"
                ),
                refresh=False,
            )
        for heading_index, theta_initial in enumerate(initial_headings):
            for velocity_index, angular_velocity in enumerate(velocities):
                (
                    detailed_time_averaged_abs_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_rms_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_final_abs_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_wrapped_time_averaged_abs_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_wrapped_rms_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_wrapped_final_abs_error[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_velocity_bias[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_mean_pva_strength[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_minimum_pva_strength[
                        snapshot_index, heading_index, velocity_index
                    ],
                    detailed_mean_bump_contrast[
                        snapshot_index, heading_index, velocity_index
                    ],
                ) = _run_one_frozen_probe(
                    config=config,
                    snapshot_state=snapshot_state,
                    theta_initial=float(theta_initial),
                    angular_velocity=float(angular_velocity),
                    cue_steps=cue_steps,
                    probe_steps=probe_steps,
                    average_start_step=average_start_step,
                    progress=progress,
                )

    # Preserve the historical snapshot-by-velocity arrays for plots and
    # downstream consumers.  The new detailed arrays retain every initial
    # heading so selection cannot hide basin-specific failures by cancellation.
    time_averaged_abs_error = np.mean(
        detailed_time_averaged_abs_error, axis=1
    )
    rms_error = np.sqrt(np.mean(np.square(detailed_rms_error), axis=1))
    final_abs_error = np.mean(detailed_final_abs_error, axis=1)
    wrapped_time_averaged_abs_error = np.mean(
        detailed_wrapped_time_averaged_abs_error, axis=1
    )
    wrapped_rms_error = np.sqrt(
        np.mean(np.square(detailed_wrapped_rms_error), axis=1)
    )
    wrapped_final_abs_error = np.mean(
        detailed_wrapped_final_abs_error, axis=1
    )
    velocity_bias = np.mean(detailed_velocity_bias, axis=1)
    mean_pva_strength = np.mean(detailed_mean_pva_strength, axis=1)
    minimum_pva_strength = np.min(detailed_minimum_pva_strength, axis=1)
    mean_bump_contrast = np.mean(detailed_mean_bump_contrast, axis=1)

    aggregate_time_averaged_abs_error = np.mean(
        time_averaged_abs_error,
        axis=1,
    )
    aggregate_rms_error = np.sqrt(np.mean(np.square(rms_error), axis=1))
    aggregate_rms_velocity_bias = np.sqrt(
        np.mean(np.square(detailed_velocity_bias), axis=(1, 2))
    )
    selection_metric = str(
        config.tests.weight_snapshot_pi_selection_metric
    ).lower()
    selection_score_by_name = {
        "mean_abs_unwrapped_error": aggregate_time_averaged_abs_error,
        "rms_velocity_bias": aggregate_rms_velocity_bias,
    }
    if selection_metric not in selection_score_by_name:
        raise ValueError(
            "tests.weight_snapshot_pi_selection_metric must be "
            "mean_abs_unwrapped_error or rms_velocity_bias"
        )
    selection_score = selection_score_by_name[selection_metric]
    params = VafidisToyParams.from_config(config)
    effective_hd_norm = np.empty(snapshot_time.shape, dtype=float)
    effective_hr_norm = np.empty(snapshot_time.shape, dtype=float)
    for snapshot_index in range(snapshot_time.size):
        effective_hd, effective_hr = effective_hd_distal_weight_matrices(
            w_hd_to_hd=w_hd_history[snapshot_index],
            w_hr_to_hd=w_hr_history[snapshot_index],
            normalization=params.hd_distal_normalization,
        )
        effective_hd_norm[snapshot_index] = float(np.linalg.norm(effective_hd))
        effective_hr_norm[snapshot_index] = float(np.linalg.norm(effective_hr))
    effective_hd_growth = np.divide(
        effective_hd_norm,
        effective_hd_norm[0],
        out=np.full_like(effective_hd_norm, np.nan),
        where=effective_hd_norm[0] > 0.0,
    )
    effective_hr_growth = np.divide(
        effective_hr_norm,
        effective_hr_norm[0],
        out=np.full_like(effective_hr_norm, np.nan),
        where=effective_hr_norm[0] > 0.0,
    )
    effective_hr_to_hd_norm_ratio = np.divide(
        effective_hr_norm,
        effective_hd_norm,
        out=np.full_like(effective_hr_norm, np.nan),
        where=effective_hd_norm > 0.0,
    )
    best_snapshot_index = int(np.nanargmin(selection_score))
    best_snapshot_time = float(snapshot_time[best_snapshot_index])
    best_error = float(aggregate_time_averaged_abs_error[best_snapshot_index])
    final_snapshot_error = float(aggregate_time_averaged_abs_error[-1])

    history = {
        "snapshot_time": snapshot_time,
        "snapshot_source_index": snapshot_source_index,
        "commanded_velocity": velocities,
        "initial_heading": initial_headings,
        "time_averaged_abs_pi_error": time_averaged_abs_error,
        "time_averaged_abs_pi_error_by_initial_heading": (
            detailed_time_averaged_abs_error
        ),
        "rms_pi_error": rms_error,
        "rms_pi_error_by_initial_heading": detailed_rms_error,
        "final_abs_pi_error": final_abs_error,
        "final_abs_pi_error_by_initial_heading": detailed_final_abs_error,
        "wrapped_time_averaged_abs_pi_error": wrapped_time_averaged_abs_error,
        "wrapped_time_averaged_abs_pi_error_by_initial_heading": (
            detailed_wrapped_time_averaged_abs_error
        ),
        "wrapped_rms_pi_error": wrapped_rms_error,
        "wrapped_rms_pi_error_by_initial_heading": detailed_wrapped_rms_error,
        "wrapped_final_abs_pi_error": wrapped_final_abs_error,
        "wrapped_final_abs_pi_error_by_initial_heading": (
            detailed_wrapped_final_abs_error
        ),
        "velocity_bias": velocity_bias,
        "velocity_bias_by_initial_heading": detailed_velocity_bias,
        "aggregate_rms_velocity_bias": aggregate_rms_velocity_bias,
        "selection_metric": np.asarray(selection_metric),
        "selection_score": np.asarray(selection_score, dtype=float),
        "effective_weight_norm_hd_to_hd": effective_hd_norm,
        "effective_weight_norm_hr_to_hd": effective_hr_norm,
        "effective_weight_growth_hd_to_hd": effective_hd_growth,
        "effective_weight_growth_hr_to_hd": effective_hr_growth,
        "effective_weight_norm_hr_to_hd_over_hd_to_hd": (
            effective_hr_to_hd_norm_ratio
        ),
        "mean_pva_strength": mean_pva_strength,
        "mean_pva_strength_by_initial_heading": detailed_mean_pva_strength,
        "minimum_pva_strength": minimum_pva_strength,
        "minimum_pva_strength_by_initial_heading": (
            detailed_minimum_pva_strength
        ),
        "mean_bump_contrast": mean_bump_contrast,
        "mean_bump_contrast_by_initial_heading": detailed_mean_bump_contrast,
        "aggregate_time_averaged_abs_pi_error": (
            aggregate_time_averaged_abs_error
        ),
        "aggregate_rms_pi_error": aggregate_rms_error,
        "best_snapshot_index": np.asarray(best_snapshot_index, dtype=int),
        "best_snapshot_time": np.asarray(best_snapshot_time, dtype=float),
    }
    metrics = {
        "weight_snapshot_pi_development_diagnostic_enabled": 1.0,
        "weight_snapshot_pi_snapshot_count": float(snapshot_time.size),
        "weight_snapshot_pi_velocity_count": float(velocities.size),
        "weight_snapshot_pi_initial_heading_count": float(
            initial_headings.size
        ),
        "weight_snapshot_pi_best_snapshot_time": best_snapshot_time,
        "weight_snapshot_pi_selection_uses_rms_velocity_bias": float(
            selection_metric == "rms_velocity_bias"
        ),
        "weight_snapshot_pi_best_selection_score": float(
            selection_score[best_snapshot_index]
        ),
        "weight_snapshot_pi_best_time_averaged_abs_error": best_error,
        "weight_snapshot_pi_best_time_averaged_abs_error_deg": float(
            np.rad2deg(best_error)
        ),
        "weight_snapshot_pi_final_time_averaged_abs_error": final_snapshot_error,
        "weight_snapshot_pi_final_time_averaged_abs_error_deg": float(
            np.rad2deg(final_snapshot_error)
        ),
        "weight_snapshot_pi_final_minus_best_error": (
            final_snapshot_error - best_error
        ),
        "weight_snapshot_pi_final_minus_best_error_deg": float(
            np.rad2deg(final_snapshot_error - best_error)
        ),
        "weight_snapshot_pi_best_rms_velocity_bias_deg_s": float(
            np.rad2deg(aggregate_rms_velocity_bias[best_snapshot_index])
        ),
        "weight_snapshot_pi_final_rms_velocity_bias_deg_s": float(
            np.rad2deg(aggregate_rms_velocity_bias[-1])
        ),
        "weight_snapshot_best_effective_hr_to_hd_over_hd_to_hd_norm": float(
            effective_hr_to_hd_norm_ratio[best_snapshot_index]
        ),
        "weight_snapshot_final_effective_hr_to_hd_over_hd_to_hd_norm": float(
            effective_hr_to_hd_norm_ratio[-1]
        ),
    }
    return history, metrics
