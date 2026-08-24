"""Shared frozen-weight intervention protocols.

The velocity probe in this module is intentionally used by both online
checkpoint selection and offline weight-snapshot development.  Keeping the
rollout and aggregation here prevents a checkpoint from being called "best"
under a weaker protocol than the one used by the final darkness figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from learning.common.angles import circular_difference, pva_vector_strength, wrap_angle
from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.dynamics.hd_dynamics import compute_hd_distal_pathway_drives
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    initialize_vafidis_toy_state,
    step_vafidis_toy,
    validate_vafidis_toy_state,
)


@dataclass(frozen=True)
class FrozenVelocityProbeSpec:
    """Timing and low-speed success criterion for a frozen PI probe."""

    cue_duration: float
    probe_duration: float
    fit_start_time: float
    velocity_during_cue: bool = False
    minimum_moving_gain: float = 0.5

    def validate(self, *, dt: float) -> None:
        if not np.isfinite(self.cue_duration) or self.cue_duration < 0.0:
            raise ValueError("frozen velocity probe cue_duration must be non-negative")
        if not np.isfinite(self.probe_duration) or self.probe_duration <= 0.0:
            raise ValueError("frozen velocity probe probe_duration must be positive")
        if (
            not np.isfinite(self.fit_start_time)
            or self.fit_start_time < 0.0
            or self.fit_start_time >= self.probe_duration
        ):
            raise ValueError(
                "frozen velocity probe fit_start_time must lie in [0, probe_duration)"
            )
        if self.probe_duration - self.fit_start_time < 2.0 * float(dt):
            raise ValueError("frozen velocity probe fit window needs at least two steps")
        if not np.isfinite(self.minimum_moving_gain) or self.minimum_moving_gain < 0.0:
            raise ValueError("frozen velocity probe minimum_moving_gain must be non-negative")


def initialize_frozen_protocol_state(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
) -> VafidisToyState:
    """Create a reproducible fresh state carrying only frozen learned parameters."""

    params = VafidisToyParams.from_config(config)
    protocol_rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(
        config=config,
        rng=protocol_rng,
        theta_true=theta_true,
    )
    state.w_hd_to_hd = trained_state.w_hd_to_hd.copy()
    state.w_hr_to_hd = trained_state.w_hr_to_hd.copy()
    state.w_hd_to_hr = trained_state.w_hd_to_hr.copy()
    state.visual_tuning_profiles = (
        None
        if trained_state.visual_tuning_profiles is None
        else trained_state.visual_tuning_profiles.copy()
    )
    (
        state.i_hd_from_hd,
        state.i_hd_from_lhr,
        state.i_hd_from_rhr,
    ) = compute_hd_distal_pathway_drives(
        w_hd_to_hd=state.w_hd_to_hd,
        r_hd=state.r_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        r_hr=state.r_hr,
        normalization=params.hd_distal_normalization,
    )
    validate_vafidis_toy_state(state, params)
    return state


def run_frozen_constant_velocity_probe(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_release: float,
    angular_velocity: float,
    spec: FrozenVelocityProbeSpec,
    progress: Any | None = None,
) -> dict[str, float]:
    """Run one deterministic frozen-weight cue-to-darkness PI probe.

    ``theta_release`` is always the desired true heading at cue removal.  When
    ``velocity_during_cue`` is enabled, the cue starts earlier on the ring and
    moves at the probe velocity so every condition still enters darkness at
    the requested phase.  This avoids testing a moving darkness response from
    a stationary HR/HD cue state that the training stream never visits.
    """

    params = VafidisToyParams.from_config(config)
    spec.validate(dt=params.dt)
    angular_velocity = float(angular_velocity)
    cue_steps = int(round(spec.cue_duration / params.dt))
    actual_cue_duration = cue_steps * params.dt
    cue_velocity = angular_velocity if spec.velocity_during_cue else 0.0
    theta_start = float(
        wrap_angle(theta_release - cue_velocity * actual_cue_duration)
    )
    state = initialize_frozen_protocol_state(
        config=config,
        trained_state=trained_state,
        theta_true=theta_start,
    )
    for _step_index in range(cue_steps):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=cue_velocity,
            visual_teacher=True,
            training=False,
            visual_noise=None,
        )
        if progress is not None:
            progress.update(1)

    probe_steps = max(1, int(round(spec.probe_duration / params.dt)))
    fit_start_step = int(round(spec.fit_start_time / params.dt))
    previous_decoded = float(state.theta_hd_decoded)
    decoded_displacement = 0.0
    true_displacement = 0.0
    sample_count = 0
    sum_time = 0.0
    sum_decoded_displacement = 0.0
    sum_time_squared = 0.0
    sum_time_decoded_displacement = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    wrapped_absolute_error_sum = 0.0
    wrapped_squared_error_sum = 0.0
    pva_strength_sum = 0.0
    minimum_pva_strength = float("inf")
    bump_contrast_sum = 0.0
    minimum_bump_contrast = float("inf")
    final_abs_error = float("nan")
    final_wrapped_abs_error = float("nan")

    for probe_step in range(1, probe_steps + 1):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=angular_velocity,
            visual_teacher=False,
            training=False,
            visual_noise=None,
        )
        current_decoded = float(state.theta_hd_decoded)
        decoded_displacement += float(
            circular_difference(current_decoded, previous_decoded)
        )
        previous_decoded = current_decoded
        true_displacement += angular_velocity * params.dt
        accumulated_error = decoded_displacement - true_displacement
        wrapped_accumulated_error = float(wrap_angle(accumulated_error))
        final_abs_error = abs(accumulated_error)
        final_wrapped_abs_error = abs(wrapped_accumulated_error)

        if probe_step >= fit_start_step:
            sample_time = probe_step * params.dt
            sample_count += 1
            sum_time += sample_time
            sum_decoded_displacement += decoded_displacement
            sum_time_squared += sample_time * sample_time
            sum_time_decoded_displacement += sample_time * decoded_displacement
            absolute_error_sum += final_abs_error
            squared_error_sum += accumulated_error**2
            wrapped_absolute_error_sum += final_wrapped_abs_error
            wrapped_squared_error_sum += wrapped_accumulated_error**2
            strength = float(pva_vector_strength(state.theta_hd_pref, state.r_hd))
            contrast = float(np.max(state.r_hd) - np.min(state.r_hd))
            pva_strength_sum += strength
            minimum_pva_strength = min(minimum_pva_strength, strength)
            bump_contrast_sum += contrast
            minimum_bump_contrast = min(minimum_bump_contrast, contrast)
        if progress is not None:
            progress.update(1)

    fit_denominator = (
        sample_count * sum_time_squared - sum_time * sum_time
    )
    decoded_velocity = (
        (
            sample_count * sum_time_decoded_displacement
            - sum_time * sum_decoded_displacement
        )
        / fit_denominator
        if sample_count >= 2 and fit_denominator > 0.0
        else float("nan")
    )
    denominator = max(sample_count, 1)
    return {
        "time_averaged_abs_error": absolute_error_sum / denominator,
        "rms_error": float(np.sqrt(squared_error_sum / denominator)),
        "final_abs_error": final_abs_error,
        "wrapped_time_averaged_abs_error": (
            wrapped_absolute_error_sum / denominator
        ),
        "wrapped_rms_error": float(
            np.sqrt(wrapped_squared_error_sum / denominator)
        ),
        "wrapped_final_abs_error": final_wrapped_abs_error,
        "decoded_velocity": float(decoded_velocity),
        "velocity_bias": float(decoded_velocity - angular_velocity),
        "mean_pva_strength": pva_strength_sum / denominator,
        "minimum_pva_strength": minimum_pva_strength,
        "mean_bump_contrast": bump_contrast_sum / denominator,
        "minimum_bump_contrast": minimum_bump_contrast,
    }


def _depinning_threshold_for_sign(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    sign: float,
    minimum_moving_gain: float,
) -> float:
    sign_mask = sign * commanded_velocity > 1e-12
    magnitudes = np.unique(np.abs(commanded_velocity[sign_mask]))
    if magnitudes.size == 0:
        return float("nan")
    successful_by_magnitude: list[bool] = []
    for magnitude in np.sort(magnitudes):
        velocity_index = np.flatnonzero(
            sign_mask & np.isclose(np.abs(commanded_velocity), magnitude)
        )
        decoded = decoded_velocity[:, velocity_index]
        command = commanded_velocity[velocity_index][None, :]
        gain = np.divide(decoded, command)
        successful_by_magnitude.append(
            bool(
                np.isfinite(decoded).all()
                and np.all(command * decoded > 0.0)
                and np.all(gain >= minimum_moving_gain)
            )
        )
    successful = np.asarray(successful_by_magnitude, dtype=bool)
    for magnitude_index, magnitude in enumerate(np.sort(magnitudes)):
        if np.all(successful[magnitude_index:]):
            return float(magnitude)
    return float("inf")


def summarize_frozen_velocity_probe_grid(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    minimum_pva_strength: np.ndarray,
    minimum_bump_contrast: np.ndarray,
    minimum_moving_gain: float,
) -> dict[str, float]:
    """Aggregate a heading-by-velocity grid using worst-case PI criteria."""

    commanded_velocity = np.asarray(commanded_velocity, dtype=float)
    decoded_velocity = np.asarray(decoded_velocity, dtype=float)
    if (
        decoded_velocity.ndim != 2
        or decoded_velocity.shape[1] != commanded_velocity.size
    ):
        raise ValueError("decoded_velocity must have shape (heading, velocity)")
    expected_shape = decoded_velocity.shape
    if np.asarray(minimum_pva_strength).shape != expected_shape:
        raise ValueError("minimum_pva_strength must match decoded_velocity")
    if np.asarray(minimum_bump_contrast).shape != expected_shape:
        raise ValueError("minimum_bump_contrast must match decoded_velocity")
    if not np.isfinite(minimum_moving_gain) or minimum_moving_gain < 0.0:
        raise ValueError("minimum_moving_gain must be finite and non-negative")

    velocity_bias = decoded_velocity - commanded_velocity[None, :]
    zero_mask = np.abs(commanded_velocity) <= 1e-12
    zero_drift = decoded_velocity[:, zero_mask]
    finite_bias = np.isfinite(velocity_bias)
    fully_defined = bool(
        np.all(finite_bias)
        and np.isfinite(minimum_pva_strength).all()
        and np.isfinite(minimum_bump_contrast).all()
    )
    rms_velocity_bias = (
        float(np.sqrt(np.mean(np.square(velocity_bias[finite_bias]))))
        if np.any(finite_bias)
        else float("inf")
    )
    maximum_abs_velocity_bias = (
        float(np.max(np.abs(velocity_bias[finite_bias])))
        if np.any(finite_bias)
        else float("inf")
    )
    maximum_abs_zero_velocity_drift = (
        float(np.max(np.abs(zero_drift[np.isfinite(zero_drift)])))
        if np.any(np.isfinite(zero_drift))
        else float("inf")
    )
    negative_depinning_velocity = _depinning_threshold_for_sign(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
        sign=-1.0,
        minimum_moving_gain=minimum_moving_gain,
    )
    positive_depinning_velocity = _depinning_threshold_for_sign(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
        sign=1.0,
        minimum_moving_gain=minimum_moving_gain,
    )
    sign_thresholds = np.asarray(
        [negative_depinning_velocity, positive_depinning_velocity], dtype=float
    )
    depinning_velocity = (
        float(np.max(sign_thresholds[~np.isnan(sign_thresholds)]))
        if np.any(~np.isnan(sign_thresholds))
        else float("inf")
    )
    moving_mask = np.abs(commanded_velocity) > 1e-12
    moving_command = commanded_velocity[moving_mask][None, :]
    moving_decoded = decoded_velocity[:, moving_mask]
    stalled = (
        ~np.isfinite(moving_decoded)
        | (moving_command * moving_decoded <= 0.0)
        | (moving_decoded / moving_command < minimum_moving_gain)
    )
    return {
        "rms_velocity_bias": rms_velocity_bias,
        "maximum_abs_velocity_bias": maximum_abs_velocity_bias,
        "maximum_abs_zero_velocity_drift": maximum_abs_zero_velocity_drift,
        "negative_depinning_velocity": negative_depinning_velocity,
        "positive_depinning_velocity": positive_depinning_velocity,
        "depinning_velocity": depinning_velocity,
        "stall_fraction": float(np.mean(stalled)) if stalled.size else 0.0,
        "minimum_pva_strength": float(np.min(minimum_pva_strength)),
        "minimum_bump_contrast": float(np.min(minimum_bump_contrast)),
        "fully_defined": float(fully_defined),
    }


def run_frozen_velocity_probe_grid(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    initial_headings: np.ndarray,
    velocities: np.ndarray,
    spec: FrozenVelocityProbeSpec,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run and summarize a deterministic heading-by-velocity probe grid."""

    initial_headings = np.asarray(initial_headings, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if initial_headings.ndim != 1 or initial_headings.size == 0:
        raise ValueError("initial_headings must be a non-empty 1D array")
    if velocities.ndim != 1 or velocities.size == 0:
        raise ValueError("velocities must be a non-empty 1D array")
    if not np.isfinite(initial_headings).all() or not np.isfinite(velocities).all():
        raise ValueError("frozen velocity probe grid values must be finite")
    spec.validate(dt=config.simulation.dt)

    shape = (initial_headings.size, velocities.size)
    metric_names = (
        "time_averaged_abs_error",
        "rms_error",
        "final_abs_error",
        "wrapped_time_averaged_abs_error",
        "wrapped_rms_error",
        "wrapped_final_abs_error",
        "decoded_velocity",
        "velocity_bias",
        "mean_pva_strength",
        "minimum_pva_strength",
        "mean_bump_contrast",
        "minimum_bump_contrast",
    )
    details = {metric_name: np.empty(shape, dtype=float) for metric_name in metric_names}
    for heading_index, theta_release in enumerate(initial_headings):
        for velocity_index, angular_velocity in enumerate(velocities):
            result = run_frozen_constant_velocity_probe(
                config=config,
                trained_state=trained_state,
                theta_release=float(theta_release),
                angular_velocity=float(angular_velocity),
                spec=spec,
                progress=progress,
            )
            for metric_name in metric_names:
                details[metric_name][heading_index, velocity_index] = result[
                    metric_name
                ]
    nonzero_velocity = np.abs(velocities) > 1e-12
    details["velocity_gain"] = np.divide(
        details["decoded_velocity"],
        velocities[None, :],
        out=np.full(shape, np.nan, dtype=float),
        where=np.broadcast_to(nonzero_velocity[None, :], shape),
    )
    details["stalled"] = np.asarray(
        nonzero_velocity[None, :]
        & (
            ~np.isfinite(details["decoded_velocity"])
            | (velocities[None, :] * details["decoded_velocity"] <= 0.0)
            | (details["velocity_gain"] < spec.minimum_moving_gain)
        ),
        dtype=float,
    )
    summary = summarize_frozen_velocity_probe_grid(
        commanded_velocity=velocities,
        decoded_velocity=details["decoded_velocity"],
        minimum_pva_strength=details["minimum_pva_strength"],
        minimum_bump_contrast=details["minimum_bump_contrast"],
        minimum_moving_gain=spec.minimum_moving_gain,
    )
    return details, summary
