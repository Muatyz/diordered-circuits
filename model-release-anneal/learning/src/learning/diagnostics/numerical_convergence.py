"""Whole-step convergence audit for the coupled Vafidis toy dynamics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from learning.common.angles import circular_difference
from learning.config.schema import ExperimentConfig
from learning.diagnostics.protocols import initialize_frozen_protocol_state
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    step_vafidis_toy,
)


def _validated_settings(
    config: ExperimentConfig,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    settings = config.tests
    dt_values = np.asarray(settings.numerical_convergence_dt_values, dtype=float)
    methods = tuple(
        str(method).lower()
        for method in settings.numerical_convergence_methods
    )
    reference_dt = float(settings.numerical_convergence_reference_dt)
    cue_duration = float(settings.numerical_convergence_cue_duration)
    duration = float(settings.numerical_convergence_duration)
    sample_interval = float(settings.numerical_convergence_sample_interval)
    angular_velocity = float(settings.numerical_convergence_angular_velocity)
    max_heading_error = float(
        np.deg2rad(settings.numerical_convergence_max_heading_error_deg)
    )
    max_rate_error = float(settings.numerical_convergence_max_rate_rms_error)
    if (
        dt_values.ndim != 1
        or dt_values.size == 0
        or not np.isfinite(dt_values).all()
        or np.any(dt_values <= 0.0)
        or np.unique(dt_values).size != dt_values.size
    ):
        raise ValueError(
            "tests.numerical_convergence_dt_values must be unique positive values"
        )
    if not methods or len(set(methods)) != len(methods):
        raise ValueError(
            "tests.numerical_convergence_methods must be non-empty and unique"
        )
    allowed_methods = {"forward_euler", "exact_linear"}
    if set(methods) - allowed_methods:
        raise ValueError(
            "tests.numerical_convergence_methods contains an unknown method"
        )
    positive_scalars = {
        "reference_dt": reference_dt,
        "cue_duration": cue_duration,
        "duration": duration,
        "sample_interval": sample_interval,
        "max_heading_error": max_heading_error,
        "max_rate_error": max_rate_error,
    }
    for name, value in positive_scalars.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"numerical convergence {name} must be positive")
    if not np.isfinite(angular_velocity):
        raise ValueError("numerical convergence angular velocity must be finite")
    for dt in np.concatenate([dt_values, np.asarray([reference_dt])]):
        if not np.isclose(sample_interval / dt, round(sample_interval / dt)):
            raise ValueError(
                "numerical convergence sample_interval must be divisible by every dt"
            )
        if not np.isclose(duration / dt, round(duration / dt)):
            raise ValueError(
                "numerical convergence duration must be divisible by every dt"
            )
    return (
        dt_values,
        methods,
        reference_dt,
        cue_duration,
        duration,
        sample_interval,
        angular_velocity,
        max_heading_error,
        max_rate_error,
    )


def _config_with_integrator(
    config: ExperimentConfig,
    *,
    dt: float,
    method: str,
) -> ExperimentConfig:
    result = deepcopy(config)
    result.simulation.dt = float(dt)
    result.simulation.proximal_integration_method = str(method)
    return result


def _run_deterministic_trace(
    *,
    initial_state: VafidisToyState,
    params: VafidisToyParams,
    duration: float,
    sample_interval: float,
    angular_velocity: float,
) -> dict[str, np.ndarray]:
    total_steps = int(round(duration / params.dt))
    sample_stride = int(round(sample_interval / params.dt))
    sample_steps = np.arange(0, total_steps + 1, sample_stride, dtype=int)
    state = initial_state.copy()
    time: list[float] = []
    theta: list[float] = []
    r_hd: list[np.ndarray] = []
    v_hd_proximal: list[np.ndarray] = []
    v_hd_distal: list[np.ndarray] = []
    r_hd_to_hr_lp: list[np.ndarray] = []

    def record(step_index: int) -> None:
        time.append(float(step_index * params.dt))
        theta.append(float(state.theta_hd_decoded))
        r_hd.append(state.r_hd.copy())
        v_hd_proximal.append(state.v_hd_proximal.copy())
        v_hd_distal.append(state.v_hd_distal.copy())
        r_hd_to_hr_lp.append(state.r_hd_to_hr_lp.copy())

    record(0)
    sample_slot = 1
    for step_index in range(1, total_steps + 1):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=angular_velocity,
            visual_teacher=False,
            training=False,
            visual_noise=None,
        )
        if sample_slot < sample_steps.size and step_index == sample_steps[sample_slot]:
            record(step_index)
            sample_slot += 1
    return {
        "time": np.asarray(time, dtype=float),
        "theta_hd_decoded": np.asarray(theta, dtype=float),
        "r_hd": np.asarray(r_hd, dtype=float),
        "v_hd_proximal": np.asarray(v_hd_proximal, dtype=float),
        "v_hd_distal": np.asarray(v_hd_distal, dtype=float),
        "r_hd_to_hr_lp": np.asarray(r_hd_to_hr_lp, dtype=float),
    }


def _settle_shared_release_state(
    *,
    initial_state: VafidisToyState,
    params: VafidisToyParams,
    cue_duration: float,
) -> VafidisToyState:
    cue_steps = int(round(cue_duration / params.dt))
    state = initial_state.copy()
    for _step_index in range(cue_steps):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=0.0,
            visual_teacher=True,
            training=False,
            visual_noise=None,
        )
    return state


def run_numerical_convergence_diagnostic(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compare complete ordered steps against a high-resolution reference.

    This is intentionally a short deterministic intervention on copies of one
    shared frozen state.  It checks the complete coupled update, not only the
    analytically solvable proximal Eq. (4) substep.
    """

    (
        dt_values,
        methods,
        reference_dt,
        cue_duration,
        duration,
        sample_interval,
        angular_velocity,
        max_heading_error,
        max_rate_error,
    ) = _validated_settings(config)
    initial_state = initialize_frozen_protocol_state(
        config=config,
        trained_state=trained_state,
        theta_true=float(config.simulation.theta0),
    )
    reference_config = _config_with_integrator(
        config,
        dt=reference_dt,
        method="exact_linear",
    )
    reference_params = VafidisToyParams.from_config(reference_config)
    release_state = _settle_shared_release_state(
        initial_state=initial_state,
        params=reference_params,
        cue_duration=cue_duration,
    )
    reference = _run_deterministic_trace(
        initial_state=release_state,
        params=reference_params,
        duration=duration,
        sample_interval=sample_interval,
        angular_velocity=angular_velocity,
    )

    row_dt = np.asarray(
        [dt for method in methods for dt in dt_values],
        dtype=float,
    )
    row_method = np.asarray(
        [method for method in methods for _dt in dt_values],
        dtype="U32",
    )
    row_count = row_dt.size
    sample_count = reference["time"].size
    n_hd = initial_state.r_hd.size
    theta = np.full((row_count, sample_count), np.nan, dtype=float)
    heading_error = np.full_like(theta, np.nan)
    rate_rms_error = np.full_like(theta, np.nan)
    proximal_rms_error = np.full_like(theta, np.nan)
    distal_rms_error = np.full_like(theta, np.nan)
    lowpass_rms_error = np.full_like(theta, np.nan)
    finite = np.zeros(row_count, dtype=bool)
    valid = np.zeros(row_count, dtype=bool)
    proximal_amplification = np.full(row_count, np.nan, dtype=float)
    release_literal_effective_tau = np.full(row_count, np.nan, dtype=float)

    total_conductance = (
        float(config.model.g_l_hd_proximal)
        + float(config.model.g_d_hd_to_proximal)
    )
    capacitance = float(config.model.c_hd_proximal)
    tau_hd_to_hr = (
        float(config.model.tau_s)
        if config.model.tau_hd_to_hr is None
        else float(config.model.tau_hd_to_hr)
    )
    for row_index, (dt, method) in enumerate(zip(row_dt, row_method)):
        if progress is not None:
            progress.set_postfix(
                job=f"numerics {method} dt={1e3 * dt:g} ms",
                refresh=False,
            )
        if method == "forward_euler":
            proximal_amplification[row_index] = (
                1.0 - dt * total_conductance / capacitance
            )
        else:
            proximal_amplification[row_index] = np.exp(
                -dt * total_conductance / capacitance
            )
        # The released Python line omitted dt while using tau in ms.  This is
        # its equivalent physical time constant, reported only for parity
        # auditing; production dynamics continue to implement paper Eq. (9).
        release_literal_effective_tau[row_index] = tau_hd_to_hr * dt * 1000.0
        try:
            row_config = _config_with_integrator(
                config,
                dt=float(dt),
                method=str(method),
            )
            params = VafidisToyParams.from_config(row_config)
            trace = _run_deterministic_trace(
                initial_state=release_state,
                params=params,
                duration=duration,
                sample_interval=sample_interval,
                angular_velocity=angular_velocity,
            )
            valid[row_index] = True
        except (ValueError, FloatingPointError):
            if progress is not None:
                progress.update(1)
            continue
        theta[row_index] = trace["theta_hd_decoded"]
        heading_error[row_index] = circular_difference(
            trace["theta_hd_decoded"],
            reference["theta_hd_decoded"],
        )
        rate_rms_error[row_index] = np.sqrt(
            np.mean(np.square(trace["r_hd"] - reference["r_hd"]), axis=1)
        )
        proximal_rms_error[row_index] = np.sqrt(
            np.mean(
                np.square(
                    trace["v_hd_proximal"] - reference["v_hd_proximal"]
                ),
                axis=1,
            )
        )
        distal_rms_error[row_index] = np.sqrt(
            np.mean(
                np.square(trace["v_hd_distal"] - reference["v_hd_distal"]),
                axis=1,
            )
        )
        lowpass_rms_error[row_index] = np.sqrt(
            np.mean(
                np.square(
                    trace["r_hd_to_hr_lp"] - reference["r_hd_to_hr_lp"]
                ),
                axis=1,
            )
        )
        finite[row_index] = all(
            np.isfinite(values).all()
            for values in (
                theta[row_index],
                heading_error[row_index],
                rate_rms_error[row_index],
                proximal_rms_error[row_index],
                distal_rms_error[row_index],
                lowpass_rms_error[row_index],
            )
        )
        if progress is not None:
            progress.update(1)

    max_abs_heading_error = np.full(row_count, np.nan, dtype=float)
    max_rate_rms_error = np.full(row_count, np.nan, dtype=float)
    valid_finite = valid & finite
    if np.any(valid_finite):
        max_abs_heading_error[valid_finite] = np.max(
            np.abs(heading_error[valid_finite]), axis=1
        )
        max_rate_rms_error[valid_finite] = np.max(
            rate_rms_error[valid_finite], axis=1
        )
    convergence_passed = (
        valid_finite
        & (max_abs_heading_error <= max_heading_error)
        & (max_rate_rms_error <= max_rate_error)
    )

    recommended_dt_by_method: dict[str, float] = {}
    for method in methods:
        method_passed = convergence_passed & (row_method == method)
        recommended_dt_by_method[method] = (
            float(np.max(row_dt[method_passed]))
            if np.any(method_passed)
            else float("nan")
        )
    history = {
        "time": reference["time"],
        "dt": row_dt,
        "integration_method": row_method,
        "reference_dt": np.asarray(reference_dt, dtype=float),
        "cue_duration": np.asarray(cue_duration, dtype=float),
        "reference_integration_method": np.asarray("exact_linear"),
        "reference_theta_hd_decoded": reference["theta_hd_decoded"],
        "reference_r_hd": reference["r_hd"],
        "theta_hd_decoded": theta,
        "heading_error": heading_error,
        "rate_rms_error": rate_rms_error,
        "proximal_voltage_rms_error": proximal_rms_error,
        "distal_voltage_rms_error": distal_rms_error,
        "hd_to_hr_lowpass_rms_error": lowpass_rms_error,
        "max_abs_heading_error": max_abs_heading_error,
        "max_rate_rms_error": max_rate_rms_error,
        "valid_configuration": valid,
        "finite_trajectory": finite,
        "convergence_passed": convergence_passed,
        "proximal_homogeneous_amplification": proximal_amplification,
        "paper_hd_to_hr_time_constant": np.full(row_count, tau_hd_to_hr),
        "release_literal_hd_to_hr_effective_time_constant": (
            release_literal_effective_tau
        ),
        "n_hd": np.asarray(n_hd, dtype=int),
    }
    metrics: dict[str, float] = {
        "numerical_convergence_diagnostic_enabled": 1.0,
        "numerical_convergence_row_count": float(row_count),
        "numerical_convergence_valid_fraction": float(np.mean(valid)),
        "numerical_convergence_finite_fraction": float(np.mean(finite)),
        "numerical_convergence_pass_fraction": float(
            np.mean(convergence_passed)
        ),
        "numerical_convergence_reference_dt": reference_dt,
        "numerical_convergence_duration": duration,
        "numerical_convergence_cue_duration": cue_duration,
        "numerical_convergence_heading_tolerance_deg": float(
            np.rad2deg(max_heading_error)
        ),
        "numerical_convergence_rate_rms_tolerance": max_rate_error,
    }
    for method, recommended_dt in recommended_dt_by_method.items():
        metrics[
            f"numerical_convergence_recommended_max_dt_{method}"
        ] = recommended_dt
    return history, metrics
