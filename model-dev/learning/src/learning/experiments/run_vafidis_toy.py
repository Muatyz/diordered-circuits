"""Train the Vafidis-style predictive local plasticity toy model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

try:
    from tqdm import tqdm, trange
except ImportError:  # pragma: no cover
    class _NoOpProgress:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback

        def update(self, count):
            del count

        def set_postfix(self, *args, **kwargs):
            del args, kwargs

    def tqdm(*args, **kwargs):
        return _NoOpProgress(*args, **kwargs)

    def trange(*args, **kwargs):
        del kwargs
        return range(*args)

from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.analysis.slow_manifold import (
    analyze_ramesan_firing_rate_geometry,
    analyze_ramesan_phase_landscape,
    analyze_slow_manifold_candidates,
    candidate_angular_bin_counts,
    empty_slow_manifold_result,
    select_slow_candidate_indices,
    summarize_candidate_angle_clusters,
)
from learning.analysis.metrics import (
    angular_first_passage_time,
    classify_endpoint_map_fixed_points,
    decode_heading_by_clark_overlap,
    circular_error_trace,
    empirical_tuning_preferred_directions,
    estimate_relaxation_e_folding_time,
    estimate_decoded_velocity,
    estimate_effective_diffusion_coefficient,
    estimate_velocity_tracking_operating_range,
    fit_anomalous_diffusion_power_law,
    final_abs_circular_error,
    full_protocol_pi_error_trace,
    linear_fit_slope_intercept,
    nearest_closed_manifold_distance,
    release_relative_pi_error_trace,
    rms_circular_error,
    summarize_ensemble_diffusion_coefficient,
    summarize_ensemble_diffusion_trajectories,
    summarize_pi_error_ensemble,
    summarize_velocity_gain,
    summarize_velocity_tracking,
)
from learning.analysis.weights import summarize_weight_structure
from learning.common.angles import (
    circular_difference,
    collapse_activity_by_theta,
    pva_vector_strength,
    unwrap_heading_trace,
    wrap_angle,
)
from learning.common.arrays import l2_norm
from learning.common.random import make_rng
from learning.config.diagnostics import (
    diagnostic_is_enabled,
    selected_diagnostics,
)
from learning.config.load_config import (
    find_project_root,
    find_repository_root,
    load_experiment_config,
    save_yaml,
)
from learning.config.schema import (
    TRAINING_INTEGRATION_BLOCK_MULTIRATE,
    ExperimentConfig,
)
from learning.dynamics.activation import apply_activation
from learning.dynamics.autonomous import FrozenAutonomousDynamics
from learning.dynamics.hd_dynamics import (
    compute_hd_distal_pathway_drives,
    effective_hd_distal_weight_matrices,
)
from learning.dynamics.hr_dynamics import compute_i_hr
from learning.diagnostics.protocols import (
    FrozenVelocityProbeSpec,
    initialize_frozen_protocol_state,
    run_frozen_velocity_probe_grid,
)
from learning.diagnostics.numerical_convergence import (
    run_numerical_convergence_diagnostic,
)
from learning.diagnostics.training_error import (
    TrainingAbsoluteLearningErrorRecorder,
)
from learning.diagnostics.weight_development import (
    run_weight_snapshot_pi_development_diagnostic,
    select_weight_snapshot_indices,
)
from learning.io.run_dir import create_run_dir
from learning.io.save_load import save_json, save_npz
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    initialize_vafidis_toy_state,
    validate_vafidis_toy_state,
    step_vafidis_toy,
)
from learning.plasticity.predictive_local import (
    update_predictive_local_weights_block,
)
from learning.stimuli.visual import ScheduledVisualAmplitude, VisualCurrentNoiseProcess
from learning.stimuli.velocity import OUAngularVelocity, ScheduledOUAngularVelocity


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path.resolve()
    for project_root_candidate in [
        Path.cwd(),
        Path.cwd() / "learning",
        find_project_root(),
    ]:
        candidate_config_path = project_root_candidate / config_path
        if candidate_config_path.exists():
            return candidate_config_path.resolve()
    raise FileNotFoundError(f"Could not resolve config path: {config_arg}")


def build_training_velocity_process(
    *,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Callable[[float], float]:
    if config.velocity.process == "ou":
        if config.velocity.training_ou_std_schedule:
            process = ScheduledOUAngularVelocity(
                mean=config.velocity.mean,
                tau=config.velocity.tau,
                clip=config.velocity.clip,
                rng=rng,
                total_duration=config.simulation.train_duration,
                std_schedule=config.velocity.training_ou_std_schedule,
            )
            return process.step
        process = OUAngularVelocity(
            mean=config.velocity.mean,
            std=config.velocity.std,
            tau=config.velocity.tau,
            clip=config.velocity.clip,
            rng=rng,
        )
        return process.step
    raise ValueError(f"Unknown velocity process: {config.velocity.process}")


def build_visual_noise_process(
    *,
    config: ExperimentConfig,
    params: VafidisToyParams,
    rng: np.random.Generator,
    enabled: bool,
) -> VisualCurrentNoiseProcess | None:
    if not enabled or config.visual.noise_std <= 0.0:
        return None
    return VisualCurrentNoiseProcess(
        mode=config.visual.noise_process,
        std=config.visual.noise_std,
        shape=(params.n_theta,),
        rng=rng,
        correlation_time=config.visual.noise_correlation_time,
    )


def step_visual_noise(
    *,
    process: VisualCurrentNoiseProcess | None,
    dt: float,
    visual_teacher: bool,
) -> np.ndarray | None:
    if process is None:
        return None
    noise_sample = process.step(dt)
    return noise_sample if visual_teacher else None


def _history_to_arrays(history: dict[str, list[np.ndarray | float]]) -> dict[str, np.ndarray]:
    return {history_name: np.asarray(history_value) for history_name, history_value in history.items()}


def _new_history(include_activity: bool) -> dict[str, list[np.ndarray | float]]:
    history: dict[str, list[np.ndarray | float]] = {
        "time": [],
        "theta_true": [],
        "theta_hd_decoded": [],
        "theta_hd_decoded_peak": [],
        "angular_velocity": [],
        "visual_teacher": [],
        "phase_id": [],
        "mean_e_hd": [],
        "rms_e_hd": [],
        "mean_r_hd": [],
        "pva_strength_hd": [],
        "bump_contrast_hd": [],
        "mean_r_lhr": [],
        "mean_r_rhr": [],
        "contrast_r_lhr": [],
        "contrast_r_rhr": [],
        "weight_norm_hd_to_hd": [],
        "weight_norm_hr_to_hd": [],
        "mean_i_hd_from_hd": [],
        "rms_i_hd_from_hd": [],
        "mean_i_hd_from_lhr": [],
        "rms_i_hd_from_lhr": [],
        "mean_i_hd_from_rhr": [],
        "rms_i_hd_from_rhr": [],
    }
    if include_activity:
        history["r_hd"] = []
        history["i_vis_to_hd"] = []
    return history


def record_state(
    *,
    history: dict[str, list[np.ndarray | float]],
    state: VafidisToyState,
    include_activity: bool,
    visual_teacher: bool | None = None,
    phase_id: int | None = None,
) -> None:
    history["time"].append(float(state.time))
    history["theta_true"].append(float(state.theta_true))
    history["theta_hd_decoded"].append(float(state.theta_hd_decoded))
    history["theta_hd_decoded_peak"].append(float(state.theta_hd_decoded_peak))
    history["angular_velocity"].append(float(state.angular_velocity))
    history["visual_teacher"].append(float("nan") if visual_teacher is None else float(visual_teacher))
    history["phase_id"].append(float("nan") if phase_id is None else float(phase_id))
    history["mean_e_hd"].append(float(np.mean(state.e_hd)))
    history["rms_e_hd"].append(float(np.sqrt(np.mean(np.square(state.e_hd)))))
    history["mean_r_hd"].append(float(np.mean(state.r_hd)))
    history["pva_strength_hd"].append(
        pva_vector_strength(state.theta_hd_pref, state.r_hd)
    )
    history["bump_contrast_hd"].append(float(np.max(state.r_hd) - np.min(state.r_hd)))
    n_hr_per_wing = state.r_hr.size // 2
    r_lhr = state.r_hr[:n_hr_per_wing]
    r_rhr = state.r_hr[n_hr_per_wing:]
    history["mean_r_lhr"].append(float(np.mean(r_lhr)))
    history["mean_r_rhr"].append(float(np.mean(r_rhr)))
    history["contrast_r_lhr"].append(float(np.max(r_lhr) - np.min(r_lhr)))
    history["contrast_r_rhr"].append(float(np.max(r_rhr) - np.min(r_rhr)))
    history["weight_norm_hd_to_hd"].append(l2_norm(state.w_hd_to_hd))
    history["weight_norm_hr_to_hd"].append(l2_norm(state.w_hr_to_hd))
    for pathway_name in ["hd", "lhr", "rhr"]:
        pathway_current = getattr(state, f"i_hd_from_{pathway_name}")
        history[f"mean_i_hd_from_{pathway_name}"].append(
            float(np.mean(pathway_current))
        )
        history[f"rms_i_hd_from_{pathway_name}"].append(
            float(np.sqrt(np.mean(np.square(pathway_current))))
        )
    if include_activity:
        history["r_hd"].append(state.r_hd.copy())
        history["i_vis_to_hd"].append(state.i_vis_to_hd.copy())


def _relative_array_change(current: np.ndarray, reference: np.ndarray) -> float:
    """Return a scale-free change without introducing a reference population."""
    denominator = max(l2_norm(current), l2_norm(reference), 1e-12)
    return float(l2_norm(current - reference) / denominator)


def _validate_training_early_stopping(config: ExperimentConfig) -> None:
    settings = config.simulation.early_stopping
    if not settings.enabled:
        return
    if settings.min_duration <= 0.0:
        raise ValueError("simulation.early_stopping.min_duration must be positive")
    if settings.min_duration > config.simulation.train_duration:
        raise ValueError(
            "simulation.early_stopping.min_duration must not exceed train_duration"
        )
    if settings.check_interval <= 0.0:
        raise ValueError("simulation.early_stopping.check_interval must be positive")
    if settings.window_duration < settings.check_interval:
        raise ValueError(
            "simulation.early_stopping.window_duration must be at least check_interval"
        )
    if int(settings.patience_checks) < 1:
        raise ValueError("simulation.early_stopping.patience_checks must be at least one")
    if settings.relative_weight_change_tolerance < 0.0:
        raise ValueError(
            "simulation.early_stopping.relative_weight_change_tolerance "
            "must be non-negative"
        )
    if settings.relative_error_change_tolerance < 0.0:
        raise ValueError(
            "simulation.early_stopping.relative_error_change_tolerance "
            "must be non-negative"
        )


def _validate_training_checkpoint_selection(config: ExperimentConfig) -> None:
    settings = config.simulation.checkpoint_selection
    if not settings.enabled:
        return
    if settings.min_duration <= 0.0:
        raise ValueError(
            "simulation.checkpoint_selection.min_duration must be positive"
        )
    if settings.min_duration > config.simulation.train_duration:
        raise ValueError(
            "simulation.checkpoint_selection.min_duration must not exceed "
            "train_duration"
        )
    if settings.check_interval <= 0.0:
        raise ValueError(
            "simulation.checkpoint_selection.check_interval must be positive"
        )
    if int(settings.patience_checks) < 1:
        raise ValueError(
            "simulation.checkpoint_selection.patience_checks must be at least one"
        )
    if int(settings.success_checks) < 1:
        raise ValueError(
            "simulation.checkpoint_selection.success_checks must be at least one"
        )
    if settings.minimum_improvement < 0.0:
        raise ValueError(
            "simulation.checkpoint_selection.minimum_improvement must be "
            "non-negative"
        )
    if settings.cue_duration < 0.0:
        raise ValueError(
            "simulation.checkpoint_selection.cue_duration must be non-negative"
        )
    if settings.probe_duration <= 0.0:
        raise ValueError(
            "simulation.checkpoint_selection.probe_duration must be positive"
        )
    if not 0.0 <= settings.fit_start_time < settings.probe_duration:
        raise ValueError(
            "simulation.checkpoint_selection.fit_start_time must lie in "
            "[0, probe_duration)"
        )
    velocities = np.asarray(settings.velocities, dtype=float)
    headings = np.asarray(settings.initial_headings, dtype=float)
    if velocities.ndim != 1 or velocities.size == 0 or not np.isfinite(velocities).all():
        raise ValueError(
            "simulation.checkpoint_selection.velocities must be a non-empty "
            "finite list"
        )
    if headings.ndim != 1 or headings.size == 0 or not np.isfinite(headings).all():
        raise ValueError(
            "simulation.checkpoint_selection.initial_headings must be a "
            "non-empty finite list"
        )
    if not np.any(np.abs(velocities) <= 1e-12):
        raise ValueError(
            "simulation.checkpoint_selection.velocities must include zero"
        )
    if not np.any(velocities < -1e-12) or not np.any(velocities > 1e-12):
        raise ValueError(
            "simulation.checkpoint_selection.velocities must include both "
            "negative and positive probes"
        )
    if settings.probe_duration - settings.fit_start_time < 2.0 * config.simulation.dt:
        raise ValueError(
            "simulation.checkpoint_selection velocity fit must contain at "
            "least two timesteps"
        )
    if settings.selection_metric not in {
        "rms_velocity_bias",
        "maximum_abs_velocity_bias",
        "depinning_velocity",
    }:
        raise ValueError(
            "simulation.checkpoint_selection.selection_metric must be "
            "rms_velocity_bias, maximum_abs_velocity_bias, or depinning_velocity"
        )
    if (
        not np.isfinite(settings.minimum_moving_gain)
        or settings.minimum_moving_gain < 0.0
    ):
        raise ValueError(
            "simulation.checkpoint_selection.minimum_moving_gain must be "
            "finite and non-negative"
        )
    for field_name in (
        "maximum_rms_velocity_bias",
        "maximum_abs_velocity_bias",
        "maximum_abs_zero_velocity_drift",
        "minimum_pva_strength",
        "minimum_bump_contrast",
    ):
        value = float(getattr(settings, field_name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"simulation.checkpoint_selection.{field_name} must be "
                "finite and non-negative"
            )
    if (
        np.isnan(settings.maximum_depinning_velocity)
        or settings.maximum_depinning_velocity < 0.0
    ):
        raise ValueError(
            "simulation.checkpoint_selection.maximum_depinning_velocity must "
            "be non-negative"
        )


def _interval_duration_to_steps(
    interval_duration: float | None,
    *,
    dt: float,
    field_name: str,
) -> int | None:
    """Convert a physical sampling duration to the nearest positive step count."""

    if interval_duration is None:
        return None
    duration = float(interval_duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"simulation.{field_name} must be finite and positive")
    return max(1, int(round(duration / float(dt))))


def evaluate_training_checkpoint_behavior(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> dict[str, float]:
    """Evaluate a checkpoint without changing weights or training streams."""
    settings = config.simulation.checkpoint_selection
    _details, summary = run_frozen_velocity_probe_grid(
        config=config,
        trained_state=trained_state,
        initial_headings=np.asarray(settings.initial_headings, dtype=float),
        velocities=np.asarray(settings.velocities, dtype=float),
        spec=FrozenVelocityProbeSpec(
            cue_duration=float(settings.cue_duration),
            probe_duration=float(settings.probe_duration),
            fit_start_time=float(settings.fit_start_time),
            velocity_during_cue=bool(settings.velocity_during_cue),
            minimum_moving_gain=float(settings.minimum_moving_gain),
        ),
    )
    acceptance_passed = bool(
        summary["fully_defined"]
        and summary["rms_velocity_bias"]
        <= float(settings.maximum_rms_velocity_bias)
        and summary["maximum_abs_velocity_bias"]
        <= float(settings.maximum_abs_velocity_bias)
        and summary["maximum_abs_zero_velocity_drift"]
        <= float(settings.maximum_abs_zero_velocity_drift)
        and summary["depinning_velocity"]
        <= float(settings.maximum_depinning_velocity)
        and summary["minimum_pva_strength"]
        >= float(settings.minimum_pva_strength)
        and summary["minimum_bump_contrast"]
        >= float(settings.minimum_bump_contrast)
    )
    result = {
        metric_name: float(metric_value)
        for metric_name, metric_value in summary.items()
    }
    result.update({
        "score": float(summary[settings.selection_metric]),
        "acceptance_passed": float(acceptance_passed),
    })
    return result


def _restore_state_from_weight_archive(
    *,
    state: VafidisToyState,
    archive_path: Path,
    params: VafidisToyParams,
) -> None:
    """Copy plastic matrices and static geometry from a saved weight archive.

    Supported archives are the run artifacts written by this package
    (``trained_weights.npz`` / ``best_weights.npz`` / ``final_weights.npz``):
    they store ``theta_hd_pref``, ``w_hd_to_hd``, ``w_hr_to_hd``,
    ``w_hd_to_hr`` and (for heterogeneous profiles) ``visual_tuning_profiles``.
    Dynamic state (voltages, currents, PSP traces) is left at the freshly
    initialized values so every resume is reproducible.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"init-weights archive not found: {archive_path}")
    archive = np.load(archive_path)
    required = ("theta_hd_pref", "w_hd_to_hd", "w_hr_to_hd", "w_hd_to_hr")
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise ValueError(
            f"init-weights archive {archive_path} is missing fields {missing}; "
            "expected a run weight archive (trained_weights.npz, best_weights.npz, "
            "or final_weights.npz)"
        )
    theta_hd_pref = np.asarray(archive["theta_hd_pref"], dtype=float)
    w_hd_to_hd = np.asarray(archive["w_hd_to_hd"], dtype=float)
    w_hr_to_hd = np.asarray(archive["w_hr_to_hd"], dtype=float)
    w_hd_to_hr = np.asarray(archive["w_hd_to_hr"], dtype=float)
    if theta_hd_pref.shape != (params.n_theta,):
        raise ValueError(
            f"init-weights theta_hd_pref shape {theta_hd_pref.shape} does not "
            f"match model.n_theta={params.n_theta}"
        )
    if w_hd_to_hd.shape != (params.n_theta, params.n_theta) or (
        w_hr_to_hd.shape != (params.n_theta, params.n_hr)
    ):
        raise ValueError(
            "init-weights plastic matrix shapes do not match the model dimensions"
        )
    if w_hd_to_hr.shape != (params.n_hr, params.n_theta):
        raise ValueError(
            "init-weights w_hd_to_hr shape does not match (n_hr, n_theta)"
        )
    state.theta_hd_pref = theta_hd_pref
    state.w_hd_to_hd = w_hd_to_hd.copy()
    state.w_hr_to_hd = w_hr_to_hd.copy()
    state.w_hd_to_hr = w_hd_to_hr.copy()
    if "visual_tuning_profiles" in archive.files:
        profiles = np.asarray(archive["visual_tuning_profiles"], dtype=float)
        if profiles.size:
            if profiles.ndim != 2 or profiles.shape[0] != params.n_theta:
                raise ValueError(
                    "init-weights visual_tuning_profiles do not match model.n_theta"
                )
            state.visual_tuning_profiles = profiles
    validate_vafidis_toy_state(state, params)


def run_training(
    *,
    config: ExperimentConfig,
    rng: np.random.Generator,
    recovery_checkpoint_callback: Callable[[VafidisToyState], None] | None = None,
    init_weights_path: str | Path | None = None,
) -> tuple[VafidisToyState, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Train the toy model, optionally starting from pretrained weights.

    ``init_weights_path`` points to a saved ``*.npz`` weight archive (e.g. a
    previous run's ``best_weights.npz`` / ``trained_weights.npz``).  The
    plastic matrices, fixed HD-to-HR projection, preferred directions and the
    optional heterogeneous visual tuning profiles are copied onto a freshly
    initialized state, so the protocol starts from the pretrained structure
    while its dynamic (voltage/current) state is reset reproducibly.
    """
    _validate_training_early_stopping(config)
    _validate_training_checkpoint_selection(config)
    params = VafidisToyParams.from_config(config)
    random_stream_mode = config.simulation.random_stream_mode.lower()
    if random_stream_mode == "legacy_shared":
        initialization_rng = rng
        velocity_rng = rng
    elif random_stream_mode == "component_split":
        initialization_rng = make_rng(
            config.simulation.seed + config.simulation.initialization_seed_offset
        )
        velocity_rng = make_rng(
            config.simulation.seed + config.simulation.velocity_seed_offset
        )
    else:
        raise ValueError(
            "simulation.random_stream_mode must be legacy_shared or component_split"
        )
    state = initialize_vafidis_toy_state(config=config, rng=initialization_rng)
    if init_weights_path is not None:
        _restore_state_from_weight_archive(
            state=state,
            archive_path=Path(init_weights_path),
            params=params,
        )
    visual_rng = make_rng(config.simulation.seed + config.visual.noise_seed_offset)
    visual_noise_process = build_visual_noise_process(
        config=config,
        params=params,
        rng=visual_rng,
        enabled=config.visual.apply_noise_during_training,
    )
    angular_velocity_step = build_training_velocity_process(config=config, rng=velocity_rng)
    # Optional "night-vision" curriculum: anneal the teacher amplitude during
    # training so the learned recurrent weights must sustain the bump as the
    # cue weakens.  Frozen-weight diagnostics keep the configured amplitude.
    if config.visual.training_amplitude_schedule:
        visual_amplitude_step = ScheduledVisualAmplitude(
            total_duration=config.simulation.train_duration,
            amplitude_schedule=config.visual.training_amplitude_schedule,
        )
    else:
        visual_amplitude_step = None
    train_steps = int(round(config.simulation.train_duration / params.dt))
    save_interval_steps = _interval_duration_to_steps(
        config.simulation.save_interval_duration,
        dt=params.dt,
        field_name="save_interval_duration",
    )
    if save_interval_steps is None:
        raise ValueError("simulation.save_interval_duration cannot be null")
    weight_snapshot_interval_steps = _interval_duration_to_steps(
        config.simulation.weight_snapshot_interval_duration,
        dt=params.dt,
        field_name="weight_snapshot_interval_duration",
    )
    recovery_checkpoint_interval_steps = (
        weight_snapshot_interval_steps
        if weight_snapshot_interval_steps is not None
        else save_interval_steps
    )
    configured_plasticity_block_steps = int(
        round(params.plasticity_update_interval_duration / params.dt)
    )
    use_block_multirate = bool(
        config.simulation.plasticity_enabled
        and params.training_integration_method
        == TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    plasticity_block_steps = (
        configured_plasticity_block_steps if use_block_multirate else 1
    )
    history = _new_history(include_activity=True)
    learning_error_recorder = (
        TrainingAbsoluteLearningErrorRecorder(
            total_steps=train_steps,
            n_hd=params.n_theta,
            dt=params.dt,
            window_duration=config.tests.learning_error_window_duration,
            interval_fraction=config.tests.learning_error_interval_fraction,
            rate_scale_to_spikes_per_second=(
                config.tests.learning_error_rate_scale_to_spikes_per_second
            ),
        )
        if diagnostic_is_enabled(config, "learning_error_development")
        else None
    )
    weight_history: dict[str, list[np.ndarray | float]] = {
        "time": [0.0],
        "w_hd_to_hd": [state.w_hd_to_hd.copy()],
        "w_hr_to_hd": [state.w_hr_to_hd.copy()],
        "weight_norm_hd_to_hd": [l2_norm(state.w_hd_to_hd)],
        "weight_norm_hr_to_hd": [l2_norm(state.w_hr_to_hd)],
    }
    effective_w_hd_to_hd, effective_w_hr_to_hd = effective_hd_distal_weight_matrices(
        w_hd_to_hd=state.w_hd_to_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        normalization=params.hd_distal_normalization,
    )
    weight_history["effective_weight_norm_hd_to_hd"] = [
        l2_norm(effective_w_hd_to_hd)
    ]
    weight_history["effective_weight_norm_hr_to_hd"] = [
        l2_norm(effective_w_hr_to_hd)
    ]
    early_stopping = config.simulation.early_stopping
    early_stopping_enabled = bool(early_stopping.enabled)
    early_check_steps = max(
        1,
        int(round(float(early_stopping.check_interval) / params.dt)),
    )
    early_window_steps = max(
        early_check_steps,
        int(round(float(early_stopping.window_duration) / params.dt)),
    )
    early_min_steps = max(
        1,
        int(round(float(early_stopping.min_duration) / params.dt)),
    )
    convergence_checkpoints: list[
        tuple[int, np.ndarray, np.ndarray, float]
    ] = []
    convergence_history: dict[str, list[float]] = {
        "time": [],
        "relative_change_w_hd_to_hd": [],
        "relative_change_w_hr_to_hd": [],
        "relative_change_rms_e_hd": [],
        "interval_mean_rms_e_hd": [],
        "consecutive_converged_checks": [],
    }
    interval_error_sum = 0.0
    interval_error_count = 0
    consecutive_converged_checks = 0
    stopped_early = False
    stop_reason = "hard_cap"
    checkpoint_selection = config.simulation.checkpoint_selection
    checkpoint_selection_enabled = bool(checkpoint_selection.enabled)
    checkpoint_selection_check_steps = max(
        1,
        int(round(float(checkpoint_selection.check_interval) / params.dt)),
    )
    checkpoint_selection_min_steps = max(
        1,
        int(round(float(checkpoint_selection.min_duration) / params.dt)),
    )
    checkpoint_selection_history: dict[str, list[float]] = {
        "time": [],
        "score": [],
        "rms_velocity_bias": [],
        "maximum_abs_velocity_bias": [],
        "maximum_abs_zero_velocity_drift": [],
        "negative_depinning_velocity": [],
        "positive_depinning_velocity": [],
        "depinning_velocity": [],
        "stall_fraction": [],
        "minimum_pva_strength": [],
        "minimum_bump_contrast": [],
        "fully_defined": [],
        "acceptance_passed": [],
        "improved": [],
        "checks_without_improvement": [],
        "consecutive_successful_checks": [],
        "best_score": [],
        "best_time": [],
        "best_acceptance_passed": [],
    }
    best_checkpoint_state: VafidisToyState | None = None
    best_checkpoint_score = float("inf")
    best_checkpoint_time = float("nan")
    best_checkpoint_acceptance_passed = False
    checks_without_behavioral_improvement = 0
    consecutive_behavioral_successes = 0

    def append_weight_snapshot() -> None:
        effective_hd, effective_hr = effective_hd_distal_weight_matrices(
            w_hd_to_hd=state.w_hd_to_hd,
            w_hr_to_hd=state.w_hr_to_hd,
            normalization=params.hd_distal_normalization,
        )
        weight_history["time"].append(float(state.time))
        weight_history["w_hd_to_hd"].append(state.w_hd_to_hd.copy())
        weight_history["w_hr_to_hd"].append(state.w_hr_to_hd.copy())
        weight_history["weight_norm_hd_to_hd"].append(l2_norm(state.w_hd_to_hd))
        weight_history["weight_norm_hr_to_hd"].append(l2_norm(state.w_hr_to_hd))
        weight_history["effective_weight_norm_hd_to_hd"].append(l2_norm(effective_hd))
        weight_history["effective_weight_norm_hr_to_hd"].append(l2_norm(effective_hr))

    if recovery_checkpoint_callback is not None:
        recovery_checkpoint_callback(state)

    total_steps = train_steps
    block_e_hd = np.empty((plasticity_block_steps, params.n_theta), dtype=float)
    block_p_hd = np.empty((plasticity_block_steps, params.n_theta), dtype=float)
    block_p_hr = np.empty((plasticity_block_steps, params.n_hr), dtype=float)
    block_sample_count = 0
    progress_iterator = trange(total_steps, disable=not config.simulation.progress, desc="training")
    for step_index in progress_iterator:
        completed_steps = step_index + 1
        angular_velocity = angular_velocity_step(params.dt)
        current_visual_amplitude = (
            visual_amplitude_step.step(params.dt)
            if visual_amplitude_step is not None
            else None
        )
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=angular_velocity,
            visual_teacher=True,
            training=(
                config.simulation.plasticity_enabled
                and not use_block_multirate
            ),
            visual_noise=step_visual_noise(
                process=visual_noise_process,
                dt=params.dt,
                visual_teacher=True,
            ),
            visual_amplitude=current_visual_amplitude,
            validate_state=not use_block_multirate,
            copy_frozen_plasticity_state=not use_block_multirate,
        )
        if use_block_multirate:
            block_e_hd[block_sample_count] = state.e_hd
            block_p_hd[block_sample_count] = state.p_hd
            block_p_hr[block_sample_count] = state.p_hr
            block_sample_count += 1
            checkpoint_boundary = bool(
                (
                    weight_snapshot_interval_steps is not None
                    and completed_steps % weight_snapshot_interval_steps == 0
                )
                or (
                    recovery_checkpoint_callback is not None
                    and completed_steps % recovery_checkpoint_interval_steps == 0
                )
                or (
                    checkpoint_selection_enabled
                    and completed_steps >= checkpoint_selection_min_steps
                    and completed_steps % checkpoint_selection_check_steps == 0
                )
                or (
                    early_stopping_enabled
                    and completed_steps % early_check_steps == 0
                )
            )
            flush_plasticity_block = bool(
                block_sample_count == plasticity_block_steps
                or completed_steps == total_steps
                or checkpoint_boundary
            )
            if flush_plasticity_block:
                (
                    state.w_hd_to_hd,
                    state.w_hr_to_hd,
                    state.delta_w_hd_to_hd,
                    state.delta_w_hr_to_hd,
                ) = update_predictive_local_weights_block(
                    w_hd_to_hd=state.w_hd_to_hd,
                    w_hr_to_hd=state.w_hr_to_hd,
                    delta_w_hd_to_hd=state.delta_w_hd_to_hd,
                    delta_w_hr_to_hd=state.delta_w_hr_to_hd,
                    e_hd_history=block_e_hd[:block_sample_count],
                    p_hd_history=block_p_hd[:block_sample_count],
                    p_hr_history=block_p_hr[:block_sample_count],
                    dt=params.dt,
                    tau_delta=params.tau_delta,
                    eta_hd_to_hd=params.eta_hd_to_hd,
                    eta_hr_to_hd=params.eta_hr_to_hd,
                    w_hd_to_hd_min=params.w_hd_to_hd_min,
                    w_hd_to_hd_max=params.w_hd_to_hd_max,
                    w_hr_to_hd_min=params.w_hr_to_hd_min,
                    w_hr_to_hd_max=params.w_hr_to_hd_max,
                    hd_to_hd_symmetry_mode=params.hd_to_hd_symmetry_mode,
                    hd_to_hd_balance_mode=params.hd_to_hd_balance_mode,
                    hr_to_hd_balance_mode=params.hr_to_hd_balance_mode,
                    zero_hd_to_hd_diagonal=params.zero_hd_to_hd_diagonal,
                )
                validate_vafidis_toy_state(state, params)
                block_sample_count = 0
        interval_error_sum += float(np.sqrt(np.mean(np.square(state.e_hd))))
        interval_error_count += 1
        if learning_error_recorder is not None:
            learning_error_recorder.update(
                completed_step=step_index + 1,
                e_hd=state.e_hd,
            )
        recorded_state = step_index % save_interval_steps == 0 or step_index == total_steps - 1
        if recorded_state:
            record_state(history=history, state=state, include_activity=True)
        recorded_weight_snapshot = (
            weight_snapshot_interval_steps is not None
            and (
                (step_index + 1) % weight_snapshot_interval_steps == 0
                or step_index == total_steps - 1
            )
        )
        if recorded_weight_snapshot:
            append_weight_snapshot()

        recorded_recovery_checkpoint = bool(
            recovery_checkpoint_callback is not None
            and (
                completed_steps % recovery_checkpoint_interval_steps == 0
                or completed_steps == total_steps
            )
        )
        if recorded_recovery_checkpoint:
            recovery_checkpoint_callback(state)

        run_checkpoint_selection = (
            checkpoint_selection_enabled
            and completed_steps >= checkpoint_selection_min_steps
            and (
                completed_steps % checkpoint_selection_check_steps == 0
                or completed_steps == total_steps
            )
        )
        if run_checkpoint_selection:
            behavioral_metrics = evaluate_training_checkpoint_behavior(
                config=config,
                trained_state=state,
            )
            candidate_score = (
                float(behavioral_metrics["score"])
                if behavioral_metrics["fully_defined"]
                else float("inf")
            )
            acceptance_passed = bool(behavioral_metrics["acceptance_passed"])
            improved = bool(
                np.isfinite(candidate_score)
                and (
                    best_checkpoint_state is None
                    or (
                        acceptance_passed
                        and not best_checkpoint_acceptance_passed
                    )
                    or (
                        acceptance_passed == best_checkpoint_acceptance_passed
                        and candidate_score
                        < best_checkpoint_score
                        - float(checkpoint_selection.minimum_improvement)
                    )
                )
            )
            if improved:
                best_checkpoint_state = state.copy()
                best_checkpoint_score = candidate_score
                best_checkpoint_time = float(state.time)
                best_checkpoint_acceptance_passed = acceptance_passed
                checks_without_behavioral_improvement = 0
            else:
                checks_without_behavioral_improvement += 1
            consecutive_behavioral_successes = (
                consecutive_behavioral_successes + 1
                if acceptance_passed
                else 0
            )
            for metric_name in (
                "score",
                "rms_velocity_bias",
                "maximum_abs_velocity_bias",
                "maximum_abs_zero_velocity_drift",
                "negative_depinning_velocity",
                "positive_depinning_velocity",
                "depinning_velocity",
                "stall_fraction",
                "minimum_pva_strength",
                "minimum_bump_contrast",
                "fully_defined",
                "acceptance_passed",
            ):
                checkpoint_selection_history[metric_name].append(
                    float(behavioral_metrics.get(metric_name, float("nan")))
                )
            checkpoint_selection_history["time"].append(float(state.time))
            checkpoint_selection_history["improved"].append(float(improved))
            checkpoint_selection_history["checks_without_improvement"].append(
                float(checks_without_behavioral_improvement)
            )
            checkpoint_selection_history["consecutive_successful_checks"].append(
                float(consecutive_behavioral_successes)
            )
            checkpoint_selection_history["best_score"].append(
                float(best_checkpoint_score)
            )
            checkpoint_selection_history["best_time"].append(
                float(best_checkpoint_time)
            )
            checkpoint_selection_history["best_acceptance_passed"].append(
                float(best_checkpoint_acceptance_passed)
            )

            behavioral_success_stop = (
                consecutive_behavioral_successes
                >= int(checkpoint_selection.success_checks)
            )
            behavioral_patience_stop = (
                checks_without_behavioral_improvement
                >= int(checkpoint_selection.patience_checks)
            )
            if (
                completed_steps < total_steps
                and (behavioral_success_stop or behavioral_patience_stop)
            ):
                stopped_early = True
                stop_reason = (
                    "behavior_success"
                    if behavioral_success_stop
                    else "behavior_patience"
                )
                if not recorded_state:
                    record_state(history=history, state=state, include_activity=True)
                if not recorded_weight_snapshot:
                    append_weight_snapshot()
                if (
                    recovery_checkpoint_callback is not None
                    and not recorded_recovery_checkpoint
                ):
                    recovery_checkpoint_callback(state)
                break

        if early_stopping_enabled and completed_steps % early_check_steps == 0:
            interval_mean_error = interval_error_sum / max(interval_error_count, 1)
            interval_error_sum = 0.0
            interval_error_count = 0
            convergence_checkpoints.append(
                (
                    completed_steps,
                    state.w_hd_to_hd.copy(),
                    state.w_hr_to_hd.copy(),
                    float(interval_mean_error),
                )
            )
            reference_index: int | None = None
            target_reference_step = completed_steps - early_window_steps
            for checkpoint_index, checkpoint in enumerate(convergence_checkpoints):
                if checkpoint[0] <= target_reference_step:
                    reference_index = checkpoint_index
                else:
                    break
            relative_hd_change = float("nan")
            relative_hr_change = float("nan")
            relative_error_change = float("nan")
            if reference_index is not None:
                reference = convergence_checkpoints[reference_index]
                relative_hd_change = _relative_array_change(
                    state.w_hd_to_hd,
                    reference[1],
                )
                relative_hr_change = _relative_array_change(
                    state.w_hr_to_hd,
                    reference[2],
                )
                error_denominator = max(
                    abs(float(interval_mean_error)),
                    abs(float(reference[3])),
                    1e-12,
                )
                relative_error_change = float(
                    abs(float(interval_mean_error) - float(reference[3]))
                    / error_denominator
                )
                if reference_index > 0:
                    del convergence_checkpoints[:reference_index]

            convergence_criteria_met = (
                completed_steps >= early_min_steps
                and reference_index is not None
                and relative_hd_change
                <= float(early_stopping.relative_weight_change_tolerance)
                and relative_hr_change
                <= float(early_stopping.relative_weight_change_tolerance)
                and relative_error_change
                <= float(early_stopping.relative_error_change_tolerance)
            )
            consecutive_converged_checks = (
                consecutive_converged_checks + 1
                if convergence_criteria_met
                else 0
            )
            convergence_history["time"].append(float(state.time))
            convergence_history["relative_change_w_hd_to_hd"].append(
                relative_hd_change
            )
            convergence_history["relative_change_w_hr_to_hd"].append(
                relative_hr_change
            )
            convergence_history["relative_change_rms_e_hd"].append(
                relative_error_change
            )
            convergence_history["interval_mean_rms_e_hd"].append(
                float(interval_mean_error)
            )
            convergence_history["consecutive_converged_checks"].append(
                float(consecutive_converged_checks)
            )
            if (
                consecutive_converged_checks >= int(early_stopping.patience_checks)
                and completed_steps < total_steps
            ):
                stopped_early = True
                stop_reason = "convergence"
                if not recorded_state:
                    record_state(history=history, state=state, include_activity=True)
                if not recorded_weight_snapshot:
                    append_weight_snapshot()
                if (
                    recovery_checkpoint_callback is not None
                    and not recorded_recovery_checkpoint
                ):
                    recovery_checkpoint_callback(state)
                break

    executed_duration = float(state.time)
    restored_best_checkpoint = bool(
        checkpoint_selection_enabled
        and bool(checkpoint_selection.restore_best)
        and best_checkpoint_state is not None
    )
    if restored_best_checkpoint:
        state = best_checkpoint_state
    training_history_arrays = _history_to_arrays(history)
    training_history_arrays["training_requested_duration"] = np.asarray(
        float(config.simulation.train_duration)
    )
    training_history_arrays["training_actual_duration"] = np.asarray(
        executed_duration
    )
    training_history_arrays["training_integration_method"] = np.asarray(
        params.training_integration_method
    )
    training_history_arrays["plasticity_update_interval_duration"] = np.asarray(
        plasticity_block_steps * params.dt
    )
    training_history_arrays["plasticity_update_interval_steps"] = np.asarray(
        plasticity_block_steps
    )
    training_history_arrays["training_early_stopped"] = np.asarray(float(stopped_early))
    training_history_arrays["training_stop_reason"] = np.asarray(stop_reason)
    training_history_arrays["training_checkpoint_selection_enabled"] = np.asarray(
        float(checkpoint_selection_enabled)
    )
    training_history_arrays["training_checkpoint_selection_restored_best"] = (
        np.asarray(float(restored_best_checkpoint))
    )
    training_history_arrays["training_selected_checkpoint_time"] = np.asarray(
        float(best_checkpoint_time)
        if restored_best_checkpoint
        else executed_duration
    )
    training_history_arrays["training_selected_checkpoint_score"] = np.asarray(
        float(best_checkpoint_score)
        if best_checkpoint_state is not None
        else float("nan")
    )
    training_history_arrays["training_behavioral_acceptance_passed"] = np.asarray(
        float(best_checkpoint_acceptance_passed)
    )
    training_history_arrays["training_checkpoint_selection_was_fallback"] = (
        np.asarray(
            float(
                checkpoint_selection_enabled
                and not best_checkpoint_acceptance_passed
            )
        )
    )
    if learning_error_recorder is not None:
        training_history_arrays.update(learning_error_recorder.to_history())
    for convergence_name, convergence_values in convergence_history.items():
        training_history_arrays[f"early_stopping_{convergence_name}"] = np.asarray(
            convergence_values,
            dtype=float,
        )
    for selection_name, selection_values in checkpoint_selection_history.items():
        training_history_arrays[f"checkpoint_selection_{selection_name}"] = (
            np.asarray(selection_values, dtype=float)
        )
    return state, training_history_arrays, _history_to_arrays(weight_history)


def initialize_protocol_state(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
) -> VafidisToyState:
    """Backward-compatible alias for the shared frozen protocol initializer."""

    return initialize_frozen_protocol_state(
        config=config,
        trained_state=trained_state,
        theta_true=theta_true,
    )


VISUAL_CUE_PHASE_ID = 0
DARKNESS_PHASE_ID = 1
VISUAL_RECUE_PHASE_ID = 2
HD_SATURATION_RATE_THRESHOLD = 0.99
HD_NEAR_PEAK_RELATIVE_TOLERANCE = 5e-3


def get_pi_cue_duration(config: ExperimentConfig) -> float:
    """Return the visual cue duration used for PI tests.

    The short ``cue_duration`` is useful for bump maintenance probes.  Vafidis
    Figure 2A / Appendix 1 examples instead show a visual-dark-visual PI trial
    with a longer initial visual segment, so PI tests can override it without
    changing the bump-maintenance protocol.
    """
    if config.simulation.pi_cue_duration is None:
        return float(config.simulation.cue_duration)
    return float(config.simulation.pi_cue_duration)


def run_visual_dark_visual_protocol(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
    darkness_duration: float,
    angular_velocity_step: Callable[[float], float],
    cue_duration: float | None = None,
    recue_duration: float | None = None,
    synaptic_input_noise_std: float = 0.0,
    synaptic_input_noise_seed: int | None = None,
    progress: Any | None = None,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    params = VafidisToyParams.from_config(config)
    state = initialize_protocol_state(
        config=config,
        trained_state=trained_state,
        theta_true=theta_true,
    )
    history = _new_history(include_activity=True)
    visual_rng = make_rng(config.simulation.seed + config.visual.noise_seed_offset + 100_000)
    visual_noise_process = build_visual_noise_process(
        config=config,
        params=params,
        rng=visual_rng,
        enabled=config.visual.apply_noise_during_visual_test,
    )
    synaptic_input_noise_std = float(synaptic_input_noise_std)
    if synaptic_input_noise_std < 0.0:
        raise ValueError("synaptic_input_noise_std must be non-negative")
    synaptic_noise_rng = (
        None
        if synaptic_input_noise_std == 0.0
        else make_rng(
            config.simulation.seed
            if synaptic_input_noise_seed is None
            else int(synaptic_input_noise_seed)
        )
    )
    record_state(
        history=history,
        state=state,
        include_activity=True,
        visual_teacher=True,
        phase_id=VISUAL_CUE_PHASE_ID,
    )
    phase_specs = [
        (
            cue_duration if cue_duration is not None else config.simulation.cue_duration,
            True,
            VISUAL_CUE_PHASE_ID,
        ),
        (darkness_duration, False, DARKNESS_PHASE_ID),
        (
            recue_duration if recue_duration is not None else config.simulation.recue_duration,
            True,
            VISUAL_RECUE_PHASE_ID,
        ),
    ]
    if progress is not None and progress_label:
        progress.set_postfix(job=progress_label, refresh=False)
    progress_update_interval = max(1, int(round(1.0 / params.dt)))
    pending_progress_steps = 0
    for phase_duration, visual_teacher, phase_id in phase_specs:
        phase_steps = int(round(max(phase_duration, 0.0) / params.dt))
        for step_index in range(phase_steps):
            angular_velocity = angular_velocity_step(params.dt)
            visual_noise = step_visual_noise(
                process=visual_noise_process,
                dt=params.dt,
                visual_teacher=visual_teacher,
            )
            if synaptic_noise_rng is None:
                i_hd_distal_noise = None
                i_hd_proximal_noise = None
                i_hr_noise = None
            else:
                i_hd_distal_noise = synaptic_noise_rng.normal(
                    0.0, synaptic_input_noise_std, size=params.n_theta
                )
                i_hd_proximal_noise = synaptic_noise_rng.normal(
                    0.0, synaptic_input_noise_std, size=params.n_theta
                )
                i_hr_noise = synaptic_noise_rng.normal(
                    0.0, synaptic_input_noise_std, size=params.n_hr
                )
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=angular_velocity,
                visual_teacher=visual_teacher,
                training=False,
                visual_noise=visual_noise,
                i_hd_distal_noise=i_hd_distal_noise,
                i_hd_proximal_noise=i_hd_proximal_noise,
                i_hr_noise=i_hr_noise,
            )
            record_state(
                history=history,
                state=state,
                include_activity=True,
                visual_teacher=visual_teacher,
                phase_id=phase_id,
            )
            if progress is not None:
                pending_progress_steps += 1
                if (
                    pending_progress_steps >= progress_update_interval
                    or step_index + 1 == phase_steps
                ):
                    progress.update(pending_progress_steps)
                    pending_progress_steps = 0
    if progress is not None and pending_progress_steps:
        progress.update(pending_progress_steps)
    return _history_to_arrays(history)


def run_constant_velocity_visual_dark_visual_protocol(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
    darkness_duration: float,
    angular_velocity: float,
    cue_duration: float | None = None,
    recue_duration: float | None = None,
    synaptic_input_noise_std: float = 0.0,
    synaptic_input_noise_seed: int | None = None,
    progress: Any | None = None,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    constant_angular_velocity = float(angular_velocity)

    def angular_velocity_step(_dt: float) -> float:
        return constant_angular_velocity

    return run_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=theta_true,
        darkness_duration=darkness_duration,
        angular_velocity_step=angular_velocity_step,
        cue_duration=cue_duration,
        recue_duration=recue_duration,
        synaptic_input_noise_std=synaptic_input_noise_std,
        synaptic_input_noise_seed=synaptic_input_noise_seed,
        progress=progress,
        progress_label=progress_label,
    )


def run_ou_visual_dark_visual_protocol(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
    darkness_duration: float,
    cue_duration: float | None = None,
    recue_duration: float | None = None,
    protocol_seed: int | None = None,
    progress: Any | None = None,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    protocol_rng = make_rng(
        config.simulation.seed + 10_000
        if protocol_seed is None
        else int(protocol_seed)
    )
    # ``tests.ou_test_clip`` (if set) overrides the training ``velocity.clip``
    # for frozen-weight OU tests only, matching the original paper's
    # ``bound_vel=True, v_max=500`` PI-error protocol without changing the
    # training OU distribution.
    test_clip = (
        config.velocity.clip
        if config.tests.ou_test_clip is None
        else float(config.tests.ou_test_clip)
    )
    ou_process = OUAngularVelocity(
        mean=config.velocity.mean,
        std=config.velocity.std,
        tau=config.velocity.tau,
        clip=test_clip,
        rng=protocol_rng,
    )
    return run_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=theta_true,
        darkness_duration=darkness_duration,
        angular_velocity_step=ou_process.step,
        cue_duration=cue_duration,
        recue_duration=recue_duration,
        progress=progress,
        progress_label=progress_label,
    )


def phase_mask(history: dict[str, np.ndarray], phase_id: int) -> np.ndarray:
    phase_id_history = history.get("phase_id", np.empty(0))
    if phase_id_history.size == 0:
        return np.zeros_like(history.get("time", np.empty(0)), dtype=bool)
    return np.asarray(phase_id_history, dtype=float) == float(phase_id)


def nanmean_or_nan(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def count_saturated_hd_bins(
    *,
    theta_hd_pref: np.ndarray,
    r_hd: np.ndarray,
    threshold: float = HD_SATURATION_RATE_THRESHOLD,
) -> int:
    """Count paired-HD angular bins whose collapsed rate is near saturation."""
    _unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    return int(np.count_nonzero(collapsed_rates >= threshold))


def summarize_hd_saturation(
    *,
    history: dict[str, np.ndarray],
    theta_hd_pref: np.ndarray,
    mask: np.ndarray,
    metric_prefix: str,
    threshold: float = HD_SATURATION_RATE_THRESHOLD,
) -> dict[str, float]:
    if "r_hd" not in history or not np.any(mask):
        return {
            f"{metric_prefix}_mean_saturated_hd_bins": float("nan"),
            f"{metric_prefix}_max_saturated_hd_bins": float("nan"),
            f"{metric_prefix}_final_saturated_hd_bins": float("nan"),
        }
    saturated_bin_counts = np.asarray(
        [
            count_saturated_hd_bins(
                theta_hd_pref=theta_hd_pref,
                r_hd=r_hd,
                threshold=threshold,
            )
            for r_hd in history["r_hd"][mask]
        ],
        dtype=float,
    )
    return {
        f"{metric_prefix}_mean_saturated_hd_bins": float(np.mean(saturated_bin_counts)),
        f"{metric_prefix}_max_saturated_hd_bins": float(np.max(saturated_bin_counts)),
        f"{metric_prefix}_final_saturated_hd_bins": float(saturated_bin_counts[-1]),
    }


def count_near_peak_hd_bins(
    *,
    theta_hd_pref: np.ndarray,
    r_hd: np.ndarray,
    relative_tolerance: float = HD_NEAR_PEAK_RELATIVE_TOLERANCE,
) -> int:
    """Count angular bins that belong to the same near-saturated peak top."""
    _unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    if collapsed_rates.size == 0:
        return 0
    max_rate = float(np.max(collapsed_rates))
    min_rate = float(np.min(collapsed_rates))
    tolerance = max(1e-9, relative_tolerance * max(max_rate - min_rate, abs(max_rate), 1.0))
    return int(np.count_nonzero(collapsed_rates >= max_rate - tolerance))


def count_local_hd_peaks(
    *,
    theta_hd_pref: np.ndarray,
    r_hd: np.ndarray,
    relative_height_threshold: float = 0.25,
) -> int:
    """Count circular local maxima above a fraction of the activity range."""
    _unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    collapsed_rates = np.asarray(collapsed_rates, dtype=float)
    if collapsed_rates.size < 3 or not np.all(np.isfinite(collapsed_rates)):
        return 0
    minimum = float(np.min(collapsed_rates))
    maximum = float(np.max(collapsed_rates))
    threshold = minimum + float(relative_height_threshold) * (maximum - minimum)
    is_peak = (
        (collapsed_rates > np.roll(collapsed_rates, 1))
        & (collapsed_rates >= np.roll(collapsed_rates, -1))
        & (collapsed_rates >= threshold)
    )
    return int(np.count_nonzero(is_peak))


def summarize_hd_local_peaks(
    *,
    history: dict[str, np.ndarray],
    theta_hd_pref: np.ndarray,
    mask: np.ndarray,
    metric_prefix: str,
) -> dict[str, float]:
    if "r_hd" not in history or not np.any(mask):
        return {
            f"{metric_prefix}_mean_local_peak_count_25pct": float("nan"),
            f"{metric_prefix}_max_local_peak_count_25pct": float("nan"),
            f"{metric_prefix}_final_local_peak_count_25pct": float("nan"),
        }
    counts = np.asarray(
        [
            count_local_hd_peaks(theta_hd_pref=theta_hd_pref, r_hd=r_hd)
            for r_hd in history["r_hd"][mask]
        ],
        dtype=float,
    )
    return {
        f"{metric_prefix}_mean_local_peak_count_25pct": float(np.mean(counts)),
        f"{metric_prefix}_max_local_peak_count_25pct": float(np.max(counts)),
        f"{metric_prefix}_final_local_peak_count_25pct": float(counts[-1]),
    }


def summarize_collapsed_peak_top(
    *,
    theta_hd_pref: np.ndarray,
    r_hd: np.ndarray,
    metric_prefix: str,
    relative_tolerance: float = HD_NEAR_PEAK_RELATIVE_TOLERANCE,
) -> dict[str, float]:
    """Summarize the top of a paired-HD collapsed activity bump."""
    unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    finite_mask = np.isfinite(unique_theta) & np.isfinite(collapsed_rates)
    unique_theta = unique_theta[finite_mask]
    collapsed_rates = collapsed_rates[finite_mask]
    if collapsed_rates.size == 0:
        return {
            f"{metric_prefix}_top1_angle": float("nan"),
            f"{metric_prefix}_top1_angle_deg": float("nan"),
            f"{metric_prefix}_top1_rate": float("nan"),
            f"{metric_prefix}_top2_angle": float("nan"),
            f"{metric_prefix}_top2_angle_deg": float("nan"),
            f"{metric_prefix}_top2_rate": float("nan"),
            f"{metric_prefix}_top1_minus_top2_rate": float("nan"),
            f"{metric_prefix}_near_peak_hd_bins": float("nan"),
            f"{metric_prefix}_near_peak_span_deg": float("nan"),
        }
    sorted_peak_indices = np.argsort(collapsed_rates)[::-1]
    top1_index = int(sorted_peak_indices[0])
    top2_index = int(sorted_peak_indices[1]) if sorted_peak_indices.size > 1 else top1_index
    max_rate = float(collapsed_rates[top1_index])
    min_rate = float(np.min(collapsed_rates))
    tolerance = max(1e-9, relative_tolerance * max(max_rate - min_rate, abs(max_rate), 1.0))
    near_peak_bins = int(np.count_nonzero(collapsed_rates >= max_rate - tolerance))
    theta_step = (
        float(np.median(np.diff(unique_theta)))
        if unique_theta.size > 1
        else float("nan")
    )
    near_peak_span = (
        float(np.rad2deg(theta_step * max(near_peak_bins - 1, 0)))
        if np.isfinite(theta_step)
        else float("nan")
    )
    top1_angle = float(unique_theta[top1_index])
    top2_angle = float(unique_theta[top2_index])
    return {
        f"{metric_prefix}_top1_angle": top1_angle,
        f"{metric_prefix}_top1_angle_deg": float(np.rad2deg(top1_angle)),
        f"{metric_prefix}_top1_rate": max_rate,
        f"{metric_prefix}_top2_angle": top2_angle,
        f"{metric_prefix}_top2_angle_deg": float(np.rad2deg(top2_angle)),
        f"{metric_prefix}_top2_rate": float(collapsed_rates[top2_index]),
        f"{metric_prefix}_top1_minus_top2_rate": float(
            max_rate - float(collapsed_rates[top2_index])
        ),
        f"{metric_prefix}_near_peak_hd_bins": float(near_peak_bins),
        f"{metric_prefix}_near_peak_span_deg": near_peak_span,
    }


def estimate_peak_transition_time_after_dark_onset(
    *,
    time: np.ndarray,
    theta_peak: np.ndarray,
    dark_mask: np.ndarray,
    theta_reference: float,
    theta_hd_pref: np.ndarray,
) -> float:
    """Return first dark time where peak decode leaves its cue basin."""
    if not np.any(dark_mask):
        return float("nan")
    dark_indices = np.flatnonzero(dark_mask)
    unique_theta = np.unique(np.asarray(theta_hd_pref, dtype=float))
    if unique_theta.size < 2:
        threshold = 1e-9
    else:
        threshold = 0.5 * float(np.median(np.diff(unique_theta)))
    dark_time = np.asarray(time[dark_indices], dtype=float)
    dark_peak = np.asarray(theta_peak[dark_indices], dtype=float)
    finite_mask = np.isfinite(dark_time) & np.isfinite(dark_peak)
    if np.count_nonzero(finite_mask) == 0:
        return float("nan")
    dark_time = dark_time[finite_mask]
    dark_peak = dark_peak[finite_mask]
    dark_onset_time = float(dark_time[0])
    left_basin_mask = np.abs(circular_difference(dark_peak, theta_reference)) > threshold
    if not np.any(left_basin_mask):
        return float("nan")
    return float(dark_time[np.flatnonzero(left_basin_mask)[0]] - dark_onset_time)


def summarize_hd_near_peak(
    *,
    history: dict[str, np.ndarray],
    theta_hd_pref: np.ndarray,
    mask: np.ndarray,
    metric_prefix: str,
    relative_tolerance: float = HD_NEAR_PEAK_RELATIVE_TOLERANCE,
) -> dict[str, float]:
    if "r_hd" not in history or not np.any(mask):
        return {
            f"{metric_prefix}_mean_near_peak_hd_bins": float("nan"),
            f"{metric_prefix}_max_near_peak_hd_bins": float("nan"),
            f"{metric_prefix}_final_near_peak_hd_bins": float("nan"),
        }
    near_peak_counts = np.asarray(
        [
            count_near_peak_hd_bins(
                theta_hd_pref=theta_hd_pref,
                r_hd=r_hd,
                relative_tolerance=relative_tolerance,
            )
            for r_hd in history["r_hd"][mask]
        ],
        dtype=float,
    )
    return {
        f"{metric_prefix}_mean_near_peak_hd_bins": float(np.mean(near_peak_counts)),
        f"{metric_prefix}_max_near_peak_hd_bins": float(np.max(near_peak_counts)),
        f"{metric_prefix}_final_near_peak_hd_bins": float(near_peak_counts[-1]),
    }


def summarize_bump_release_peak_diagnostics(
    *,
    history: dict[str, np.ndarray],
    theta_hd_pref: np.ndarray,
) -> dict[str, float]:
    """Diagnose cue-off basin jumps in bump-maintenance peak decoding."""
    cue_mask = phase_mask(history, VISUAL_CUE_PHASE_ID)
    dark_mask = phase_mask(history, DARKNESS_PHASE_ID)
    if "r_hd" not in history or history["r_hd"].size == 0 or not np.any(dark_mask):
        empty_summary = {
            "bump_peak_transition_time_after_dark_onset": float("nan"),
            "bump_peak_cue_to_final_shift": float("nan"),
            "bump_peak_cue_to_final_shift_deg": float("nan"),
        }
        empty_activity = np.empty(0, dtype=float)
        empty_theta = np.empty(0, dtype=float)
        for empty_prefix in ["bump_cue_final", "bump_dark_initial", "bump_dark_final"]:
            empty_summary.update(
                summarize_collapsed_peak_top(
                    theta_hd_pref=empty_theta,
                    r_hd=empty_activity,
                    metric_prefix=empty_prefix,
                )
            )
            empty_summary[f"{empty_prefix}_local_peak_count_25pct"] = float("nan")
        return empty_summary

    dark_indices = np.flatnonzero(dark_mask)
    cue_indices = np.flatnonzero(cue_mask)
    cue_final_index = int(cue_indices[-1]) if cue_indices.size else int(dark_indices[0])
    dark_initial_index = int(dark_indices[0])
    dark_final_index = int(dark_indices[-1])
    cue_peak = float(history["theta_hd_decoded_peak"][cue_final_index])
    dark_final_peak = float(history["theta_hd_decoded_peak"][dark_final_index])
    peak_shift = float(circular_difference(dark_final_peak, cue_peak))
    return {
        "bump_peak_transition_time_after_dark_onset": (
            estimate_peak_transition_time_after_dark_onset(
                time=history["time"],
                theta_peak=history["theta_hd_decoded_peak"],
                dark_mask=dark_mask,
                theta_reference=cue_peak,
                theta_hd_pref=theta_hd_pref,
            )
        ),
        "bump_peak_cue_to_final_shift": peak_shift,
        "bump_peak_cue_to_final_shift_deg": float(np.rad2deg(peak_shift)),
        **summarize_collapsed_peak_top(
            theta_hd_pref=theta_hd_pref,
            r_hd=history["r_hd"][cue_final_index],
            metric_prefix="bump_cue_final",
        ),
        **summarize_collapsed_peak_top(
            theta_hd_pref=theta_hd_pref,
            r_hd=history["r_hd"][dark_initial_index],
            metric_prefix="bump_dark_initial",
        ),
        **summarize_collapsed_peak_top(
            theta_hd_pref=theta_hd_pref,
            r_hd=history["r_hd"][dark_final_index],
            metric_prefix="bump_dark_final",
        ),
        "bump_cue_final_local_peak_count_25pct": float(
            count_local_hd_peaks(
                theta_hd_pref=theta_hd_pref,
                r_hd=history["r_hd"][cue_final_index],
            )
        ),
        "bump_dark_initial_local_peak_count_25pct": float(
            count_local_hd_peaks(
                theta_hd_pref=theta_hd_pref,
                r_hd=history["r_hd"][dark_initial_index],
            )
        ),
        "bump_dark_final_local_peak_count_25pct": float(
            count_local_hd_peaks(
                theta_hd_pref=theta_hd_pref,
                r_hd=history["r_hd"][dark_final_index],
            )
        ),
    }


def run_bump_maintenance_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    bump_history = run_constant_velocity_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.bump_test_duration,
        angular_velocity=0.0,
        cue_duration=config.simulation.cue_duration,
        recue_duration=0.0,
    )
    cue_mask = phase_mask(bump_history, DARKNESS_PHASE_ID)
    theta_dark = bump_history["theta_hd_decoded"][cue_mask]
    theta_peak_dark = bump_history["theta_hd_decoded_peak"][cue_mask]
    drift = final_abs_circular_error(
        theta_dark,
        theta_reference=config.simulation.theta0,
    )
    intrinsic_drift_velocity = estimate_decoded_velocity(
        time=bump_history["time"][cue_mask] - bump_history["time"][cue_mask][0],
        theta_decoded=theta_dark,
        start_fraction=0.25,
    )
    peak_drift = final_abs_circular_error(
        theta_peak_dark,
        theta_reference=config.simulation.theta0,
    )
    peak_intrinsic_drift_velocity = estimate_decoded_velocity(
        time=bump_history["time"][cue_mask] - bump_history["time"][cue_mask][0],
        theta_decoded=theta_peak_dark,
        start_fraction=0.25,
    )
    bump_release_shift = float(circular_difference(theta_dark[-1], theta_dark[0]))
    bump_peak_release_shift = float(circular_difference(theta_peak_dark[-1], theta_peak_dark[0]))
    bump_release_displacement = np.abs(circular_difference(theta_dark, theta_dark[0]))
    bump_peak_release_displacement = np.abs(circular_difference(theta_peak_dark, theta_peak_dark[0]))
    bump_release_displacement = bump_release_displacement[np.isfinite(bump_release_displacement)]
    bump_peak_release_displacement = bump_peak_release_displacement[
        np.isfinite(bump_peak_release_displacement)
    ]
    bump_max_abs_release_displacement = (
        float(np.max(bump_release_displacement))
        if bump_release_displacement.size
        else float("nan")
    )
    bump_peak_max_abs_release_displacement = (
        float(np.max(bump_peak_release_displacement))
        if bump_peak_release_displacement.size
        else float("nan")
    )
    return bump_history, {
        "bump_final_abs_drift": drift,
        "bump_intrinsic_drift_velocity": intrinsic_drift_velocity,
        "bump_abs_intrinsic_drift_velocity": float(abs(intrinsic_drift_velocity)),
        "bump_intrinsic_drift_velocity_deg_s": float(np.rad2deg(intrinsic_drift_velocity)),
        "bump_abs_intrinsic_drift_velocity_deg_s": float(abs(np.rad2deg(intrinsic_drift_velocity))),
        "bump_final_abs_peak_drift": peak_drift,
        "bump_release_shift": bump_release_shift,
        "bump_abs_release_shift": float(abs(bump_release_shift)),
        "bump_release_shift_deg": float(np.rad2deg(bump_release_shift)),
        "bump_abs_release_shift_deg": float(abs(np.rad2deg(bump_release_shift))),
        "bump_peak_release_shift": bump_peak_release_shift,
        "bump_peak_abs_release_shift": float(abs(bump_peak_release_shift)),
        "bump_peak_release_shift_deg": float(np.rad2deg(bump_peak_release_shift)),
        "bump_peak_abs_release_shift_deg": float(abs(np.rad2deg(bump_peak_release_shift))),
        "bump_max_abs_release_displacement": bump_max_abs_release_displacement,
        "bump_max_abs_release_displacement_deg": float(
            np.rad2deg(bump_max_abs_release_displacement)
        ),
        "bump_peak_max_abs_release_displacement": bump_peak_max_abs_release_displacement,
        "bump_peak_max_abs_release_displacement_deg": float(
            np.rad2deg(bump_peak_max_abs_release_displacement)
        ),
        "bump_effective_diffusion_coefficient": estimate_effective_diffusion_coefficient(
            time=bump_history["time"][cue_mask],
            theta_decoded=theta_dark,
            theta_reference=config.simulation.theta0,
            start_fraction=0.0,
        ),
        "bump_peak_effective_diffusion_coefficient": estimate_effective_diffusion_coefficient(
            time=bump_history["time"][cue_mask],
            theta_decoded=theta_peak_dark,
            theta_reference=config.simulation.theta0,
            start_fraction=0.0,
        ),
        "bump_peak_intrinsic_drift_velocity": peak_intrinsic_drift_velocity,
        "bump_peak_intrinsic_drift_velocity_deg_s": float(np.rad2deg(peak_intrinsic_drift_velocity)),
        "bump_final_pva_strength": float(bump_history["pva_strength_hd"][cue_mask][-1]),
        "bump_final_contrast": float(bump_history["bump_contrast_hd"][cue_mask][-1]),
        **summarize_hd_saturation(
            history=bump_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="bump",
        ),
        **summarize_hd_near_peak(
            history=bump_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="bump",
        ),
        **summarize_hd_local_peaks(
            history=bump_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="bump",
        ),
        **summarize_bump_release_peak_diagnostics(
            history=bump_history,
            theta_hd_pref=trained_state.theta_hd_pref,
        ),
    }


def _summarize_bump_attractor_decoder(
    *,
    decoder_name: str,
    theta_decoded: np.ndarray,
    theta_initial: np.ndarray,
    time: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Summarize flatness and drift for one circular heading decoder."""
    theta_decoded = np.asarray(theta_decoded, dtype=float)
    theta_initial = np.asarray(theta_initial, dtype=float)
    time = np.asarray(time, dtype=float)
    if (
        theta_decoded.ndim != 2
        or theta_initial.ndim != 1
        or time.ndim != 1
        or theta_decoded.shape != (theta_initial.size, time.size)
    ):
        raise ValueError("decoded attractor trajectories must have shape (start, time)")
    theta_unwrapped = np.vstack(
        [unwrap_heading_trace(theta_trace) for theta_trace in theta_decoded]
    )
    angular_displacement = theta_unwrapped - theta_unwrapped[:, :1]
    final_displacement = angular_displacement[:, -1]
    max_abs_displacement = np.full(theta_initial.size, np.nan, dtype=float)
    for trial_index, displacement_trace in enumerate(angular_displacement):
        finite_displacement = displacement_trace[np.isfinite(displacement_trace)]
        if finite_displacement.size:
            max_abs_displacement[trial_index] = float(
                np.max(np.abs(finite_displacement))
            )
    initial_alignment_error = circular_difference(theta_decoded[:, 0], theta_initial)
    trajectory_defined_mask = np.all(np.isfinite(theta_decoded), axis=1)
    fit_mask = time >= 0.25 * float(time[-1])
    drift_velocity = np.full(theta_initial.size, np.nan, dtype=float)
    for trial_index, displacement_trace in enumerate(angular_displacement):
        if not trajectory_defined_mask[trial_index]:
            continue
        trial_fit_mask = fit_mask & np.isfinite(displacement_trace)
        if np.count_nonzero(trial_fit_mask) >= 2:
            drift_velocity[trial_index], _intercept = linear_fit_slope_intercept(
                time[trial_fit_mask],
                displacement_trace[trial_fit_mask],
            )

    final_abs_displacement = np.abs(final_displacement)
    finite_max_abs_displacement = max_abs_displacement[
        trajectory_defined_mask & np.isfinite(max_abs_displacement)
    ]
    finite_final_abs_displacement = final_abs_displacement[
        np.isfinite(final_abs_displacement)
    ]
    finite_initial_alignment_error = initial_alignment_error[
        np.isfinite(initial_alignment_error)
    ]
    finite_angular_displacement = angular_displacement[trajectory_defined_mask]
    finite_angular_displacement = finite_angular_displacement[
        np.isfinite(finite_angular_displacement)
    ]
    finite_abs_drift_velocity_deg = np.abs(
        np.rad2deg(drift_velocity[np.isfinite(drift_velocity)])
    )

    def finite_statistic(values: np.ndarray, statistic: str) -> float:
        if values.size == 0:
            return float("nan")
        if statistic == "median":
            return float(np.median(values))
        if statistic == "p95":
            return float(np.percentile(values, 95.0))
        if statistic == "max":
            return float(np.max(values))
        if statistic == "rms":
            return float(np.sqrt(np.mean(np.square(values))))
        raise ValueError(f"unknown finite statistic: {statistic}")

    if finite_max_abs_displacement.size:
        eligible_max_abs_displacement = np.where(
            trajectory_defined_mask,
            max_abs_displacement,
            np.nan,
        )
        worst_trial_index = int(np.nanargmax(eligible_max_abs_displacement))
        worst_initial_angle_deg = float(np.rad2deg(theta_initial[worst_trial_index]))
    else:
        worst_initial_angle_deg = float("nan")
    metric_prefix = f"bump_attractor_{decoder_name}"
    history = {
        f"{decoder_name}_angular_displacement": angular_displacement,
        f"final_theta_{decoder_name}": np.asarray(
            wrap_angle(theta_decoded[:, -1]),
            dtype=float,
        ),
        f"{decoder_name}_final_angular_displacement": final_displacement,
        f"{decoder_name}_max_abs_angular_displacement": max_abs_displacement,
        f"{decoder_name}_drift_velocity": drift_velocity,
    }
    metrics = {
        f"{metric_prefix}_defined_sample_fraction": float(
            np.mean(np.isfinite(theta_decoded))
        ),
        f"{metric_prefix}_fully_defined_trajectory_fraction": float(
            np.mean(trajectory_defined_mask)
        ),
        f"{metric_prefix}_defined_final_fraction": float(
            np.mean(np.isfinite(theta_decoded[:, -1]))
        ),
        f"{metric_prefix}_initial_alignment_rms_deg": float(
            np.rad2deg(finite_statistic(finite_initial_alignment_error, "rms"))
        ),
        f"{metric_prefix}_trajectory_flatness_rmse_deg": float(
            np.rad2deg(finite_statistic(finite_angular_displacement, "rms"))
        ),
        f"{metric_prefix}_final_displacement_rms_deg": float(
            np.rad2deg(finite_statistic(finite_final_abs_displacement, "rms"))
        ),
        f"{metric_prefix}_final_abs_displacement_median_deg": float(
            np.rad2deg(finite_statistic(finite_final_abs_displacement, "median"))
        ),
        f"{metric_prefix}_final_abs_displacement_p95_deg": float(
            np.rad2deg(finite_statistic(finite_final_abs_displacement, "p95"))
        ),
        f"{metric_prefix}_final_abs_displacement_max_deg": float(
            np.rad2deg(finite_statistic(finite_final_abs_displacement, "max"))
        ),
        f"{metric_prefix}_max_abs_displacement_median_deg": float(
            np.rad2deg(finite_statistic(finite_max_abs_displacement, "median"))
        ),
        f"{metric_prefix}_max_abs_displacement_p95_deg": float(
            np.rad2deg(finite_statistic(finite_max_abs_displacement, "p95"))
        ),
        f"{metric_prefix}_max_abs_displacement_max_deg": float(
            np.rad2deg(finite_statistic(finite_max_abs_displacement, "max"))
        ),
        f"{metric_prefix}_stable_within_5deg_fraction": float(
            np.mean(
                trajectory_defined_mask
                & np.isfinite(max_abs_displacement)
                & (max_abs_displacement <= np.deg2rad(5.0))
            )
        ),
        f"{metric_prefix}_stable_within_10deg_fraction": float(
            np.mean(
                trajectory_defined_mask
                & np.isfinite(max_abs_displacement)
                & (max_abs_displacement <= np.deg2rad(10.0))
            )
        ),
        f"{metric_prefix}_abs_drift_velocity_median_deg_s": float(
            finite_statistic(finite_abs_drift_velocity_deg, "median")
        ),
        f"{metric_prefix}_abs_drift_velocity_p95_deg_s": float(
            finite_statistic(finite_abs_drift_velocity_deg, "p95")
        ),
        f"{metric_prefix}_abs_drift_velocity_max_deg_s": float(
            finite_statistic(finite_abs_drift_velocity_deg, "max")
        ),
        f"{metric_prefix}_worst_initial_angle_deg": worst_initial_angle_deg,
    }
    return history, metrics


def _summarize_cue_release_map(
    *,
    decoder_name: str,
    theta_initial: np.ndarray,
    theta_release: np.ndarray,
) -> dict[str, float]:
    """Quantify how faithfully a stationary cue sets the decoded phase."""
    theta_initial = np.asarray(theta_initial, dtype=float)
    theta_release = np.asarray(theta_release, dtype=float)
    if theta_initial.ndim != 1 or theta_release.shape != theta_initial.shape:
        raise ValueError("cue-release map arrays must have matching 1D shapes")
    finite = np.isfinite(theta_initial) & np.isfinite(theta_release)
    metric_prefix = f"bump_attractor_{decoder_name}_cue_release"
    if np.count_nonzero(finite) < 2:
        return {
            f"{metric_prefix}_alignment_rms_deg": float("nan"),
            f"{metric_prefix}_linearity_slope": float("nan"),
            f"{metric_prefix}_linearity_r_squared": float("nan"),
            f"{metric_prefix}_local_gain_median": float("nan"),
            f"{metric_prefix}_local_gain_rmse_from_one": float("nan"),
            f"{metric_prefix}_local_gain_p05": float("nan"),
            f"{metric_prefix}_local_gain_p95": float("nan"),
            f"{metric_prefix}_near_zero_local_gain_fraction": float("nan"),
        }

    cue = theta_initial[finite]
    release = cue + circular_difference(theta_release[finite], cue)
    alignment_error = circular_difference(theta_release[finite], cue)
    slope, intercept = linear_fit_slope_intercept(cue, release)
    fitted_release = slope * cue + intercept
    residual_sum_squares = float(np.sum(np.square(release - fitted_release)))
    centered_sum_squares = float(np.sum(np.square(release - np.mean(release))))
    r_squared = (
        1.0 - residual_sum_squares / centered_sum_squares
        if centered_sum_squares > 1e-12
        else float("nan")
    )
    cue_spacing = np.diff(cue)
    local_gain = np.divide(
        np.diff(release),
        cue_spacing,
        out=np.full(cue_spacing.shape, np.nan, dtype=float),
        where=np.abs(cue_spacing) > 1e-12,
    )
    finite_local_gain = local_gain[np.isfinite(local_gain)]
    return {
        f"{metric_prefix}_alignment_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean(np.square(alignment_error))))
        ),
        f"{metric_prefix}_linearity_slope": float(slope),
        f"{metric_prefix}_linearity_r_squared": float(r_squared),
        f"{metric_prefix}_local_gain_median": (
            float(np.median(finite_local_gain))
            if finite_local_gain.size
            else float("nan")
        ),
        f"{metric_prefix}_local_gain_rmse_from_one": (
            float(np.sqrt(np.mean(np.square(finite_local_gain - 1.0))))
            if finite_local_gain.size
            else float("nan")
        ),
        f"{metric_prefix}_local_gain_p05": (
            float(np.percentile(finite_local_gain, 5.0))
            if finite_local_gain.size
            else float("nan")
        ),
        f"{metric_prefix}_local_gain_p95": (
            float(np.percentile(finite_local_gain, 95.0))
            if finite_local_gain.size
            else float("nan")
        ),
        f"{metric_prefix}_near_zero_local_gain_fraction": (
            float(np.mean(np.abs(finite_local_gain) < 0.1))
            if finite_local_gain.size
            else float("nan")
        ),
    }


def _summarize_autonomous_endpoint_landscape(
    *,
    decoder_name: str,
    theta_probe: np.ndarray,
    theta_release: np.ndarray,
    theta_final: np.ndarray,
    n_theta: int | None = None,
) -> dict[str, float]:
    """Count endpoint attractors and strictly bracketed basin boundaries.

    When ``n_theta`` is provided (the paired HD population size, so the
    number of unique headings is ``n_theta // 2``), the endpoint fixed points
    are additionally aligned to the geometric heading/midpoint grid.  A
    ``heading`` is classified as stable when a detected attractor lies within
    a quarter-spacing tolerance, and a ``midpoint`` is unstable when a
    detected repeller lies there.

    The heading/midpoint grid assumes the paired-HD geometry, where every two
    cells share one preferred direction, so the heading count is
    ``n_theta // 2`` (30 for the N=60 baseline, 60 for the N=120
    finite-size control).  It is only a meaningful success criterion when the
    network has pinned into a discrete, ``n_heading``-fold symmetric ring of
    attractors: for a genuinely continuous ring there are no discrete fixed
    points, so the aligned counts are expected to be low and the raw
    ``attractor_count`` (bounded by the probe count) is the right readout
    instead.  This alignment is therefore a diagnostic of the discrete-pinned
    regime, not a universal fixed-point test.
    """
    landscape = classify_endpoint_map_fixed_points(
        theta_initial=theta_probe,
        theta_release=theta_release,
        theta_final=theta_final,
    )
    metric_prefix = f"bump_attractor_{decoder_name}_endpoint"
    fixed_point_stability = np.asarray(
        landscape["fixed_point_stability"],
        dtype=np.int8,
    )
    resolution_limited = np.asarray(
        landscape["fixed_point_release_resolution_limited"],
        dtype=bool,
    )
    fixed_point_theta = np.asarray(
        landscape["fixed_point_theta"],
        dtype=float,
    )
    heading_grid: dict[str, float] = {}
    if n_theta is not None and n_theta > 0 and n_theta % 2 == 0:
        n_heading = n_theta // 2
        heading = np.linspace(-np.pi, np.pi, n_heading, endpoint=False)
        midpoint = np.asarray(wrap_angle(heading + np.pi / n_heading), dtype=float)
        tolerance = 0.25 * (2.0 * np.pi / n_heading)
        stable_theta = fixed_point_theta[fixed_point_stability == -1]
        unstable_theta = fixed_point_theta[fixed_point_stability == 1]
        heading_stable_count = 0
        for anchor in heading:
            if stable_theta.size and np.any(
                np.abs(circular_difference(stable_theta, anchor)) < tolerance
            ):
                heading_stable_count += 1
        midpoint_unstable_count = 0
        for anchor in midpoint:
            if unstable_theta.size and np.any(
                np.abs(circular_difference(unstable_theta, anchor)) < tolerance
            ):
                midpoint_unstable_count += 1
        heading_grid = {
            f"{metric_prefix}_heading_aligned_stable_count": float(
                heading_stable_count
            ),
            f"{metric_prefix}_heading_aligned_unstable_count": float(
                midpoint_unstable_count
            ),
            f"{metric_prefix}_expected_heading_count": float(n_heading),
        }
    return {
        f"{metric_prefix}_attractor_count": float(
            np.count_nonzero(fixed_point_stability == -1)
        ),
        f"{metric_prefix}_trajectory_inferred_unstable_count": float(
            np.count_nonzero(fixed_point_stability == 1)
        ),
        f"{metric_prefix}_release_resolution_limited_fixed_point_count": float(
            np.count_nonzero(resolution_limited)
        ),
        f"{metric_prefix}_subbin_boundary_count": float(
            np.asarray(landscape["subbin_boundary_theta"]).size
        ),
        f"{metric_prefix}_basin_boundary_bracket_count": float(
            np.asarray(landscape["basin_boundary_theta"]).size
        ),
        f"{metric_prefix}_unresolved_cluster_transition_count": float(
            np.asarray(landscape["unresolved_boundary_theta"]).size
        ),
        f"{metric_prefix}_nonmonotonic_crossing_count": float(
            np.asarray(landscape["nonmonotonic_transition_theta"]).size
        ),
        f"{metric_prefix}_missing_boundary_interval_count": float(
            np.asarray(landscape["missing_boundary_interval_theta"]).size
        ),
        f"{metric_prefix}_repeated_support_fraction": float(
            landscape["repeated_endpoint_support_fraction"]
        ),
        f"{metric_prefix}_cue_transfer_orientation_preserving_fraction": float(
            landscape["cue_transfer_orientation_preserving_fraction"]
        ),
        f"{metric_prefix}_cue_transfer_plateau_fraction": float(
            landscape["cue_transfer_plateau_fraction"]
        ),
        f"{metric_prefix}_cue_transfer_coverage_fraction": float(
            landscape["cue_transfer_coverage_fraction"]
        ),
        f"{metric_prefix}_cue_transfer_valid": float(
            bool(np.asarray(landscape["cue_transfer_valid"]))
        ),
        f"{metric_prefix}_alternation_mismatch_count": float(
            landscape["alternation_mismatch_count"]
        ),
        **heading_grid,
    }


def run_bump_attractor_trajectory_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    hd_tuning_history: dict[str, np.ndarray] | None = None,
    as_dependency: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Map deterministic zero-input bump trajectories from uniform headings.

    Each trial first settles under a stationary visual cue and then evolves
    with both visual and velocity inputs set to zero.  Only population-vector
    PVA, peak-neuron, and Clark-overlap summaries are retained. When the
    Ságodi analysis is enabled, one trajectory at a time is buffered in the
    canonical frozen Markov state and only low-speed candidate points are
    retained. The overlap template is the frozen network's post-training
    visual tuning manifold measured by :func:`run_hd_tuning_curve_test`.
    Full trajectories are retained only for the uniform coarse grid. Endpoint
    transitions are optionally refined by cue-angle bisection and saved as a
    separate endpoint-probe table for higher-resolution basin boundaries.
    """
    enabled = as_dependency or diagnostic_is_enabled(
        config,
        "bump_attractor_trajectories",
    )
    if not enabled:
        return {
            "time": np.empty(0, dtype=float),
            "cue_time": np.empty(0, dtype=float),
            "theta_initial": np.empty(0, dtype=float),
            "cue_theta_pva": np.empty((0, 0), dtype=float),
            "cue_theta_peak": np.empty((0, 0), dtype=float),
            "cue_theta_overlap": np.empty((0, 0), dtype=float),
            "cue_release_pva_strength": np.empty(0, dtype=float),
            "cue_release_max_rate_fraction": np.empty(0, dtype=float),
            "cue_release_saturated_bin_count": np.empty(0, dtype=float),
            "cue_release_visual_to_distal_modulation_ratio": np.empty(
                0,
                dtype=float,
            ),
            "theta_pva": np.empty((0, 0), dtype=float),
            "theta_peak": np.empty((0, 0), dtype=float),
            "theta_overlap": np.empty((0, 0), dtype=float),
            "theta_release_pva": np.empty(0, dtype=float),
            "theta_release_peak": np.empty(0, dtype=float),
            "theta_release_overlap": np.empty(0, dtype=float),
            "endpoint_probe_theta_initial": np.empty(0, dtype=float),
            "endpoint_probe_refinement_level": np.empty(0, dtype=int),
            "endpoint_probe_theta_release_pva": np.empty(0, dtype=float),
            "endpoint_probe_theta_release_peak": np.empty(0, dtype=float),
            "endpoint_probe_theta_release_overlap": np.empty(0, dtype=float),
            "endpoint_probe_theta_final_pva": np.empty(0, dtype=float),
            "endpoint_probe_theta_final_peak": np.empty(0, dtype=float),
            "endpoint_probe_theta_final_overlap": np.empty(0, dtype=float),
            "endpoint_probe_theta_pva_trajectory": np.empty((0, 0), dtype=float),
            "pva_angular_displacement": np.empty((0, 0), dtype=float),
            "peak_angular_displacement": np.empty((0, 0), dtype=float),
            "overlap_angular_displacement": np.empty((0, 0), dtype=float),
            "pva_strength": np.empty((0, 0), dtype=float),
            "overlap_max": np.empty((0, 0), dtype=float),
            "bump_contrast": np.empty((0, 0), dtype=float),
            "slow_candidate_theta": np.empty(0, dtype=float),
            "slow_candidate_state": np.empty((0, 0), dtype=float),
            "slow_candidate_speed": np.empty(0, dtype=float),
            "autonomous_probe_phase": np.empty(0, dtype=float),
            "autonomous_probe_decoded_theta": np.empty(0, dtype=float),
            "autonomous_probe_state": np.empty((0, 0), dtype=float),
            "slow_candidate_trajectory_index": np.empty(0, dtype=int),
            "slow_candidate_time": np.empty(0, dtype=float),
            "ramesan_trajectory_theta": np.empty(0, dtype=float),
            "ramesan_trajectory_state": np.empty((0, 0), dtype=float),
            "ramesan_trajectory_speed": np.empty(0, dtype=float),
            "ramesan_trajectory_index": np.empty(0, dtype=int),
            "ramesan_trajectory_time": np.empty(0, dtype=float),
            "slow_trajectory_max_speed": np.empty(0, dtype=float),
            "slow_trajectory_speed_threshold": np.empty(0, dtype=float),
            "slow_trajectory_candidate_count": np.empty(0, dtype=int),
        }, {"bump_attractor_trajectory_enabled": 0.0}

    cue_amplitude = (
        float(config.visual.amplitude)
        if config.tests.bump_attractor_cue_amplitude is None
        else float(config.tests.bump_attractor_cue_amplitude)
    )
    if cue_amplitude <= 0.0:
        raise ValueError("tests.bump_attractor_cue_amplitude must be positive")
    cue_config = deepcopy(config)
    cue_config.visual.amplitude = cue_amplitude
    params = VafidisToyParams.from_config(cue_config)
    n_initial_conditions = int(config.tests.bump_attractor_initial_conditions)
    boundary_bisection_depth = int(
        config.tests.bump_attractor_boundary_bisection_depth
    )
    duration = float(config.tests.bump_attractor_duration)
    cue_duration = float(config.tests.bump_attractor_cue_duration)
    sample_interval = float(config.tests.bump_attractor_sample_interval)
    if n_initial_conditions <= 0:
        raise ValueError("tests.bump_attractor_initial_conditions must be positive")
    if boundary_bisection_depth < 0 or boundary_bisection_depth > 12:
        raise ValueError(
            "tests.bump_attractor_boundary_bisection_depth must lie in [0, 12]"
        )
    if duration <= 0.0:
        raise ValueError("tests.bump_attractor_duration must be positive")
    if cue_duration < 0.0:
        raise ValueError("tests.bump_attractor_cue_duration must be non-negative")
    if sample_interval <= 0.0:
        raise ValueError("tests.bump_attractor_sample_interval must be positive")
    if hd_tuning_history is None:
        raise ValueError("enabled bump attractor trajectories require hd_tuning_history")
    if "theta_true" not in hd_tuning_history or "r_hd" not in hd_tuning_history:
        raise ValueError("hd_tuning_history must contain theta_true and r_hd")
    overlap_theta_template = np.asarray(hd_tuning_history["theta_true"], dtype=float)
    overlap_target_rate = np.asarray(hd_tuning_history["r_hd"], dtype=float).T
    if (
        overlap_target_rate.ndim != 2
        or overlap_target_rate.shape
        != (trained_state.theta_hd_pref.size, overlap_theta_template.size)
    ):
        raise ValueError("HD tuning template must have shape (heading, HD neuron)")

    cue_steps = int(round(cue_duration / params.dt))
    darkness_steps = max(1, int(round(duration / params.dt)))
    sample_interval_steps = max(1, int(round(sample_interval / params.dt)))
    sample_step_indices = np.arange(
        0,
        darkness_steps + 1,
        sample_interval_steps,
        dtype=int,
    )
    if sample_step_indices[-1] != darkness_steps:
        sample_step_indices = np.append(sample_step_indices, darkness_steps)
    time = sample_step_indices.astype(float) * params.dt
    cue_sample_step_indices = np.arange(
        0,
        cue_steps + 1,
        sample_interval_steps,
        dtype=int,
    )
    if cue_sample_step_indices[-1] != cue_steps:
        cue_sample_step_indices = np.append(cue_sample_step_indices, cue_steps)
    cue_time = (
        cue_sample_step_indices.astype(float) - float(cue_steps)
    ) * params.dt
    theta_initial = np.linspace(
        -np.pi,
        np.pi,
        n_initial_conditions,
        endpoint=False,
        dtype=float,
    )
    theta_pva = np.empty((n_initial_conditions, time.size), dtype=float)
    theta_peak = np.empty_like(theta_pva)
    theta_overlap = np.empty_like(theta_pva)
    cue_theta_pva = np.empty(
        (n_initial_conditions, cue_time.size),
        dtype=float,
    )
    cue_theta_peak = np.empty_like(cue_theta_pva)
    cue_theta_overlap = np.empty_like(cue_theta_pva)
    cue_release_pva_strength = np.empty(n_initial_conditions, dtype=float)
    cue_release_max_rate_fraction = np.empty(n_initial_conditions, dtype=float)
    cue_release_saturated_bin_count = np.empty(n_initial_conditions, dtype=float)
    cue_release_visual_to_distal_modulation_ratio = np.empty(
        n_initial_conditions,
        dtype=float,
    )
    pva_strength = np.empty_like(theta_pva)
    overlap_max = np.empty_like(theta_pva)
    bump_contrast = np.empty_like(theta_pva)
    slow_manifold_enabled = diagnostic_is_enabled(config, "slow_manifold")
    autonomous_dynamics = (
        FrozenAutonomousDynamics.from_state(params=params, state=trained_state)
        if slow_manifold_enabled
        else None
    )
    autonomous_probe_state = (
        np.empty(
            (n_initial_conditions, autonomous_dynamics.state_dimension),
            dtype=float,
        )
        if autonomous_dynamics is not None
        else np.empty((n_initial_conditions, 0), dtype=float)
    )
    autonomous_probe_decoded_theta = np.full(n_initial_conditions, np.nan)
    slow_candidate_target = int(config.tests.slow_manifold_candidate_count)
    slow_speed_fraction = float(config.tests.slow_manifold_speed_fraction)
    slow_speed_floor = (
        float(config.tests.slow_manifold_speed_floor)
        if config.tests.slow_manifold_speed_floor is not None
        else None
    )
    ramesan_trajectory_target = int(config.tests.ramesan_trajectory_sample_count)
    if slow_manifold_enabled:
        if slow_candidate_target < 4:
            raise ValueError("tests.slow_manifold_candidate_count must be at least four")
        if not 0.0 < slow_speed_fraction < 1.0:
            raise ValueError(
                "tests.slow_manifold_speed_fraction must lie between zero and one"
            )
        if slow_speed_floor is not None and (
            not np.isfinite(slow_speed_floor) or slow_speed_floor <= 0.0
        ):
            raise ValueError(
                "tests.slow_manifold_speed_floor must be finite and positive "
                "when provided"
            )
        if ramesan_trajectory_target < 0:
            raise ValueError(
                "tests.ramesan_trajectory_sample_count must be non-negative"
            )
    candidate_points_per_trajectory = max(
        1,
        int(np.ceil(slow_candidate_target / n_initial_conditions)),
    )
    slow_candidate_theta_parts: list[np.ndarray] = []
    slow_candidate_state_parts: list[np.ndarray] = []
    slow_candidate_speed_parts: list[np.ndarray] = []
    slow_candidate_trajectory_parts: list[np.ndarray] = []
    slow_candidate_time_parts: list[np.ndarray] = []
    ramesan_trajectory_theta_parts: list[np.ndarray] = []
    ramesan_trajectory_state_parts: list[np.ndarray] = []
    ramesan_trajectory_speed_parts: list[np.ndarray] = []
    ramesan_trajectory_index_parts: list[np.ndarray] = []
    ramesan_trajectory_time_parts: list[np.ndarray] = []
    slow_trajectory_max_speed = np.full(n_initial_conditions, np.nan)
    slow_trajectory_speed_threshold = np.full(n_initial_conditions, np.nan)
    slow_trajectory_candidate_count = np.zeros(n_initial_conditions, dtype=int)

    total_dynamics_steps = n_initial_conditions * (cue_steps + darkness_steps)
    progress_update_interval_steps = max(1, int(round(1.0 / params.dt)))
    dynamics_progress = tqdm(
        total=total_dynamics_steps,
        disable=not config.simulation.progress,
        desc="bump attractor trajectories",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    with dynamics_progress:
        for trial_index in range(n_initial_conditions):
            state = initialize_protocol_state(
                config=cue_config,
                trained_state=trained_state,
                theta_true=float(theta_initial[trial_index]),
            )

            def decode_current_state() -> tuple[float, float, float, float]:
                decoded_overlap, overlap_peak = decode_heading_by_clark_overlap(
                    theta_template=overlap_theta_template,
                    target_rate=overlap_target_rate,
                    population_activity=state.r_hd,
                )
                return (
                    float(state.theta_hd_decoded),
                    float(state.theta_hd_decoded_peak),
                    float(decoded_overlap[0]),
                    float(overlap_peak[0]),
                )

            def record_cue_sample(sample_index: int) -> None:
                (
                    cue_theta_pva[trial_index, sample_index],
                    cue_theta_peak[trial_index, sample_index],
                    cue_theta_overlap[trial_index, sample_index],
                    _overlap_peak,
                ) = decode_current_state()

            record_cue_sample(0)
            next_cue_sample_index = 1
            dynamics_progress.set_postfix(
                start=f"{trial_index + 1}/{n_initial_conditions}",
                angle=f"{np.rad2deg(theta_initial[trial_index]):.1f} deg",
                phase="cue",
                refresh=False,
            )
            for cue_step in range(1, cue_steps + 1):
                state = step_vafidis_toy(
                    state=state,
                    params=params,
                    angular_velocity=0.0,
                    visual_teacher=True,
                    training=False,
                    visual_noise=None,
                )
                if cue_step % progress_update_interval_steps == 0:
                    dynamics_progress.update(progress_update_interval_steps)
                if (
                    next_cue_sample_index < cue_sample_step_indices.size
                    and cue_step
                    == cue_sample_step_indices[next_cue_sample_index]
                ):
                    record_cue_sample(next_cue_sample_index)
                    next_cue_sample_index += 1
            cue_remainder_steps = (
                cue_steps % progress_update_interval_steps
            )
            if cue_remainder_steps:
                dynamics_progress.update(cue_remainder_steps)

            cue_release_pva_strength[trial_index] = pva_vector_strength(
                state.theta_hd_pref,
                state.r_hd,
            )
            cue_release_max_rate_fraction[trial_index] = float(
                np.max(state.r_hd) / params.activation_max_rate
            )
            cue_release_saturated_bin_count[trial_index] = float(
                count_saturated_hd_bins(
                    theta_hd_pref=state.theta_hd_pref,
                    r_hd=state.r_hd,
                    threshold=(
                        HD_SATURATION_RATE_THRESHOLD
                        * params.activation_max_rate
                    ),
                )
            )
            visual_modulation = state.i_vis_to_hd - np.mean(state.i_vis_to_hd)
            distal_voltage_drive = (
                params.g_d_hd_to_proximal * state.v_hd_distal
            )
            distal_modulation = distal_voltage_drive - np.mean(
                distal_voltage_drive
            )
            distal_modulation_rms = float(
                np.sqrt(np.mean(np.square(distal_modulation)))
            )
            cue_release_visual_to_distal_modulation_ratio[trial_index] = (
                float(np.sqrt(np.mean(np.square(visual_modulation))))
                / distal_modulation_rms
                if distal_modulation_rms > 1e-12
                else float("inf")
            )

            if autonomous_dynamics is not None:
                release_state_vector = autonomous_dynamics.pack_state(state)
                autonomous_probe_state[trial_index] = release_state_vector
                autonomous_probe_decoded_theta[trial_index] = (
                    autonomous_dynamics.decoded_heading(release_state_vector)
                )

            trajectory_autonomous_state = (
                np.empty((time.size, autonomous_dynamics.state_dimension), dtype=float)
                if autonomous_dynamics is not None
                else None
            )
            trajectory_autonomous_theta = (
                np.empty(time.size, dtype=float)
                if autonomous_dynamics is not None
                else None
            )
            trajectory_autonomous_speed = (
                np.empty(time.size, dtype=float)
                if autonomous_dynamics is not None
                else None
            )

            def record_sample(sample_index: int) -> None:
                (
                    theta_pva[trial_index, sample_index],
                    theta_peak[trial_index, sample_index],
                    theta_overlap[trial_index, sample_index],
                    overlap_max[trial_index, sample_index],
                ) = decode_current_state()
                pva_strength[trial_index, sample_index] = pva_vector_strength(
                    state.theta_hd_pref,
                    state.r_hd,
                )
                bump_contrast[trial_index, sample_index] = float(
                    np.max(state.r_hd) - np.min(state.r_hd)
                )
                if autonomous_dynamics is not None:
                    autonomous_state = autonomous_dynamics.pack_state(state)
                    trajectory_autonomous_state[sample_index] = autonomous_state
                    trajectory_autonomous_theta[sample_index] = (
                        autonomous_dynamics.decoded_heading(autonomous_state)
                    )
                    trajectory_autonomous_speed[sample_index] = np.linalg.norm(
                        autonomous_dynamics.flow(autonomous_state)
                    )

            record_sample(0)
            next_sample_index = 1
            dynamics_progress.set_postfix(
                start=f"{trial_index + 1}/{n_initial_conditions}",
                angle=f"{np.rad2deg(theta_initial[trial_index]):.1f} deg",
                phase="darkness",
                refresh=False,
            )
            for darkness_step in range(1, darkness_steps + 1):
                state = step_vafidis_toy(
                    state=state,
                    params=params,
                    angular_velocity=0.0,
                    visual_teacher=False,
                    training=False,
                    visual_noise=None,
                )
                if darkness_step % progress_update_interval_steps == 0:
                    dynamics_progress.update(progress_update_interval_steps)
                if (
                    next_sample_index < sample_step_indices.size
                    and darkness_step == sample_step_indices[next_sample_index]
                ):
                    record_sample(next_sample_index)
                    next_sample_index += 1
            darkness_remainder_steps = darkness_steps % progress_update_interval_steps
            if darkness_remainder_steps:
                dynamics_progress.update(darkness_remainder_steps)
            if autonomous_dynamics is not None:
                if ramesan_trajectory_target > 0:
                    points_per_trajectory = max(
                        1,
                        int(
                            np.ceil(
                                ramesan_trajectory_target
                                / n_initial_conditions
                            )
                        ),
                    )
                    retained_count = min(points_per_trajectory, time.size)
                    retained_index = np.unique(
                        np.rint(
                            np.linspace(0, time.size - 1, retained_count)
                        ).astype(int)
                    )
                    ramesan_trajectory_theta_parts.append(
                        trajectory_autonomous_theta[retained_index]
                    )
                    ramesan_trajectory_state_parts.append(
                        trajectory_autonomous_state[retained_index]
                    )
                    ramesan_trajectory_speed_parts.append(
                        trajectory_autonomous_speed[retained_index]
                    )
                    ramesan_trajectory_index_parts.append(
                        np.full(retained_index.size, trial_index, dtype=int)
                    )
                    ramesan_trajectory_time_parts.append(time[retained_index])
                selected_index, speed_threshold = select_slow_candidate_indices(
                    speed=trajectory_autonomous_speed,
                    speed_fraction=slow_speed_fraction,
                    maximum_points=candidate_points_per_trajectory,
                    speed_floor=slow_speed_floor,
                    time=time,
                )
                slow_trajectory_max_speed[trial_index] = float(
                    np.max(trajectory_autonomous_speed)
                )
                slow_trajectory_speed_threshold[trial_index] = speed_threshold
                slow_trajectory_candidate_count[trial_index] = selected_index.size
                if selected_index.size:
                    slow_candidate_theta_parts.append(
                        trajectory_autonomous_theta[selected_index]
                    )
                    slow_candidate_state_parts.append(
                        trajectory_autonomous_state[selected_index]
                    )
                    slow_candidate_speed_parts.append(
                        trajectory_autonomous_speed[selected_index]
                    )
                    slow_candidate_trajectory_parts.append(
                        np.full(selected_index.size, trial_index, dtype=int)
                    )
                    slow_candidate_time_parts.append(time[selected_index])

    # Preserve the dense time series on the uniform coarse grid, but refine
    # endpoint-map transitions with inexpensive endpoint-only rollouts.  At
    # each level, every adjacent pair assigned to different PVA endpoint
    # clusters receives a cue-angle midpoint probe.  Recomputing adjacency
    # after each level naturally follows either half of a simple boundary and
    # retains both halves if a narrow/interleaved basin introduces two.
    endpoint_probe_theta_initial = theta_initial.copy()
    endpoint_probe_refinement_level = np.zeros(n_initial_conditions, dtype=int)
    endpoint_probe_theta_release_pva = theta_pva[:, 0].copy()
    endpoint_probe_theta_release_peak = theta_peak[:, 0].copy()
    endpoint_probe_theta_release_overlap = theta_overlap[:, 0].copy()
    endpoint_probe_theta_final_pva = theta_pva[:, -1].copy()
    endpoint_probe_theta_final_peak = theta_peak[:, -1].copy()
    endpoint_probe_theta_final_overlap = theta_overlap[:, -1].copy()
    # Full darkness trajectories for every bisection probe (refined basin
    # boundaries).  These share the coarse-grid ``time`` axis, so they can be
    # overlaid on the trajectory panel alongside the 1-degree coarse grid.
    endpoint_probe_theta_pva_trajectory = np.empty(
        (0, time.size),
        dtype=float,
    )

    coarse_fixed_points = classify_endpoint_map_fixed_points(
        theta_initial=theta_initial,
        theta_release=theta_pva[:, 0],
        theta_final=theta_pva[:, -1],
    )
    coarse_fixed_theta = np.asarray(
        coarse_fixed_points["fixed_point_theta"], dtype=float
    )
    coarse_fixed_stability = np.asarray(
        coarse_fixed_points["fixed_point_stability"], dtype=np.int8
    )
    coarse_attractor_theta = coarse_fixed_theta[coarse_fixed_stability == -1]

    def endpoint_attractor_label(theta_final_value: np.ndarray) -> np.ndarray:
        theta_final_value = np.asarray(theta_final_value, dtype=float)
        if coarse_attractor_theta.size == 0:
            return np.full(theta_final_value.shape, -1, dtype=int)
        distance = np.abs(
            circular_difference(
                theta_final_value[:, None],
                coarse_attractor_theta[None, :],
            )
        )
        return np.argmin(distance, axis=1).astype(int)

    def simulate_endpoint_probe(
        theta_cue: float,
    ) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
        """Run one cue -> darkness bisection probe.

        In addition to the scalar release/final headings used by the endpoint
        map, the full darkness PVA trajectory (sampled at ``sample_interval``)
        is returned so refined basin-boundary probes can be drawn alongside
        the coarse-grid trajectories.
        """
        state = initialize_protocol_state(
            config=cue_config,
            trained_state=trained_state,
            theta_true=float(theta_cue),
        )
        for _cue_step in range(cue_steps):
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=0.0,
                visual_teacher=True,
                training=False,
                visual_noise=None,
            )
        release_overlap, _release_overlap_max = decode_heading_by_clark_overlap(
            theta_template=overlap_theta_template,
            target_rate=overlap_target_rate,
            population_activity=state.r_hd,
        )
        release_values = (
            float(state.theta_hd_decoded),
            float(state.theta_hd_decoded_peak),
            float(release_overlap[0]),
        )
        darkness_trace = np.full(time.size, np.nan, dtype=float)
        darkness_trace[0] = float(release_values[0])
        # ``sample_step_indices`` are absolute step numbers from the coarse
        # grid.  Here the darkness loop counts from 1, so sample every
        # ``sample_interval_steps`` steps instead of matching absolute
        # indices.
        next_sample_index = 1
        for darkness_step in range(1, darkness_steps + 1):
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=0.0,
                visual_teacher=False,
                training=False,
                visual_noise=None,
            )
            if darkness_step % sample_interval_steps == 0:
                if next_sample_index < darkness_trace.size:
                    darkness_trace[next_sample_index] = float(
                        state.theta_hd_decoded
                    )
                    next_sample_index += 1
        # Ensure the final sample is captured even if the duration is not an
        # exact multiple of the interval.
        if darkness_trace[-1] != darkness_trace[-1] or np.isnan(darkness_trace[-1]):
            darkness_trace[-1] = float(state.theta_hd_decoded)
        final_overlap, _final_overlap_max = decode_heading_by_clark_overlap(
            theta_template=overlap_theta_template,
            target_rate=overlap_target_rate,
            population_activity=state.r_hd,
        )
        return (
            *release_values,
            float(state.theta_hd_decoded),
            float(state.theta_hd_decoded_peak),
            float(final_overlap[0]),
            darkness_trace,
            np.asarray(release_values[0], dtype=float),
        )

    completed_bisection_depth = 0
    bisection_level_progress: tqdm | None = None
    if boundary_bisection_depth > 0 and coarse_attractor_theta.size >= 2:
        bisection_level_progress = tqdm(
            total=boundary_bisection_depth,
            disable=not config.simulation.progress,
            desc="bisection levels",
            unit="level",
        )
        for refinement_level in range(1, boundary_bisection_depth + 1):
            probe_phase = np.mod(
                np.asarray(wrap_angle(endpoint_probe_theta_initial), dtype=float)
                + np.pi,
                2.0 * np.pi,
            )
            probe_order = np.argsort(probe_phase)
            ordered_phase = probe_phase[probe_order]
            ordered_label = endpoint_attractor_label(
                endpoint_probe_theta_final_pva[probe_order]
            )
            midpoint_theta: list[float] = []
            for left_order_index in range(probe_order.size):
                right_order_index = (left_order_index + 1) % probe_order.size
                if ordered_label[left_order_index] == ordered_label[right_order_index]:
                    continue
                left_phase = float(ordered_phase[left_order_index])
                right_phase = float(ordered_phase[right_order_index])
                if right_order_index == 0:
                    right_phase += 2.0 * np.pi
                midpoint_phase = np.mod(
                    left_phase + 0.5 * (right_phase - left_phase),
                    2.0 * np.pi,
                )
                candidate_theta = float(wrap_angle(midpoint_phase - np.pi))
                if np.min(
                    np.abs(
                        circular_difference(
                            endpoint_probe_theta_initial,
                            candidate_theta,
                        )
                    )
                ) > 1e-10:
                    midpoint_theta.append(candidate_theta)
            if not midpoint_theta:
                break

            midpoint_theta_array = np.asarray(midpoint_theta, dtype=float)
            probe_progress = tqdm(
                total=midpoint_theta_array.size,
                disable=not config.simulation.progress,
                desc=f"level {refinement_level} probes",
                unit="probe",
                leave=False,
            )
            midpoint_results: list[object] = []
            for theta_cue in midpoint_theta_array:
                midpoint_results.append(
                    simulate_endpoint_probe(float(theta_cue))
                )
                probe_progress.update(1)
            probe_progress.close()
            midpoint_result = np.asarray(midpoint_results, dtype=object)
            # The last two object entries are per-probe darkness traces.
            midpoint_trajectory = np.stack(
                [entry[6] for entry in midpoint_result],
                axis=0,
            )
            endpoint_probe_theta_pva_trajectory = np.concatenate(
                [
                    endpoint_probe_theta_pva_trajectory,
                    midpoint_trajectory,
                ],
                axis=0,
            )
            midpoint_result = np.stack(
                [np.asarray(entry[:6], dtype=float) for entry in midpoint_result],
                axis=0,
            )
            endpoint_probe_theta_initial = np.concatenate(
                [endpoint_probe_theta_initial, midpoint_theta_array]
            )
            endpoint_probe_refinement_level = np.concatenate(
                [
                    endpoint_probe_refinement_level,
                    np.full(
                        midpoint_theta_array.size,
                        refinement_level,
                        dtype=int,
                    ),
                ]
            )
            endpoint_probe_theta_release_pva = np.concatenate(
                [endpoint_probe_theta_release_pva, midpoint_result[:, 0]]
            )
            endpoint_probe_theta_release_peak = np.concatenate(
                [endpoint_probe_theta_release_peak, midpoint_result[:, 1]]
            )
            endpoint_probe_theta_release_overlap = np.concatenate(
                [endpoint_probe_theta_release_overlap, midpoint_result[:, 2]]
            )
            endpoint_probe_theta_final_pva = np.concatenate(
                [endpoint_probe_theta_final_pva, midpoint_result[:, 3]]
            )
            endpoint_probe_theta_final_peak = np.concatenate(
                [endpoint_probe_theta_final_peak, midpoint_result[:, 4]]
            )
            endpoint_probe_theta_final_overlap = np.concatenate(
                [endpoint_probe_theta_final_overlap, midpoint_result[:, 5]]
            )
            completed_bisection_depth = refinement_level
            bisection_level_progress.update(1)
    if bisection_level_progress is not None:
        bisection_level_progress.close()

    endpoint_probe_order = np.argsort(
        np.mod(
            np.asarray(wrap_angle(endpoint_probe_theta_initial), dtype=float)
            + np.pi,
            2.0 * np.pi,
        )
    )
    endpoint_probe_theta_initial = endpoint_probe_theta_initial[
        endpoint_probe_order
    ]
    endpoint_probe_refinement_level = endpoint_probe_refinement_level[
        endpoint_probe_order
    ]
    endpoint_probe_theta_release_pva = endpoint_probe_theta_release_pva[
        endpoint_probe_order
    ]
    endpoint_probe_theta_release_peak = endpoint_probe_theta_release_peak[
        endpoint_probe_order
    ]
    endpoint_probe_theta_release_overlap = endpoint_probe_theta_release_overlap[
        endpoint_probe_order
    ]
    endpoint_probe_theta_final_pva = endpoint_probe_theta_final_pva[
        endpoint_probe_order
    ]
    endpoint_probe_theta_final_peak = endpoint_probe_theta_final_peak[
        endpoint_probe_order
    ]
    endpoint_probe_theta_final_overlap = endpoint_probe_theta_final_overlap[
        endpoint_probe_order
    ]
    # ``endpoint_probe_theta_pva_trajectory`` only contains the darkness
    # trajectories of the bisection (midpoint) probes; the coarse-grid
    # trajectories live in ``theta_pva``.  Its rows correspond to the probes
    # appended after the first ``n_initial_conditions`` entries, so apply the
    # same circular phase ordering restricted to that suffix.
    bisection_theta = np.mod(
        np.asarray(wrap_angle(endpoint_probe_theta_initial), dtype=float)
        + np.pi,
        2.0 * np.pi,
    )[n_initial_conditions:]
    bisection_order = np.argsort(bisection_theta, kind="stable")
    endpoint_probe_theta_pva_trajectory = (
        endpoint_probe_theta_pva_trajectory[bisection_order]
        if endpoint_probe_theta_pva_trajectory.shape[0] == bisection_order.size
        else endpoint_probe_theta_pva_trajectory
    )

    pva_history, pva_metrics = _summarize_bump_attractor_decoder(
        decoder_name="pva",
        theta_decoded=theta_pva,
        theta_initial=theta_initial,
        time=time,
    )
    peak_history, peak_metrics = _summarize_bump_attractor_decoder(
        decoder_name="peak",
        theta_decoded=theta_peak,
        theta_initial=theta_initial,
        time=time,
    )
    overlap_history, overlap_metrics = _summarize_bump_attractor_decoder(
        decoder_name="overlap",
        theta_decoded=theta_overlap,
        theta_initial=theta_initial,
        time=time,
    )

    def decoder_disagreement_rms_deg(
        first_decoder: np.ndarray,
        second_decoder: np.ndarray,
    ) -> float:
        disagreement = circular_difference(first_decoder, second_decoder)
        finite_disagreement = disagreement[np.isfinite(disagreement)]
        if finite_disagreement.size == 0:
            return float("nan")
        return float(
            np.rad2deg(np.sqrt(np.mean(np.square(finite_disagreement))))
        )

    if slow_candidate_state_parts:
        slow_candidate_theta = np.concatenate(slow_candidate_theta_parts)
        slow_candidate_state = np.concatenate(slow_candidate_state_parts, axis=0)
        slow_candidate_speed = np.concatenate(slow_candidate_speed_parts)
        slow_candidate_trajectory_index = np.concatenate(
            slow_candidate_trajectory_parts
        )
        slow_candidate_time = np.concatenate(slow_candidate_time_parts)
        if slow_candidate_speed.size > slow_candidate_target:
            global_selection = np.rint(
                np.linspace(0, slow_candidate_speed.size - 1, slow_candidate_target)
            ).astype(int)
            slow_candidate_theta = slow_candidate_theta[global_selection]
            slow_candidate_state = slow_candidate_state[global_selection]
            slow_candidate_speed = slow_candidate_speed[global_selection]
            slow_candidate_trajectory_index = slow_candidate_trajectory_index[
                global_selection
            ]
            slow_candidate_time = slow_candidate_time[global_selection]
    else:
        state_dimension = (
            autonomous_dynamics.state_dimension
            if autonomous_dynamics is not None
            else 0
        )
        slow_candidate_theta = np.empty(0, dtype=float)
        slow_candidate_state = np.empty((0, state_dimension), dtype=float)
        slow_candidate_speed = np.empty(0, dtype=float)
        slow_candidate_trajectory_index = np.empty(0, dtype=int)
        slow_candidate_time = np.empty(0, dtype=float)

    if ramesan_trajectory_state_parts:
        ramesan_trajectory_theta = np.concatenate(ramesan_trajectory_theta_parts)
        ramesan_trajectory_state = np.concatenate(
            ramesan_trajectory_state_parts, axis=0
        )
        ramesan_trajectory_speed = np.concatenate(ramesan_trajectory_speed_parts)
        ramesan_trajectory_index = np.concatenate(ramesan_trajectory_index_parts)
        ramesan_trajectory_time = np.concatenate(ramesan_trajectory_time_parts)
        if ramesan_trajectory_speed.size > ramesan_trajectory_target:
            trajectory_selection = np.rint(
                np.linspace(
                    0,
                    ramesan_trajectory_speed.size - 1,
                    ramesan_trajectory_target,
                )
            ).astype(int)
            ramesan_trajectory_theta = ramesan_trajectory_theta[
                trajectory_selection
            ]
            ramesan_trajectory_state = ramesan_trajectory_state[
                trajectory_selection
            ]
            ramesan_trajectory_speed = ramesan_trajectory_speed[
                trajectory_selection
            ]
            ramesan_trajectory_index = ramesan_trajectory_index[
                trajectory_selection
            ]
            ramesan_trajectory_time = ramesan_trajectory_time[
                trajectory_selection
            ]
    else:
        state_dimension = (
            autonomous_dynamics.state_dimension
            if autonomous_dynamics is not None
            else 0
        )
        ramesan_trajectory_theta = np.empty(0, dtype=float)
        ramesan_trajectory_state = np.empty((0, state_dimension), dtype=float)
        ramesan_trajectory_speed = np.empty(0, dtype=float)
        ramesan_trajectory_index = np.empty(0, dtype=int)
        ramesan_trajectory_time = np.empty(0, dtype=float)

    history = {
        "time": time,
        "cue_time": cue_time,
        "theta_initial": theta_initial,
        "cue_theta_pva": np.asarray(wrap_angle(cue_theta_pva), dtype=float),
        "cue_theta_peak": np.asarray(wrap_angle(cue_theta_peak), dtype=float),
        "cue_theta_overlap": np.asarray(
            wrap_angle(cue_theta_overlap),
            dtype=float,
        ),
        "cue_release_pva_strength": cue_release_pva_strength,
        "cue_release_max_rate_fraction": cue_release_max_rate_fraction,
        "cue_release_saturated_bin_count": cue_release_saturated_bin_count,
        "cue_release_visual_to_distal_modulation_ratio": (
            cue_release_visual_to_distal_modulation_ratio
        ),
        "theta_pva": np.asarray(wrap_angle(theta_pva), dtype=float),
        "theta_peak": np.asarray(wrap_angle(theta_peak), dtype=float),
        "theta_overlap": np.asarray(wrap_angle(theta_overlap), dtype=float),
        "theta_release_pva": np.asarray(
            wrap_angle(theta_pva[:, 0]),
            dtype=float,
        ),
        "theta_release_peak": np.asarray(
            wrap_angle(theta_peak[:, 0]),
            dtype=float,
        ),
        "theta_release_overlap": np.asarray(
            wrap_angle(theta_overlap[:, 0]),
            dtype=float,
        ),
        "endpoint_probe_theta_initial": np.asarray(
            wrap_angle(endpoint_probe_theta_initial), dtype=float
        ),
        "endpoint_probe_refinement_level": endpoint_probe_refinement_level,
        "endpoint_probe_theta_release_pva": np.asarray(
            wrap_angle(endpoint_probe_theta_release_pva), dtype=float
        ),
        "endpoint_probe_theta_release_peak": np.asarray(
            wrap_angle(endpoint_probe_theta_release_peak), dtype=float
        ),
        "endpoint_probe_theta_release_overlap": np.asarray(
            wrap_angle(endpoint_probe_theta_release_overlap), dtype=float
        ),
        "endpoint_probe_theta_final_pva": np.asarray(
            wrap_angle(endpoint_probe_theta_final_pva), dtype=float
        ),
        "endpoint_probe_theta_final_peak": np.asarray(
            wrap_angle(endpoint_probe_theta_final_peak), dtype=float
        ),
        "endpoint_probe_theta_final_overlap": np.asarray(
            wrap_angle(endpoint_probe_theta_final_overlap), dtype=float
        ),
        "endpoint_probe_theta_pva_trajectory": np.asarray(
            wrap_angle(endpoint_probe_theta_pva_trajectory), dtype=float
        ),
        "overlap_theta_template": overlap_theta_template,
        "pva_strength": pva_strength,
        "overlap_max": overlap_max,
        "bump_contrast": bump_contrast,
        "slow_candidate_theta": np.asarray(
            wrap_angle(slow_candidate_theta), dtype=float
        ),
        "slow_candidate_state": slow_candidate_state,
        "slow_candidate_speed": slow_candidate_speed,
        "autonomous_probe_phase": (
            theta_initial.copy()
            if autonomous_dynamics is not None
            else np.empty(0, dtype=float)
        ),
        "autonomous_probe_decoded_theta": (
            np.asarray(wrap_angle(autonomous_probe_decoded_theta), dtype=float)
            if autonomous_dynamics is not None
            else np.empty(0, dtype=float)
        ),
        "autonomous_probe_state": autonomous_probe_state,
        "slow_candidate_trajectory_index": slow_candidate_trajectory_index,
        "slow_candidate_time": slow_candidate_time,
        "ramesan_trajectory_theta": np.asarray(
            wrap_angle(ramesan_trajectory_theta), dtype=float
        ),
        "ramesan_trajectory_state": ramesan_trajectory_state,
        "ramesan_trajectory_speed": ramesan_trajectory_speed,
        "ramesan_trajectory_index": ramesan_trajectory_index,
        "ramesan_trajectory_time": ramesan_trajectory_time,
        "slow_trajectory_max_speed": slow_trajectory_max_speed,
        "slow_trajectory_speed_threshold": slow_trajectory_speed_threshold,
        "slow_trajectory_candidate_count": slow_trajectory_candidate_count,
        "slow_speed_fraction": np.asarray(slow_speed_fraction),
        **pva_history,
        **peak_history,
        **overlap_history,
    }
    metrics = {
        "bump_attractor_trajectory_enabled": 1.0,
        "bump_attractor_initial_condition_count": float(n_initial_conditions),
        "bump_attractor_boundary_bisection_depth": float(
            boundary_bisection_depth
        ),
        "bump_attractor_boundary_bisection_completed_depth": float(
            completed_bisection_depth
        ),
        "bump_attractor_endpoint_probe_count": float(
            endpoint_probe_theta_initial.size
        ),
        "bump_attractor_boundary_nominal_resolution_deg": float(
            360.0
            / n_initial_conditions
            / (2**completed_bisection_depth)
        ),
        "bump_attractor_duration": duration,
        "bump_attractor_cue_duration": float(cue_steps * params.dt),
        "bump_attractor_cue_amplitude": cue_amplitude,
        "bump_attractor_sample_interval": float(sample_interval_steps * params.dt),
        "bump_attractor_cue_release_pva_strength_median": float(
            np.median(cue_release_pva_strength)
        ),
        "bump_attractor_cue_release_max_rate_fraction_median": float(
            np.median(cue_release_max_rate_fraction)
        ),
        "bump_attractor_cue_release_saturated_bin_count_median": float(
            np.median(cue_release_saturated_bin_count)
        ),
        "bump_attractor_cue_release_saturated_trial_fraction": float(
            np.mean(cue_release_saturated_bin_count > 0.0)
        ),
        "bump_attractor_cue_release_visual_to_distal_modulation_ratio_median": (
            float(np.median(cue_release_visual_to_distal_modulation_ratio))
        ),
        "bump_attractor_overlap_template_heading_count": float(
            overlap_theta_template.size
        ),
        **pva_metrics,
        **peak_metrics,
        **overlap_metrics,
        **_summarize_cue_release_map(
            decoder_name="pva",
            theta_initial=theta_initial,
            theta_release=cue_theta_pva[:, -1],
        ),
        **_summarize_cue_release_map(
            decoder_name="peak",
            theta_initial=theta_initial,
            theta_release=cue_theta_peak[:, -1],
        ),
        **_summarize_cue_release_map(
            decoder_name="overlap",
            theta_initial=theta_initial,
            theta_release=cue_theta_overlap[:, -1],
        ),
        **_summarize_autonomous_endpoint_landscape(
            decoder_name="pva",
            theta_probe=endpoint_probe_theta_initial,
            theta_release=endpoint_probe_theta_release_pva,
            theta_final=endpoint_probe_theta_final_pva,
            n_theta=params.n_theta,
        ),
        **_summarize_autonomous_endpoint_landscape(
            decoder_name="peak",
            theta_probe=endpoint_probe_theta_initial,
            theta_release=endpoint_probe_theta_release_peak,
            theta_final=endpoint_probe_theta_final_peak,
            n_theta=params.n_theta,
        ),
        **_summarize_autonomous_endpoint_landscape(
            decoder_name="overlap",
            theta_probe=endpoint_probe_theta_initial,
            theta_release=endpoint_probe_theta_release_overlap,
            theta_final=endpoint_probe_theta_final_overlap,
            n_theta=params.n_theta,
        ),
        "bump_attractor_pva_vs_overlap_disagreement_rms_deg": (
            decoder_disagreement_rms_deg(theta_pva, theta_overlap)
        ),
        "bump_attractor_peak_vs_overlap_disagreement_rms_deg": (
            decoder_disagreement_rms_deg(theta_peak, theta_overlap)
        ),
        "bump_attractor_final_pva_strength_median": float(
            np.median(pva_strength[:, -1])
        ),
        "bump_attractor_final_pva_strength_min": float(
            np.min(pva_strength[:, -1])
        ),
        "bump_attractor_final_contrast_median": float(
            np.median(bump_contrast[:, -1])
        ),
        "bump_attractor_final_contrast_min": float(
            np.min(bump_contrast[:, -1])
        ),
        "bump_attractor_final_overlap_max_median": float(
            np.median(overlap_max[:, -1])
        ),
        "slow_manifold_candidate_capture_enabled": float(slow_manifold_enabled),
        "slow_manifold_captured_candidate_count": float(
            slow_candidate_speed.size
        ),
        "ramesan_trajectory_sample_count": float(
            ramesan_trajectory_speed.size
        ),
        "slow_manifold_trajectory_with_candidate_fraction": float(
            np.mean(slow_trajectory_candidate_count > 0)
            if slow_manifold_enabled
            else 0.0
        ),
    }
    return history, metrics


def run_autonomous_probe_ring(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> dict[str, np.ndarray]:
    """Sample only the cue-settled canonical ring for geometry diagnostics.

    This is a lightweight backfill for saved darkness trajectories produced
    before canonical release states were retained.  It freezes all weights,
    settles each uniform visual cue, then packs the state on the zero-input
    side of the release without integrating the long darkness interval.
    """
    params = VafidisToyParams.from_config(config)
    n_probes = int(config.tests.bump_attractor_initial_conditions)
    cue_duration = float(config.tests.bump_attractor_cue_duration)
    if n_probes < 4:
        raise ValueError("autonomous probe ring requires at least four cues")
    if cue_duration < 0.0:
        raise ValueError("bump_attractor_cue_duration must be non-negative")
    probe_phase = np.linspace(-np.pi, np.pi, n_probes, endpoint=False)
    dynamics = FrozenAutonomousDynamics.from_state(
        params=params,
        state=trained_state,
    )
    probe_state = np.empty((n_probes, dynamics.state_dimension), dtype=float)
    probe_decoded_theta = np.empty(n_probes, dtype=float)
    cue_steps = int(round(cue_duration / params.dt))
    progress = tqdm(
        total=n_probes * cue_steps,
        disable=not config.simulation.progress,
        desc="autonomous cue-ring probes",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    progress_interval = max(1, int(round(1.0 / params.dt)))
    with progress:
        for probe_index, current_phase in enumerate(probe_phase):
            state = initialize_protocol_state(
                config=config,
                trained_state=trained_state,
                theta_true=float(current_phase),
            )
            progress.set_postfix(
                probe=f"{probe_index + 1}/{n_probes}",
                angle=f"{np.rad2deg(current_phase):.1f} deg",
                refresh=False,
            )
            for cue_step in range(1, cue_steps + 1):
                state = step_vafidis_toy(
                    state=state,
                    params=params,
                    angular_velocity=0.0,
                    visual_teacher=True,
                    training=False,
                    visual_noise=None,
                )
                if cue_step % progress_interval == 0:
                    progress.update(progress_interval)
            remainder = cue_steps % progress_interval
            if remainder:
                progress.update(remainder)
            state_vector = dynamics.pack_state(state)
            probe_state[probe_index] = state_vector
            probe_decoded_theta[probe_index] = dynamics.decoded_heading(
                state_vector
            )
    return {
        "autonomous_probe_phase": probe_phase,
        "autonomous_probe_decoded_theta": np.asarray(
            wrap_angle(probe_decoded_theta), dtype=float
        ),
        "autonomous_probe_state": probe_state,
    }


def run_slow_manifold_diagnostic(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    bump_attractor_trajectory_history: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit and diagnose the autonomous slow ring from retained slow points."""
    if not diagnostic_is_enabled(config, "slow_manifold"):
        return empty_slow_manifold_result()
    candidate_theta = np.asarray(
        bump_attractor_trajectory_history.get("slow_candidate_theta", np.empty(0)),
        dtype=float,
    )
    candidate_state = np.asarray(
        bump_attractor_trajectory_history.get(
            "slow_candidate_state", np.empty((0, 0))
        ),
        dtype=float,
    )
    candidate_speed = np.asarray(
        bump_attractor_trajectory_history.get("slow_candidate_speed", np.empty(0)),
        dtype=float,
    )
    trajectory_theta = np.asarray(
        bump_attractor_trajectory_history.get(
            "ramesan_trajectory_theta", np.empty(0)
        ),
        dtype=float,
    )
    trajectory_state = np.asarray(
        bump_attractor_trajectory_history.get(
            "ramesan_trajectory_state", np.empty((0, 0))
        ),
        dtype=float,
    )
    trajectory_speed = np.asarray(
        bump_attractor_trajectory_history.get(
            "ramesan_trajectory_speed", np.empty(0)
        ),
        dtype=float,
    )
    params = VafidisToyParams.from_config(config)
    dynamics = FrozenAutonomousDynamics.from_state(
        params=params,
        state=trained_state,
    )
    probe_phase = np.asarray(
        bump_attractor_trajectory_history.get(
            "autonomous_probe_phase", np.empty(0)
        ),
        dtype=float,
    )
    probe_state = np.asarray(
        bump_attractor_trajectory_history.get(
            "autonomous_probe_state", np.empty((0, 0))
        ),
        dtype=float,
    )
    if candidate_state.size == 0:
        candidate_state = np.empty((0, dynamics.state_dimension), dtype=float)
    ramesan_history: dict[str, np.ndarray] = {}
    ramesan_metrics: dict[str, float] = {
        "ramesan_diagnostic_succeeded": 0.0,
        "ramesan_missing_probe_state": 1.0,
    }
    if probe_phase.size >= 4:
        ramesan_history, ramesan_metrics = analyze_ramesan_firing_rate_geometry(
            dynamics=dynamics,
            probe_phase=probe_phase,
            probe_state=probe_state,
            candidate_theta=candidate_theta,
            candidate_state=candidate_state,
            candidate_speed=candidate_speed,
            q_threshold=float(config.tests.ramesan_q_threshold),
            jacobian_anchor_count=int(
                config.tests.slow_manifold_jacobian_anchors
            ),
            jacobian_eigenvalue_count=int(
                config.tests.slow_manifold_jacobian_eigenvalues
            ),
            jacobian_dense_dimension_limit=int(
                config.tests.slow_manifold_jacobian_dense_dimension_limit
            ),
            show_progress=bool(config.simulation.progress),
        )
        if trajectory_theta.size >= 4:
            phase_history, phase_metrics = analyze_ramesan_phase_landscape(
                dynamics=dynamics,
                probe_phase=probe_phase,
                probe_state=probe_state,
                trajectory_theta=trajectory_theta,
                trajectory_state=trajectory_state,
                trajectory_speed=trajectory_speed,
                q_threshold=float(config.tests.ramesan_q_threshold),
                angular_bin_count=int(config.tests.ramesan_phase_angular_bins),
                smoothing_bins=int(config.tests.ramesan_phase_smoothing_bins),
                ambient_enabled=bool(config.tests.ramesan_ambient_enabled),
                ambient_sample_count=int(
                    config.tests.ramesan_ambient_sample_count
                ),
                ambient_perturbation_scales=np.asarray(
                    config.tests.ramesan_ambient_perturbation_scales,
                    dtype=float,
                ),
                ambient_seed=(
                    int(config.simulation.seed)
                    + int(config.tests.ramesan_ambient_seed_offset)
                ),
                pca_center=ramesan_history["ramesan_pca_center"],
                pca_components=ramesan_history["ramesan_pca_components"],
                phase_velocity_floor=float(
                    config.tests.ramesan_phase_velocity_floor
                ),
                show_progress=bool(config.simulation.progress),
            )
            ramesan_history.update(phase_history)
            ramesan_metrics.update(phase_metrics)

    def merge_ramesan_diagnostic(
        history: dict[str, np.ndarray],
        metrics: dict[str, float],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        history.update(ramesan_history)
        metrics.update(ramesan_metrics)
        return history, metrics

    angular_bin_count = int(config.tests.slow_manifold_angular_bins)
    minimum_angular_support = float(
        config.tests.slow_manifold_min_angular_support_fraction
    )
    if not 0.0 < minimum_angular_support <= 1.0:
        raise ValueError(
            "tests.slow_manifold_min_angular_support_fraction must lie in (0, 1]"
        )
    if candidate_theta.size < 4:
        history, _metrics = empty_slow_manifold_result()
        return merge_ramesan_diagnostic(
            history,
            {
                "slow_manifold_enabled": 1.0,
                "slow_manifold_fit_succeeded": 0.0,
                "slow_manifold_candidate_count": float(candidate_theta.size),
            },
        )
    angular_bin_sample_count = candidate_angular_bin_counts(
        candidate_theta=candidate_theta,
        angular_bin_count=angular_bin_count,
    )
    angular_support_fraction = float(np.mean(angular_bin_sample_count > 0))
    if angular_support_fraction < minimum_angular_support:
        angle_clusters = summarize_candidate_angle_clusters(
            bin_sample_count=angular_bin_sample_count
        )
        history, _metrics = empty_slow_manifold_result()
        history["candidate_theta"] = candidate_theta
        history["candidate_state"] = candidate_state
        history["candidate_speed"] = candidate_speed
        history["angular_bin_sample_count"] = angular_bin_sample_count
        history["low_speed_angle_cluster_theta"] = angle_clusters[
            "cluster_theta"
        ]
        history["low_speed_angle_cluster_sample_count"] = angle_clusters[
            "cluster_sample_count"
        ]
        history["low_speed_angle_cluster_bin_count"] = angle_clusters[
            "cluster_bin_count"
        ]
        return merge_ramesan_diagnostic(
            history,
            {
                "slow_manifold_enabled": 1.0,
                "slow_manifold_fit_succeeded": 0.0,
                "slow_manifold_candidate_count": float(candidate_theta.size),
                "slow_manifold_angular_support_fraction": angular_support_fraction,
                "slow_manifold_min_angular_support_fraction": minimum_angular_support,
                "slow_manifold_fit_failure_is_insufficient_coverage": 1.0,
                "slow_manifold_occupied_angular_bin_count": float(
                    np.count_nonzero(angular_bin_sample_count)
                ),
                "slow_manifold_low_speed_angle_cluster_count": float(
                    angle_clusters["cluster_theta"].size
                ),
                "slow_manifold_candidate_speed_median": float(
                    np.median(candidate_speed)
                ),
                "slow_manifold_candidate_speed_max": float(
                    np.max(candidate_speed)
                ),
                "slow_manifold_speed_fraction": float(
                    config.tests.slow_manifold_speed_fraction
                ),
                "slow_manifold_speed_floor": (
                    float(config.tests.slow_manifold_speed_floor)
                    if config.tests.slow_manifold_speed_floor is not None
                    else float("nan")
                ),
            },
        )
    try:
        history, metrics = analyze_slow_manifold_candidates(
            dynamics=dynamics,
            candidate_theta=candidate_theta,
            candidate_state=candidate_state,
            candidate_speed=candidate_speed,
            angular_bin_count=angular_bin_count,
            jacobian_anchor_count=int(
                config.tests.slow_manifold_jacobian_anchors
            ),
            jacobian_eigenvalue_count=int(
                config.tests.slow_manifold_jacobian_eigenvalues
            ),
            jacobian_dense_dimension_limit=int(
                config.tests.slow_manifold_jacobian_dense_dimension_limit
            ),
            minimum_angular_support_fraction=minimum_angular_support,
        )
    except ValueError as error:
        history, _metrics = empty_slow_manifold_result()
        return merge_ramesan_diagnostic(
            history,
            {
                "slow_manifold_enabled": 1.0,
                "slow_manifold_fit_succeeded": 0.0,
                "slow_manifold_candidate_count": float(candidate_theta.size),
                "slow_manifold_fit_failure_is_insufficient_coverage": float(
                    "angular bin" in str(error) or "candidate" in str(error)
                ),
            },
        )
    metrics["slow_manifold_fit_succeeded"] = 1.0
    metrics["slow_manifold_speed_fraction"] = float(
        config.tests.slow_manifold_speed_fraction
    )
    metrics["slow_manifold_speed_floor"] = (
        float(config.tests.slow_manifold_speed_floor)
        if config.tests.slow_manifold_speed_floor is not None
        else float("nan")
    )
    return merge_ramesan_diagnostic(history, metrics)


def _release_visual_teacher_from_state(
    *,
    state: VafidisToyState,
    params: VafidisToyParams,
) -> VafidisToyState:
    """Remove visual current without collapsing the dynamic proximal voltage."""
    released_state = state.copy()
    released_state.i_vis_to_hd = np.zeros(params.n_theta, dtype=float)
    released_state.v_hd_ss = (
        params.p_distal_to_proximal * released_state.v_hd_distal
    )
    released_state.r_hd = apply_activation(
        released_state.v_hd_proximal,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )
    released_state.e_hd = released_state.r_hd - apply_activation(
        released_state.v_hd_ss,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )
    validate_vafidis_toy_state(released_state, params)
    return released_state


def _normally_perturb_hd_current_state(
    *,
    state: VafidisToyState,
    params: VafidisToyParams,
    manifold_tangent: np.ndarray,
    perturbation_rms: float,
    rng: np.random.Generator,
) -> VafidisToyState:
    """Perturb the distal cascade in a direction normal to the ring."""
    tangent = np.asarray(manifold_tangent, dtype=float)
    if tangent.shape != (params.n_theta,):
        raise ValueError("manifold_tangent must contain one value per HD neuron")
    perturbation = rng.normal(size=params.n_theta)
    tangent_squared_norm = float(np.dot(tangent, tangent))
    if tangent_squared_norm > 1e-12:
        perturbation = perturbation - (
            float(np.dot(perturbation, tangent)) / tangent_squared_norm
        ) * tangent
    perturbation_unit_rms = float(np.sqrt(np.mean(np.square(perturbation))))
    if perturbation_unit_rms <= 1e-12:
        raise ValueError("failed to sample a non-zero normal perturbation")
    perturbation = perturbation * (perturbation_rms / perturbation_unit_rms)

    perturbed_state = state.copy()
    # Perturbing both distal cascade variables avoids an artificial one-step
    # reset by the faster filter and mirrors Clark's current-space protocol.
    # The proximal voltage remains continuous and subsequently responds through
    # Eq. (4), rather than being projected onto a steady-state manifold.
    perturbed_state.i_hd_distal = perturbed_state.i_hd_distal + perturbation
    perturbed_state.v_hd_distal = perturbed_state.v_hd_distal + perturbation
    perturbed_state.v_hd_ss = (
        params.p_distal_to_proximal * perturbed_state.v_hd_distal
    )
    perturbed_state.r_hd = apply_activation(
        perturbed_state.v_hd_proximal,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )
    perturbed_state.e_hd = perturbed_state.r_hd - apply_activation(
        perturbed_state.v_hd_ss,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )
    validate_vafidis_toy_state(perturbed_state, params)
    return perturbed_state


def run_timescale_separation_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    hd_tuning_history: dict[str, np.ndarray],
    bump_attractor_trajectory_history: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Apply an operational Clark-style separation-of-timescales assay.

    The normal timescale is measured in HD distal-current space after frozen-
    weight states are displaced away from the closed visual target manifold.
    The tangential timescale is the first-passage time of Clark-overlap bump
    trajectories in zero-input darkness.  A conservative comparison uses the
    90th percentile normal relaxation time and the 10th percentile tangential
    first-passage time.
    """
    enabled = diagnostic_is_enabled(config, "timescale_separation")
    if not enabled:
        return {
            "normal_time": np.empty(0, dtype=float),
            "theta_anchor": np.empty(0, dtype=float),
            "perturbation_scale": np.empty(0, dtype=float),
            "normal_distance_to_manifold": np.empty((0, 0, 0, 0), dtype=float),
            "normal_control_distance_to_manifold": np.empty((0, 0), dtype=float),
            "normal_e_folding_time": np.empty((0, 0, 0), dtype=float),
            "normal_recovery_observed": np.empty((0, 0, 0), dtype=bool),
            "tangential_time": np.empty(0, dtype=float),
            "tangential_overlap_displacement": np.empty((0, 0), dtype=float),
            "tangential_first_passage_time": np.empty(0, dtype=float),
            "tangential_first_passage_observed": np.empty(0, dtype=bool),
        }, {"timescale_separation_enabled": 0.0}

    params = VafidisToyParams.from_config(config)
    n_anchors = int(config.tests.timescale_separation_initial_conditions)
    normal_duration = float(config.tests.timescale_separation_normal_duration)
    sample_interval = float(config.tests.timescale_separation_sample_interval)
    perturbation_scales = np.asarray(
        config.tests.timescale_separation_perturbation_scales,
        dtype=float,
    )
    repetitions = int(config.tests.timescale_separation_perturbations_per_condition)
    seed_offset = int(config.tests.timescale_separation_seed_offset)
    tangential_threshold_deg = float(
        config.tests.timescale_separation_tangential_threshold_deg
    )
    ratio_threshold = float(config.tests.timescale_separation_ratio_threshold)
    cue_duration = float(config.tests.bump_attractor_cue_duration)
    if n_anchors <= 0:
        raise ValueError("tests.timescale_separation_initial_conditions must be positive")
    if normal_duration <= 0.0:
        raise ValueError("tests.timescale_separation_normal_duration must be positive")
    if sample_interval <= 0.0:
        raise ValueError("tests.timescale_separation_sample_interval must be positive")
    if perturbation_scales.ndim != 1 or perturbation_scales.size == 0:
        raise ValueError("tests.timescale_separation_perturbation_scales cannot be empty")
    if not np.all(np.isfinite(perturbation_scales)) or np.any(
        perturbation_scales <= 0.0
    ):
        raise ValueError("timescale perturbation scales must be finite and positive")
    if repetitions <= 0:
        raise ValueError(
            "tests.timescale_separation_perturbations_per_condition must be positive"
        )
    if tangential_threshold_deg <= 0.0 or ratio_threshold <= 0.0:
        raise ValueError("timescale thresholds must be positive")
    if "theta_true" not in hd_tuning_history or "v_hd_distal" not in hd_tuning_history:
        raise ValueError(
            "timescale separation requires theta_true and v_hd_distal from HD tuning"
        )
    if (
        "time" not in bump_attractor_trajectory_history
        or "overlap_angular_displacement"
        not in bump_attractor_trajectory_history
    ):
        raise ValueError(
            "timescale separation requires enabled bump attractor trajectories"
        )

    manifold_theta = np.asarray(hd_tuning_history["theta_true"], dtype=float)
    current_manifold = np.asarray(hd_tuning_history["v_hd_distal"], dtype=float)
    if current_manifold.shape != (manifold_theta.size, params.n_theta):
        raise ValueError("HD distal-current manifold must have shape (heading, neuron)")
    if manifold_theta.size < 8:
        raise ValueError("timescale separation requires at least eight manifold points")

    normal_steps = max(1, int(round(normal_duration / params.dt)))
    sample_interval_steps = max(1, int(round(sample_interval / params.dt)))
    sample_step_indices = np.arange(
        0,
        normal_steps + 1,
        sample_interval_steps,
        dtype=int,
    )
    if sample_step_indices[-1] != normal_steps:
        sample_step_indices = np.append(sample_step_indices, normal_steps)
    normal_time = sample_step_indices.astype(float) * params.dt
    cue_steps = max(0, int(round(cue_duration / params.dt)))
    theta_anchor = np.linspace(-np.pi, np.pi, n_anchors, endpoint=False, dtype=float)
    n_scales = perturbation_scales.size
    n_samples = normal_time.size

    released_anchor_states: list[VafidisToyState] = []
    control_voltage = np.empty(
        (n_anchors, n_samples, params.n_theta),
        dtype=float,
    )
    perturbed_voltage = np.empty(
        (n_scales, n_anchors, repetitions, n_samples, params.n_theta),
        dtype=float,
    )
    total_steps = n_anchors * cue_steps + normal_steps * (
        n_anchors + n_scales * n_anchors * repetitions
    )
    progress = tqdm(
        total=total_steps,
        disable=not config.simulation.progress,
        desc="timescale separation",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    progress_update_interval = max(1, int(round(1.0 / params.dt)))

    def advance_darkness_trajectory(
        *,
        initial_state: VafidisToyState,
        destination: np.ndarray,
    ) -> None:
        state = initial_state
        destination[0] = state.v_hd_distal
        next_sample_index = 1
        for step_index in range(1, normal_steps + 1):
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=0.0,
                visual_teacher=False,
                training=False,
                visual_noise=None,
            )
            if step_index % progress_update_interval == 0:
                progress.update(progress_update_interval)
            if (
                next_sample_index < sample_step_indices.size
                and step_index == sample_step_indices[next_sample_index]
            ):
                destination[next_sample_index] = state.v_hd_distal
                next_sample_index += 1
        remainder = normal_steps % progress_update_interval
        if remainder:
            progress.update(remainder)

    with progress:
        for anchor_index, anchor_angle in enumerate(theta_anchor):
            state = initialize_protocol_state(
                config=config,
                trained_state=trained_state,
                theta_true=float(anchor_angle),
            )
            progress.set_postfix(
                phase="cue anchors",
                anchor=f"{anchor_index + 1}/{n_anchors}",
                refresh=False,
            )
            for cue_step in range(1, cue_steps + 1):
                state = step_vafidis_toy(
                    state=state,
                    params=params,
                    angular_velocity=0.0,
                    visual_teacher=True,
                    training=False,
                    visual_noise=None,
                )
                if cue_step % progress_update_interval == 0:
                    progress.update(progress_update_interval)
            cue_remainder = cue_steps % progress_update_interval
            if cue_remainder:
                progress.update(cue_remainder)
            released_anchor_states.append(
                _release_visual_teacher_from_state(state=state, params=params)
            )

        for anchor_index, released_state in enumerate(released_anchor_states):
            progress.set_postfix(
                phase="controls",
                anchor=f"{anchor_index + 1}/{n_anchors}",
                refresh=False,
            )
            advance_darkness_trajectory(
                initial_state=released_state.copy(),
                destination=control_voltage[anchor_index],
            )

        perturbation_rng = make_rng(config.simulation.seed + seed_offset)
        for scale_index, perturbation_scale in enumerate(perturbation_scales):
            for anchor_index, anchor_angle in enumerate(theta_anchor):
                nearest_index = int(
                    np.argmin(np.abs(circular_difference(manifold_theta, anchor_angle)))
                )
                manifold_tangent = (
                    current_manifold[(nearest_index + 1) % manifold_theta.size]
                    - current_manifold[(nearest_index - 1) % manifold_theta.size]
                )
                for repetition_index in range(repetitions):
                    progress.set_postfix(
                        phase="normal recovery",
                        scale=f"{perturbation_scale:g}",
                        anchor=f"{anchor_index + 1}/{n_anchors}",
                        rep=f"{repetition_index + 1}/{repetitions}",
                        refresh=False,
                    )
                    perturbed_state = _normally_perturb_hd_current_state(
                        state=released_anchor_states[anchor_index],
                        params=params,
                        manifold_tangent=manifold_tangent,
                        perturbation_rms=float(perturbation_scale),
                        rng=perturbation_rng,
                    )
                    advance_darkness_trajectory(
                        initial_state=perturbed_state,
                        destination=perturbed_voltage[
                            scale_index,
                            anchor_index,
                            repetition_index,
                        ],
                    )

    control_distance_flat, control_coordinate_flat = nearest_closed_manifold_distance(
        control_voltage.reshape(-1, params.n_theta),
        current_manifold,
    )
    normal_control_distance = control_distance_flat.reshape(n_anchors, n_samples)
    normal_control_coordinate = control_coordinate_flat.reshape(n_anchors, n_samples)
    normal_distance_flat, normal_coordinate_flat = nearest_closed_manifold_distance(
        perturbed_voltage.reshape(-1, params.n_theta),
        current_manifold,
    )
    normal_distance = normal_distance_flat.reshape(
        n_scales,
        n_anchors,
        repetitions,
        n_samples,
    )
    normal_coordinate = normal_coordinate_flat.reshape(normal_distance.shape)
    normal_distance_to_control = np.sqrt(
        np.mean(
            np.square(
                perturbed_voltage
                - control_voltage[None, :, None, :, :]
            ),
            axis=-1,
        )
    )
    relaxation_summary = estimate_relaxation_e_folding_time(
        time=normal_time,
        distance=normal_distance,
        tail_fraction=0.2,
        peak_window=min(0.25, normal_duration),
    )
    normal_e_folding_time = np.asarray(
        relaxation_summary["e_folding_time"],
        dtype=float,
    )
    normal_recovery_observed = np.asarray(
        relaxation_summary["event_observed"],
        dtype=bool,
    )
    normal_peak_distance = np.asarray(relaxation_summary["peak_value"], dtype=float)
    strict_normal_recovery = (
        np.isfinite(normal_peak_distance)
        & np.isfinite(normal_distance[..., -1])
        & (normal_distance[..., -1] <= normal_peak_distance / np.e)
    )

    tangential_time = np.asarray(
        bump_attractor_trajectory_history["time"],
        dtype=float,
    )
    tangential_overlap_displacement = np.asarray(
        bump_attractor_trajectory_history["overlap_angular_displacement"],
        dtype=float,
    )
    tangential_threshold = float(np.deg2rad(tangential_threshold_deg))
    tangential_first_passage_time, tangential_first_passage_observed = (
        angular_first_passage_time(
            time=tangential_time,
            angular_displacement=tangential_overlap_displacement,
            threshold=tangential_threshold,
        )
    )

    finite_normal_times = normal_e_folding_time[np.isfinite(normal_e_folding_time)]
    finite_tangential_times = tangential_first_passage_time[
        np.isfinite(tangential_first_passage_time)
    ]
    normal_time_p90 = (
        float(np.quantile(finite_normal_times, 0.90))
        if finite_normal_times.size
        else float("nan")
    )
    normal_time_median = (
        float(np.median(finite_normal_times))
        if finite_normal_times.size
        else float("nan")
    )
    tangential_time_p10 = (
        float(np.quantile(finite_tangential_times, 0.10))
        if finite_tangential_times.size
        else float("nan")
    )
    tangential_time_median = (
        float(np.median(finite_tangential_times))
        if finite_tangential_times.size
        else float("nan")
    )
    conservative_ratio = (
        float(tangential_time_p10 / normal_time_p90)
        if np.isfinite(tangential_time_p10)
        and np.isfinite(normal_time_p90)
        and normal_time_p90 > 0.0
        else float("nan")
    )
    normal_recovery_fraction = float(np.mean(normal_recovery_observed))
    strict_normal_recovery_fraction = float(np.mean(strict_normal_recovery))
    criterion_passed = float(
        np.isfinite(conservative_ratio)
        and conservative_ratio >= ratio_threshold
        and normal_recovery_fraction >= 0.90
        and strict_normal_recovery_fraction >= 0.90
    )
    tangential_duration = float(tangential_time[-1] - tangential_time[0])
    ratio_is_lower_bound = float(
        np.isfinite(tangential_time_p10)
        and tangential_time_p10 >= tangential_duration - 1e-9
    )

    history = {
        "normal_time": normal_time,
        "theta_anchor": theta_anchor,
        "perturbation_scale": perturbation_scales,
        "normal_distance_to_manifold": normal_distance,
        "normal_nearest_manifold_coordinate": normal_coordinate,
        "normal_distance_to_control": normal_distance_to_control,
        "normal_control_distance_to_manifold": normal_control_distance,
        "normal_control_nearest_manifold_coordinate": normal_control_coordinate,
        "normal_e_folding_time": normal_e_folding_time,
        "normal_recovery_observed": normal_recovery_observed,
        "normal_strict_recovery_observed": strict_normal_recovery,
        "normal_distance_floor": np.asarray(relaxation_summary["floor"], dtype=float),
        "normal_peak_distance": normal_peak_distance,
        "normal_peak_time": np.asarray(relaxation_summary["peak_time"], dtype=float),
        "tangential_time": tangential_time,
        "tangential_theta_initial": np.asarray(
            bump_attractor_trajectory_history["theta_initial"],
            dtype=float,
        ),
        "tangential_overlap_displacement": tangential_overlap_displacement,
        "tangential_first_passage_time": tangential_first_passage_time,
        "tangential_first_passage_observed": tangential_first_passage_observed,
        "tangential_threshold_rad": np.asarray(tangential_threshold),
        "normal_time_p90": np.asarray(normal_time_p90),
        "tangential_time_p10": np.asarray(tangential_time_p10),
        "conservative_timescale_ratio": np.asarray(conservative_ratio),
        "criterion_ratio_threshold": np.asarray(ratio_threshold),
        "criterion_passed": np.asarray(criterion_passed),
        "criterion_ratio_is_lower_bound": np.asarray(ratio_is_lower_bound),
    }
    metrics = {
        "timescale_separation_enabled": 1.0,
        "timescale_separation_anchor_count": float(n_anchors),
        "timescale_separation_perturbation_scale_count": float(n_scales),
        "timescale_separation_repetitions_per_condition": float(repetitions),
        "timescale_separation_normal_duration_s": normal_duration,
        "timescale_separation_tangential_duration_s": tangential_duration,
        "timescale_separation_tangential_threshold_deg": tangential_threshold_deg,
        "timescale_separation_normal_time_median_s": normal_time_median,
        "timescale_separation_normal_time_p90_s": normal_time_p90,
        "timescale_separation_tangential_time_median_s": tangential_time_median,
        "timescale_separation_tangential_time_p10_s": tangential_time_p10,
        "timescale_separation_conservative_ratio": conservative_ratio,
        "timescale_separation_ratio_threshold": ratio_threshold,
        "timescale_separation_ratio_is_lower_bound": ratio_is_lower_bound,
        "timescale_separation_normal_recovery_fraction": normal_recovery_fraction,
        "timescale_separation_strict_normal_recovery_fraction": (
            strict_normal_recovery_fraction
        ),
        "timescale_separation_tangential_passage_fraction": float(
            np.mean(tangential_first_passage_observed)
        ),
        "timescale_separation_control_final_distance_median": float(
            np.median(normal_control_distance[:, -1])
        ),
        "timescale_separation_perturbed_initial_distance_median": float(
            np.median(normal_distance[..., 0])
        ),
        "timescale_separation_perturbed_final_distance_median": float(
            np.median(normal_distance[..., -1])
        ),
        "timescale_separation_criterion_passed": criterion_passed,
    }
    return history, metrics


def run_velocity_trajectory_sweep_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    hd_tuning_history: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Resolve pinning and depinning under constant positive velocity input.

    This is a frozen-weight analogue of Noorman et al. Figure 2f.  Uniformly
    spaced cue locations are released into darkness while a constant velocity
    is applied.  Retaining the full decoded trajectories exposes stationary
    pinning and stick--slip motion that an endpoint-only gain fit can hide.
    """
    enabled = diagnostic_is_enabled(config, "velocity_trajectory_sweep")
    if not enabled:
        return {
            "time": np.empty(0, dtype=float),
            "commanded_velocity": np.empty(0, dtype=float),
            "theta_initial": np.empty(0, dtype=float),
            "theta_pva": np.empty((0, 0, 0), dtype=float),
            "theta_peak": np.empty((0, 0, 0), dtype=float),
            "theta_overlap": np.empty((0, 0, 0), dtype=float),
            "pva_angular_displacement": np.empty((0, 0, 0), dtype=float),
            "overlap_angular_displacement": np.empty((0, 0, 0), dtype=float),
            "decoded_velocity_pva": np.empty((0, 0), dtype=float),
                "decoded_velocity_overlap": np.empty((0, 0), dtype=float),
                "stall_fraction": np.empty((0, 0), dtype=float),
            "robust_speed_linearity": np.empty((0, 0), dtype=float),
            "depinning_success": np.empty((0, 0), dtype=bool),
            "depinning_success_fraction": np.empty(0, dtype=float),
        }, {"velocity_trajectory_sweep_enabled": 0.0}

    params = VafidisToyParams.from_config(config)
    configured_sweep_velocities = np.asarray(
        config.tests.velocity_trajectory_sweep_velocities,
        dtype=float,
    )
    if configured_sweep_velocities.size:
        if (
            configured_sweep_velocities.ndim != 1
            or configured_sweep_velocities.size < 2
            or not np.all(np.isfinite(configured_sweep_velocities))
            or np.any(configured_sweep_velocities < 0.0)
            or np.any(np.diff(configured_sweep_velocities) <= 0.0)
            or not np.isclose(configured_sweep_velocities[0], 0.0)
        ):
            raise ValueError(
                "tests.velocity_trajectory_sweep_velocities must be a strictly "
                "increasing finite list beginning at zero"
            )
        commanded_velocity = configured_sweep_velocities.copy()
        velocity_count = int(commanded_velocity.size)
        max_velocity = float(commanded_velocity[-1])
    else:
        max_velocity = float(config.tests.velocity_trajectory_sweep_max_velocity)
        velocity_count = int(config.tests.velocity_trajectory_sweep_count)
        commanded_velocity = np.linspace(
            0.0,
            max_velocity,
            velocity_count,
            dtype=float,
        )
    ring_velocity_count = int(
        config.tests.velocity_trajectory_sweep_ring_velocity_count
    )
    phase_flow_probe_enabled = True
    phase_flow_initial_condition_count = int(
        config.tests.velocity_phase_flow_initial_conditions
    )
    phase_flow_duration = float(config.tests.velocity_phase_flow_duration)
    phase_flow_sample_interval = float(
        config.tests.velocity_phase_flow_sample_interval
    )
    phase_flow_fit_start_time = float(
        config.tests.velocity_phase_flow_fit_start_time
    )
    phase_flow_fit_duration = float(
        config.tests.velocity_phase_flow_fit_duration
    )
    phase_flow_angular_bin_count = int(
        config.tests.velocity_phase_flow_angular_bins
    )
    phase_flow_smoothing_bin_count = int(
        config.tests.velocity_phase_flow_smoothing_bins
    )
    phase_flow_empirical_lambda_speed_floor = float(
        config.tests.velocity_phase_flow_empirical_lambda_speed_floor
    )
    requested_phase_flow_velocities = np.asarray(
        config.tests.velocity_phase_flow_probe_velocities,
        dtype=float,
    )
    initial_condition_count = int(
        config.tests.velocity_trajectory_sweep_initial_conditions
    )
    initial_heading = float(
        config.tests.velocity_trajectory_sweep_initial_heading
    )
    duration = float(config.tests.velocity_trajectory_sweep_duration)
    cue_duration = float(config.tests.velocity_trajectory_sweep_cue_duration)
    sample_interval = float(config.tests.velocity_trajectory_sweep_sample_interval)
    fit_start_time = float(config.tests.velocity_trajectory_sweep_fit_start_time)
    depinning_gain_threshold = float(
        config.tests.velocity_trajectory_sweep_depinning_gain_threshold
    )
    depinning_max_stall_fraction = float(
        config.tests.velocity_trajectory_sweep_depinning_max_stall_fraction
    )
    depinning_success_fraction_threshold = float(
        config.tests.velocity_trajectory_sweep_depinning_success_fraction
    )
    if max_velocity <= 0.0:
        raise ValueError("tests.velocity_trajectory_sweep_max_velocity must be positive")
    if velocity_count < 2:
        raise ValueError("tests.velocity_trajectory_sweep_count must be at least two")
    if ring_velocity_count <= 0:
        raise ValueError(
            "tests.velocity_trajectory_sweep_ring_velocity_count must be positive"
        )
    if phase_flow_probe_enabled:
        if phase_flow_initial_condition_count < 3:
            raise ValueError(
                "tests.velocity_phase_flow_initial_conditions must be at least three"
            )
        if phase_flow_duration <= 0.0 or phase_flow_sample_interval <= 0.0:
            raise ValueError("velocity phase-flow duration and interval must be positive")
        if (
            phase_flow_fit_start_time < 0.0
            or phase_flow_fit_duration <= 0.0
            or phase_flow_fit_start_time + phase_flow_fit_duration
            > phase_flow_duration + 1e-12
        ):
            raise ValueError("velocity phase-flow fit window must lie within the probe")
        if phase_flow_angular_bin_count < 8:
            raise ValueError("velocity phase-flow angular bins must be at least eight")
        if (
            phase_flow_smoothing_bin_count <= 0
            or phase_flow_smoothing_bin_count % 2 == 0
            or phase_flow_smoothing_bin_count > phase_flow_angular_bin_count
        ):
            raise ValueError(
                "velocity phase-flow smoothing bins must be positive, odd, "
                "and no larger than the angular grid"
            )
        if phase_flow_empirical_lambda_speed_floor <= 0.0:
            raise ValueError(
                "velocity phase-flow empirical-lambda speed floor must be positive"
            )
        if requested_phase_flow_velocities.size:
            if (
                requested_phase_flow_velocities.ndim != 1
                or not np.all(np.isfinite(requested_phase_flow_velocities))
                or np.any(np.diff(requested_phase_flow_velocities) <= 0.0)
            ):
                raise ValueError(
                    "tests.velocity_phase_flow_probe_velocities must be a "
                    "strictly increasing finite list"
                )
            for requested_velocity in requested_phase_flow_velocities:
                if not np.any(
                    np.isclose(commanded_velocity, requested_velocity, atol=1e-12)
                ):
                    raise ValueError(
                        "every velocity_phase_flow_probe_velocities value must "
                        "also occur in velocity_trajectory_sweep_velocities"
                    )
    if initial_condition_count <= 0:
        raise ValueError(
            "tests.velocity_trajectory_sweep_initial_conditions must be positive"
        )
    if not np.isfinite(initial_heading):
        raise ValueError(
            "tests.velocity_trajectory_sweep_initial_heading must be finite"
        )
    if duration <= 0.0 or cue_duration < 0.0 or sample_interval <= 0.0:
        raise ValueError("velocity trajectory durations and sample interval are invalid")
    if fit_start_time < 0.0 or fit_start_time >= duration:
        raise ValueError(
            "tests.velocity_trajectory_sweep_fit_start_time must lie within the trial"
        )
    if depinning_gain_threshold <= 0.0:
        raise ValueError("velocity trajectory depinning gain threshold must be positive")
    if not 0.0 <= depinning_max_stall_fraction <= 1.0:
        raise ValueError("velocity trajectory maximum stall fraction must lie in [0, 1]")
    if not 0.0 < depinning_success_fraction_threshold <= 1.0:
        raise ValueError("velocity trajectory success fraction must lie in (0, 1]")
    if "theta_true" not in hd_tuning_history or "r_hd" not in hd_tuning_history:
        raise ValueError("velocity trajectory sweep requires the HD tuning template")

    overlap_theta_template = np.asarray(hd_tuning_history["theta_true"], dtype=float)
    overlap_target_rate = np.asarray(hd_tuning_history["r_hd"], dtype=float).T
    if overlap_target_rate.shape != (params.n_theta, overlap_theta_template.size):
        raise ValueError("HD tuning template must have shape (heading, HD neuron)")

    theta_initial = np.asarray(
        wrap_angle(
            initial_heading
            + np.linspace(
                0.0,
                2.0 * np.pi,
                initial_condition_count,
                endpoint=False,
                dtype=float,
            )
        ),
        dtype=float,
    )
    cue_steps = max(0, int(round(cue_duration / params.dt)))
    trial_steps = max(1, int(round(duration / params.dt)))
    sample_interval_steps = max(1, int(round(sample_interval / params.dt)))
    sample_step_indices = np.arange(
        0,
        trial_steps + 1,
        sample_interval_steps,
        dtype=int,
    )
    if sample_step_indices[-1] != trial_steps:
        sample_step_indices = np.append(sample_step_indices, trial_steps)
    time = sample_step_indices.astype(float) * params.dt
    trajectory_shape = (velocity_count, initial_condition_count, time.size)
    theta_pva = np.empty(trajectory_shape, dtype=float)
    theta_peak = np.empty(trajectory_shape, dtype=float)
    theta_overlap = np.empty(trajectory_shape, dtype=float)
    pva_strength = np.empty(trajectory_shape, dtype=float)
    overlap_max = np.empty(trajectory_shape, dtype=float)
    settled_states: list[VafidisToyState] = []

    total_steps = initial_condition_count * cue_steps + (
        velocity_count * initial_condition_count * trial_steps
    )
    progress = tqdm(
        total=total_steps,
        disable=not config.simulation.progress,
        desc="velocity trajectory sweep",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    progress_update_interval = max(1, int(round(1.0 / params.dt)))

    def update_progress(step_index: int, total_phase_steps: int) -> None:
        if step_index % progress_update_interval == 0:
            progress.update(progress_update_interval)
        elif step_index == total_phase_steps:
            remainder = total_phase_steps % progress_update_interval
            if remainder:
                progress.update(remainder)

    with progress:
        for initial_index, initial_angle in enumerate(theta_initial):
            state = initialize_protocol_state(
                config=config,
                trained_state=trained_state,
                theta_true=float(initial_angle),
            )
            progress.set_postfix(
                phase="cue anchors",
                start=f"{initial_index + 1}/{initial_condition_count}",
                refresh=False,
            )
            for cue_step in range(1, cue_steps + 1):
                state = step_vafidis_toy(
                    state=state,
                    params=params,
                    angular_velocity=0.0,
                    visual_teacher=True,
                    training=False,
                    visual_noise=None,
                )
                update_progress(cue_step, cue_steps)
            settled_states.append(state)

        for velocity_index, velocity_value in enumerate(commanded_velocity):
            for initial_index, settled_state in enumerate(settled_states):
                state = settled_state.copy()
                progress.set_postfix(
                    phase="constant-velocity darkness",
                    velocity=f"{velocity_value:.2f}",
                    start=f"{initial_index + 1}/{initial_condition_count}",
                    refresh=False,
                )
                sampled_hd_activity = np.empty(
                    (time.size, params.n_theta),
                    dtype=float,
                )

                def record_sample(sample_index: int) -> None:
                    theta_pva[velocity_index, initial_index, sample_index] = float(
                        state.theta_hd_decoded
                    )
                    theta_peak[velocity_index, initial_index, sample_index] = float(
                        state.theta_hd_decoded_peak
                    )
                    sampled_hd_activity[sample_index] = state.r_hd
                    pva_strength[velocity_index, initial_index, sample_index] = (
                        pva_vector_strength(state.theta_hd_pref, state.r_hd)
                    )

                record_sample(0)
                next_sample_index = 1
                for trial_step in range(1, trial_steps + 1):
                    state = step_vafidis_toy(
                        state=state,
                        params=params,
                        angular_velocity=float(velocity_value),
                        visual_teacher=False,
                        training=False,
                        visual_noise=None,
                    )
                    update_progress(trial_step, trial_steps)
                    if (
                        next_sample_index < sample_step_indices.size
                        and trial_step == sample_step_indices[next_sample_index]
                        ):
                            record_sample(next_sample_index)
                            next_sample_index += 1
                decoded_overlap, maximum_overlap = decode_heading_by_clark_overlap(
                    theta_template=overlap_theta_template,
                    target_rate=overlap_target_rate,
                    population_activity=sampled_hd_activity,
                )
                theta_overlap[velocity_index, initial_index] = decoded_overlap
                overlap_max[velocity_index, initial_index] = maximum_overlap

    phase_flow_velocity_index = np.empty(0, dtype=int)
    phase_flow_commanded_velocity = np.empty(0, dtype=float)
    phase_flow_time = np.empty(0, dtype=float)
    phase_flow_theta_initial = np.empty(0, dtype=float)
    phase_flow_theta_pva = np.empty((0, 0, 0), dtype=float)
    if phase_flow_probe_enabled:
        if requested_phase_flow_velocities.size:
            phase_flow_velocity_index = np.asarray(
                [
                    int(
                        np.flatnonzero(
                            np.isclose(
                                commanded_velocity,
                                requested_velocity,
                                atol=1e-12,
                            )
                        )[0]
                    )
                    for requested_velocity in requested_phase_flow_velocities
                ],
                dtype=int,
            )
        else:
            selected_count = min(ring_velocity_count, velocity_count)
            phase_flow_velocity_index = np.unique(
                np.rint(
                    np.linspace(0, velocity_count - 1, selected_count)
                ).astype(int)
            )
        phase_flow_commanded_velocity = commanded_velocity[
            phase_flow_velocity_index
        ]
        phase_flow_theta_initial = np.linspace(
            -np.pi,
            np.pi,
            phase_flow_initial_condition_count,
            endpoint=False,
            dtype=float,
        )
        phase_flow_steps = max(1, int(round(phase_flow_duration / params.dt)))
        phase_flow_sample_steps = max(
            1,
            int(round(phase_flow_sample_interval / params.dt)),
        )
        phase_flow_sample_indices = np.arange(
            0,
            phase_flow_steps + 1,
            phase_flow_sample_steps,
            dtype=int,
        )
        if phase_flow_sample_indices[-1] != phase_flow_steps:
            phase_flow_sample_indices = np.append(
                phase_flow_sample_indices,
                phase_flow_steps,
            )
        phase_flow_time = phase_flow_sample_indices.astype(float) * params.dt
        phase_flow_theta_pva = np.empty(
            (
                phase_flow_velocity_index.size,
                phase_flow_initial_condition_count,
                phase_flow_time.size,
            ),
            dtype=float,
        )
        phase_flow_settled_states: list[VafidisToyState] = []
        phase_flow_total_steps = phase_flow_initial_condition_count * cue_steps + (
            phase_flow_velocity_index.size
            * phase_flow_initial_condition_count
            * phase_flow_steps
        )
        phase_flow_progress = tqdm(
            total=phase_flow_total_steps,
            disable=not config.simulation.progress,
            desc="velocity phase-flow probe",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
        )

        def update_phase_flow_progress(
            step_index: int,
            total_phase_steps: int,
        ) -> None:
            if step_index % progress_update_interval == 0:
                phase_flow_progress.update(progress_update_interval)
            elif step_index == total_phase_steps:
                remainder = total_phase_steps % progress_update_interval
                if remainder:
                    phase_flow_progress.update(remainder)

        with phase_flow_progress:
            for initial_index, initial_angle in enumerate(
                phase_flow_theta_initial
            ):
                state = initialize_protocol_state(
                    config=config,
                    trained_state=trained_state,
                    theta_true=float(initial_angle),
                )
                phase_flow_progress.set_postfix(
                    phase="cue anchors",
                    start=(
                        f"{initial_index + 1}/"
                        f"{phase_flow_initial_condition_count}"
                    ),
                    refresh=False,
                )
                for cue_step in range(1, cue_steps + 1):
                    state = step_vafidis_toy(
                        state=state,
                        params=params,
                        angular_velocity=0.0,
                        visual_teacher=True,
                        training=False,
                        visual_noise=None,
                    )
                    update_phase_flow_progress(cue_step, cue_steps)
                phase_flow_settled_states.append(state)

            for selected_index, velocity_index in enumerate(
                phase_flow_velocity_index
            ):
                velocity_value = float(commanded_velocity[velocity_index])
                for initial_index, settled_state in enumerate(
                    phase_flow_settled_states
                ):
                    state = settled_state.copy()
                    phase_flow_progress.set_postfix(
                        phase="dense short rollouts",
                        velocity=f"{velocity_value:.2f}",
                        start=(
                            f"{initial_index + 1}/"
                            f"{phase_flow_initial_condition_count}"
                        ),
                        refresh=False,
                    )
                    phase_flow_theta_pva[
                        selected_index,
                        initial_index,
                        0,
                    ] = float(state.theta_hd_decoded)
                    next_sample_index = 1
                    for probe_step in range(1, phase_flow_steps + 1):
                        state = step_vafidis_toy(
                            state=state,
                            params=params,
                            angular_velocity=velocity_value,
                            visual_teacher=False,
                            training=False,
                            visual_noise=None,
                        )
                        update_phase_flow_progress(
                            probe_step,
                            phase_flow_steps,
                        )
                        if (
                            next_sample_index < phase_flow_sample_indices.size
                            and probe_step
                            == phase_flow_sample_indices[next_sample_index]
                        ):
                            phase_flow_theta_pva[
                                selected_index,
                                initial_index,
                                next_sample_index,
                            ] = float(state.theta_hd_decoded)
                            next_sample_index += 1

    pva_displacement = np.unwrap(theta_pva, axis=-1) - np.unwrap(
        theta_pva,
        axis=-1,
    )[..., :1]
    peak_displacement = np.unwrap(theta_peak, axis=-1) - np.unwrap(
        theta_peak,
        axis=-1,
    )[..., :1]
    overlap_displacement = np.unwrap(theta_overlap, axis=-1) - np.unwrap(
        theta_overlap,
        axis=-1,
    )[..., :1]
    fit_mask = time >= fit_start_time
    fit_time = time[fit_mask]
    decoded_velocity_pva = np.empty(
        (velocity_count, initial_condition_count),
        dtype=float,
    )
    decoded_velocity_overlap = np.empty_like(decoded_velocity_pva)
    for velocity_index in range(velocity_count):
        for initial_index in range(initial_condition_count):
            decoded_velocity_pva[velocity_index, initial_index], _intercept = (
                linear_fit_slope_intercept(
                    fit_time,
                    pva_displacement[velocity_index, initial_index, fit_mask],
                )
            )
            decoded_velocity_overlap[velocity_index, initial_index], _intercept = (
                linear_fit_slope_intercept(
                    fit_time,
                    overlap_displacement[velocity_index, initial_index, fit_mask],
                )
            )

    instantaneous_velocity_pva = np.gradient(
        pva_displacement,
        time,
        axis=-1,
    )
    stall_fraction = np.full(decoded_velocity_pva.shape, np.nan, dtype=float)
    robust_speed_linearity = np.full(decoded_velocity_pva.shape, np.nan, dtype=float)
    velocity_gain_per_trial = np.full(decoded_velocity_pva.shape, np.nan, dtype=float)
    depinning_success = np.zeros(decoded_velocity_pva.shape, dtype=bool)
    for velocity_index, velocity_value in enumerate(commanded_velocity):
        if velocity_value <= 0.0:
            continue
        evaluation_speed = instantaneous_velocity_pva[velocity_index][:, fit_mask]
        stall_speed_threshold = max(0.05 * float(velocity_value), 0.01)
        stall_fraction[velocity_index] = np.mean(
            evaluation_speed < stall_speed_threshold,
            axis=-1,
        )
        speed_p10 = np.quantile(evaluation_speed, 0.10, axis=-1)
        speed_p90 = np.quantile(evaluation_speed, 0.90, axis=-1)
        robust_speed_linearity[velocity_index] = np.divide(
            speed_p10,
            speed_p90,
            out=np.full(initial_condition_count, np.nan, dtype=float),
            where=speed_p90 > 1e-12,
        )
        velocity_gain_per_trial[velocity_index] = (
            decoded_velocity_pva[velocity_index] / float(velocity_value)
        )
        depinning_success[velocity_index] = (
            velocity_gain_per_trial[velocity_index] >= depinning_gain_threshold
        ) & (stall_fraction[velocity_index] <= depinning_max_stall_fraction)

    depinning_success_fraction = np.mean(depinning_success, axis=1)
    depinning_velocity_indices = np.flatnonzero(
        (commanded_velocity > 0.0)
        & (depinning_success_fraction >= depinning_success_fraction_threshold)
    )
    estimated_depinning_threshold = (
        float(commanded_velocity[depinning_velocity_indices[0]])
        if depinning_velocity_indices.size
        else float("nan")
    )
    decoder_disagreement = circular_difference(theta_pva, theta_overlap)
    finite_disagreement = decoder_disagreement[np.isfinite(decoder_disagreement)]

    history = {
        "time": time,
        "commanded_velocity": commanded_velocity,
        "theta_initial": theta_initial,
        "theta_pva": np.asarray(wrap_angle(theta_pva), dtype=float),
        "theta_peak": np.asarray(wrap_angle(theta_peak), dtype=float),
        "theta_overlap": np.asarray(wrap_angle(theta_overlap), dtype=float),
        "pva_angular_displacement": pva_displacement,
        "peak_angular_displacement": peak_displacement,
        "overlap_angular_displacement": overlap_displacement,
        "pva_strength": pva_strength,
        "overlap_max": overlap_max,
        "instantaneous_velocity_pva": instantaneous_velocity_pva,
        "decoded_velocity_pva": decoded_velocity_pva,
        "decoded_velocity_overlap": decoded_velocity_overlap,
        "velocity_gain_per_trial": velocity_gain_per_trial,
        "stall_fraction": stall_fraction,
        "robust_speed_linearity": robust_speed_linearity,
        "depinning_success": depinning_success,
        "depinning_success_fraction": depinning_success_fraction,
        "estimated_depinning_threshold": np.asarray(estimated_depinning_threshold),
        "fit_start_time": np.asarray(fit_start_time),
        "phase_flow_probe_enabled": np.asarray(phase_flow_probe_enabled),
        "phase_flow_velocity_index": phase_flow_velocity_index,
        "phase_flow_commanded_velocity": phase_flow_commanded_velocity,
        "phase_flow_time": phase_flow_time,
        "phase_flow_theta_initial": phase_flow_theta_initial,
        "phase_flow_theta_pva": np.asarray(
            wrap_angle(phase_flow_theta_pva),
            dtype=float,
        ),
        "phase_flow_fit_start_time": np.asarray(phase_flow_fit_start_time),
        "phase_flow_fit_duration": np.asarray(phase_flow_fit_duration),
    }
    positive_velocity_mask = commanded_velocity > 0.0
    metrics = {
        "velocity_trajectory_sweep_enabled": 1.0,
        "velocity_trajectory_sweep_max_velocity_rad_s": max_velocity,
        "velocity_trajectory_sweep_velocity_count": float(velocity_count),
        "velocity_trajectory_sweep_initial_condition_count": float(
            initial_condition_count
        ),
        "velocity_trajectory_sweep_initial_heading_rad": initial_heading,
        "velocity_trajectory_sweep_ring_velocity_count": float(ring_velocity_count),
        "velocity_phase_flow_probe_enabled": float(phase_flow_probe_enabled),
        "velocity_phase_flow_initial_condition_count": float(
            phase_flow_initial_condition_count if phase_flow_probe_enabled else 0
        ),
        "velocity_phase_flow_duration_s": (
            phase_flow_duration if phase_flow_probe_enabled else 0.0
        ),
        "velocity_phase_flow_fit_start_time_s": (
            phase_flow_fit_start_time if phase_flow_probe_enabled else 0.0
        ),
        "velocity_phase_flow_fit_duration_s": (
            phase_flow_fit_duration if phase_flow_probe_enabled else 0.0
        ),
        "velocity_phase_flow_angular_bin_count": float(
            phase_flow_angular_bin_count if phase_flow_probe_enabled else 0
        ),
        "velocity_phase_flow_smoothing_bin_count": float(
            phase_flow_smoothing_bin_count if phase_flow_probe_enabled else 0
        ),
        "velocity_phase_flow_empirical_lambda_speed_floor": (
            phase_flow_empirical_lambda_speed_floor
            if phase_flow_probe_enabled
            else 0.0
        ),
        "velocity_trajectory_sweep_duration_s": duration,
        "velocity_trajectory_sweep_fit_start_time_s": fit_start_time,
        "velocity_trajectory_sweep_depinning_gain_threshold": (
            depinning_gain_threshold
        ),
        "velocity_trajectory_sweep_depinning_max_stall_fraction": (
            depinning_max_stall_fraction
        ),
        "velocity_trajectory_sweep_depinning_success_fraction_threshold": (
            depinning_success_fraction_threshold
        ),
        "velocity_trajectory_sweep_estimated_depinning_threshold_rad_s": (
            estimated_depinning_threshold
        ),
        "velocity_trajectory_sweep_success_fraction_at_max_velocity": float(
            depinning_success_fraction[-1]
        ),
        "velocity_trajectory_sweep_median_gain_at_max_velocity": float(
            np.nanmedian(velocity_gain_per_trial[-1])
        ),
        "velocity_trajectory_sweep_median_stall_fraction_at_max_velocity": float(
            np.nanmedian(stall_fraction[-1])
        ),
        "velocity_trajectory_sweep_median_linearity_at_max_velocity": float(
            np.nanmedian(robust_speed_linearity[-1])
        ),
        "velocity_trajectory_sweep_zero_input_abs_drift_median_deg_s": float(
            np.rad2deg(np.median(np.abs(decoded_velocity_pva[0])))
        ),
        "velocity_trajectory_sweep_positive_velocity_median_gain": float(
            np.nanmedian(velocity_gain_per_trial[positive_velocity_mask])
        ),
        "velocity_trajectory_sweep_pva_overlap_disagreement_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean(np.square(finite_disagreement))))
            if finite_disagreement.size
            else float("nan")
        ),
    }
    return history, metrics


def run_bump_diffusion_ensemble_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Reproduce the zero-velocity diffusion assay in ``original/stability.py``.

    ``bump_diffusion_duration`` is the complete protocol duration. A stronger,
    wider visual cue initializes one of the discrete HD locations, after which
    zero-velocity darkness occupies the remaining time. For every configured
    test-noise level the primary estimate is the original endpoint statistic
    ``D = Var(Delta theta) / T_dark``. Time-resolved ensemble moments are kept
    without retaining every full trial trajectory.
    """
    theta_initial_values = np.unique(trained_state.theta_hd_pref)
    requested_trials = int(config.tests.bump_diffusion_trials)
    total_duration = float(config.tests.bump_diffusion_duration)
    cue_duration = float(config.tests.bump_diffusion_cue_duration)
    darkness_duration = total_duration - cue_duration
    cue_amplitude = float(config.tests.bump_diffusion_cue_amplitude)
    cue_sigma = float(config.tests.bump_diffusion_cue_sigma)
    release_skip_steps = int(config.tests.bump_diffusion_release_skip_steps)
    integration_method = str(
        config.tests.bump_diffusion_integration_method
    ).lower()
    test_noise_stds = np.asarray(
        config.tests.bump_diffusion_test_noise_stds,
        dtype=float,
    )
    primary_noise_std = float(config.tests.bump_diffusion_test_noise_std)
    if requested_trials <= 0:
        raise ValueError("tests.bump_diffusion_trials must be positive")
    if total_duration <= 0.0:
        raise ValueError("tests.bump_diffusion_duration must be positive")
    if cue_duration < 0.0 or darkness_duration <= 0.0:
        raise ValueError(
            "tests.bump_diffusion_cue_duration must lie in "
            "[0, bump_diffusion_duration)"
        )
    if cue_amplitude <= 0.0 or cue_sigma <= 0.0:
        raise ValueError("bump diffusion cue amplitude and sigma must be positive")
    if release_skip_steps < 0:
        raise ValueError("tests.bump_diffusion_release_skip_steps must be non-negative")
    if (
        test_noise_stds.ndim != 1
        or test_noise_stds.size == 0
        or not np.all(np.isfinite(test_noise_stds))
        or np.any(test_noise_stds < 0.0)
    ):
        raise ValueError(
            "tests.bump_diffusion_test_noise_stds must be a non-empty list "
            "of finite non-negative values"
        )
    if np.unique(test_noise_stds).size != test_noise_stds.size:
        raise ValueError("tests.bump_diffusion_test_noise_stds must be unique")
    primary_matches = np.flatnonzero(
        np.isclose(test_noise_stds, primary_noise_std, rtol=0.0, atol=1e-12)
    )
    if primary_matches.size != 1:
        raise ValueError(
            "tests.bump_diffusion_test_noise_std must occur exactly once in "
            "tests.bump_diffusion_test_noise_stds; "
            f"got primary={primary_noise_std:g}, "
            f"sweep={test_noise_stds.tolist()}"
        )
    primary_noise_index = int(primary_matches[0])

    n_unique_heading = theta_initial_values.size
    source_bump_size_bins = max(1, int(trained_state.theta_hd_pref.size / 12))
    source_wrap_threshold = (
        source_bump_size_bins * 2.0 * np.pi / float(n_unique_heading)
    )

    diffusion_config = deepcopy(config)
    diffusion_config.visual.amplitude = cue_amplitude
    if diffusion_config.visual.profile.lower() == "von_mises":
        diffusion_config.visual.kappa = 1.0 / (4.0 * cue_sigma**2)
    diffusion_config.simulation.proximal_integration_method = integration_method
    # Validate the assay-specific solver and parameter overrides before the
    # expensive ensemble starts.
    diffusion_params = VafidisToyParams.from_config(diffusion_config)

    ensemble_rng = make_rng(
        config.simulation.seed + int(config.tests.bump_diffusion_seed_offset)
    )
    # Match original/stability.py: independently sample a discrete HD location
    # on every trial, including the deterministic zero-noise row.
    trial_initial_values = ensemble_rng.choice(
        theta_initial_values,
        size=(test_noise_stds.size, requested_trials),
        replace=True,
    )

    endpoint_displacement = np.empty(
        (test_noise_stds.size, requested_trials),
        dtype=float,
    )
    darkness_time: np.ndarray | None = None
    displacement_sum: np.ndarray | None = None
    displacement_square_sum: np.ndarray | None = None
    total_trial_count = test_noise_stds.size * requested_trials
    trial_progress = trange(
        total_trial_count,
        disable=not config.simulation.progress,
        desc="bump diffusion ensemble",
        unit="trial",
    )
    for flat_trial_index in trial_progress:
        noise_index, trial_index = divmod(flat_trial_index, requested_trials)
        test_noise_std = float(test_noise_stds[noise_index])
        theta_initial = float(trial_initial_values[noise_index, trial_index])
        history = run_constant_velocity_visual_dark_visual_protocol(
            config=diffusion_config,
            trained_state=trained_state,
            theta_true=theta_initial,
            darkness_duration=darkness_duration,
            angular_velocity=0.0,
            cue_duration=cue_duration,
            recue_duration=0.0,
            synaptic_input_noise_std=test_noise_std,
            synaptic_input_noise_seed=(
                config.simulation.seed
                + int(config.tests.bump_diffusion_seed_offset)
                + flat_trial_index
                + 1
            ),
        )
        dark_mask = phase_mask(history, DARKNESS_PHASE_ID)
        if np.count_nonzero(dark_mask) < 2:
            raise RuntimeError("diffusion trial produced fewer than two darkness samples")
        trial_time = history["time"][dark_mask]
        reference_index = min(release_skip_steps, trial_time.size - 2)
        trial_time = trial_time[reference_index:]
        trial_time = trial_time - float(trial_time[0])
        if darkness_time is None:
            darkness_time = trial_time
            trace_shape = (test_noise_stds.size, trial_time.size)
            displacement_sum = np.zeros(trace_shape, dtype=float)
            displacement_square_sum = np.zeros(trace_shape, dtype=float)
        elif trial_time.shape != darkness_time.shape or not np.allclose(trial_time, darkness_time):
            raise RuntimeError("diffusion trials must share the same darkness time grid")
        pva_darkness = np.asarray(
            history["theta_hd_decoded"][dark_mask][reference_index:],
            dtype=float,
        )
        # Match stability.py's ``Dx`` accumulation and its bump-size wrap
        # correction (60 degrees for the 60-unit/30-position network).
        pva_step_displacement = np.diff(pva_darkness)
        pva_step_displacement[pva_step_displacement < -source_wrap_threshold] += (
            2.0 * np.pi
        )
        pva_step_displacement[pva_step_displacement > source_wrap_threshold] -= (
            2.0 * np.pi
        )
        pva_displacement = np.concatenate(
            [np.zeros(1, dtype=float), np.cumsum(pva_step_displacement)]
        )
        endpoint_displacement[noise_index, trial_index] = pva_displacement[-1]
        assert displacement_sum is not None
        assert displacement_square_sum is not None
        displacement_sum[noise_index] += pva_displacement
        displacement_square_sum[noise_index] += np.square(pva_displacement)

    if darkness_time is None or displacement_sum is None or displacement_square_sum is None:
        raise RuntimeError("diffusion protocol produced no darkness samples")
    displacement_mean_by_noise = displacement_sum / float(requested_trials)
    displacement_variance_by_noise = np.maximum(
        displacement_square_sum / float(requested_trials)
        - np.square(displacement_mean_by_noise),
        0.0,
    )
    diffusion_trace_by_noise = np.full_like(displacement_variance_by_noise, np.nan)
    positive_time = darkness_time > 0.0
    diffusion_trace_by_noise[:, positive_time] = (
        displacement_variance_by_noise[:, positive_time]
        / darkness_time[positive_time]
    )

    endpoint_diffusion = np.empty(test_noise_stds.size, dtype=float)
    endpoint_mean = np.empty_like(endpoint_diffusion)
    endpoint_std = np.empty_like(endpoint_diffusion)
    systematic_drift_velocity = np.empty_like(endpoint_diffusion)
    anomalous_exponent = np.empty_like(endpoint_diffusion)
    generalized_diffusion = np.empty_like(endpoint_diffusion)
    anomalous_r_squared = np.empty_like(endpoint_diffusion)
    anomalous_fit_trace = np.empty_like(displacement_variance_by_noise)
    anomalous_fit_n_points = np.empty_like(endpoint_diffusion)
    anomalous_fit_start = np.empty_like(endpoint_diffusion)
    anomalous_fit_end = np.empty_like(endpoint_diffusion)
    for noise_index in range(test_noise_stds.size):
        endpoint_summary = summarize_ensemble_diffusion_coefficient(
            angular_displacement=endpoint_displacement[noise_index],
            duration=darkness_duration,
        )
        anomalous_summary = fit_anomalous_diffusion_power_law(
            time=darkness_time,
            displacement_variance=displacement_variance_by_noise[noise_index],
            fit_start_time=float(config.tests.bump_diffusion_fit_start_time),
            fit_end_time=config.tests.bump_diffusion_fit_end_time,
        )
        endpoint_diffusion[noise_index] = float(
            endpoint_summary["diffusion_coefficient"]
        )
        endpoint_mean[noise_index] = float(endpoint_summary["displacement_mean"])
        endpoint_std[noise_index] = float(endpoint_summary["displacement_std"])
        systematic_drift_velocity[noise_index] = (
            endpoint_mean[noise_index] / darkness_duration
        )
        anomalous_exponent[noise_index] = float(
            anomalous_summary["anomalous_diffusion_exponent"]
        )
        generalized_diffusion[noise_index] = float(
            anomalous_summary["generalized_diffusion_coefficient"]
        )
        anomalous_r_squared[noise_index] = float(
            anomalous_summary["anomalous_diffusion_log_r_squared"]
        )
        anomalous_fit_trace[noise_index] = np.asarray(
            anomalous_summary["anomalous_diffusion_fit_trace"], dtype=float
        )
        anomalous_fit_n_points[noise_index] = float(
            anomalous_summary["anomalous_diffusion_fit_n_points"]
        )
        anomalous_fit_start[noise_index] = float(
            anomalous_summary["anomalous_diffusion_fit_start_time"]
        )
        anomalous_fit_end[noise_index] = float(
            anomalous_summary["anomalous_diffusion_fit_end_time"]
        )

    primary = primary_noise_index
    ensemble_history = {
        "time": darkness_time,
        "test_noise_std": test_noise_stds,
        "theta_initial": np.asarray(trial_initial_values, dtype=float),
        "pva_endpoint_angular_displacement": endpoint_displacement,
        "pva_endpoint_displacement_mean": endpoint_mean,
        "pva_endpoint_displacement_std": endpoint_std,
        "pva_endpoint_diffusion_coefficient": endpoint_diffusion,
        "pva_endpoint_diffusion_coefficient_deg2_s": (
            np.rad2deg(1.0) ** 2 * endpoint_diffusion
        ),
        "pva_systematic_drift_velocity": systematic_drift_velocity,
        "pva_displacement_mean_by_noise": displacement_mean_by_noise,
        "pva_displacement_variance_by_noise": displacement_variance_by_noise,
        "pva_diffusion_coefficient_by_noise": diffusion_trace_by_noise,
        "pva_anomalous_diffusion_exponent_by_noise": anomalous_exponent,
        "pva_generalized_diffusion_coefficient_by_noise": generalized_diffusion,
        "pva_anomalous_diffusion_log_r_squared_by_noise": anomalous_r_squared,
        "pva_anomalous_diffusion_fit_by_noise": anomalous_fit_trace,
        # Primary-noise aliases preserve the existing figure/data contract.
        "pva_displacement_mean": displacement_mean_by_noise[primary],
        "pva_displacement_variance": displacement_variance_by_noise[primary],
        "pva_diffusion_coefficient": diffusion_trace_by_noise[primary],
        "pva_anomalous_diffusion_fit": anomalous_fit_trace[primary],
    }
    primary_diffusion = float(endpoint_diffusion[primary])
    primary_generalized_diffusion = float(generalized_diffusion[primary])
    metrics = {
        "bump_diffusion_diagnostic_enabled": 1.0,
        "bump_diffusion_original_protocol": 1.0,
        "bump_diffusion_noise_level_count": float(test_noise_stds.size),
        "bump_diffusion_primary_noise_index": float(primary),
        "bump_ensemble_diffusion_coefficient": primary_diffusion,
        "bump_ensemble_diffusion_coefficient_deg2_s": float(
            np.rad2deg(1.0) ** 2 * primary_diffusion
        ),
        "bump_ensemble_displacement_mean": float(endpoint_mean[primary]),
        "bump_ensemble_displacement_std": float(endpoint_std[primary]),
        "bump_ensemble_diffusion_n_trials": float(requested_trials),
        "bump_ensemble_systematic_drift_velocity": float(
            systematic_drift_velocity[primary]
        ),
        "bump_ensemble_abs_systematic_drift_velocity": abs(
            float(systematic_drift_velocity[primary])
        ),
        "bump_ensemble_diffusion_total_duration": total_duration,
        "bump_ensemble_diffusion_cue_duration": cue_duration,
        "bump_ensemble_diffusion_duration": darkness_duration,
        "bump_ensemble_diffusion_release_skip_steps": float(release_skip_steps),
        "bump_ensemble_diffusion_wrap_threshold_deg": float(
            np.rad2deg(source_wrap_threshold)
        ),
        "bump_ensemble_diffusion_cue_amplitude": cue_amplitude,
        "bump_ensemble_diffusion_cue_sigma": cue_sigma,
        "bump_ensemble_diffusion_forward_euler": float(
            integration_method == "forward_euler"
        ),
        "bump_ensemble_test_noise_std": primary_noise_std,
        "bump_ensemble_anomalous_diffusion_exponent": float(
            anomalous_exponent[primary]
        ),
        "bump_ensemble_generalized_diffusion_coefficient": (
            primary_generalized_diffusion
        ),
        "bump_ensemble_generalized_diffusion_coefficient_deg2_s_alpha": float(
            np.rad2deg(1.0) ** 2 * primary_generalized_diffusion
        ),
        "bump_ensemble_anomalous_diffusion_log_r_squared": float(
            anomalous_r_squared[primary]
        ),
        "bump_ensemble_anomalous_diffusion_fit_n_points": float(
            anomalous_fit_n_points[primary]
        ),
        "bump_ensemble_anomalous_diffusion_fit_start_time": float(
            anomalous_fit_start[primary]
        ),
        "bump_ensemble_anomalous_diffusion_fit_end_time": float(
            anomalous_fit_end[primary]
        ),
        "bump_ensemble_diffusion_dt": float(diffusion_params.dt),
    }
    return ensemble_history, metrics


def run_darkness_path_integration_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    angular_velocity: float,
    progress: Any | None = None,
    progress_label: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    darkness_history = run_constant_velocity_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.darkness_test_duration,
        angular_velocity=angular_velocity,
        cue_duration=get_pi_cue_duration(config),
        progress=progress,
        progress_label=progress_label,
    )
    visual_cue_mask = phase_mask(darkness_history, VISUAL_CUE_PHASE_ID)
    cue_mask = phase_mask(darkness_history, DARKNESS_PHASE_ID)
    recue_mask = phase_mask(darkness_history, VISUAL_RECUE_PHASE_ID)
    visual_cue_pi_error = circular_error_trace(
        darkness_history["theta_hd_decoded"][visual_cue_mask],
        darkness_history["theta_true"][visual_cue_mask],
    )
    darkness_theta_decoded = darkness_history["theta_hd_decoded"][cue_mask]
    darkness_theta_true = darkness_history["theta_true"][cue_mask]
    absolute_decode_error = circular_error_trace(
        darkness_theta_decoded,
        darkness_theta_true,
    )
    pi_error = release_relative_pi_error_trace(
        darkness_theta_decoded,
        darkness_theta_true,
    )
    wrapped_pi_error = np.asarray(wrap_angle(pi_error), dtype=float)
    visual_cue_final_abs_error = (
        final_abs_circular_error(
            darkness_history["theta_hd_decoded"][visual_cue_mask],
            theta_reference=darkness_history["theta_true"][visual_cue_mask][-1],
        )
        if np.any(visual_cue_mask)
        else float("nan")
    )
    decoded_velocity = estimate_decoded_velocity(
        time=darkness_history["time"][cue_mask] - darkness_history["time"][cue_mask][0],
        theta_decoded=darkness_history["theta_hd_decoded"][cue_mask],
        start_fraction=0.25,
    )
    pi_error_peak = release_relative_pi_error_trace(
        darkness_history["theta_hd_decoded_peak"][cue_mask],
        darkness_theta_true,
    )
    wrapped_pi_error_peak = np.asarray(wrap_angle(pi_error_peak), dtype=float)
    decoded_velocity_peak = estimate_decoded_velocity(
        time=darkness_history["time"][cue_mask] - darkness_history["time"][cue_mask][0],
        theta_decoded=darkness_history["theta_hd_decoded_peak"][cue_mask],
        start_fraction=0.25,
    )
    darkness_rms_pi_error = float(np.sqrt(np.nanmean(np.square(pi_error))))
    darkness_final_abs_pi_error = float(abs(pi_error[-1]))
    # Full-protocol accumulated PI error, referenced to darkness onset and
    # unwrapped continuously across the cue-darkness-recue window.  The cue
    # segment shows any pre-release drift (should be ~0 because the visual
    # anchor holds the bump), the darkness segment accumulates the integrator
    # error, and the recue segment reveals whether re-anchoring pulls the
    # accumulated error back toward zero.
    full_theta_decoded = np.asarray(
        darkness_history["theta_hd_decoded"], dtype=float
    )
    full_theta_true = np.asarray(darkness_history["theta_true"], dtype=float)
    darkness_onset_index = int(np.flatnonzero(cue_mask)[0])
    full_protocol_pi_error = full_protocol_pi_error_trace(
        full_theta_decoded,
        full_theta_true,
        reference_index=darkness_onset_index,
    )
    full_protocol_pi_error_peak = full_protocol_pi_error_trace(
        np.asarray(darkness_history["theta_hd_decoded_peak"], dtype=float),
        full_theta_true,
        reference_index=darkness_onset_index,
    )
    darkness_history["pi_error_full_protocol"] = full_protocol_pi_error
    darkness_history["pi_error_full_protocol_wrapped"] = np.asarray(
        wrap_angle(full_protocol_pi_error), dtype=float
    )
    darkness_history["peak_pi_error_full_protocol"] = full_protocol_pi_error_peak
    darkness_history["pi_error_release_relative"] = np.full(
        darkness_history["time"].shape,
        np.nan,
        dtype=float,
    )
    darkness_history["pi_error_release_relative"][cue_mask] = pi_error
    darkness_history["pi_error_release_relative_wrapped"] = np.full(
        darkness_history["time"].shape,
        np.nan,
        dtype=float,
    )
    darkness_history["pi_error_release_relative_wrapped"][cue_mask] = (
        wrapped_pi_error
    )
    darkness_history["peak_pi_error_release_relative"] = np.full(
        darkness_history["time"].shape,
        np.nan,
        dtype=float,
    )
    darkness_history["peak_pi_error_release_relative"][cue_mask] = pi_error_peak
    darkness_metrics = {
        "visual_cue_hd_decode_rms_error": rms_circular_error(
            darkness_history["theta_hd_decoded"][visual_cue_mask],
            darkness_history["theta_true"][visual_cue_mask],
        ),
        "visual_cue_hd_decode_mean_abs_error": nanmean_or_nan(np.abs(visual_cue_pi_error)),
        "visual_cue_hd_decode_final_abs_error": visual_cue_final_abs_error,
        "darkness_rms_pi_error": darkness_rms_pi_error,
        "darkness_hd_decode_rms_error": float(
            np.sqrt(np.nanmean(np.square(absolute_decode_error)))
        ),
        "darkness_hd_decode_mean_abs_error": nanmean_or_nan(
            np.abs(absolute_decode_error)
        ),
        "darkness_hd_decode_signed_bias": nanmean_or_nan(absolute_decode_error),
        "darkness_final_abs_pi_error": darkness_final_abs_pi_error,
        "darkness_hd_decode_final_abs_error": float(
            abs(absolute_decode_error[-1])
        ),
        "darkness_mean_pi_error": nanmean_or_nan(pi_error),
        "darkness_mean_abs_pi_error": nanmean_or_nan(np.abs(pi_error)),
        "darkness_wrapped_rms_pi_error": float(
            np.sqrt(np.nanmean(np.square(wrapped_pi_error)))
        ),
        "darkness_wrapped_final_abs_pi_error": float(abs(wrapped_pi_error[-1])),
        "darkness_mean_pva_strength": float(np.nanmean(darkness_history["pva_strength_hd"][cue_mask])),
        "darkness_final_pva_strength": float(darkness_history["pva_strength_hd"][cue_mask][-1]),
        "darkness_mean_bump_contrast": float(np.nanmean(darkness_history["bump_contrast_hd"][cue_mask])),
        "darkness_final_bump_contrast": float(darkness_history["bump_contrast_hd"][cue_mask][-1]),
        "darkness_decoded_velocity": decoded_velocity,
        "darkness_commanded_angular_velocity": float(angular_velocity),
        "darkness_decoded_velocity_deg_s": float(np.rad2deg(decoded_velocity)),
        "darkness_velocity_bias": decoded_velocity - angular_velocity,
        "darkness_abs_velocity_bias": float(abs(decoded_velocity - angular_velocity)),
        "darkness_velocity_bias_deg_s": float(np.rad2deg(decoded_velocity - angular_velocity)),
        "darkness_abs_velocity_bias_deg_s": float(abs(np.rad2deg(decoded_velocity - angular_velocity))),
        "darkness_peak_rms_pi_error": float(
            np.sqrt(np.nanmean(np.square(pi_error_peak)))
        ),
        "darkness_peak_final_abs_pi_error": float(abs(pi_error_peak[-1])),
        "darkness_peak_mean_pi_error": nanmean_or_nan(pi_error_peak),
        "darkness_peak_wrapped_rms_pi_error": float(
            np.sqrt(np.nanmean(np.square(wrapped_pi_error_peak)))
        ),
        "darkness_peak_decoded_velocity": decoded_velocity_peak,
        "darkness_peak_decoded_velocity_deg_s": float(np.rad2deg(decoded_velocity_peak)),
        "darkness_peak_velocity_bias": decoded_velocity_peak - angular_velocity,
        "darkness_peak_abs_velocity_bias": float(abs(decoded_velocity_peak - angular_velocity)),
        "darkness_peak_velocity_bias_deg_s": float(np.rad2deg(decoded_velocity_peak - angular_velocity)),
        "darkness_peak_abs_velocity_bias_deg_s": float(abs(np.rad2deg(decoded_velocity_peak - angular_velocity))),
        **summarize_hd_saturation(
            history=darkness_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="darkness",
        ),
        **summarize_hd_near_peak(
            history=darkness_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="darkness",
        ),
        **summarize_hd_local_peaks(
            history=darkness_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=cue_mask,
            metric_prefix="darkness",
        ),
    }
    if np.any(recue_mask):
        darkness_metrics.update(
            {
                "darkness_recue_initial_abs_pi_error": final_abs_circular_error(
                    darkness_history["theta_hd_decoded"][cue_mask],
                    theta_reference=darkness_history["theta_true"][cue_mask][-1],
                ),
                "darkness_recue_final_abs_pi_error": final_abs_circular_error(
                    darkness_history["theta_hd_decoded"][recue_mask],
                    theta_reference=darkness_history["theta_true"][recue_mask][-1],
                ),
                "darkness_recue_rms_pi_error": rms_circular_error(
                    darkness_history["theta_hd_decoded"][recue_mask],
                    darkness_history["theta_true"][recue_mask],
                ),
                "darkness_recue_final_pva_strength": float(
                    darkness_history["pva_strength_hd"][recue_mask][-1]
                ),
            }
        )
    return darkness_history, darkness_metrics


def run_constant_velocity_pi_error_sweep_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    primary_history: dict[str, np.ndarray] | None = None,
    primary_metrics: dict[str, float] | None = None,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compare frozen-network PI error over a compact velocity grid.

    The legacy scalar darkness trace remains the primary detailed diagnostic.
    These arrays add a matched set of constant-velocity protocols so gain and
    accumulated PI error are not inferred from one commanded speed alone.
    """
    commanded_velocities = np.asarray(
        config.tests.constant_pi_velocities,
        dtype=float,
    )
    if (
        commanded_velocities.ndim != 1
        or commanded_velocities.size == 0
        or not np.isfinite(commanded_velocities).all()
    ):
        raise ValueError(
            "tests.constant_pi_velocities must be a non-empty finite list"
        )
    if np.unique(commanded_velocities).size != commanded_velocities.size:
        raise ValueError("tests.constant_pi_velocities must not contain duplicates")

    primary_velocity = float(config.tests.darkness_angular_velocity)
    histories: list[dict[str, np.ndarray]] = []
    decoded_velocities: list[float] = []
    decoded_peak_velocities: list[float] = []
    for commanded_velocity in commanded_velocities:
        reuse_primary = bool(
            primary_history is not None
            and primary_metrics is not None
            and np.isclose(
                float(commanded_velocity),
                primary_velocity,
                atol=1e-12,
                rtol=0.0,
            )
        )
        if reuse_primary:
            history = primary_history
            metrics = primary_metrics
        else:
            history, metrics = run_darkness_path_integration_test(
                config=config,
                trained_state=trained_state,
                angular_velocity=float(commanded_velocity),
                progress=progress,
                progress_label=(
                    "constant PI "
                    f"{np.rad2deg(float(commanded_velocity)):+.0f} deg/s"
                ),
            )
        histories.append(history)
        decoded_velocities.append(float(metrics["darkness_decoded_velocity"]))
        decoded_peak_velocities.append(
            float(metrics["darkness_peak_decoded_velocity"])
        )

    common_time = np.asarray(histories[0]["time"], dtype=float)
    common_phase_id = np.asarray(histories[0]["phase_id"], dtype=float)
    for history in histories[1:]:
        if not np.array_equal(np.asarray(history["time"]), common_time):
            raise RuntimeError("constant PI protocols produced inconsistent time grids")
        if not np.array_equal(np.asarray(history["phase_id"]), common_phase_id):
            raise RuntimeError("constant PI protocols produced inconsistent phases")

    theta_true = np.stack(
        [np.asarray(history["theta_true"], dtype=float) for history in histories]
    )
    theta_hd_decoded = np.stack(
        [
            np.asarray(history["theta_hd_decoded"], dtype=float)
            for history in histories
        ]
    )
    theta_hd_decoded_peak = np.stack(
        [
            np.asarray(history["theta_hd_decoded_peak"], dtype=float)
            for history in histories
        ]
    )
    darkness_mask = np.isclose(common_phase_id, DARKNESS_PHASE_ID)
    if not np.any(darkness_mask):
        raise RuntimeError("constant PI protocols contain no darkness samples")
    absolute_decode_error = np.stack(
        [
            circular_error_trace(decoded_trace, true_trace)
            for decoded_trace, true_trace in zip(
                theta_hd_decoded,
                theta_true,
                strict=True,
            )
        ]
    )
    pi_error = np.full(theta_true.shape, np.nan, dtype=float)
    peak_pi_error = np.full(theta_true.shape, np.nan, dtype=float)
    pi_error[:, darkness_mask] = release_relative_pi_error_trace(
        theta_hd_decoded[:, darkness_mask],
        theta_true[:, darkness_mask],
        axis=1,
    )
    peak_pi_error[:, darkness_mask] = release_relative_pi_error_trace(
        theta_hd_decoded_peak[:, darkness_mask],
        theta_true[:, darkness_mask],
        axis=1,
    )
    # The release-relative accumulated error is only defined over the
    # darkness phase; cue and recue phases stay NaN and would erase those
    # segments from the grid figure.  Fill every NaN sample with the absolute
    # circular decode error so the whole visual-dark-visual window is drawn.
    missing_pva = np.isnan(pi_error)
    if np.any(missing_pva):
        pi_error[missing_pva] = absolute_decode_error[missing_pva]
    missing_peak = np.isnan(peak_pi_error)
    if np.any(missing_peak):
        peak_decode_error = circular_error_trace(
            theta_hd_decoded_peak,
            theta_true,
        )
        peak_pi_error[missing_peak] = peak_decode_error[missing_peak]
    wrapped_pi_error = np.asarray(wrap_angle(pi_error), dtype=float)
    decoded_velocity = np.asarray(decoded_velocities, dtype=float)
    decoded_peak_velocity = np.asarray(decoded_peak_velocities, dtype=float)
    velocity_bias = decoded_velocity - commanded_velocities
    moving_velocity_mask = ~np.isclose(
        commanded_velocities,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )
    velocity_gain = np.divide(
        decoded_velocity,
        commanded_velocities,
        out=np.full(decoded_velocity.shape, np.nan, dtype=float),
        where=moving_velocity_mask,
    )
    peak_velocity_gain = np.divide(
        decoded_peak_velocity,
        commanded_velocities,
        out=np.full(decoded_peak_velocity.shape, np.nan, dtype=float),
        where=moving_velocity_mask,
    )
    rms_pi_error = np.sqrt(np.nanmean(pi_error[:, darkness_mask] ** 2, axis=1))
    final_abs_pi_error = np.abs(pi_error[:, darkness_mask][:, -1])

    history = {
        "constant_pi_time": common_time,
        "constant_pi_phase_id": common_phase_id,
        "constant_pi_commanded_velocity": commanded_velocities,
        "constant_pi_decoded_velocity": decoded_velocity,
        "constant_pi_decoded_peak_velocity": decoded_peak_velocity,
        "constant_pi_velocity_bias": velocity_bias,
        "constant_pi_velocity_gain": velocity_gain,
        "constant_pi_peak_velocity_gain": peak_velocity_gain,
        "constant_pi_rms_error": rms_pi_error,
        "constant_pi_final_abs_error": final_abs_pi_error,
        "constant_pi_wrapped_rms_error": np.sqrt(
            np.nanmean(wrapped_pi_error[:, darkness_mask] ** 2, axis=1)
        ),
        "constant_pi_wrapped_final_abs_error": np.abs(
            wrapped_pi_error[:, darkness_mask][:, -1]
        ),
        "constant_pi_theta_true": theta_true,
        "constant_pi_theta_hd_decoded": theta_hd_decoded,
        "constant_pi_theta_hd_decoded_peak": theta_hd_decoded_peak,
        "constant_pi_error": pi_error,
        "constant_pi_peak_error": peak_pi_error,
        "constant_pi_absolute_decode_error": absolute_decode_error,
    }
    metrics = {
        "constant_pi_velocity_count": float(commanded_velocities.size),
        "constant_pi_rms_velocity_bias": float(
            np.sqrt(np.mean(velocity_bias**2))
        ),
        "constant_pi_max_abs_velocity_bias": float(
            np.max(np.abs(velocity_bias))
        ),
        "constant_pi_rms_gain_error": float(
            np.sqrt(np.mean((velocity_gain[moving_velocity_mask] - 1.0) ** 2))
            if np.any(moving_velocity_mask)
            else float("nan")
        ),
        "constant_pi_zero_velocity_probe_count": float(
            np.count_nonzero(~moving_velocity_mask)
        ),
        "constant_pi_zero_velocity_drift": (
            float(decoded_velocity[~moving_velocity_mask][0])
            if np.any(~moving_velocity_mask)
            else float("nan")
        ),
        "constant_pi_zero_velocity_abs_drift": (
            float(abs(decoded_velocity[~moving_velocity_mask][0]))
            if np.any(~moving_velocity_mask)
            else float("nan")
        ),
        "constant_pi_zero_velocity_drift_deg_s": (
            float(np.rad2deg(decoded_velocity[~moving_velocity_mask][0]))
            if np.any(~moving_velocity_mask)
            else float("nan")
        ),
        "constant_pi_rms_final_abs_error": float(
            np.sqrt(np.mean(final_abs_pi_error**2))
        ),
        "constant_pi_max_final_abs_error": float(np.max(final_abs_pi_error)),
        "constant_pi_rms_predicted_60s_drift_error": float(
            60.0 * np.sqrt(np.mean(velocity_bias**2))
        ),
    }
    return history, metrics


def summarize_zero_velocity_drive(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    bump_history: dict[str, np.ndarray],
) -> dict[str, float]:
    """Project zero-velocity HD and HR drives onto the bump tangent."""
    params = VafidisToyParams.from_config(config)
    cue_mask = phase_mask(bump_history, DARKNESS_PHASE_ID)
    if "r_hd" not in bump_history or not np.any(cue_mask):
        return {
            "zero_velocity_hd_tangent_drive": float("nan"),
            "zero_velocity_hr_tangent_drive": float("nan"),
            "zero_velocity_total_tangent_drive": float("nan"),
        }
    cue_end_index = int(np.flatnonzero(cue_mask)[0])
    r_hd = bump_history["r_hd"][cue_end_index]
    theta_hd_decoded = float(bump_history["theta_hd_decoded"][cue_end_index])
    tangent_basis = np.sin(trained_state.theta_hd_pref - theta_hd_decoded)
    tangent_norm = l2_norm(tangent_basis)
    if tangent_norm <= 1e-12:
        return {
            "zero_velocity_hd_tangent_drive": float("nan"),
            "zero_velocity_hr_tangent_drive": float("nan"),
            "zero_velocity_total_tangent_drive": float("nan"),
        }
    tangent_basis = tangent_basis / tangent_norm
    i_hr = compute_i_hr(
        w_hd_to_hr=trained_state.w_hd_to_hr,
        r_hd_to_hr_lp=r_hd,
        i_vel_to_hr=np.zeros(params.n_hr, dtype=float),
        b_hr=params.b_hr,
    )
    r_hr = apply_activation(
        i_hr,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )
    hd_drive, lhr_drive, rhr_drive = compute_hd_distal_pathway_drives(
        w_hd_to_hd=trained_state.w_hd_to_hd,
        r_hd=r_hd,
        w_hr_to_hd=trained_state.w_hr_to_hd,
        r_hr=r_hr,
        normalization=params.hd_distal_normalization,
    )
    hr_drive = lhr_drive + rhr_drive
    total_drive = hd_drive + hr_drive - params.b_hd
    return {
        "zero_velocity_hd_tangent_drive": float(np.dot(hd_drive, tangent_basis)),
        "zero_velocity_hr_tangent_drive": float(np.dot(hr_drive, tangent_basis)),
        "zero_velocity_total_tangent_drive": float(np.dot(total_drive, tangent_basis)),
    }


def _estimate_phase_decoded_velocity(
    *,
    history: dict[str, np.ndarray],
    phase_id: int,
    trace_name: str,
    start_fraction: float = 0.25,
) -> float:
    phase = phase_mask(history, phase_id)
    if np.count_nonzero(phase) < 3:
        return float("nan")
    phase_time = history["time"][phase]
    return estimate_decoded_velocity(
        time=phase_time - phase_time[0],
        theta_decoded=history[trace_name][phase],
        start_fraction=start_fraction,
    )


def _prefixed_velocity_gain_metrics(
    *,
    metric_prefix: str,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    decoded_peak_velocity: np.ndarray,
) -> dict[str, float]:
    gain_summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
    )
    tracking_summary = summarize_velocity_tracking(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
    )
    peak_gain_summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_peak_velocity,
    )
    peak_tracking_summary = summarize_velocity_tracking(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_peak_velocity,
    )
    return {
        f"{metric_prefix}_velocity_gain": gain_summary["gain"],
        f"{metric_prefix}_velocity_gain_abs_error": float(abs(gain_summary["gain"] - 1.0)),
        f"{metric_prefix}_velocity_gain_intercept": gain_summary["intercept"],
        f"{metric_prefix}_velocity_gain_intercept_abs": float(abs(gain_summary["intercept"])),
        f"{metric_prefix}_velocity_gain_r_squared": gain_summary["r_squared"],
        f"{metric_prefix}_velocity_gain_linear_fit_rmse": gain_summary["linear_fit_rmse"],
        f"{metric_prefix}_velocity_tracking_operating_range_max_abs_velocity": (
            estimate_velocity_tracking_operating_range(
                commanded_velocity=commanded_velocity,
                decoded_velocity=decoded_velocity,
            )
        ),
        **{f"{metric_prefix}_{key}": value for key, value in tracking_summary.items()},
        f"{metric_prefix}_velocity_gain_peak": peak_gain_summary["gain"],
        f"{metric_prefix}_velocity_gain_peak_abs_error": float(abs(peak_gain_summary["gain"] - 1.0)),
        f"{metric_prefix}_velocity_gain_peak_intercept": peak_gain_summary["intercept"],
        f"{metric_prefix}_velocity_gain_peak_intercept_abs": float(abs(peak_gain_summary["intercept"])),
        f"{metric_prefix}_velocity_gain_peak_r_squared": peak_gain_summary["r_squared"],
        f"{metric_prefix}_velocity_gain_peak_linear_fit_rmse": peak_gain_summary["linear_fit_rmse"],
        f"{metric_prefix}_peak_velocity_tracking_operating_range_max_abs_velocity": (
            estimate_velocity_tracking_operating_range(
                commanded_velocity=commanded_velocity,
                decoded_velocity=decoded_peak_velocity,
            )
        ),
        **{f"{metric_prefix}_peak_{key}": value for key, value in peak_tracking_summary.items()},
    }


def run_velocity_gain_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    commanded_velocity_values = np.asarray(config.tests.gain_velocities, dtype=float)
    visual_decoded_velocity_values: list[float] = []
    visual_decoded_peak_velocity_values: list[float] = []
    darkness_decoded_velocity_values: list[float] = []
    darkness_decoded_peak_velocity_values: list[float] = []
    for commanded_velocity in commanded_velocity_values:
        gain_history = run_constant_velocity_visual_dark_visual_protocol(
            config=config,
            trained_state=trained_state,
            theta_true=config.simulation.theta0,
            darkness_duration=config.simulation.darkness_test_duration,
            angular_velocity=float(commanded_velocity),
            cue_duration=get_pi_cue_duration(config),
            recue_duration=0.0,
        )
        visual_decoded_velocity = _estimate_phase_decoded_velocity(
            history=gain_history,
            phase_id=VISUAL_CUE_PHASE_ID,
            trace_name="theta_hd_decoded",
        )
        visual_decoded_peak_velocity = _estimate_phase_decoded_velocity(
            history=gain_history,
            phase_id=VISUAL_CUE_PHASE_ID,
            trace_name="theta_hd_decoded_peak",
        )
        darkness_decoded_velocity = _estimate_phase_decoded_velocity(
            history=gain_history,
            phase_id=DARKNESS_PHASE_ID,
            trace_name="theta_hd_decoded",
        )
        darkness_decoded_peak_velocity = _estimate_phase_decoded_velocity(
            history=gain_history,
            phase_id=DARKNESS_PHASE_ID,
            trace_name="theta_hd_decoded_peak",
        )
        visual_decoded_velocity_values.append(visual_decoded_velocity)
        visual_decoded_peak_velocity_values.append(visual_decoded_peak_velocity)
        darkness_decoded_velocity_values.append(darkness_decoded_velocity)
        darkness_decoded_peak_velocity_values.append(darkness_decoded_peak_velocity)
    visual_decoded_velocity_array = np.asarray(visual_decoded_velocity_values, dtype=float)
    visual_decoded_peak_velocity_array = np.asarray(visual_decoded_peak_velocity_values, dtype=float)
    darkness_decoded_velocity_array = np.asarray(darkness_decoded_velocity_values, dtype=float)
    darkness_decoded_peak_velocity_array = np.asarray(darkness_decoded_peak_velocity_values, dtype=float)
    visual_metrics = _prefixed_velocity_gain_metrics(
        metric_prefix="visual",
        commanded_velocity=commanded_velocity_values,
        decoded_velocity=visual_decoded_velocity_array,
        decoded_peak_velocity=visual_decoded_peak_velocity_array,
    )
    darkness_metrics = _prefixed_velocity_gain_metrics(
        metric_prefix="darkness",
        commanded_velocity=commanded_velocity_values,
        decoded_velocity=darkness_decoded_velocity_array,
        decoded_peak_velocity=darkness_decoded_peak_velocity_array,
    )
    return {
        "commanded_velocity": commanded_velocity_values,
        "decoded_velocity": darkness_decoded_velocity_array,
        "decoded_velocity_peak": darkness_decoded_peak_velocity_array,
        "decoded_velocity_darkness": darkness_decoded_velocity_array,
        "decoded_velocity_darkness_peak": darkness_decoded_peak_velocity_array,
        "decoded_velocity_visual": visual_decoded_velocity_array,
        "decoded_velocity_visual_peak": visual_decoded_peak_velocity_array,
    }, {
        **visual_metrics,
        **darkness_metrics,
        "darkness_minus_visual_velocity_gain": (
            darkness_metrics["darkness_velocity_gain"] - visual_metrics["visual_velocity_gain"]
        ),
        "darkness_minus_visual_velocity_gain_abs_error": (
            darkness_metrics["darkness_velocity_gain_abs_error"]
            - visual_metrics["visual_velocity_gain_abs_error"]
        ),
        "darkness_minus_visual_velocity_tracking_rmse": (
            darkness_metrics["darkness_velocity_tracking_rmse"]
            - visual_metrics["visual_velocity_tracking_rmse"]
        ),
        "velocity_gain": darkness_metrics["darkness_velocity_gain"],
        "velocity_gain_abs_error": darkness_metrics["darkness_velocity_gain_abs_error"],
        "velocity_gain_intercept": darkness_metrics["darkness_velocity_gain_intercept"],
        "velocity_gain_intercept_abs": darkness_metrics["darkness_velocity_gain_intercept_abs"],
        "velocity_gain_r_squared": darkness_metrics["darkness_velocity_gain_r_squared"],
        "velocity_gain_linear_fit_rmse": darkness_metrics[
            "darkness_velocity_gain_linear_fit_rmse"
        ],
        "velocity_tracking_operating_range_max_abs_velocity": darkness_metrics[
            "darkness_velocity_tracking_operating_range_max_abs_velocity"
        ],
        **{
            key.removeprefix("darkness_"): value
            for key, value in darkness_metrics.items()
            if key.startswith("darkness_velocity_tracking_")
            or key == "darkness_velocity_direction_match_fraction"
        },
        "velocity_gain_peak": darkness_metrics["darkness_velocity_gain_peak"],
        "velocity_gain_peak_abs_error": darkness_metrics["darkness_velocity_gain_peak_abs_error"],
        "velocity_gain_peak_intercept": darkness_metrics["darkness_velocity_gain_peak_intercept"],
        "velocity_gain_peak_intercept_abs": darkness_metrics["darkness_velocity_gain_peak_intercept_abs"],
        **{
            f"peak_{key.removeprefix('darkness_peak_')}": value
            for key, value in darkness_metrics.items()
            if key.startswith("darkness_peak_velocity_tracking_")
            or key == "darkness_peak_velocity_direction_match_fraction"
        },
    }


def run_ou_path_integration_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ou_history = run_ou_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.darkness_test_duration,
        cue_duration=get_pi_cue_duration(config),
        progress=progress,
        progress_label="OU PI single trial",
    )
    dark_mask = phase_mask(ou_history, DARKNESS_PHASE_ID)
    recue_mask = phase_mask(ou_history, VISUAL_RECUE_PHASE_ID)
    dark_theta_decoded = ou_history["theta_hd_decoded"][dark_mask]
    dark_theta_true = ou_history["theta_true"][dark_mask]
    dark_absolute_decode_error = circular_error_trace(
        dark_theta_decoded,
        dark_theta_true,
    )
    dark_pi_error = release_relative_pi_error_trace(
        dark_theta_decoded,
        dark_theta_true,
    )
    dark_wrapped_pi_error = np.asarray(wrap_angle(dark_pi_error), dtype=float)
    dark_peak_pi_error = release_relative_pi_error_trace(
        ou_history["theta_hd_decoded_peak"][dark_mask],
        dark_theta_true,
    )
    # Full-protocol accumulated PI error referenced to darkness onset, unwrapped
    # continuously over the whole cue-darkness-recue OU window.  This exposes
    # whether the recue visual anchor re-anchors the accumulated error to zero.
    full_ou_decoded = np.asarray(ou_history["theta_hd_decoded"], dtype=float)
    full_ou_true = np.asarray(ou_history["theta_true"], dtype=float)
    darkness_onset_index = int(np.flatnonzero(dark_mask)[0])
    ou_full_protocol_pi_error = full_protocol_pi_error_trace(
        full_ou_decoded,
        full_ou_true,
        reference_index=darkness_onset_index,
    )
    ou_full_protocol_pi_error_peak = full_protocol_pi_error_trace(
        np.asarray(ou_history["theta_hd_decoded_peak"], dtype=float),
        full_ou_true,
        reference_index=darkness_onset_index,
    )
    ou_history["pi_error_full_protocol"] = ou_full_protocol_pi_error
    ou_history["pi_error_full_protocol_wrapped"] = np.asarray(
        wrap_angle(ou_full_protocol_pi_error), dtype=float
    )
    ou_history["peak_pi_error_full_protocol"] = ou_full_protocol_pi_error_peak
    ou_history["pi_error_release_relative"] = np.full(
        ou_history["time"].shape,
        np.nan,
        dtype=float,
    )
    ou_history["pi_error_release_relative"][dark_mask] = dark_pi_error
    ou_history["pi_error_release_relative_wrapped"] = np.full(
        ou_history["time"].shape,
        np.nan,
        dtype=float,
    )
    ou_history["pi_error_release_relative_wrapped"][dark_mask] = (
        dark_wrapped_pi_error
    )
    ou_history["peak_pi_error_release_relative"] = np.full(
        ou_history["time"].shape,
        np.nan,
        dtype=float,
    )
    ou_history["peak_pi_error_release_relative"][dark_mask] = dark_peak_pi_error
    metrics = {
        "ou_darkness_rms_pi_error": float(
            np.sqrt(np.nanmean(np.square(dark_pi_error)))
        ),
        "ou_darkness_mean_abs_pi_error": nanmean_or_nan(np.abs(dark_pi_error)),
        "ou_darkness_final_abs_pi_error": float(abs(dark_pi_error[-1])),
        "ou_darkness_wrapped_rms_pi_error": float(
            np.sqrt(np.nanmean(np.square(dark_wrapped_pi_error)))
        ),
        "ou_darkness_wrapped_final_abs_pi_error": float(
            abs(dark_wrapped_pi_error[-1])
        ),
        "ou_darkness_absolute_decode_rms_error": float(
            np.sqrt(np.nanmean(np.square(dark_absolute_decode_error)))
        ),
        "ou_darkness_final_pva_strength": float(ou_history["pva_strength_hd"][dark_mask][-1])
        if np.any(dark_mask)
        else float("nan"),
        "ou_darkness_max_abs_angular_velocity": float(
            np.nanmax(np.abs(ou_history["angular_velocity"][dark_mask]))
        )
        if np.any(dark_mask)
        else float("nan"),
        "ou_darkness_rms_angular_velocity": float(
            np.sqrt(np.nanmean(ou_history["angular_velocity"][dark_mask] ** 2))
        )
        if np.any(dark_mask)
        else float("nan"),
        **summarize_hd_saturation(
            history=ou_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=dark_mask,
            metric_prefix="ou_darkness",
        ),
        **summarize_hd_near_peak(
            history=ou_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=dark_mask,
            metric_prefix="ou_darkness",
        ),
        **summarize_hd_local_peaks(
            history=ou_history,
            theta_hd_pref=trained_state.theta_hd_pref,
            mask=dark_mask,
            metric_prefix="ou_darkness",
        ),
    }
    if np.any(recue_mask):
        metrics.update(
            {
                "ou_darkness_recue_final_abs_pi_error": final_abs_circular_error(
                    ou_history["theta_hd_decoded"][recue_mask],
                    theta_reference=ou_history["theta_true"][recue_mask][-1],
                ),
                "ou_darkness_recue_rms_pi_error": rms_circular_error(
                    ou_history["theta_hd_decoded"][recue_mask],
                    ou_history["theta_true"][recue_mask],
                ),
                "ou_darkness_recue_final_pva_strength": float(
                    ou_history["pva_strength_hd"][recue_mask][-1]
                ),
            }
        )
    return ou_history, metrics


def run_hd_tuning_curve_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Measure frozen-weight visual tuning after per-heading steady settling.

    ``hd_tuning_curve_settle_duration`` is the minimum settling time.  When a
    convergence tolerance is configured, each heading continues until the
    maximum HD-rate change over the configured trailing window falls below the
    tolerance, or until ``hd_tuning_curve_max_settle_duration`` is reached.
    """
    params = VafidisToyParams.from_config(config)
    n_angles = int(config.tests.hd_tuning_curve_angles)
    minimum_settle_duration = float(config.tests.hd_tuning_curve_settle_duration)
    configured_maximum_settle_duration = config.tests.hd_tuning_curve_max_settle_duration
    maximum_settle_duration = (
        minimum_settle_duration
        if configured_maximum_settle_duration is None
        else float(configured_maximum_settle_duration)
    )
    convergence_window = float(config.tests.hd_tuning_curve_convergence_window)
    convergence_tolerance = config.tests.hd_tuning_curve_convergence_tolerance
    convergence_enabled = convergence_tolerance is not None
    if convergence_enabled:
        convergence_tolerance = float(convergence_tolerance)
    if n_angles < 8:
        raise ValueError("tests.hd_tuning_curve_angles must be at least 8")
    if minimum_settle_duration <= 0.0:
        raise ValueError("tests.hd_tuning_curve_settle_duration must be positive")
    if maximum_settle_duration < minimum_settle_duration:
        raise ValueError(
            "tests.hd_tuning_curve_max_settle_duration must be at least the minimum"
        )
    if convergence_window <= 0.0:
        raise ValueError("tests.hd_tuning_curve_convergence_window must be positive")
    if convergence_enabled and convergence_tolerance <= 0.0:
        raise ValueError("tests.hd_tuning_curve_convergence_tolerance must be positive")
    minimum_settle_steps = max(1, int(round(minimum_settle_duration / params.dt)))
    maximum_settle_steps = max(
        minimum_settle_steps,
        int(round(maximum_settle_duration / params.dt)),
    )
    convergence_window_steps = max(1, int(round(convergence_window / params.dt)))
    theta_grid = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=float)
    rate_rows: list[np.ndarray] = []
    distal_voltage_rows: list[np.ndarray] = []
    distal_current_rows: list[np.ndarray] = []
    visual_only_rate_rows: list[np.ndarray] = []
    decoded_heading: list[float] = []
    pva_strength: list[float] = []
    actual_settle_durations: list[float] = []
    convergence_flags: list[float] = []
    final_window_rate_changes: list[float] = []
    for theta_true in theta_grid:
        state = initialize_protocol_state(
            config=config,
            trained_state=trained_state,
            theta_true=float(theta_true),
        )
        visual_only_rate_rows.append(state.r_hd.copy())
        rate_window: list[np.ndarray] = [state.r_hd.copy()]
        heading_converged = False
        final_window_rate_change = float("nan")
        completed_steps = 0
        for step_index in range(maximum_settle_steps):
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=0.0,
                visual_teacher=True,
                training=False,
            )
            completed_steps = step_index + 1
            rate_window.append(state.r_hd.copy())
            if len(rate_window) > convergence_window_steps + 1:
                rate_window.pop(0)
            if len(rate_window) == convergence_window_steps + 1:
                final_window_rate_change = float(
                    np.max(np.abs(rate_window[-1] - rate_window[0]))
                )
            if (
                convergence_enabled
                and completed_steps >= minimum_settle_steps
                and np.isfinite(final_window_rate_change)
                and final_window_rate_change <= convergence_tolerance
            ):
                heading_converged = True
                break
        rate_rows.append(state.r_hd.copy())
        distal_voltage_rows.append(state.v_hd_distal.copy())
        distal_current_rows.append(state.i_hd_distal.copy())
        decoded_heading.append(state.theta_hd_decoded)
        pva_strength.append(float(pva_vector_strength(state.theta_hd_pref, state.r_hd)))
        actual_settle_durations.append(float(completed_steps * params.dt))
        convergence_flags.append(float(heading_converged) if convergence_enabled else float("nan"))
        final_window_rate_changes.append(final_window_rate_change)
    r_hd_by_heading = np.asarray(rate_rows, dtype=float)
    r_hd_visual_only_by_heading = np.asarray(visual_only_rate_rows, dtype=float)
    empirical_preference, tuning_strength = empirical_tuning_preferred_directions(
        theta_true=theta_grid,
        r_hd_by_heading=r_hd_by_heading,
    )
    visual_only_preference, visual_only_tuning_strength = empirical_tuning_preferred_directions(
        theta_true=theta_grid,
        r_hd_by_heading=r_hd_visual_only_by_heading,
    )
    centered_post_training = r_hd_by_heading - np.mean(r_hd_by_heading, axis=0, keepdims=True)
    centered_visual_only = r_hd_visual_only_by_heading - np.mean(
        r_hd_visual_only_by_heading,
        axis=0,
        keepdims=True,
    )
    shape_correlation_denominator = np.sqrt(
        np.sum(centered_post_training**2, axis=0)
        * np.sum(centered_visual_only**2, axis=0)
    )
    post_vs_visual_only_correlation = np.divide(
        np.sum(centered_post_training * centered_visual_only, axis=0),
        shape_correlation_denominator,
        out=np.full(r_hd_by_heading.shape[1], np.nan, dtype=float),
        where=shape_correlation_denominator > 1e-12,
    )
    preference_error = np.abs(
        circular_difference(empirical_preference, trained_state.theta_hd_pref)
    )
    empirical_sort_order = np.argsort(
        np.where(np.isfinite(empirical_preference), empirical_preference, np.inf),
        kind="stable",
    )
    history = {
        "theta_true": theta_grid,
        "r_hd": r_hd_by_heading,
        "v_hd_distal": np.asarray(distal_voltage_rows, dtype=float),
        "i_hd_distal": np.asarray(distal_current_rows, dtype=float),
        "r_hd_visual_only": r_hd_visual_only_by_heading,
        "theta_hd_decoded": np.asarray(decoded_heading, dtype=float),
        "pva_strength_hd": np.asarray(pva_strength, dtype=float),
        "empirical_preferred_direction": empirical_preference,
        "empirical_tuning_strength": tuning_strength,
        "empirical_sort_order": empirical_sort_order,
        "visual_only_preferred_direction": visual_only_preference,
        "visual_only_tuning_strength": visual_only_tuning_strength,
        "post_vs_visual_only_tuning_correlation": post_vs_visual_only_correlation,
        "actual_settle_duration": np.asarray(actual_settle_durations, dtype=float),
        "settle_converged": np.asarray(convergence_flags, dtype=float),
        "final_window_max_rate_change": np.asarray(
            final_window_rate_changes,
            dtype=float,
        ),
    }
    finite_shape_correlation = post_vs_visual_only_correlation[
        np.isfinite(post_vs_visual_only_correlation)
    ]
    finite_final_window_rate_changes = np.asarray(final_window_rate_changes, dtype=float)
    finite_final_window_rate_changes = finite_final_window_rate_changes[
        np.isfinite(finite_final_window_rate_changes)
    ]
    convergence_fraction = (
        float(np.mean(np.asarray(convergence_flags, dtype=float)))
        if convergence_enabled
        else float("nan")
    )
    metrics = {
        "hd_tuning_curve_n_angles": float(n_angles),
        "hd_tuning_curve_settle_duration": minimum_settle_duration,
        "hd_tuning_curve_max_settle_duration": maximum_settle_duration,
        "hd_tuning_curve_convergence_window": convergence_window,
        "hd_tuning_curve_convergence_tolerance": (
            float(convergence_tolerance) if convergence_enabled else float("nan")
        ),
        "hd_tuning_curve_converged_fraction": convergence_fraction,
        "hd_tuning_curve_mean_actual_settle_duration": float(
            np.mean(actual_settle_durations)
        ),
        "hd_tuning_curve_max_actual_settle_duration": float(
            np.max(actual_settle_durations)
        ),
        "hd_tuning_curve_max_final_window_rate_change": float(
            np.max(finite_final_window_rate_changes)
            if finite_final_window_rate_changes.size
            else float("nan")
        ),
        "hd_tuning_mean_com_strength": float(np.nanmean(tuning_strength)),
        "hd_tuning_visual_only_mean_com_strength": float(
            np.nanmean(visual_only_tuning_strength)
        ),
        "hd_tuning_post_vs_visual_only_median_correlation": (
            float(np.median(finite_shape_correlation))
            if finite_shape_correlation.size
            else float("nan")
        ),
        "hd_tuning_post_vs_visual_only_p10_correlation": (
            float(np.quantile(finite_shape_correlation, 0.10))
            if finite_shape_correlation.size
            else float("nan")
        ),
        "hd_tuning_median_abs_preference_error": float(np.nanmedian(preference_error)),
        "hd_tuning_median_abs_preference_error_deg": float(
            np.rad2deg(np.nanmedian(preference_error))
        ),
        "hd_tuning_empirical_sort_moved_fraction": float(
            np.mean(empirical_sort_order != np.arange(empirical_sort_order.size))
        ),
    }
    return history, metrics


def run_ou_path_integration_ensemble_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Average PVA/COM PI errors across independent frozen-weight OU trials."""
    trial_count = int(config.tests.ou_pi_ensemble_trials)
    if trial_count <= 0:
        raise ValueError("tests.ou_pi_ensemble_trials must be positive")
    error_trials: list[np.ndarray] = []
    darkness_time: np.ndarray | None = None
    trial_progress = (
        range(trial_count)
        if progress is not None
        else trange(
            trial_count,
            disable=not config.simulation.progress,
            desc="OU PI ensemble",
            unit="trial",
        )
    )
    for trial_index in trial_progress:
        history = run_ou_visual_dark_visual_protocol(
            config=config,
            trained_state=trained_state,
            theta_true=config.simulation.theta0,
            darkness_duration=config.simulation.darkness_test_duration,
            cue_duration=get_pi_cue_duration(config),
            recue_duration=0.0,
            protocol_seed=(
                config.simulation.seed
                + int(config.tests.ou_pi_ensemble_seed_offset)
                + trial_index
            ),
            progress=progress,
            progress_label=f"OU PI ensemble {trial_index + 1}/{trial_count}",
        )
        dark_mask = phase_mask(history, DARKNESS_PHASE_ID)
        trial_time = np.asarray(history["time"][dark_mask], dtype=float)
        trial_time = trial_time - float(trial_time[0])
        if darkness_time is None:
            darkness_time = trial_time
        elif trial_time.shape != darkness_time.shape or not np.allclose(trial_time, darkness_time):
            raise RuntimeError("OU PI ensemble trials must share the same time grid")
        # PVA/COM decode only: peak decode is intentionally excluded here.
        trial_error = release_relative_pi_error_trace(
            history["theta_hd_decoded"][dark_mask],
            history["theta_true"][dark_mask],
        )
        error_trials.append(np.asarray(trial_error, dtype=float))
    if darkness_time is None:
        raise RuntimeError("OU PI ensemble produced no darkness samples")
    pi_error = np.asarray(error_trials, dtype=float)
    summary = summarize_pi_error_ensemble(
        time=darkness_time,
        pi_error=pi_error,
        fit_start_time=float(config.tests.ou_pi_ensemble_fit_start_time),
    )
    diffusion_summary = summarize_ensemble_diffusion_trajectories(
        time=darkness_time,
        angular_displacement=pi_error,
        fit_start_time=float(config.tests.ou_pi_ensemble_fit_start_time),
    )
    history = {
        "time": np.asarray(summary["time"], dtype=float),
        "pva_pi_error": pi_error,
        "pva_pi_error_mean": np.asarray(summary["pi_error_mean"], dtype=float),
        "pva_pi_error_std": np.asarray(summary["pi_error_std"], dtype=float),
        "pva_pi_error_sem": np.asarray(summary["pi_error_sem"], dtype=float),
        "pva_pi_error_variance": np.asarray(
            diffusion_summary["displacement_variance_trace"],
            dtype=float,
        ),
        "pva_pi_error_diffusion_coefficient": np.asarray(
            diffusion_summary["diffusion_coefficient_trace"],
            dtype=float,
        ),
    }
    metrics = {
        "ou_pi_ensemble_n_trials": summary["n_trials"],
        "ou_pi_ensemble_systematic_drift_velocity": summary["systematic_drift_velocity"],
        "ou_pi_ensemble_systematic_drift_velocity_deg_s": float(
            np.rad2deg(summary["systematic_drift_velocity"])
        ),
        "ou_pi_ensemble_drift_intercept": summary["drift_intercept"],
        "ou_pi_ensemble_rms_mean_pi_error": summary["rms_mean_pi_error"],
        "ou_pi_ensemble_final_mean_pi_error": summary["final_mean_pi_error"],
        "ou_pi_ensemble_final_pi_error_std": summary["final_pi_error_std"],
        "ou_pi_ensemble_effective_diffusion_coefficient": (
            diffusion_summary["diffusion_coefficient"]
        ),
        "ou_pi_ensemble_effective_diffusion_coefficient_deg2_s": float(
            np.rad2deg(1.0) ** 2
            * diffusion_summary["diffusion_coefficient"]
        ),
        "ou_pi_ensemble_anomalous_diffusion_exponent": (
            diffusion_summary["anomalous_diffusion_exponent"]
        ),
        "ou_pi_ensemble_fit_start_time": float(config.tests.ou_pi_ensemble_fit_start_time),
    }
    return history, metrics


def _visual_dark_visual_protocol_step_count(
    *,
    config: ExperimentConfig,
    cue_duration: float,
    darkness_duration: float,
    recue_duration: float,
) -> int:
    dt = float(config.simulation.dt)
    return sum(
        int(round(max(float(duration), 0.0) / dt))
        for duration in (cue_duration, darkness_duration, recue_duration)
    )


def run_path_integration_diagnostics(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    enabled: frozenset[str],
    diagnostic_recorder: IncrementalDiagnosticRecorder | None = None,
    continue_on_error: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
    dict[str, float],
]:
    """Run the PI diagnostic group under one step-based ETA progress bar."""
    pi_jobs = {
        "darkness_path_integration",
        "ou_path_integration",
        "ou_pi_ensemble",
    }
    active_jobs = enabled & pi_jobs
    if not active_jobs:
        return (
            {},
            {"darkness_path_integration_diagnostic_enabled": 0.0},
            {},
            {"ou_path_integration_diagnostic_enabled": 0.0},
            {},
            {"ou_pi_ensemble_diagnostic_enabled": 0.0},
        )

    cue_duration = get_pi_cue_duration(config)
    darkness_duration = float(config.simulation.darkness_test_duration)
    recue_duration = float(config.simulation.recue_duration)
    full_protocol_steps = _visual_dark_visual_protocol_step_count(
        config=config,
        cue_duration=cue_duration,
        darkness_duration=darkness_duration,
        recue_duration=recue_duration,
    )
    total_steps = 0
    if "darkness_path_integration" in active_jobs:
        commanded_velocities = np.asarray(
            config.tests.constant_pi_velocities,
            dtype=float,
        )
        primary_is_reused = bool(
            commanded_velocities.ndim == 1
            and np.any(
                np.isclose(
                    commanded_velocities,
                    float(config.tests.darkness_angular_velocity),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        )
        constant_protocol_count = 1 + int(commanded_velocities.size) - int(
            primary_is_reused
        )
        total_steps += constant_protocol_count * full_protocol_steps
    if "ou_path_integration" in active_jobs:
        total_steps += full_protocol_steps
    if "ou_pi_ensemble" in active_jobs:
        ensemble_protocol_steps = _visual_dark_visual_protocol_step_count(
            config=config,
            cue_duration=cue_duration,
            darkness_duration=darkness_duration,
            recue_duration=0.0,
        )
        total_steps += (
            int(config.tests.ou_pi_ensemble_trials) * ensemble_protocol_steps
        )

    progress = tqdm(
        total=total_steps,
        disable=not config.simulation.progress,
        desc="path integration diagnostics",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    with progress:
        if "darkness_path_integration" in active_jobs:
            primary_velocity = float(config.tests.darkness_angular_velocity)
            darkness_history, darkness_metrics = _run_isolated_diagnostic(
                name="darkness_path_integration",
                operation=lambda: run_darkness_path_integration_test(
                    config=config,
                    trained_state=trained_state,
                    angular_velocity=primary_velocity,
                    progress=progress,
                    progress_label=(
                        "constant PI primary "
                        f"{np.rad2deg(primary_velocity):+.0f} deg/s"
                    ),
                ),
                recorder=diagnostic_recorder,
                continue_on_error=continue_on_error,
            )
            if darkness_history:
                try:
                    constant_pi_history, constant_pi_metrics = (
                        run_constant_velocity_pi_error_sweep_test(
                            config=config,
                            trained_state=trained_state,
                            primary_history=darkness_history,
                            primary_metrics=darkness_metrics,
                            progress=progress,
                        )
                    )
                except Exception as error:
                    if diagnostic_recorder is not None:
                        diagnostic_recorder.record_failure(
                            "constant_velocity_pi_sweep",
                            error,
                        )
                    if not continue_on_error:
                        raise
                    darkness_metrics["constant_velocity_pi_sweep_failed"] = 1.0
                else:
                    darkness_history.update(constant_pi_history)
                    darkness_metrics.update(constant_pi_metrics)
                    if diagnostic_recorder is not None:
                        diagnostic_recorder.record_success(
                            "darkness_path_integration",
                            darkness_history,
                            darkness_metrics,
                        )
                        diagnostic_recorder.record_success(
                            "constant_velocity_pi_sweep",
                            {},
                            constant_pi_metrics,
                        )
            elif diagnostic_recorder is not None:
                diagnostic_recorder.record_blocked(
                    "constant_velocity_pi_sweep",
                    dependency="darkness_path_integration",
                )
        else:
            darkness_history = {}
            darkness_metrics = {
                "darkness_path_integration_diagnostic_enabled": 0.0
            }

        if "ou_path_integration" in active_jobs:
            ou_darkness_history, ou_darkness_metrics = _run_isolated_diagnostic(
                name="ou_path_integration",
                operation=lambda: run_ou_path_integration_test(
                    config=config,
                    trained_state=trained_state,
                    progress=progress,
                ),
                recorder=diagnostic_recorder,
                continue_on_error=continue_on_error,
            )
        else:
            ou_darkness_history = {}
            ou_darkness_metrics = {
                "ou_path_integration_diagnostic_enabled": 0.0
            }

        if "ou_pi_ensemble" in active_jobs:
            ou_pi_ensemble_history, ou_pi_ensemble_metrics = (
                _run_isolated_diagnostic(
                    name="ou_pi_ensemble",
                    operation=lambda: run_ou_path_integration_ensemble_test(
                        config=config,
                        trained_state=trained_state,
                        progress=progress,
                    ),
                    recorder=diagnostic_recorder,
                    continue_on_error=continue_on_error,
                )
            )
        else:
            ou_pi_ensemble_history = {}
            ou_pi_ensemble_metrics = {
                "ou_pi_ensemble_diagnostic_enabled": 0.0
            }

    return (
        darkness_history,
        darkness_metrics,
        ou_darkness_history,
        ou_darkness_metrics,
        ou_pi_ensemble_history,
        ou_pi_ensemble_metrics,
    )


def run_weight_performance_selection(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Evaluate saved snapshots and identify the best frozen PI weights."""

    saved_snapshot_time = np.asarray(
        (weight_history or {}).get("time", np.empty(0)),
        dtype=float,
    )
    selected_snapshot_count = select_weight_snapshot_indices(
        snapshot_time=saved_snapshot_time,
        interval_fraction=config.tests.weight_snapshot_pi_interval_fraction,
    ).size
    snapshot_count = max(1, int(selected_snapshot_count))
    initial_heading_count = max(
        1,
        len(config.tests.weight_snapshot_pi_initial_headings),
    )
    snapshot_pi_steps = snapshot_count * initial_heading_count * len(
        config.tests.weight_snapshot_pi_velocities
    ) * (
        int(
            round(
                config.tests.weight_snapshot_pi_cue_duration
                / config.simulation.dt
            )
        )
        + max(
            1,
            int(
                round(
                    config.tests.weight_snapshot_pi_duration
                    / config.simulation.dt
                )
            ),
        )
    )
    progress = tqdm(
        total=snapshot_pi_steps,
        disable=not config.simulation.progress,
        desc="select best frozen weights",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
    )
    with progress:
        return run_weight_snapshot_pi_development_diagnostic(
            config=config,
            trained_state=trained_state,
            weight_history=weight_history,
            progress=progress,
        )


def state_from_weight_snapshot(
    *,
    template_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None,
    snapshot_index: int,
) -> VafidisToyState:
    """Copy a template state and install one full-history weight snapshot."""

    weight_history = weight_history or {}
    snapshot_time = np.asarray(weight_history.get("time", np.empty(0)), dtype=float)
    w_hd_history = np.asarray(
        weight_history.get("w_hd_to_hd", np.empty((0, 0, 0))), dtype=float
    )
    w_hr_history = np.asarray(
        weight_history.get("w_hr_to_hd", np.empty((0, 0, 0))), dtype=float
    )
    if snapshot_time.size == 0:
        if snapshot_index not in (0, -1):
            raise ValueError(
                "weight snapshot selection requires a non-empty weight_history"
            )
        return template_state.copy()
    snapshot_count = snapshot_time.size
    resolved_index = int(snapshot_index)
    if resolved_index < 0:
        resolved_index += snapshot_count
    if resolved_index < 0 or resolved_index >= snapshot_count:
        raise ValueError(
            f"weight snapshot index {snapshot_index} is outside [0, "
            f"{snapshot_count - 1}]"
        )
    if (
        w_hd_history.shape
        != (snapshot_count, *template_state.w_hd_to_hd.shape)
        or w_hr_history.shape
        != (snapshot_count, *template_state.w_hr_to_hd.shape)
    ):
        raise ValueError("weight_history matrices do not match the template state")
    selected_state = template_state.copy()
    selected_state.w_hd_to_hd = w_hd_history[resolved_index].copy()
    selected_state.w_hr_to_hd = w_hr_history[resolved_index].copy()
    selected_state.time = float(snapshot_time[resolved_index])
    return selected_state


def resolve_weight_states(
    *,
    config: ExperimentConfig,
    training_selected_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None,
    weight_selection_history: dict[str, np.ndarray],
) -> tuple[VafidisToyState, VafidisToyState, VafidisToyState, dict[str, np.ndarray]]:
    """Resolve final, performance-best, and configured diagnostic states."""

    validate_diagnostic_weight_selection(config)
    weight_history = weight_history or {}
    saved_time = np.asarray(weight_history.get("time", np.empty(0)), dtype=float)
    final_snapshot_index = int(saved_time.size - 1) if saved_time.size else 0
    final_state = state_from_weight_snapshot(
        template_state=training_selected_state,
        weight_history=weight_history,
        snapshot_index=final_snapshot_index,
    )

    selected_source_index = np.asarray(
        weight_selection_history.get("snapshot_source_index", np.asarray([0])),
        dtype=int,
    )
    best_selected_index = int(
        np.asarray(
            weight_selection_history.get("best_snapshot_index", np.asarray(0))
        )
    )
    if (
        selected_source_index.ndim != 1
        or best_selected_index < 0
        or best_selected_index >= selected_source_index.size
    ):
        raise ValueError("weight selection history contains an invalid best index")
    best_source_index = int(selected_source_index[best_selected_index])
    best_state = state_from_weight_snapshot(
        template_state=training_selected_state,
        weight_history=weight_history,
        snapshot_index=best_source_index,
    )

    source = str(config.diagnostics.weight_source).lower()
    configured_snapshot_index = config.diagnostics.weight_snapshot_index
    if source == "best":
        diagnostic_state = best_state.copy()
        diagnostic_source_index = best_source_index
    elif source == "final":
        diagnostic_state = final_state.copy()
        diagnostic_source_index = final_snapshot_index
    elif source == "training_selected":
        diagnostic_state = training_selected_state.copy()
        diagnostic_source_index = -1
    elif source == "snapshot":
        if configured_snapshot_index is None:
            raise ValueError(
                "diagnostics.weight_snapshot_index is required for snapshot source"
            )
        diagnostic_state = state_from_weight_snapshot(
            template_state=training_selected_state,
            weight_history=weight_history,
            snapshot_index=int(configured_snapshot_index),
        )
        diagnostic_source_index = int(configured_snapshot_index)
        if diagnostic_source_index < 0 and saved_time.size:
            diagnostic_source_index += saved_time.size
    else:
        raise ValueError(
            "diagnostics.weight_source must be best, final, training_selected, "
            "or snapshot"
        )

    selection_metadata = {
        "best_snapshot_source_index": np.asarray(best_source_index, dtype=int),
        "best_snapshot_time": np.asarray(float(best_state.time)),
        "final_snapshot_source_index": np.asarray(final_snapshot_index, dtype=int),
        "final_snapshot_time": np.asarray(float(final_state.time)),
        "diagnostic_weight_source": np.asarray(source),
        "diagnostic_snapshot_source_index": np.asarray(
            diagnostic_source_index, dtype=int
        ),
        "diagnostic_snapshot_time": np.asarray(float(diagnostic_state.time)),
    }
    return final_state, best_state, diagnostic_state, selection_metadata


def validate_diagnostic_weight_selection(config: ExperimentConfig) -> None:
    """Validate the configured frozen-weight source before expensive work."""

    source = str(config.diagnostics.weight_source).lower()
    if source not in {"best", "final", "training_selected", "snapshot"}:
        raise ValueError(
            "diagnostics.weight_source must be best, final, training_selected, "
            "or snapshot"
        )
    snapshot_index = config.diagnostics.weight_snapshot_index
    if snapshot_index is not None and (
        not isinstance(snapshot_index, int) or isinstance(snapshot_index, bool)
    ):
        raise ValueError("diagnostics.weight_snapshot_index must be an integer or null")
    if source == "snapshot" and snapshot_index is None:
        raise ValueError(
            "diagnostics.weight_snapshot_index is required for snapshot source"
        )


DIAGNOSTIC_HISTORY_FILENAMES = {
    "hd_tuning": "hd_tuning_history.npz",
    "bump_maintenance": "bump_history.npz",
    "bump_attractor_trajectories": "bump_attractor_trajectory_history.npz",
    "slow_manifold": "slow_manifold_diagnostics.npz",
    "timescale_separation": "timescale_separation_history.npz",
    "velocity_trajectory_sweep": "velocity_trajectory_sweep_history.npz",
    "bump_diffusion": "bump_diffusion_history.npz",
    "darkness_path_integration": "darkness_history.npz",
    "ou_path_integration": "ou_darkness_history.npz",
    "ou_pi_ensemble": "ou_pi_ensemble_history.npz",
    "velocity_gain": "velocity_gain_history.npz",
    "weight_snapshot_pi_development": "weight_snapshot_pi_development.npz",
    "numerical_convergence": "numerical_convergence_history.npz",
}


class IncrementalDiagnosticRecorder:
    """Persist each diagnostic result before the next diagnostic starts."""

    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.metrics: dict[str, float] = {}
        self.status: dict[str, dict[str, Any]] = {}

    def _flush(self) -> None:
        save_json(self.run_dir / "test_metrics.partial.json", self.metrics)
        save_json(
            self.run_dir / "diagnostic_status.json",
            {
                "diagnostics": self.status,
                "completed_count": sum(
                    entry["status"] == "completed"
                    for entry in self.status.values()
                ),
                "failed_count": sum(
                    entry["status"] == "failed"
                    for entry in self.status.values()
                ),
                "blocked_count": sum(
                    entry["status"] == "blocked"
                    for entry in self.status.values()
                ),
            },
        )

    def record_success(
        self,
        name: str,
        history: dict[str, np.ndarray],
        metrics: dict[str, float],
    ) -> None:
        filename = DIAGNOSTIC_HISTORY_FILENAMES.get(name)
        if filename is not None and history:
            save_npz(self.run_dir / filename, **history)
        self.metrics.update(metrics)
        self.status[name] = {
            "status": "completed",
            "history_file": filename,
        }
        self._flush()

    def record_started(self, name: str) -> None:
        self.status[name] = {"status": "running"}
        self._flush()

    def record_failure(self, name: str, error: Exception) -> None:
        self.metrics[f"{name}_diagnostic_failed"] = 1.0
        self.status[name] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        self._flush()

    def record_blocked(self, name: str, *, dependency: str) -> None:
        self.metrics[f"{name}_diagnostic_blocked"] = 1.0
        self.status[name] = {
            "status": "blocked",
            "dependency": dependency,
        }
        self._flush()

    def finalize(self, metrics: dict[str, float]) -> None:
        metrics["diagnostic_failure_count"] = float(
            sum(
                entry["status"] == "failed"
                for entry in self.status.values()
            )
        )
        metrics["diagnostic_blocked_count"] = float(
            sum(
                entry["status"] == "blocked"
                for entry in self.status.values()
            )
        )
        self.metrics.update(metrics)
        self._flush()


def _run_isolated_diagnostic(
    *,
    name: str,
    operation: Callable[
        [], tuple[dict[str, np.ndarray], dict[str, float]]
    ],
    recorder: IncrementalDiagnosticRecorder | None,
    continue_on_error: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if recorder is not None:
        recorder.record_started(name)
    try:
        history, metrics = operation()
    except Exception as error:
        if recorder is not None:
            recorder.record_failure(name, error)
        if not continue_on_error:
            raise
        return {}, {f"{name}_diagnostic_failed": 1.0}
    if recorder is not None:
        recorder.record_success(name, history, metrics)
    return history, metrics


def run_all_tests(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    weight_history: dict[str, np.ndarray] | None = None,
    cached_histories: dict[str, dict[str, np.ndarray]] | None = None,
    precomputed_weight_snapshot_pi: tuple[
        dict[str, np.ndarray], dict[str, float]
    ]
    | None = None,
    diagnostic_recorder: IncrementalDiagnosticRecorder | None = None,
    continue_on_error: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    enabled = selected_diagnostics(config)
    cached_histories = cached_histories or {}

    def run_selected_diagnostic(
        name: str,
        operation: Callable[
            [], tuple[dict[str, np.ndarray], dict[str, float]]
        ],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        return _run_isolated_diagnostic(
            name=name,
            operation=operation,
            recorder=diagnostic_recorder,
            continue_on_error=continue_on_error,
        )

    hd_tuning_dependents = {
        "bump_attractor_trajectories",
        "timescale_separation",
        "velocity_trajectory_sweep",
    }
    hd_tuning_required = bool(enabled & hd_tuning_dependents)
    cached_hd_tuning = cached_histories.get("hd_tuning", {})
    cached_hd_tuning_is_usable = all(
        key in cached_hd_tuning for key in ("theta_true", "r_hd")
    )
    if "hd_tuning" in enabled:
        hd_tuning_history, hd_tuning_metrics = run_selected_diagnostic(
            "hd_tuning",
            lambda: run_hd_tuning_curve_test(
                config=config,
                trained_state=trained_state,
            ),
        )
    elif (
        hd_tuning_required
        and config.diagnostics.reuse_cached_dependencies
        and cached_hd_tuning_is_usable
    ):
        hd_tuning_history = cached_hd_tuning
        hd_tuning_metrics = {
            "hd_tuning_diagnostic_enabled": 0.0,
            "hd_tuning_dependency_reused": 1.0,
        }
        if diagnostic_recorder is not None:
            diagnostic_recorder.record_success(
                "hd_tuning", hd_tuning_history, hd_tuning_metrics
            )
    elif hd_tuning_required:
        def compute_hd_tuning_dependency():
            history, metrics = run_hd_tuning_curve_test(
                config=config,
                trained_state=trained_state,
            )
            metrics["hd_tuning_dependency_computed"] = 1.0
            return history, metrics

        hd_tuning_history, hd_tuning_metrics = run_selected_diagnostic(
            "hd_tuning",
            compute_hd_tuning_dependency,
        )
    else:
        hd_tuning_history = {}
        hd_tuning_metrics = {"hd_tuning_diagnostic_enabled": 0.0}

    if "bump_maintenance" in enabled:
        bump_history, bump_metrics = run_selected_diagnostic(
            "bump_maintenance",
            lambda: run_bump_maintenance_test(
                config=config,
                trained_state=trained_state,
            ),
        )
    else:
        bump_history = {}
        bump_metrics = {"bump_maintenance_diagnostic_enabled": 0.0}

    trajectory_dependency_required = bool(
        enabled & {"slow_manifold", "timescale_separation"}
    )
    cached_trajectory_history = cached_histories.get(
        "bump_attractor_trajectories",
        {},
    )
    cached_trajectory_is_usable = np.asarray(
        cached_trajectory_history.get("time", np.empty(0)),
        dtype=float,
    ).size > 0
    reuse_cached_trajectory = (
        "bump_attractor_trajectories" not in enabled
        and trajectory_dependency_required
        and config.diagnostics.reuse_cached_dependencies
        and cached_trajectory_is_usable
    )
    if reuse_cached_trajectory:
        bump_attractor_trajectory_history = cached_trajectory_history
        bump_attractor_trajectory_metrics = {
            "bump_attractor_dependency_reused": 1.0
        }
        if diagnostic_recorder is not None:
            diagnostic_recorder.record_success(
                "bump_attractor_trajectories",
                bump_attractor_trajectory_history,
                bump_attractor_trajectory_metrics,
            )
    elif "bump_attractor_trajectories" in enabled or trajectory_dependency_required:
        def compute_bump_attractor_trajectories():
            history, metrics = run_bump_attractor_trajectory_test(
                config=config,
                trained_state=trained_state,
                hd_tuning_history=hd_tuning_history,
                as_dependency=(
                    trajectory_dependency_required
                    and "bump_attractor_trajectories" not in enabled
                ),
            )
            if (
                trajectory_dependency_required
                and "bump_attractor_trajectories" not in enabled
            ):
                metrics["bump_attractor_dependency_computed"] = 1.0
            return history, metrics

        (
            bump_attractor_trajectory_history,
            bump_attractor_trajectory_metrics,
        ) = run_selected_diagnostic(
            "bump_attractor_trajectories",
            compute_bump_attractor_trajectories,
        )
    else:
        (
            bump_attractor_trajectory_history,
            bump_attractor_trajectory_metrics,
        ) = run_bump_attractor_trajectory_test(
            config=config,
            trained_state=trained_state,
            hd_tuning_history=hd_tuning_history,
        )

    if "slow_manifold" in enabled:
        slow_manifold_history, slow_manifold_metrics = run_selected_diagnostic(
            "slow_manifold",
            lambda: run_slow_manifold_diagnostic(
                config=config,
                trained_state=trained_state,
                bump_attractor_trajectory_history=(
                    bump_attractor_trajectory_history
                ),
            ),
        )
    else:
        slow_manifold_history, slow_manifold_metrics = run_slow_manifold_diagnostic(
            config=config,
            trained_state=trained_state,
            bump_attractor_trajectory_history=bump_attractor_trajectory_history,
        )

    if "timescale_separation" in enabled:
        (
            timescale_separation_history,
            timescale_separation_metrics,
        ) = run_selected_diagnostic(
            "timescale_separation",
            lambda: run_timescale_separation_test(
                config=config,
                trained_state=trained_state,
                hd_tuning_history=hd_tuning_history,
                bump_attractor_trajectory_history=(
                    bump_attractor_trajectory_history
                ),
            ),
        )
    else:
        (
            timescale_separation_history,
            timescale_separation_metrics,
        ) = run_timescale_separation_test(
            config=config,
            trained_state=trained_state,
            hd_tuning_history=hd_tuning_history,
            bump_attractor_trajectory_history=bump_attractor_trajectory_history,
        )

    if "velocity_trajectory_sweep" in enabled:
        (
            velocity_trajectory_sweep_history,
            velocity_trajectory_sweep_metrics,
        ) = run_selected_diagnostic(
            "velocity_trajectory_sweep",
            lambda: run_velocity_trajectory_sweep_test(
                config=config,
                trained_state=trained_state,
                hd_tuning_history=hd_tuning_history,
            ),
        )
    else:
        (
            velocity_trajectory_sweep_history,
            velocity_trajectory_sweep_metrics,
        ) = run_velocity_trajectory_sweep_test(
            config=config,
            trained_state=trained_state,
            hd_tuning_history=hd_tuning_history,
        )

    if "bump_diffusion" in enabled:
        bump_diffusion_history, bump_ensemble_diffusion_metrics = (
            run_selected_diagnostic(
                "bump_diffusion",
                lambda: run_bump_diffusion_ensemble_test(
                    config=config,
                    trained_state=trained_state,
                ),
            )
        )
    else:
        bump_diffusion_history = {}
        bump_ensemble_diffusion_metrics = {
            "bump_diffusion_diagnostic_enabled": 0.0
        }

    (
        darkness_history,
        darkness_metrics,
        ou_darkness_history,
        ou_darkness_metrics,
        ou_pi_ensemble_history,
        ou_pi_ensemble_metrics,
    ) = run_path_integration_diagnostics(
        config=config,
        trained_state=trained_state,
        enabled=enabled,
        diagnostic_recorder=diagnostic_recorder,
        continue_on_error=continue_on_error,
    )

    if "velocity_gain" in enabled:
        velocity_gain_history, velocity_gain_metrics = run_selected_diagnostic(
            "velocity_gain",
            lambda: run_velocity_gain_test(
                config=config,
                trained_state=trained_state,
            ),
        )
    else:
        velocity_gain_history = {}
        velocity_gain_metrics = {"velocity_gain_diagnostic_enabled": 0.0}

    if "weight_structure" in enabled:
        _, weight_metrics = run_selected_diagnostic(
            "weight_structure",
            lambda: (
                {},
                summarize_weight_structure(
                    trained_state.w_hd_to_hd,
                    trained_state.w_hr_to_hd,
                ),
            ),
        )
    else:
        weight_metrics = {"weight_structure_diagnostic_enabled": 0.0}

    def compute_weight_snapshot_pi():
        saved_snapshot_time = np.asarray(
            (weight_history or {}).get("time", np.empty(0)),
            dtype=float,
        )
        snapshot_count = max(
            1,
            int(
                select_weight_snapshot_indices(
                    snapshot_time=saved_snapshot_time,
                    interval_fraction=(
                        config.tests.weight_snapshot_pi_interval_fraction
                    ),
                ).size
            ),
        )
        snapshot_pi_steps = snapshot_count * len(
            config.tests.weight_snapshot_pi_velocities
        ) * max(
            1,
            len(config.tests.weight_snapshot_pi_initial_headings),
        ) * (
            int(
                round(
                    config.tests.weight_snapshot_pi_cue_duration
                    / config.simulation.dt
                )
            )
            + max(
                1,
                int(
                    round(
                        config.tests.weight_snapshot_pi_duration
                        / config.simulation.dt
                    )
                ),
            )
        )
        snapshot_pi_progress = tqdm(
            total=snapshot_pi_steps,
            disable=not config.simulation.progress,
            desc="frozen snapshot PI development",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
        )
        with snapshot_pi_progress:
            return run_weight_snapshot_pi_development_diagnostic(
                config=config,
                trained_state=trained_state,
                weight_history=weight_history,
                progress=snapshot_pi_progress,
            )

    if precomputed_weight_snapshot_pi is not None:
        (
            weight_snapshot_pi_history,
            weight_snapshot_pi_metrics,
        ) = precomputed_weight_snapshot_pi
        if (
            diagnostic_recorder is not None
            and "weight_snapshot_pi_development" in enabled
        ):
            diagnostic_recorder.record_success(
                "weight_snapshot_pi_development",
                weight_snapshot_pi_history,
                weight_snapshot_pi_metrics,
            )
    elif "weight_snapshot_pi_development" in enabled:
        (
            weight_snapshot_pi_history,
            weight_snapshot_pi_metrics,
        ) = run_selected_diagnostic(
            "weight_snapshot_pi_development",
            compute_weight_snapshot_pi,
        )
    else:
        weight_snapshot_pi_history = {}
        weight_snapshot_pi_metrics = {
            "weight_snapshot_pi_development_diagnostic_enabled": 0.0
        }

    def compute_numerical_convergence():
        numerical_progress = tqdm(
            total=(
                len(config.tests.numerical_convergence_dt_values)
                * len(config.tests.numerical_convergence_methods)
            ),
            disable=not config.simulation.progress,
            desc="numerical convergence",
            unit="configuration",
        )
        with numerical_progress:
            return run_numerical_convergence_diagnostic(
                config=config,
                trained_state=trained_state,
                progress=numerical_progress,
            )

    if "numerical_convergence" in enabled:
        (
            numerical_convergence_history,
            numerical_convergence_metrics,
        ) = run_selected_diagnostic(
            "numerical_convergence",
            compute_numerical_convergence,
        )
    else:
        numerical_convergence_history = {}
        numerical_convergence_metrics = {
            "numerical_convergence_diagnostic_enabled": 0.0
        }

    if "bump_maintenance" in enabled and bump_history:
        _, zero_velocity_drive_metrics = run_selected_diagnostic(
            "zero_velocity_drive",
            lambda: (
                {},
                summarize_zero_velocity_drive(
                    config=config,
                    trained_state=trained_state,
                    bump_history=bump_history,
                ),
            ),
        )
    elif "bump_maintenance" in enabled:
        zero_velocity_drive_metrics = {
            "zero_velocity_drive_diagnostic_blocked": 1.0
        }
        if diagnostic_recorder is not None:
            diagnostic_recorder.record_blocked(
                "zero_velocity_drive",
                dependency="bump_maintenance",
            )
    else:
        zero_velocity_drive_metrics = {}
    metrics = {
        **hd_tuning_metrics,
        **bump_metrics,
        **bump_attractor_trajectory_metrics,
        **slow_manifold_metrics,
        **timescale_separation_metrics,
        **velocity_trajectory_sweep_metrics,
        **bump_ensemble_diffusion_metrics,
        **darkness_metrics,
        **ou_darkness_metrics,
        **ou_pi_ensemble_metrics,
        **velocity_gain_metrics,
        **weight_metrics,
        **weight_snapshot_pi_metrics,
        **numerical_convergence_metrics,
        **zero_velocity_drive_metrics,
    }
    if diagnostic_recorder is not None:
        diagnostic_recorder.finalize(metrics)
    return (
        hd_tuning_history,
        bump_history,
        bump_attractor_trajectory_history,
        slow_manifold_history,
        timescale_separation_history,
        velocity_trajectory_sweep_history,
        bump_diffusion_history,
        darkness_history,
        ou_darkness_history,
        ou_pi_ensemble_history,
        velocity_gain_history,
        weight_snapshot_pi_history,
        numerical_convergence_history,
        metrics,
    )


def _save_weight_state(
    *,
    path: Path,
    state: VafidisToyState,
    params: VafidisToyParams,
    source: str,
    source_index: int,
) -> None:
    """Persist one frozen weight state without depending on diagnostics."""
    effective_w_hd_to_hd, effective_w_hr_to_hd = (
        effective_hd_distal_weight_matrices(
            w_hd_to_hd=state.w_hd_to_hd,
            w_hr_to_hd=state.w_hr_to_hd,
            normalization=params.hd_distal_normalization,
        )
    )
    save_npz(
        path,
        theta_hd_pref=state.theta_hd_pref,
        w_hd_to_hd=state.w_hd_to_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        effective_w_hd_to_hd=effective_w_hd_to_hd,
        effective_w_hr_to_hd=effective_w_hr_to_hd,
        w_lhr_to_hd=state.w_lhr_to_hd,
        w_rhr_to_hd=state.w_rhr_to_hd,
        w_hd_to_hr=state.w_hd_to_hr,
        visual_tuning_profiles=(
            np.empty((0, 0), dtype=float)
            if state.visual_tuning_profiles is None
            else state.visual_tuning_profiles
        ),
        weight_source=np.asarray(source),
        weight_snapshot_source_index=np.asarray(source_index, dtype=int),
        weight_snapshot_time=np.asarray(float(state.time)),
    )


def save_training_completion_outputs(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    params: VafidisToyParams,
    training_selected_state: VafidisToyState,
    training_history: dict[str, np.ndarray],
    weight_history: dict[str, np.ndarray],
) -> None:
    """Save completed training before any frozen post-training evaluation."""
    save_yaml(run_dir / "config_resolved.yaml", config.to_dict())
    save_json(run_dir / "params.json", asdict(params))
    _save_weight_state(
        path=run_dir / "training_selected_weights.npz",
        state=training_selected_state,
        params=params,
        source="training_selected",
        source_index=-1,
    )
    save_npz(run_dir / "training_history.npz", **training_history)
    save_npz(run_dir / "weight_history.npz", **weight_history)


def save_weight_selection_outputs(
    *,
    run_dir: Path,
    params: VafidisToyParams,
    diagnostic_state: VafidisToyState,
    final_state: VafidisToyState,
    best_state: VafidisToyState,
    weight_selection_metadata: dict[str, np.ndarray],
    weight_snapshot_pi_history: dict[str, np.ndarray],
) -> None:
    """Save all selected weight variants before optional diagnostics run."""
    final_source_index = int(
        np.asarray(
            weight_selection_metadata.get("final_snapshot_source_index", -1)
        )
    )
    best_source_index = int(
        np.asarray(
            weight_selection_metadata.get("best_snapshot_source_index", -1)
        )
    )
    diagnostic_source = str(
        np.asarray(
            weight_selection_metadata.get(
                "diagnostic_weight_source", np.asarray("training_selected")
            )
        )
    )
    diagnostic_source_index = int(
        np.asarray(
            weight_selection_metadata.get(
                "diagnostic_snapshot_source_index", -1
            )
        )
    )
    _save_weight_state(
        path=run_dir / "final_weights.npz",
        state=final_state,
        params=params,
        source="final",
        source_index=final_source_index,
    )
    _save_weight_state(
        path=run_dir / "best_weights.npz",
        state=best_state,
        params=params,
        source="best",
        source_index=best_source_index,
    )
    _save_weight_state(
        path=run_dir / "diagnostic_weights.npz",
        state=diagnostic_state,
        params=params,
        source=diagnostic_source,
        source_index=diagnostic_source_index,
    )
    # Backward-compatible alias: new runs expose the configured diagnostic
    # weights here, while final_weights.npz always preserves the true endpoint.
    _save_weight_state(
        path=run_dir / "trained_weights.npz",
        state=diagnostic_state,
        params=params,
        source=diagnostic_source,
        source_index=diagnostic_source_index,
    )
    save_npz(
        run_dir / "weight_selection_history.npz",
        **{**weight_snapshot_pi_history, **weight_selection_metadata},
    )


def save_run_outputs(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    params: VafidisToyParams,
    trained_state: VafidisToyState,
    final_state: VafidisToyState | None = None,
    best_state: VafidisToyState | None = None,
    training_selected_state: VafidisToyState | None = None,
    weight_selection_metadata: dict[str, np.ndarray] | None = None,
    training_history: dict[str, np.ndarray],
    weight_history: dict[str, np.ndarray],
    hd_tuning_history: dict[str, np.ndarray],
    bump_history: dict[str, np.ndarray],
    bump_attractor_trajectory_history: dict[str, np.ndarray],
    slow_manifold_history: dict[str, np.ndarray],
    timescale_separation_history: dict[str, np.ndarray],
    velocity_trajectory_sweep_history: dict[str, np.ndarray],
    bump_diffusion_history: dict[str, np.ndarray],
    darkness_history: dict[str, np.ndarray],
    ou_darkness_history: dict[str, np.ndarray],
    ou_pi_ensemble_history: dict[str, np.ndarray],
    velocity_gain_history: dict[str, np.ndarray],
    weight_snapshot_pi_history: dict[str, np.ndarray],
    numerical_convergence_history: dict[str, np.ndarray],
    test_metrics: dict[str, float],
) -> None:
    final_state = trained_state if final_state is None else final_state
    best_state = trained_state if best_state is None else best_state
    training_selected_state = (
        trained_state
        if training_selected_state is None
        else training_selected_state
    )
    weight_selection_metadata = weight_selection_metadata or {
        "diagnostic_weight_source": np.asarray("training_selected"),
        "diagnostic_snapshot_source_index": np.asarray(-1, dtype=int),
        "diagnostic_snapshot_time": np.asarray(float(trained_state.time)),
    }
    save_training_completion_outputs(
        run_dir=run_dir,
        config=config,
        params=params,
        training_selected_state=training_selected_state,
        training_history=training_history,
        weight_history=weight_history,
    )
    save_weight_selection_outputs(
        run_dir=run_dir,
        params=params,
        diagnostic_state=trained_state,
        final_state=final_state,
        best_state=best_state,
        weight_selection_metadata=weight_selection_metadata,
        weight_snapshot_pi_history=weight_snapshot_pi_history,
    )
    enabled = selected_diagnostics(config)
    histories_to_save = {
        "hd_tuning": ("hd_tuning_history.npz", hd_tuning_history),
        "bump_maintenance": ("bump_history.npz", bump_history),
        "bump_attractor_trajectories": (
            "bump_attractor_trajectory_history.npz",
            bump_attractor_trajectory_history,
        ),
        "slow_manifold": ("slow_manifold_diagnostics.npz", slow_manifold_history),
        "timescale_separation": (
            "timescale_separation_history.npz",
            timescale_separation_history,
        ),
        "velocity_trajectory_sweep": (
            "velocity_trajectory_sweep_history.npz",
            velocity_trajectory_sweep_history,
        ),
        "bump_diffusion": ("bump_diffusion_history.npz", bump_diffusion_history),
        "darkness_path_integration": ("darkness_history.npz", darkness_history),
        "ou_path_integration": ("ou_darkness_history.npz", ou_darkness_history),
        "ou_pi_ensemble": ("ou_pi_ensemble_history.npz", ou_pi_ensemble_history),
        "velocity_gain": ("velocity_gain_history.npz", velocity_gain_history),
        "weight_snapshot_pi_development": (
            "weight_snapshot_pi_development.npz",
            weight_snapshot_pi_history,
        ),
        "numerical_convergence": (
            "numerical_convergence_history.npz",
            numerical_convergence_history,
        ),
    }
    dependency_outputs = set()
    if "hd_tuning_dependency_computed" in test_metrics:
        dependency_outputs.add("hd_tuning")
    if "bump_attractor_dependency_computed" in test_metrics:
        dependency_outputs.add("bump_attractor_trajectories")
    for diagnostic_name, (filename, history) in histories_to_save.items():
        if history and (
            diagnostic_name in enabled or diagnostic_name in dependency_outputs
        ):
            save_npz(run_dir / filename, **history)
    save_json(run_dir / "test_metrics.json", test_metrics)


def run_experiment(
    *,
    config: ExperimentConfig,
    project_root: Path,
    run_id: str | None = None,
    make_figures: bool = True,
    init_weights_path: str | Path | None = None,
) -> Path:
    validate_diagnostic_weight_selection(config)
    rng = make_rng(config.simulation.seed)
    params = VafidisToyParams.from_config(config)
    experiment_name = config.experiment_name
    if params.hd_distal_normalization == "presynaptic_population_mean":
        neuron_count_suffix = f"_n{params.n_theta}"
        if not experiment_name.endswith(neuron_count_suffix):
            experiment_name = f"{experiment_name}{neuron_count_suffix}"
    run_dir = create_run_dir(
        project_root=project_root,
        runs_root=config.paths.runs_root,
        experiment_name=experiment_name,
        seed=config.simulation.seed,
        run_id=run_id,
    )
    # Persist the resolved setup immediately, then maintain one atomic rolling
    # checkpoint during plasticity.  A failure inside training can recover the
    # latest configured snapshot; a post-training failure keeps the exact
    # completed state saved below.
    save_yaml(run_dir / "config_resolved.yaml", config.to_dict())
    save_json(run_dir / "params.json", asdict(params))

    def save_recovery_checkpoint(state: VafidisToyState) -> None:
        _save_weight_state(
            path=run_dir / "training_checkpoint_latest.npz",
            state=state,
            params=params,
            source="training_checkpoint",
            source_index=-1,
        )

    training_selected_state, training_history, weight_history = run_training(
        config=config,
        rng=rng,
        recovery_checkpoint_callback=save_recovery_checkpoint,
        init_weights_path=init_weights_path,
    )
    save_training_completion_outputs(
        run_dir=run_dir,
        config=config,
        params=params,
        training_selected_state=training_selected_state,
        training_history=training_history,
        weight_history=weight_history,
    )
    weight_snapshot_pi_history, weight_snapshot_pi_metrics = (
        run_weight_performance_selection(
            config=config,
            trained_state=training_selected_state,
            weight_history=weight_history,
        )
    )
    (
        final_state,
        best_state,
        diagnostic_state,
        weight_selection_metadata,
    ) = resolve_weight_states(
        config=config,
        training_selected_state=training_selected_state,
        weight_history=weight_history,
        weight_selection_history=weight_snapshot_pi_history,
    )
    save_weight_selection_outputs(
        run_dir=run_dir,
        params=params,
        diagnostic_state=diagnostic_state,
        final_state=final_state,
        best_state=best_state,
        weight_selection_metadata=weight_selection_metadata,
        weight_snapshot_pi_history=weight_snapshot_pi_history,
    )
    diagnostic_recorder = IncrementalDiagnosticRecorder(run_dir=run_dir)
    (
        hd_tuning_history,
        bump_history,
        bump_attractor_trajectory_history,
        slow_manifold_history,
        timescale_separation_history,
        velocity_trajectory_sweep_history,
        bump_diffusion_history,
        darkness_history,
        ou_darkness_history,
        ou_pi_ensemble_history,
        velocity_gain_history,
        weight_snapshot_pi_history,
        numerical_convergence_history,
        test_metrics,
    ) = run_all_tests(
        config=config,
        trained_state=diagnostic_state,
        weight_history=weight_history,
        precomputed_weight_snapshot_pi=(
            weight_snapshot_pi_history,
            weight_snapshot_pi_metrics,
        ),
        diagnostic_recorder=diagnostic_recorder,
        continue_on_error=True,
    )
    diagnostic_source = str(weight_selection_metadata["diagnostic_weight_source"])
    test_metrics.update(
        {
            "diagnostic_weight_source_is_best": float(
                diagnostic_source == "best"
            ),
            "diagnostic_weight_source_is_final": float(
                diagnostic_source == "final"
            ),
            "diagnostic_weight_snapshot_source_index": float(
                weight_selection_metadata["diagnostic_snapshot_source_index"]
            ),
            "diagnostic_weight_snapshot_time": float(
                weight_selection_metadata["diagnostic_snapshot_time"]
            ),
        }
    )
    diagnostic_recorder.finalize(test_metrics)
    save_run_outputs(
        run_dir=run_dir,
        config=config,
        params=params,
        trained_state=diagnostic_state,
        final_state=final_state,
        best_state=best_state,
        training_selected_state=training_selected_state,
        weight_selection_metadata=weight_selection_metadata,
        training_history=training_history,
        weight_history=weight_history,
        hd_tuning_history=hd_tuning_history,
        bump_history=bump_history,
        bump_attractor_trajectory_history=bump_attractor_trajectory_history,
        slow_manifold_history=slow_manifold_history,
        timescale_separation_history=timescale_separation_history,
        velocity_trajectory_sweep_history=velocity_trajectory_sweep_history,
        bump_diffusion_history=bump_diffusion_history,
        darkness_history=darkness_history,
        ou_darkness_history=ou_darkness_history,
        ou_pi_ensemble_history=ou_pi_ensemble_history,
        velocity_gain_history=velocity_gain_history,
        weight_snapshot_pi_history=weight_snapshot_pi_history,
        numerical_convergence_history=numerical_convergence_history,
        test_metrics=test_metrics,
    )
    if make_figures and selected_diagnostics(config):
        make_vafidis_figures_for_run(run_dir=run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument(
        "--diagnostics-config",
        default=None,
        help=(
            "Optional grouped diagnostics hyper config to run after training; "
            "omit it for training only."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Reusable partial config YAML. Repeat to compose profiles in order; "
            "later profiles win."
        ),
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override one resolved config field using a dotted path and YAML value. "
            "Repeat as needed; later assignments win."
        ),
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        metavar="NPZ",
        help=(
            "Optional pretrained weight archive (trained_weights.npz / "
            "best_weights.npz / final_weights.npz) to resume training from. "
            "Plastic matrices and static geometry are copied onto a freshly "
            "initialized dynamic state."
        ),
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully composed config and exit without creating a run.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    project_root = find_repository_root(config_path)
    diagnostics_path = (
        None
        if args.diagnostics_config is None
        else resolve_config_path(args.diagnostics_config)
    )
    profile_paths = [resolve_config_path(profile_path) for profile_path in args.profile]
    config = load_experiment_config(
        config_path,
        diagnostics_path=diagnostics_path,
        profile_paths=profile_paths,
        overrides=args.config_overrides,
    )
    # Parameter construction performs cross-field and Euler-stability checks,
    # which makes --print-config a useful zero-cost validation command.
    VafidisToyParams.from_config(config)
    if args.print_config:
        print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))
        return
    run_dir = run_experiment(
        config=config,
        project_root=project_root,
        run_id=args.run_id,
        make_figures=not args.no_figures,
        init_weights_path=args.init_weights,
    )
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
