"""Train the Vafidis-style predictive local plasticity toy model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from tqdm import trange
except ImportError:  # pragma: no cover
    def trange(*args, **kwargs):
        del kwargs
        return range(*args)

from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.analysis.metrics import (
    circular_error_trace,
    estimate_decoded_velocity,
    final_abs_circular_error,
    rms_circular_error,
    summarize_velocity_gain,
)
from learning.analysis.weights import summarize_weight_structure
from learning.common.angles import collapse_activity_by_theta, pva_vector_strength
from learning.common.arrays import l2_norm
from learning.common.random import make_rng
from learning.config.load_config import find_project_root, load_experiment_config, save_yaml
from learning.config.schema import ExperimentConfig
from learning.dynamics.activation import apply_activation
from learning.dynamics.hr_dynamics import compute_i_hr
from learning.io.run_dir import create_run_dir
from learning.io.save_load import save_json, save_npz
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    initialize_vafidis_toy_state,
    validate_vafidis_toy_state,
    step_vafidis_toy,
)
from learning.stimuli.velocity import OUAngularVelocity


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path.resolve()
    for project_root_candidate in [Path.cwd(), Path.cwd() / "learning"]:
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
        process = OUAngularVelocity(
            mean=config.velocity.mean,
            std=config.velocity.std,
            tau=config.velocity.tau,
            clip=config.velocity.clip,
            rng=rng,
        )
        return process.step
    raise ValueError(f"Unknown velocity process: {config.velocity.process}")


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
        "pva_strength_hd": [],
        "bump_contrast_hd": [],
        "mean_r_lhr": [],
        "mean_r_rhr": [],
        "contrast_r_lhr": [],
        "contrast_r_rhr": [],
        "weight_norm_hd_to_hd": [],
        "weight_norm_hr_to_hd": [],
    }
    if include_activity:
        history["r_hd"] = []
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
    if include_activity:
        history["r_hd"].append(state.r_hd.copy())


def run_training(
    *,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[VafidisToyState, dict[str, np.ndarray]]:
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=rng)
    angular_velocity_step = build_training_velocity_process(config=config, rng=rng)
    warmup_steps = int(round(config.simulation.recurrent_warmup_duration / params.dt))
    train_steps = int(round(config.simulation.train_duration / params.dt))
    save_interval_steps = max(1, int(config.simulation.save_interval_steps))
    history = _new_history(include_activity=True)
    total_steps = warmup_steps + train_steps
    warmup_params = replace(params, eta_hr_to_hd=0.0)
    if config.simulation.freeze_hd_to_hd_after_warmup:
        main_params = replace(params, eta_hd_to_hd=0.0)
    else:
        main_params = params
    main_params_hd_frozen = replace(main_params, eta_hd_to_hd=0.0)
    progress_iterator = trange(total_steps, disable=not config.simulation.progress, desc="training")
    for step_index in progress_iterator:
        in_warmup = step_index < warmup_steps
        if in_warmup:
            angular_velocity = 0.0
            step_params = warmup_params
        else:
            angular_velocity = angular_velocity_step(params.dt)
            velocity_threshold = config.learning_rule.hd_to_hd_learning_velocity_threshold
            if velocity_threshold is not None and abs(angular_velocity) > velocity_threshold:
                step_params = main_params_hd_frozen
            else:
                step_params = main_params
        state = step_vafidis_toy(
            state=state,
            params=step_params,
            angular_velocity=angular_velocity,
            visual_teacher=True,
            training=True,
        )
        if step_index % save_interval_steps == 0 or step_index == total_steps - 1:
            record_state(history=history, state=state, include_activity=True)
    return state, _history_to_arrays(history)


def initialize_protocol_state(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
) -> VafidisToyState:
    protocol_rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(config=config, rng=protocol_rng, theta_true=theta_true)
    state.w_hd_to_hd = trained_state.w_hd_to_hd.copy()
    state.w_hr_to_hd = trained_state.w_hr_to_hd.copy()
    state.w_hd_to_hr = trained_state.w_hd_to_hr.copy()
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    return state


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
) -> dict[str, np.ndarray]:
    params = VafidisToyParams.from_config(config)
    state = initialize_protocol_state(
        config=config,
        trained_state=trained_state,
        theta_true=theta_true,
    )
    history = _new_history(include_activity=True)
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
    for phase_duration, visual_teacher, phase_id in phase_specs:
        phase_steps = int(round(max(phase_duration, 0.0) / params.dt))
        for _step_index in range(phase_steps):
            angular_velocity = angular_velocity_step(params.dt)
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=angular_velocity,
                visual_teacher=visual_teacher,
                training=False,
            )
            record_state(
                history=history,
                state=state,
                include_activity=True,
                visual_teacher=visual_teacher,
                phase_id=phase_id,
            )
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
    )


def run_ou_visual_dark_visual_protocol(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
    darkness_duration: float,
    cue_duration: float | None = None,
    recue_duration: float | None = None,
) -> dict[str, np.ndarray]:
    protocol_rng = make_rng(config.simulation.seed + 10_000)
    ou_process = OUAngularVelocity(
        mean=config.velocity.mean,
        std=config.velocity.std,
        tau=config.velocity.tau,
        clip=config.velocity.clip,
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
    drift = final_abs_circular_error(
        bump_history["theta_hd_decoded"][cue_mask],
        theta_reference=config.simulation.theta0,
    )
    intrinsic_drift_velocity = estimate_decoded_velocity(
        time=bump_history["time"][cue_mask] - bump_history["time"][cue_mask][0],
        theta_decoded=bump_history["theta_hd_decoded"][cue_mask],
        start_fraction=0.25,
    )
    peak_drift = final_abs_circular_error(
        bump_history["theta_hd_decoded_peak"][cue_mask],
        theta_reference=config.simulation.theta0,
    )
    peak_intrinsic_drift_velocity = estimate_decoded_velocity(
        time=bump_history["time"][cue_mask] - bump_history["time"][cue_mask][0],
        theta_decoded=bump_history["theta_hd_decoded_peak"][cue_mask],
        start_fraction=0.25,
    )
    return bump_history, {
        "bump_final_abs_drift": drift,
        "bump_intrinsic_drift_velocity": intrinsic_drift_velocity,
        "bump_intrinsic_drift_velocity_deg_s": float(np.rad2deg(intrinsic_drift_velocity)),
        "bump_final_abs_peak_drift": peak_drift,
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
    }


def run_darkness_path_integration_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    angular_velocity: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    darkness_history = run_constant_velocity_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.darkness_test_duration,
        angular_velocity=angular_velocity,
        cue_duration=get_pi_cue_duration(config),
    )
    cue_mask = phase_mask(darkness_history, DARKNESS_PHASE_ID)
    recue_mask = phase_mask(darkness_history, VISUAL_RECUE_PHASE_ID)
    pi_error = circular_error_trace(
        darkness_history["theta_hd_decoded"][cue_mask],
        darkness_history["theta_true"][cue_mask],
    )
    decoded_velocity = estimate_decoded_velocity(
        time=darkness_history["time"][cue_mask] - darkness_history["time"][cue_mask][0],
        theta_decoded=darkness_history["theta_hd_decoded"][cue_mask],
        start_fraction=0.25,
    )
    pi_error_peak = circular_error_trace(
        darkness_history["theta_hd_decoded_peak"][cue_mask],
        darkness_history["theta_true"][cue_mask],
    )
    decoded_velocity_peak = estimate_decoded_velocity(
        time=darkness_history["time"][cue_mask] - darkness_history["time"][cue_mask][0],
        theta_decoded=darkness_history["theta_hd_decoded_peak"][cue_mask],
        start_fraction=0.25,
    )
    darkness_metrics = {
        "darkness_rms_pi_error": rms_circular_error(
            darkness_history["theta_hd_decoded"][cue_mask],
            darkness_history["theta_true"][cue_mask],
        ),
        "darkness_final_abs_pi_error": final_abs_circular_error(
            darkness_history["theta_hd_decoded"][cue_mask],
            theta_reference=darkness_history["theta_true"][cue_mask][-1],
        ),
        "darkness_mean_pi_error": nanmean_or_nan(pi_error),
        "darkness_mean_pva_strength": float(np.nanmean(darkness_history["pva_strength_hd"][cue_mask])),
        "darkness_final_pva_strength": float(darkness_history["pva_strength_hd"][cue_mask][-1]),
        "darkness_mean_bump_contrast": float(np.nanmean(darkness_history["bump_contrast_hd"][cue_mask])),
        "darkness_final_bump_contrast": float(darkness_history["bump_contrast_hd"][cue_mask][-1]),
        "darkness_decoded_velocity": decoded_velocity,
        "darkness_decoded_velocity_deg_s": float(np.rad2deg(decoded_velocity)),
        "darkness_velocity_bias": decoded_velocity - angular_velocity,
        "darkness_velocity_bias_deg_s": float(np.rad2deg(decoded_velocity - angular_velocity)),
        "darkness_peak_rms_pi_error": rms_circular_error(
            darkness_history["theta_hd_decoded_peak"][cue_mask],
            darkness_history["theta_true"][cue_mask],
        ),
        "darkness_peak_final_abs_pi_error": final_abs_circular_error(
            darkness_history["theta_hd_decoded_peak"][cue_mask],
            theta_reference=darkness_history["theta_true"][cue_mask][-1],
        ),
        "darkness_peak_mean_pi_error": nanmean_or_nan(pi_error_peak),
        "darkness_peak_decoded_velocity": decoded_velocity_peak,
        "darkness_peak_decoded_velocity_deg_s": float(np.rad2deg(decoded_velocity_peak)),
        "darkness_peak_velocity_bias": decoded_velocity_peak - angular_velocity,
        "darkness_peak_velocity_bias_deg_s": float(np.rad2deg(decoded_velocity_peak - angular_velocity)),
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
    )
    hd_drive = trained_state.w_hd_to_hd @ r_hd
    hr_drive = trained_state.w_hr_to_hd @ r_hr
    total_drive = hd_drive + hr_drive - params.b_hd
    return {
        "zero_velocity_hd_tangent_drive": float(np.dot(hd_drive, tangent_basis)),
        "zero_velocity_hr_tangent_drive": float(np.dot(hr_drive, tangent_basis)),
        "zero_velocity_total_tangent_drive": float(np.dot(total_drive, tangent_basis)),
    }


def run_velocity_gain_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    commanded_velocity_values = np.asarray(config.tests.gain_velocities, dtype=float)
    decoded_velocity_values: list[float] = []
    decoded_peak_velocity_values: list[float] = []
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
        cue_mask = phase_mask(gain_history, DARKNESS_PHASE_ID)
        decoded_velocity = estimate_decoded_velocity(
            time=gain_history["time"][cue_mask] - gain_history["time"][cue_mask][0],
            theta_decoded=gain_history["theta_hd_decoded"][cue_mask],
            start_fraction=0.25,
        )
        decoded_peak_velocity = estimate_decoded_velocity(
            time=gain_history["time"][cue_mask] - gain_history["time"][cue_mask][0],
            theta_decoded=gain_history["theta_hd_decoded_peak"][cue_mask],
            start_fraction=0.25,
        )
        decoded_velocity_values.append(decoded_velocity)
        decoded_peak_velocity_values.append(decoded_peak_velocity)
    decoded_velocity_array = np.asarray(decoded_velocity_values, dtype=float)
    decoded_peak_velocity_array = np.asarray(decoded_peak_velocity_values, dtype=float)
    gain_summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity_values,
        decoded_velocity=decoded_velocity_array,
    )
    peak_gain_summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity_values,
        decoded_velocity=decoded_peak_velocity_array,
    )
    return {
        "commanded_velocity": commanded_velocity_values,
        "decoded_velocity": decoded_velocity_array,
        "decoded_velocity_peak": decoded_peak_velocity_array,
    }, {
        "velocity_gain": gain_summary["gain"],
        "velocity_gain_intercept": gain_summary["intercept"],
        "velocity_gain_peak": peak_gain_summary["gain"],
        "velocity_gain_peak_intercept": peak_gain_summary["intercept"],
    }


def run_ou_path_integration_test(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    ou_history = run_ou_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.darkness_test_duration,
        cue_duration=get_pi_cue_duration(config),
    )
    dark_mask = phase_mask(ou_history, DARKNESS_PHASE_ID)
    recue_mask = phase_mask(ou_history, VISUAL_RECUE_PHASE_ID)
    dark_pi_error = circular_error_trace(
        ou_history["theta_hd_decoded"][dark_mask],
        ou_history["theta_true"][dark_mask],
    )
    metrics = {
        "ou_darkness_rms_pi_error": rms_circular_error(
            ou_history["theta_hd_decoded"][dark_mask],
            ou_history["theta_true"][dark_mask],
        ),
        "ou_darkness_mean_abs_pi_error": nanmean_or_nan(np.abs(dark_pi_error)),
        "ou_darkness_final_abs_pi_error": final_abs_circular_error(
            ou_history["theta_hd_decoded"][dark_mask],
            theta_reference=ou_history["theta_true"][dark_mask][-1],
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


def run_all_tests(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    bump_history, bump_metrics = run_bump_maintenance_test(config=config, trained_state=trained_state)
    darkness_history, darkness_metrics = run_darkness_path_integration_test(
        config=config,
        trained_state=trained_state,
        angular_velocity=config.tests.darkness_angular_velocity,
    )
    ou_darkness_history, ou_darkness_metrics = run_ou_path_integration_test(
        config=config,
        trained_state=trained_state,
    )
    velocity_gain_history, velocity_gain_metrics = run_velocity_gain_test(
        config=config,
        trained_state=trained_state,
    )
    weight_metrics = summarize_weight_structure(
        trained_state.w_hd_to_hd,
        trained_state.w_hr_to_hd,
    )
    zero_velocity_drive_metrics = summarize_zero_velocity_drive(
        config=config,
        trained_state=trained_state,
        bump_history=bump_history,
    )
    metrics = {
        **bump_metrics,
        **darkness_metrics,
        **ou_darkness_metrics,
        **velocity_gain_metrics,
        **weight_metrics,
        **zero_velocity_drive_metrics,
    }
    return bump_history, darkness_history, ou_darkness_history, velocity_gain_history, metrics


def save_run_outputs(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    params: VafidisToyParams,
    trained_state: VafidisToyState,
    training_history: dict[str, np.ndarray],
    bump_history: dict[str, np.ndarray],
    darkness_history: dict[str, np.ndarray],
    ou_darkness_history: dict[str, np.ndarray],
    velocity_gain_history: dict[str, np.ndarray],
    test_metrics: dict[str, float],
) -> None:
    save_yaml(run_dir / "config_resolved.yaml", config.to_dict())
    save_json(run_dir / "params.json", asdict(params))
    save_npz(
        run_dir / "trained_weights.npz",
        theta_hd_pref=trained_state.theta_hd_pref,
        w_hd_to_hd=trained_state.w_hd_to_hd,
        w_hr_to_hd=trained_state.w_hr_to_hd,
        w_lhr_to_hd=trained_state.w_lhr_to_hd,
        w_rhr_to_hd=trained_state.w_rhr_to_hd,
        w_hd_to_hr=trained_state.w_hd_to_hr,
    )
    save_npz(run_dir / "training_history.npz", **training_history)
    save_npz(run_dir / "bump_history.npz", **bump_history)
    save_npz(run_dir / "darkness_history.npz", **darkness_history)
    save_npz(run_dir / "ou_darkness_history.npz", **ou_darkness_history)
    save_npz(run_dir / "velocity_gain_history.npz", **velocity_gain_history)
    save_json(run_dir / "test_metrics.json", test_metrics)


def run_experiment(
    *,
    config: ExperimentConfig,
    project_root: Path,
    run_id: str | None = None,
    make_figures: bool = True,
) -> Path:
    rng = make_rng(config.simulation.seed)
    params = VafidisToyParams.from_config(config)
    run_dir = create_run_dir(
        project_root=project_root,
        runs_root=config.paths.runs_root,
        experiment_name=config.experiment_name,
        seed=config.simulation.seed,
        run_id=run_id,
    )
    trained_state, training_history = run_training(config=config, rng=rng)
    (
        bump_history,
        darkness_history,
        ou_darkness_history,
        velocity_gain_history,
        test_metrics,
    ) = run_all_tests(
        config=config,
        trained_state=trained_state,
    )
    save_run_outputs(
        run_dir=run_dir,
        config=config,
        params=params,
        trained_state=trained_state,
        training_history=training_history,
        bump_history=bump_history,
        darkness_history=darkness_history,
        ou_darkness_history=ou_darkness_history,
        velocity_gain_history=velocity_gain_history,
        test_metrics=test_metrics,
    )
    if make_figures:
        make_vafidis_figures_for_run(run_dir=run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    project_root = find_project_root(config_path)
    config = load_experiment_config(config_path)
    run_dir = run_experiment(
        config=config,
        project_root=project_root,
        run_id=args.run_id,
        make_figures=not args.no_figures,
    )
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
