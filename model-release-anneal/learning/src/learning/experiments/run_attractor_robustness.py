"""Run attractor robustness and manifold-perturbation probes."""

from __future__ import annotations

import argparse
import copy
import csv
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from learning.analysis.metrics import (
    benjamini_hochberg_adjusted_p_values,
    circular_error_trace,
    empirical_two_point_correlation,
    estimate_velocity_tracking_operating_range,
    final_abs_circular_error,
    kuiper_uniformity_test_asymptotic,
    relative_circulant_error,
    summarize_velocity_gain,
    summarize_velocity_tracking,
)
from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.analysis.weights import summarize_weight_structure
from learning.common.random import make_rng
from learning.config.load_config import find_project_root, load_experiment_config, load_yaml, save_yaml
from learning.config.schema import ExperimentConfig
from learning.experiments.run_vafidis_toy import (
    DARKNESS_PHASE_ID,
    VISUAL_CUE_PHASE_ID,
    VISUAL_RECUE_PHASE_ID,
    get_pi_cue_duration,
    phase_mask,
    run_bump_diffusion_ensemble_test,
    run_constant_velocity_visual_dark_visual_protocol,
    run_experiment,
)
from learning.io.save_load import load_json, load_npz, save_json, save_npz
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    initialize_vafidis_toy_state,
    step_vafidis_toy,
    validate_vafidis_toy_state,
)
from learning.plotting.backend import use_headless_backend
from learning.plotting.heading import plot_pi_error, plot_true_vs_decoded_heading

use_headless_backend()

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


DEFAULT_SWEEP_METRICS = [
    "visual_velocity_gain",
    "darkness_velocity_gain",
    "darkness_velocity_gain_r_squared",
    "darkness_velocity_tracking_operating_range_max_abs_velocity",
    "darkness_minus_visual_velocity_gain",
    "visual_cue_hd_decode_rms_error",
    "darkness_hd_decode_rms_error",
    "visual_velocity_tracking_rmse",
    "darkness_velocity_tracking_rmse",
    "darkness_velocity_tracking_max_abs_error",
    "darkness_abs_velocity_bias",
    "bump_abs_intrinsic_drift_velocity",
    "bump_ensemble_diffusion_coefficient",
    "darkness_final_pva_strength",
    "darkness_mean_near_peak_hd_bins",
    "hd_tuning_curve_converged_fraction",
]

SWEEP_TARGET_VALUE_METRICS = {
    "visual_velocity_gain": 1.0,
    "darkness_velocity_gain": 1.0,
    "darkness_minus_visual_velocity_gain": 0.0,
    "hd_tuning_curve_converged_fraction": 1.0,
}

SWEEP_LOWER_IS_BETTER_METRICS = {
    "visual_cue_hd_decode_rms_error",
    "darkness_hd_decode_rms_error",
    "visual_velocity_tracking_rmse",
    "darkness_velocity_tracking_rmse",
    "darkness_velocity_tracking_max_abs_error",
    "darkness_abs_velocity_bias",
    "bump_abs_intrinsic_drift_velocity",
    "bump_ensemble_diffusion_coefficient",
    "darkness_mean_near_peak_hd_bins",
}

SWEEP_HIGHER_IS_BETTER_METRICS = {
    "darkness_velocity_gain_r_squared",
    "darkness_velocity_tracking_operating_range_max_abs_velocity",
    "darkness_final_pva_strength",
}

SWEEP_IDEAL_TIE_VALUES = {
    **SWEEP_TARGET_VALUE_METRICS,
    "visual_cue_hd_decode_rms_error": 0.0,
    "darkness_hd_decode_rms_error": 0.0,
    "visual_velocity_tracking_rmse": 0.0,
    "darkness_velocity_tracking_rmse": 0.0,
    "darkness_velocity_tracking_max_abs_error": 0.0,
    "darkness_abs_velocity_bias": 0.0,
    "bump_abs_intrinsic_drift_velocity": 0.0,
    "bump_ensemble_diffusion_coefficient": 0.0,
    "darkness_mean_near_peak_hd_bins": 1.0,
    "darkness_final_pva_strength": 1.0,
    "darkness_velocity_gain_r_squared": 1.0,
}

STANDARD_RUN_FIGURE_SENTINELS = [
    "hd_tuning_com_aligned.npz",
    (
        "figures/pva_spectrum_and_visualization/"
        "training_visual_input_heatmap.png"
    ),
    (
        "figures/path_integration_and_pi_error/"
        "darkness_hd_activity_heatmap.png"
    ),
    (
        "figures/path_integration_and_pi_error/"
        "darkness_heading_and_pi_error.png"
    ),
    "figures/path_integration_and_pi_error/darkness_pi_error.png",
    (
        "figures/weight_snapshots_and_development/"
        "training_weight_matrices_side_by_side.png"
    ),
    "figures/velocity_gain/velocity_gain_curve.png",
    "figures/bump_diffusion/bump_ensemble_diffusion_variance.png",
]

SWEEP_METRIC_PLOT_LABELS = {
    "visual_velocity_gain": (
        "visual-cue velocity gain g",
        "decoded velocity / commanded velocity [unitless]",
    ),
    "darkness_velocity_gain": (
        "darkness velocity gain g",
        "decoded velocity / commanded velocity [unitless]",
    ),
    "darkness_minus_visual_velocity_gain": (
        "darkness - visual gain",
        "gain difference [unitless]",
    ),
    "visual_cue_hd_decode_rms_error": (
        "visual-cue HD decode RMS error",
        "wrapped decoded-true heading error [rad]",
    ),
    "darkness_hd_decode_rms_error": (
        "darkness HD decode RMS error",
        "wrapped decoded-true heading error [rad]",
    ),
    "visual_velocity_tracking_rmse": (
        "visual-cue velocity RMSE",
        "decoded-commanded angular velocity [rad/s]",
    ),
    "darkness_velocity_tracking_rmse": (
        "darkness velocity RMSE",
        "decoded-commanded angular velocity [rad/s]",
    ),
    "darkness_velocity_tracking_max_abs_error": (
        "darkness worst velocity error",
        "max |decoded-commanded| angular velocity [rad/s]",
    ),
    "darkness_abs_velocity_bias": (
        "single-speed darkness bias",
        "|decoded-commanded| angular velocity [rad/s]",
    ),
    "bump_abs_intrinsic_drift_velocity": (
        "zero-speed bump drift",
        "|decoded drift velocity| [rad/s]",
    ),
    "bump_ensemble_diffusion_coefficient": (
        "zero-speed ensemble diffusion",
        "D [rad^2/s]",
    ),
    "darkness_final_pva_strength": (
        "final PVA vector strength",
        "PVA resultant length [unitless]",
    ),
    "darkness_mean_near_peak_hd_bins": (
        "near-peak bump width",
        "near-peak angular bins [bins]",
    ),
    "hd_tuning_curve_converged_fraction": (
        "steady tuning headings",
        "converged heading fraction [unitless]",
    ),
}

SWEEP_X_AXIS_LABELS = {
    "visual_noise_std": "visual input noise std [HD current]",
    "n_theta": "HD/HR neuron count [cells]",
}

NOISE_DELTA_METRICS = [
    "delta_visual_velocity_gain_abs_error",
    "delta_darkness_velocity_gain_abs_error",
    "delta_darkness_minus_visual_velocity_gain",
    "delta_visual_cue_hd_decode_rms_error",
    "delta_darkness_hd_decode_rms_error",
    "delta_visual_velocity_tracking_rmse",
    "delta_darkness_velocity_tracking_rmse",
    "delta_darkness_velocity_tracking_max_abs_error",
    "delta_darkness_abs_velocity_bias",
    "delta_bump_abs_intrinsic_drift_velocity",
    "delta_bump_ensemble_diffusion_coefficient",
    "delta_darkness_final_pva_strength",
    "delta_darkness_mean_near_peak_hd_bins",
]

NOISE_DELTA_METRIC_PLOT_LABELS = {
    "delta_visual_velocity_gain_abs_error": (
        "change in visual gain error",
        "delta |g - 1| [unitless]",
    ),
    "delta_darkness_velocity_gain_abs_error": (
        "change in darkness gain error",
        "delta |g - 1| [unitless]",
    ),
    "delta_darkness_minus_visual_velocity_gain": (
        "change in darkness - visual gain",
        "delta gain difference [unitless]",
    ),
    "delta_visual_cue_hd_decode_rms_error": (
        "change in visual-cue HD decode RMS error",
        "delta wrapped heading error [rad]",
    ),
    "delta_darkness_hd_decode_rms_error": (
        "change in darkness HD decode RMS error",
        "delta wrapped heading error [rad]",
    ),
    "delta_visual_velocity_tracking_rmse": (
        "change in visual velocity RMSE",
        "delta RMSE [rad/s]",
    ),
    "delta_darkness_velocity_tracking_rmse": (
        "change in darkness velocity RMSE",
        "delta RMSE [rad/s]",
    ),
    "delta_darkness_velocity_tracking_max_abs_error": (
        "change in darkness worst velocity error",
        "delta max |decoded-commanded| [rad/s]",
    ),
    "delta_darkness_abs_velocity_bias": (
        "change in single-speed bias",
        "delta |decoded-commanded| [rad/s]",
    ),
    "delta_bump_abs_intrinsic_drift_velocity": (
        "change in zero-speed drift",
        "delta |drift velocity| [rad/s]",
    ),
    "delta_bump_ensemble_diffusion_coefficient": (
        "change in ensemble diffusion",
        "delta D [rad^2/s]",
    ),
    "delta_darkness_final_pva_strength": (
        "change in final PVA strength",
        "delta PVA resultant length [unitless]",
    ),
    "delta_darkness_mean_near_peak_hd_bins": (
        "change in near-peak width",
        "delta near-peak angular bins [bins]",
    ),
}


def _wrap_plot_label(label: str, *, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _coerce_float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_float_list(value)
    if isinstance(value, list):
        return [float(item) for item in value]
    raise TypeError("Expected a comma-separated string or list of floats")


def _coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_int_list(value)
    if isinstance(value, list):
        return [int(item) for item in value]
    raise TypeError("Expected a comma-separated string or list of ints")


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError("Expected a comma-separated string or list of strings")


def _safe_number_label(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _safe_path_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    return label or "report"


def _create_timestamped_report_dir(
    *,
    output_root: Path,
    label: str,
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped robustness-report directory under output_root."""
    selected_timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    report_id = f"{selected_timestamp}_{_safe_path_label(label)}"
    report_dir = output_root / report_id
    if report_dir.exists():
        counter = 1
        while True:
            candidate_report_dir = output_root / f"{report_id}_{counter:02d}"
            if not candidate_report_dir.exists():
                report_dir = candidate_report_dir
                break
            counter += 1
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    return report_dir


def _visual_noise_run_label(config: ExperimentConfig) -> str:
    noise_std = float(config.visual.noise_std)
    if noise_std <= 0.0:
        return f"noise_{_safe_number_label(noise_std)}"
    process_label = str(config.visual.noise_process).replace("-", "_")
    tau_label = _safe_number_label(float(config.visual.noise_correlation_time))
    return f"noise_{process_label}_{_safe_number_label(noise_std)}_tau_{tau_label}"


def _visual_noise_metadata(config: ExperimentConfig) -> dict[str, float | int | str]:
    if config.visual.normalize_peak:
        profile_peak_to_trough = 1.0 - np.exp(-2.0 * config.visual.kappa)
    else:
        profile_peak_to_trough = np.exp(config.visual.kappa) - np.exp(
            -config.visual.kappa
        )
    effective_peak_to_trough = float(
        config.visual.proximal_scale
        * config.visual.amplitude
        * profile_peak_to_trough
    )
    return {
        "visual_noise_process": str(config.visual.noise_process),
        "visual_noise_correlation_time": float(config.visual.noise_correlation_time),
        "visual_noise_apply_training": int(bool(config.visual.apply_noise_during_training)),
        "visual_noise_apply_visual_test": int(bool(config.visual.apply_noise_during_visual_test)),
        "visual_effective_peak_to_trough": effective_peak_to_trough,
        "visual_noise_to_signal_ratio": (
            float(config.visual.noise_std) / effective_peak_to_trough
            if effective_peak_to_trough > 0.0
            else float("nan")
        ),
    }


def _normalize_seed_offsets(seed_offsets: list[int]) -> list[int]:
    normalized_offsets = [int(seed_offset) for seed_offset in seed_offsets]
    if not normalized_offsets:
        return [0]
    if len(set(normalized_offsets)) != len(normalized_offsets):
        raise ValueError("Seed offsets must be unique")
    return normalized_offsets


def _nested_get(config: dict[str, object], path: tuple[str, ...], default: object = None) -> object:
    current_value: object = config
    for key in path:
        if not isinstance(current_value, dict) or key not in current_value:
            return default
        current_value = current_value[key]
    return current_value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _visual_sigma_from_kappa(kappa: float) -> float:
    if kappa <= 0.0:
        return float("inf")
    return float(0.5 / np.sqrt(kappa))


def _visual_kappa_from_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("visual sigma must be positive")
    return float(1.0 / (4.0 * sigma * sigma))


def _retune_visual_width_for_neuron_count(
    *,
    config: ExperimentConfig,
    n_theta: int,
    min_visual_sigma_bins: float | None,
) -> None:
    """Keep visual teacher width resolvable on coarse paired-HD grids."""
    if min_visual_sigma_bins is None:
        return
    min_visual_sigma_bins = float(min_visual_sigma_bins)
    if min_visual_sigma_bins <= 0.0:
        return
    n_angle_bins = n_theta // 2
    if n_angle_bins <= 0:
        raise ValueError("n_theta must provide at least one paired angular bin")
    angular_bin_width = 2.0 * np.pi / float(n_angle_bins)
    current_sigma = _visual_sigma_from_kappa(float(config.visual.kappa))
    target_sigma = max(current_sigma, min_visual_sigma_bins * angular_bin_width)
    config.visual.kappa = _visual_kappa_from_sigma(target_sigma)


def _progress_message(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def _progress_iter(values, *, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return values
    return tqdm(values, desc=desc, unit="run")


def _resolve_path_argument(
    path_value: str,
    *,
    project_root: Path,
    relative_to: Path | None = None,
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    candidate_roots = []
    if relative_to is not None:
        candidate_roots.append(relative_to)
    candidate_roots.extend(
        [project_root, Path.cwd(), Path.cwd() / "learning", find_project_root()]
    )
    for candidate_root in candidate_roots:
        candidate_path = candidate_root / path
        if candidate_path.exists():
            return candidate_path.resolve()
    return (project_root / path).resolve()


def _resolve_existing_cli_path(path_value: str) -> Path:
    """Resolve a CLI path accepted from either repo root or learning/."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    for candidate_root in [Path.cwd(), Path.cwd() / "learning", find_project_root()]:
        candidate_path = candidate_root / path
        if candidate_path.exists():
            return candidate_path.resolve()
    return path.resolve()


def _clone_config_with_runtime_overrides(
    config: ExperimentConfig,
    *,
    train_duration: float | None,
    save_interval_duration: float | None,
    training_progress: bool,
) -> ExperimentConfig:
    config_copy = copy.deepcopy(config)
    config_copy.simulation.progress = bool(training_progress)
    if train_duration is not None:
        config_copy.simulation.train_duration = float(train_duration)
    if save_interval_duration is not None:
        config_copy.simulation.save_interval_duration = float(save_interval_duration)
    return config_copy


def _required_files_exist(run_dir: Path, required_files: tuple[str, ...]) -> bool:
    return run_dir.exists() and all((run_dir / file_name).exists() for file_name in required_files)


def _find_existing_run_dir(
    *,
    project_root: Path,
    runs_root: str,
    run_id: str,
    required_files: tuple[str, ...],
) -> Path | None:
    runs_root_path = project_root / runs_root
    candidates = [runs_root_path / run_id]
    if runs_root_path.exists():
        candidates.extend(sorted(runs_root_path.glob(f"{run_id}_*")))
    for candidate_run_dir in candidates:
        if _required_files_exist(candidate_run_dir, required_files):
            return candidate_run_dir
    return None


def _run_figures_complete(run_dir: Path) -> bool:
    if not all(
        (run_dir / relative_path).exists()
        for relative_path in STANDARD_RUN_FIGURE_SENTINELS
    ):
        return False
    try:
        tuning_summary = load_npz(run_dir / "hd_tuning_com_aligned.npz")
    except Exception:
        return False
    required_tuning_keys = {
        "theta_aligned",
        "r_hd_peak_normalized_com_aligned_mean",
        "r_hd_peak_normalized_com_aligned_std",
        "r_hd_unit_mean_com_aligned_mean",
        "r_hd_unit_mean_com_aligned_std",
    }
    return required_tuning_keys.issubset(tuning_summary)


def _run_velocity_gain_comparison_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "test_metrics.json"
    history_path = run_dir / "velocity_gain_history.npz"
    if not metrics_path.exists() or not history_path.exists():
        return False
    try:
        metrics = load_json(metrics_path)
        velocity_history = load_npz(history_path)
    except Exception:
        return False
    return (
        "visual_velocity_gain" in metrics
        and "darkness_velocity_gain" in metrics
        and "darkness_minus_visual_velocity_gain" in metrics
        and "decoded_velocity_visual" in velocity_history
        and "decoded_velocity_darkness" in velocity_history
    )


def _ensure_run_figures(*, run_dir: Path, progress: bool) -> None:
    if _run_figures_complete(run_dir):
        return
    _progress_message(f"[figures] generating standard run figures for {run_dir.name}", enabled=progress)
    make_vafidis_figures_for_run(run_dir=run_dir)


def _run_or_reuse_experiment(
    *,
    config: ExperimentConfig,
    project_root: Path,
    run_id: str,
    required_files: tuple[str, ...],
    skip_existing_runs: bool,
    progress: bool,
) -> Path:
    if skip_existing_runs:
        existing_run_dir = _find_existing_run_dir(
            project_root=project_root,
            runs_root=config.paths.runs_root,
            run_id=run_id,
            required_files=required_files,
        )
        if existing_run_dir is not None:
            if _run_velocity_gain_comparison_complete(existing_run_dir):
                _progress_message(f"[reuse] found existing run={existing_run_dir.name}", enabled=progress)
                _ensure_run_figures(run_dir=existing_run_dir, progress=progress)
                return existing_run_dir
            _progress_message(
                (
                    f"[reuse] existing run={existing_run_dir.name} lacks visual/darkness "
                    "gain comparison; running a fresh experiment"
                ),
                enabled=progress,
            )
    run_dir = run_experiment(
        config=config,
        project_root=project_root,
        run_id=run_id,
        make_figures=True,
    )
    _ensure_run_figures(run_dir=run_dir, progress=progress)
    return run_dir


def load_trained_state_from_run(run_dir: Path) -> tuple[ExperimentConfig, VafidisToyState]:
    config = load_experiment_config(run_dir / "config_resolved.yaml")
    trained_weights = load_npz(run_dir / "trained_weights.npz")
    state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
        theta_true=config.simulation.theta0,
    )
    state.w_hd_to_hd = trained_weights["w_hd_to_hd"].copy()
    state.w_hr_to_hd = trained_weights["w_hr_to_hd"].copy()
    state.w_hd_to_hr = trained_weights["w_hd_to_hr"].copy()
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    return config, state


def _metric_row(
    *,
    run_dir: Path,
    metrics: dict[str, float],
    extra_values: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"run_dir": str(run_dir), **extra_values}
    for metric_name in DEFAULT_SWEEP_METRICS:
        row[metric_name] = metrics.get(metric_name, float("nan"))
    return row


def _metrics_with_backfilled_comparison_fields(
    *,
    run_dir: Path,
    metrics: dict[str, float],
) -> dict[str, float]:
    enriched_metrics = dict(metrics)
    if "visual_cue_hd_decode_rms_error" not in enriched_metrics:
        darkness_history_path = run_dir / "darkness_history.npz"
        if darkness_history_path.exists():
            try:
                darkness_history = load_npz(darkness_history_path)
                visual_cue_mask = phase_mask(darkness_history, VISUAL_CUE_PHASE_ID)
                visual_cue_error = circular_error_trace(
                    darkness_history["theta_hd_decoded"][visual_cue_mask],
                    darkness_history["theta_true"][visual_cue_mask],
                )
                finite_error = visual_cue_error[np.isfinite(visual_cue_error)]
                if finite_error.size:
                    enriched_metrics["visual_cue_hd_decode_rms_error"] = float(
                        np.sqrt(np.mean(finite_error**2))
                    )
                    enriched_metrics["visual_cue_hd_decode_mean_abs_error"] = float(
                        np.mean(np.abs(finite_error))
                    )
                    enriched_metrics["visual_cue_hd_decode_final_abs_error"] = float(abs(finite_error[-1]))
            except Exception as exc:  # pragma: no cover - best-effort legacy backfill
                _progress_message(
                    f"[warn] could not backfill visual-cue HD decode error for run={run_dir.name}: {exc}",
                    enabled=True,
                )
    if "darkness_hd_decode_rms_error" not in enriched_metrics and "darkness_rms_pi_error" in enriched_metrics:
        enriched_metrics["darkness_hd_decode_rms_error"] = float(enriched_metrics["darkness_rms_pi_error"])
    if (
        "darkness_hd_decode_final_abs_error" not in enriched_metrics
        and "darkness_final_abs_pi_error" in enriched_metrics
    ):
        enriched_metrics["darkness_hd_decode_final_abs_error"] = float(enriched_metrics["darkness_final_abs_pi_error"])
    if "darkness_velocity_gain" not in enriched_metrics and "velocity_gain" in enriched_metrics:
        enriched_metrics["darkness_velocity_gain"] = float(enriched_metrics["velocity_gain"])
    if "darkness_velocity_gain_abs_error" not in enriched_metrics and "velocity_gain_abs_error" in enriched_metrics:
        enriched_metrics["darkness_velocity_gain_abs_error"] = float(enriched_metrics["velocity_gain_abs_error"])
    if "darkness_velocity_tracking_rmse" not in enriched_metrics and "velocity_tracking_rmse" in enriched_metrics:
        enriched_metrics["darkness_velocity_tracking_rmse"] = float(enriched_metrics["velocity_tracking_rmse"])
    if (
        "darkness_velocity_tracking_max_abs_error" not in enriched_metrics
        and "velocity_tracking_max_abs_error" in enriched_metrics
    ):
        enriched_metrics["darkness_velocity_tracking_max_abs_error"] = float(
            enriched_metrics["velocity_tracking_max_abs_error"]
        )
    if "velocity_gain_abs_error" not in enriched_metrics and "velocity_gain" in enriched_metrics:
        enriched_metrics["velocity_gain_abs_error"] = float(abs(float(enriched_metrics["velocity_gain"]) - 1.0))
    if "velocity_gain_intercept_abs" not in enriched_metrics and "velocity_gain_intercept" in enriched_metrics:
        enriched_metrics["velocity_gain_intercept_abs"] = float(abs(float(enriched_metrics["velocity_gain_intercept"])))
    if "darkness_velocity_gain_abs_error" not in enriched_metrics and "darkness_velocity_gain" in enriched_metrics:
        enriched_metrics["darkness_velocity_gain_abs_error"] = float(
            abs(float(enriched_metrics["darkness_velocity_gain"]) - 1.0)
        )
    if "visual_velocity_gain_abs_error" not in enriched_metrics and "visual_velocity_gain" in enriched_metrics:
        enriched_metrics["visual_velocity_gain_abs_error"] = float(
            abs(float(enriched_metrics["visual_velocity_gain"]) - 1.0)
        )
    if (
        "darkness_minus_visual_velocity_gain" not in enriched_metrics
        and "darkness_velocity_gain" in enriched_metrics
        and "visual_velocity_gain" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_gain"] = float(
            enriched_metrics["darkness_velocity_gain"]
        ) - float(enriched_metrics["visual_velocity_gain"])
    if (
        "darkness_minus_visual_velocity_gain_abs_error" not in enriched_metrics
        and "darkness_velocity_gain_abs_error" in enriched_metrics
        and "visual_velocity_gain_abs_error" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_gain_abs_error"] = float(
            enriched_metrics["darkness_velocity_gain_abs_error"]
        ) - float(enriched_metrics["visual_velocity_gain_abs_error"])
    if (
        "darkness_minus_visual_velocity_tracking_rmse" not in enriched_metrics
        and "darkness_velocity_tracking_rmse" in enriched_metrics
        and "visual_velocity_tracking_rmse" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_tracking_rmse"] = float(
            enriched_metrics["darkness_velocity_tracking_rmse"]
        ) - float(enriched_metrics["visual_velocity_tracking_rmse"])
    if "darkness_abs_velocity_bias" not in enriched_metrics and "darkness_velocity_bias" in enriched_metrics:
        enriched_metrics["darkness_abs_velocity_bias"] = float(abs(float(enriched_metrics["darkness_velocity_bias"])))
    if (
        "bump_abs_intrinsic_drift_velocity" not in enriched_metrics
        and "bump_intrinsic_drift_velocity" in enriched_metrics
    ):
        enriched_metrics["bump_abs_intrinsic_drift_velocity"] = float(
            abs(float(enriched_metrics["bump_intrinsic_drift_velocity"]))
        )

    if "bump_ensemble_diffusion_coefficient" not in enriched_metrics:
        try:
            trained_config, trained_state = load_trained_state_from_run(run_dir)
            _diffusion_history, diffusion_metrics = run_bump_diffusion_ensemble_test(
                config=trained_config,
                trained_state=trained_state,
            )
            enriched_metrics.update(diffusion_metrics)
        except Exception as exc:  # pragma: no cover - best-effort legacy backfill
            _progress_message(
                f"[warn] could not backfill ensemble diffusion for run={run_dir.name}: {exc}",
                enabled=True,
            )

    velocity_history_path = run_dir / "velocity_gain_history.npz"
    needs_velocity_tracking = any(
        metric_name not in enriched_metrics
        for metric_name in (
            "visual_velocity_gain",
            "visual_velocity_tracking_rmse",
            "visual_velocity_tracking_max_abs_error",
            "darkness_velocity_gain",
            "darkness_velocity_tracking_rmse",
            "darkness_velocity_tracking_max_abs_error",
            "darkness_velocity_gain_r_squared",
            "darkness_velocity_tracking_operating_range_max_abs_velocity",
            "darkness_minus_visual_velocity_gain",
            "velocity_tracking_rmse",
            "velocity_tracking_mae",
            "velocity_tracking_max_abs_error",
            "velocity_tracking_bias",
            "velocity_tracking_rmse_fraction_of_max_command",
            "velocity_direction_match_fraction",
        )
    )
    if needs_velocity_tracking and velocity_history_path.exists():
        velocity_history = load_npz(velocity_history_path)
        commanded_velocity = velocity_history.get("commanded_velocity", None)
        if commanded_velocity is not None:
            phase_history_fields = {
                "darkness": "decoded_velocity_darkness"
                if "decoded_velocity_darkness" in velocity_history
                else "decoded_velocity",
                "visual": "decoded_velocity_visual",
            }
            for phase_name, decoded_field in phase_history_fields.items():
                if decoded_field not in velocity_history:
                    continue
                decoded_velocity = velocity_history[decoded_field]
                gain_summary = summarize_velocity_gain(
                    commanded_velocity=commanded_velocity,
                    decoded_velocity=decoded_velocity,
                )
                tracking_summary = summarize_velocity_tracking(
                    commanded_velocity=commanded_velocity,
                    decoded_velocity=decoded_velocity,
                )
                enriched_metrics.setdefault(f"{phase_name}_velocity_gain", gain_summary["gain"])
                enriched_metrics.setdefault(
                    f"{phase_name}_velocity_gain_r_squared",
                    gain_summary["r_squared"],
                )
                enriched_metrics.setdefault(
                    f"{phase_name}_velocity_gain_linear_fit_rmse",
                    gain_summary["linear_fit_rmse"],
                )
                enriched_metrics.setdefault(
                    f"{phase_name}_velocity_tracking_operating_range_max_abs_velocity",
                    estimate_velocity_tracking_operating_range(
                        commanded_velocity=commanded_velocity,
                        decoded_velocity=decoded_velocity,
                    ),
                )
                enriched_metrics.setdefault(
                    f"{phase_name}_velocity_gain_abs_error",
                    float(abs(gain_summary["gain"] - 1.0)),
                )
                enriched_metrics.setdefault(f"{phase_name}_velocity_gain_intercept", gain_summary["intercept"])
                for key, value in tracking_summary.items():
                    enriched_metrics.setdefault(f"{phase_name}_{key}", value)
            if "decoded_velocity" in velocity_history:
                enriched_metrics.update(
                    {
                        key: enriched_metrics.get(f"darkness_{key}", value)
                        for key, value in summarize_velocity_tracking(
                            commanded_velocity=commanded_velocity,
                            decoded_velocity=velocity_history["decoded_velocity"],
                        ).items()
                    }
                )
    if (
        "darkness_minus_visual_velocity_gain" not in enriched_metrics
        and "darkness_velocity_gain" in enriched_metrics
        and "visual_velocity_gain" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_gain"] = float(
            enriched_metrics["darkness_velocity_gain"]
        ) - float(enriched_metrics["visual_velocity_gain"])
    if (
        "darkness_minus_visual_velocity_gain_abs_error" not in enriched_metrics
        and "darkness_velocity_gain_abs_error" in enriched_metrics
        and "visual_velocity_gain_abs_error" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_gain_abs_error"] = float(
            enriched_metrics["darkness_velocity_gain_abs_error"]
        ) - float(enriched_metrics["visual_velocity_gain_abs_error"])
    if (
        "darkness_minus_visual_velocity_tracking_rmse" not in enriched_metrics
        and "darkness_velocity_tracking_rmse" in enriched_metrics
        and "visual_velocity_tracking_rmse" in enriched_metrics
    ):
        enriched_metrics["darkness_minus_visual_velocity_tracking_rmse"] = float(
            enriched_metrics["darkness_velocity_tracking_rmse"]
        ) - float(enriched_metrics["visual_velocity_tracking_rmse"])
    if "velocity_gain" not in enriched_metrics and "darkness_velocity_gain" in enriched_metrics:
        enriched_metrics["velocity_gain"] = float(enriched_metrics["darkness_velocity_gain"])
    if "velocity_gain_abs_error" not in enriched_metrics and "darkness_velocity_gain_abs_error" in enriched_metrics:
        enriched_metrics["velocity_gain_abs_error"] = float(enriched_metrics["darkness_velocity_gain_abs_error"])
    return enriched_metrics


def _aggregate_sweep_rows(
    rows: list[dict[str, float | int | str]],
    *,
    x_key: str,
) -> list[dict[str, float | int | str]]:
    grouped_rows: dict[float, list[dict[str, float | int | str]]] = {}
    representative_x_values: dict[float, float | int | str] = {}
    for row in rows:
        x_value = row[x_key]
        x_float = float(x_value)
        grouped_rows.setdefault(x_float, []).append(row)
        representative_x_values.setdefault(x_float, x_value)

    aggregate_rows: list[dict[str, float | int | str]] = []
    for x_float in sorted(grouped_rows):
        group_rows = grouped_rows[x_float]
        aggregate_row: dict[str, float | int | str] = {
            x_key: representative_x_values[x_float],
            "repeat_count": len(group_rows),
        }
        for metric_name in DEFAULT_SWEEP_METRICS:
            values = np.asarray(
                [float(row.get(metric_name, float("nan"))) for row in group_rows],
                dtype=float,
            )
            finite_values = values[np.isfinite(values)]
            metric_n = int(finite_values.size)
            if metric_n == 0:
                metric_mean = float("nan")
                metric_std = float("nan")
                metric_sem = float("nan")
            else:
                metric_mean = float(np.mean(finite_values))
                metric_std = float(np.std(finite_values, ddof=1)) if metric_n > 1 else 0.0
                metric_sem = metric_std / float(np.sqrt(metric_n)) if metric_n > 1 else 0.0
            aggregate_row[f"{metric_name}_mean"] = metric_mean
            aggregate_row[f"{metric_name}_std"] = metric_std
            aggregate_row[f"{metric_name}_sem"] = metric_sem
            aggregate_row[f"{metric_name}_n"] = metric_n
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _aggregate_sweep_grid_rows(
    rows: list[dict[str, float | int | str]],
    *,
    x_key: str,
    y_key: str,
) -> list[dict[str, float | int | str]]:
    grouped_rows: dict[tuple[float, float], list[dict[str, float | int | str]]] = {}
    representative_values: dict[tuple[float, float], tuple[float | int | str, float | int | str]] = {}
    for row in rows:
        x_value = row[x_key]
        y_value = row[y_key]
        grid_key = (float(x_value), float(y_value))
        grouped_rows.setdefault(grid_key, []).append(row)
        representative_values.setdefault(grid_key, (x_value, y_value))

    aggregate_rows: list[dict[str, float | int | str]] = []
    for grid_key in sorted(grouped_rows, key=lambda value: (value[1], value[0])):
        group_rows = grouped_rows[grid_key]
        representative_x, representative_y = representative_values[grid_key]
        aggregate_row: dict[str, float | int | str] = {
            x_key: representative_x,
            y_key: representative_y,
            "repeat_count": len(group_rows),
        }
        for metric_name in DEFAULT_SWEEP_METRICS:
            values = np.asarray(
                [float(row.get(metric_name, float("nan"))) for row in group_rows],
                dtype=float,
            )
            finite_values = values[np.isfinite(values)]
            metric_n = int(finite_values.size)
            if metric_n == 0:
                metric_mean = float("nan")
                metric_std = float("nan")
                metric_sem = float("nan")
            else:
                metric_mean = float(np.mean(finite_values))
                metric_std = float(np.std(finite_values, ddof=1)) if metric_n > 1 else 0.0
                metric_sem = metric_std / float(np.sqrt(metric_n)) if metric_n > 1 else 0.0
            aggregate_row[f"{metric_name}_mean"] = metric_mean
            aggregate_row[f"{metric_name}_std"] = metric_std
            aggregate_row[f"{metric_name}_sem"] = metric_sem
            aggregate_row[f"{metric_name}_n"] = metric_n
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _metric_value(row: dict[str, float | int | str], metric_name: str) -> float:
    return float(row.get(metric_name, float("nan")))


def _compute_noise_paired_delta_rows(
    rows: list[dict[str, float | int | str]],
    *,
    baseline_noise_std: float = 0.0,
) -> list[dict[str, float | int | str]]:
    baseline_by_seed_offset: dict[int, dict[str, float | int | str]] = {}
    for row in rows:
        if np.isclose(float(row["visual_noise_std"]), baseline_noise_std):
            baseline_by_seed_offset[int(row["seed_offset"])] = row

    delta_rows: list[dict[str, float | int | str]] = []
    for row in rows:
        seed_offset = int(row["seed_offset"])
        baseline_row = baseline_by_seed_offset.get(seed_offset)
        if baseline_row is None:
            continue
        delta_row: dict[str, float | int | str] = {
            "run_dir": row["run_dir"],
            "baseline_run_dir": baseline_row["run_dir"],
            "visual_noise_std": float(row["visual_noise_std"]),
            "repeat_index": int(row["repeat_index"]),
            "seed_offset": seed_offset,
            "seed": int(row["seed"]),
            "baseline_seed": int(baseline_row["seed"]),
        }
        visual_gain_value = _metric_value(row, "visual_velocity_gain")
        baseline_visual_gain_value = _metric_value(baseline_row, "visual_velocity_gain")
        darkness_gain_value = _metric_value(row, "darkness_velocity_gain")
        baseline_darkness_gain_value = _metric_value(baseline_row, "darkness_velocity_gain")
        delta_row["delta_visual_velocity_gain_abs_error"] = (
            abs(visual_gain_value - 1.0) - abs(baseline_visual_gain_value - 1.0)
        )
        delta_row["delta_darkness_velocity_gain_abs_error"] = (
            abs(darkness_gain_value - 1.0) - abs(baseline_darkness_gain_value - 1.0)
        )
        for metric_name in DEFAULT_SWEEP_METRICS:
            delta_row[f"delta_{metric_name}"] = _metric_value(row, metric_name) - _metric_value(
                baseline_row,
                metric_name,
            )
        delta_rows.append(delta_row)
    return delta_rows


def _bootstrap_median_interval(values: np.ndarray, *, rng_seed: int = 1729) -> tuple[float, float, float]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    median_value = float(np.median(finite_values))
    if finite_values.size == 1:
        return median_value, median_value, median_value
    rng = np.random.default_rng(rng_seed)
    bootstrap_samples = rng.choice(finite_values, size=(2000, finite_values.size), replace=True)
    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    return (
        median_value,
        float(np.percentile(bootstrap_medians, 2.5)),
        float(np.percentile(bootstrap_medians, 97.5)),
    )


def _aggregate_noise_delta_rows(
    delta_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped_rows: dict[float, list[dict[str, float | int | str]]] = {}
    for row in delta_rows:
        grouped_rows.setdefault(float(row["visual_noise_std"]), []).append(row)

    aggregate_rows: list[dict[str, float | int | str]] = []
    for noise_std in sorted(grouped_rows):
        group_rows = grouped_rows[noise_std]
        aggregate_row: dict[str, float | int | str] = {
            "visual_noise_std": noise_std,
            "paired_count": len(group_rows),
        }
        for metric_name in NOISE_DELTA_METRICS:
            values = np.asarray([_metric_value(row, metric_name) for row in group_rows], dtype=float)
            finite_values = values[np.isfinite(values)]
            median_value, ci_low, ci_high = _bootstrap_median_interval(
                finite_values,
                rng_seed=1729 + int(round(noise_std * 1000.0)),
            )
            aggregate_row[f"{metric_name}_median"] = median_value
            aggregate_row[f"{metric_name}_ci_low"] = ci_low
            aggregate_row[f"{metric_name}_ci_high"] = ci_high
            aggregate_row[f"{metric_name}_n"] = int(finite_values.size)
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _plot_sweep_summary(
    *,
    rows: list[dict[str, float | int | str]],
    x_key: str,
    path: Path,
    title: str,
) -> None:
    if not rows:
        return
    aggregate_rows = _aggregate_sweep_rows(rows, x_key=x_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray([float(row[x_key]) for row in aggregate_rows], dtype=float)
    n_metrics = len(DEFAULT_SWEEP_METRICS)
    n_columns = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, metric_name in zip(flat_axes, DEFAULT_SWEEP_METRICS, strict=False):
        y_values = np.asarray(
            [float(row.get(f"{metric_name}_mean", float("nan"))) for row in aggregate_rows],
            dtype=float,
        )
        y_errors = np.asarray(
            [float(row.get(f"{metric_name}_sem", float("nan"))) for row in aggregate_rows],
            dtype=float,
        )
        y_errors = np.where(np.isfinite(y_errors), y_errors, 0.0)
        axis.errorbar(x_values, y_values, yerr=y_errors, marker="o", linewidth=1.3, capsize=3)
        title, y_label = SWEEP_METRIC_PLOT_LABELS.get(metric_name, (metric_name, "mean +/- SEM"))
        axis.set_title(_wrap_plot_label(title, width=26), fontsize=9)
        axis.set_xlabel(_wrap_plot_label(SWEEP_X_AXIS_LABELS.get(x_key, x_key), width=28), fontsize=8)
        axis.set_ylabel(f"{_wrap_plot_label(y_label, width=30)}\nmean +/- SEM", fontsize=8, labelpad=8)
        axis.tick_params(axis="both", labelsize=8)
        if metric_name in {"velocity_gain", "visual_velocity_gain", "darkness_velocity_gain"}:
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        elif metric_name == "darkness_minus_visual_velocity_gain":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.grid(alpha=0.25)
    for axis in flat_axes[n_metrics:]:
        axis.set_visible(False)
    fig.suptitle(f"{title} (mean +/- SEM)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format_meshgrid_value(value: float, sem: float, n_value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if np.isfinite(sem) and sem > 0.0 and n_value > 1:
        return f"{value:.3g}\n+/-{sem:.2g}"
    return f"{value:.3g}"


def _meshgrid_text_color(cmap, norm: mcolors.Normalize, value: float) -> str:
    if not np.isfinite(value):
        return "0.35"
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.48 else "black"


def _metric_performance_score_grid(metric_name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    performance_values = np.full_like(values, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return performance_values

    if metric_name in SWEEP_TARGET_VALUE_METRICS:
        target_value = SWEEP_TARGET_VALUE_METRICS[metric_name]
        raw_scores = -np.abs(values - target_value)
    elif metric_name in SWEEP_LOWER_IS_BETTER_METRICS:
        raw_scores = -values
    elif metric_name in SWEEP_HIGHER_IS_BETTER_METRICS:
        raw_scores = values
    else:
        raw_scores = values

    finite_scores = raw_scores[finite_mask]
    min_score = float(np.min(finite_scores))
    max_score = float(np.max(finite_scores))
    if np.isclose(min_score, max_score):
        ideal_value = SWEEP_IDEAL_TIE_VALUES.get(metric_name)
        if ideal_value is not None and np.allclose(values[finite_mask], ideal_value):
            performance_values[finite_mask] = 1.0
        else:
            performance_values[finite_mask] = 0.5
        return performance_values

    performance_values[finite_mask] = (finite_scores - min_score) / (max_score - min_score)
    return performance_values


def _plot_sweep_metric_meshgrid(
    *,
    aggregate_rows: list[dict[str, float | int | str]],
    x_key: str,
    y_key: str,
    path: Path,
    title: str,
    metric_names: list[str] | None = None,
) -> None:
    if not aggregate_rows:
        return
    selected_metric_names = DEFAULT_SWEEP_METRICS if metric_names is None else metric_names
    if not selected_metric_names:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = sorted({float(row[x_key]) for row in aggregate_rows})
    y_values = sorted({float(row[y_key]) for row in aggregate_rows})
    x_index_by_value = {x_value: index for index, x_value in enumerate(x_values)}
    y_index_by_value = {y_value: index for index, y_value in enumerate(y_values)}

    n_metrics = len(selected_metric_names)
    n_columns = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.6 * n_columns, 3.8 * n_rows),
        squeeze=False,
    )
    top_margin = 0.82 if n_rows == 1 else 0.93
    bottom_margin = 0.15 if n_rows == 1 else 0.07
    left_margin = 0.18 if n_columns == 1 else 0.07
    right_margin = 0.88 if n_columns == 1 else 0.96
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=0.34,
        hspace=0.42,
    )
    flat_axes = axes.ravel()
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    performance_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for axis, metric_name in zip(flat_axes, selected_metric_names, strict=False):
        value_grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
        sem_grid = np.full_like(value_grid, np.nan)
        n_grid = np.zeros_like(value_grid)
        for row in aggregate_rows:
            x_index = x_index_by_value[float(row[x_key])]
            y_index = y_index_by_value[float(row[y_key])]
            value_grid[y_index, x_index] = float(row.get(f"{metric_name}_mean", float("nan")))
            sem_grid[y_index, x_index] = float(row.get(f"{metric_name}_sem", float("nan")))
            n_grid[y_index, x_index] = float(row.get(f"{metric_name}_n", 0.0))

        performance_grid = _metric_performance_score_grid(metric_name, value_grid)
        image = axis.imshow(
            np.ma.masked_invalid(performance_grid),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=performance_norm,
        )
        for y_index in range(len(y_values)):
            for x_index in range(len(x_values)):
                value = value_grid[y_index, x_index]
                performance_score = performance_grid[y_index, x_index]
                axis.text(
                    x_index,
                    y_index,
                    _format_meshgrid_value(value, sem_grid[y_index, x_index], n_grid[y_index, x_index]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_meshgrid_text_color(cmap, performance_norm, performance_score),
                )

        metric_title, _ = SWEEP_METRIC_PLOT_LABELS.get(metric_name, (metric_name, metric_name))
        axis.set_title(_wrap_plot_label(metric_title, width=28), fontsize=8.5)
        axis.set_xticks(np.arange(len(x_values)))
        axis.set_yticks(np.arange(len(y_values)))
        axis.set_xticklabels([f"{value:g}" for value in x_values], rotation=45, ha="right")
        axis.set_yticklabels([f"{value:g}" for value in y_values])
        axis.set_xlabel(_wrap_plot_label(SWEEP_X_AXIS_LABELS.get(x_key, x_key), width=28), fontsize=8)
        axis.set_ylabel(_wrap_plot_label(SWEEP_X_AXIS_LABELS.get(y_key, y_key), width=28), fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
        colorbar.set_ticks([0.0, 0.5, 1.0])
        colorbar.set_ticklabels(["worse", "mid", "better"])
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("relative performance", fontsize=7)

    for axis in flat_axes[n_metrics:]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_noise_seed_lines(
    *,
    rows: list[dict[str, float | int | str]],
    path: Path,
    title: str = "Visual-noise raw seed trajectories",
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped_rows: dict[int, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped_rows.setdefault(int(row["seed"]), []).append(row)

    n_metrics = len(DEFAULT_SWEEP_METRICS)
    n_columns = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(grouped_rows))))
    flat_axes = axes.ravel()
    for axis, metric_name in zip(flat_axes, DEFAULT_SWEEP_METRICS, strict=False):
        for color, seed in zip(colors, sorted(grouped_rows), strict=False):
            seed_rows = sorted(grouped_rows[seed], key=lambda row: float(row["visual_noise_std"]))
            x_values = np.asarray([float(row["visual_noise_std"]) for row in seed_rows], dtype=float)
            y_values = np.asarray([_metric_value(row, metric_name) for row in seed_rows], dtype=float)
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.0,
                alpha=0.8,
                color=color,
                label=f"seed {seed}",
            )
        title, y_label = SWEEP_METRIC_PLOT_LABELS.get(metric_name, (metric_name, metric_name))
        axis.set_title(_wrap_plot_label(title, width=26), fontsize=9)
        axis.set_xlabel(_wrap_plot_label(SWEEP_X_AXIS_LABELS["visual_noise_std"], width=28), fontsize=8)
        axis.set_ylabel(_wrap_plot_label(y_label, width=30), fontsize=8, labelpad=8)
        axis.tick_params(axis="both", labelsize=8)
        if metric_name in {"velocity_gain", "visual_velocity_gain", "darkness_velocity_gain"}:
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        elif metric_name == "darkness_minus_visual_velocity_gain":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.grid(alpha=0.25)
    for axis in flat_axes[n_metrics:]:
        axis.set_visible(False)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=min(4, len(handles)), fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_noise_delta_summary(
    *,
    delta_rows: list[dict[str, float | int | str]],
    path: Path,
    title: str = "Visual-noise paired change from sigma=0 baseline",
) -> None:
    if not delta_rows:
        return
    aggregate_rows = _aggregate_noise_delta_rows(delta_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray([float(row["visual_noise_std"]) for row in aggregate_rows], dtype=float)
    n_metrics = len(NOISE_DELTA_METRICS)
    n_columns = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, metric_name in zip(flat_axes, NOISE_DELTA_METRICS, strict=False):
        y_values = np.asarray(
            [float(row.get(f"{metric_name}_median", float("nan"))) for row in aggregate_rows],
            dtype=float,
        )
        ci_low = np.asarray(
            [float(row.get(f"{metric_name}_ci_low", float("nan"))) for row in aggregate_rows],
            dtype=float,
        )
        ci_high = np.asarray(
            [float(row.get(f"{metric_name}_ci_high", float("nan"))) for row in aggregate_rows],
            dtype=float,
        )
        lower_error = np.where(np.isfinite(ci_low), y_values - ci_low, 0.0)
        upper_error = np.where(np.isfinite(ci_high), ci_high - y_values, 0.0)
        y_errors = np.vstack([lower_error, upper_error])
        axis.errorbar(x_values, y_values, yerr=y_errors, marker="o", linewidth=1.3, capsize=3)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        title, y_label = NOISE_DELTA_METRIC_PLOT_LABELS.get(metric_name, (metric_name, metric_name))
        axis.set_title(_wrap_plot_label(title, width=26), fontsize=9)
        axis.set_xlabel(_wrap_plot_label(SWEEP_X_AXIS_LABELS["visual_noise_std"], width=28), fontsize=8)
        axis.set_ylabel(f"{_wrap_plot_label(y_label, width=30)}\nmedian, 95% bootstrap CI", fontsize=8, labelpad=8)
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(alpha=0.25)
    for axis in flat_axes[n_metrics:]:
        axis.set_visible(False)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_noise_sweep(
    *,
    base_config: ExperimentConfig,
    project_root: Path,
    output_dir: Path,
    noise_stds: list[float],
    run_id_prefix: str,
    train_duration: float | None,
    save_interval_duration: float | None,
    training_progress: bool,
    progress: bool,
    skip_existing_runs: bool,
    seed_offsets: list[int],
    min_visual_sigma_bins: float | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    seed_offsets = _normalize_seed_offsets(seed_offsets)
    run_specs = [
        (noise_index, noise_std, repeat_index, seed_offset)
        for noise_index, noise_std in enumerate(noise_stds, start=1)
        for repeat_index, seed_offset in enumerate(seed_offsets, start=1)
    ]
    indexed_run_specs = list(enumerate(run_specs, start=1))
    for run_position, (noise_index, noise_std, repeat_index, seed_offset) in _progress_iter(
        indexed_run_specs,
        desc="visual-noise sweep",
        enabled=progress,
    ):
        seed = int(base_config.simulation.seed) + int(seed_offset)
        _progress_message(
            (
                f"[noise run {run_position}/{len(run_specs)}; "
                f"value {noise_index}/{len(noise_stds)}; repeat {repeat_index}/{len(seed_offsets)}] "
                f"start visual.noise_std={noise_std}, seed={seed}"
            ),
            enabled=progress,
        )
        config = _clone_config_with_runtime_overrides(
            base_config,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
        )
        config.simulation.seed = seed
        _retune_visual_width_for_neuron_count(
            config=config,
            n_theta=int(config.model.n_theta),
            min_visual_sigma_bins=min_visual_sigma_bins,
        )
        config.visual.noise_std = float(noise_std)
        run_id = f"{run_id_prefix}_{_visual_noise_run_label(config)}"
        if seed_offset != 0:
            run_id = f"{run_id}_seed_{seed}"
        run_dir = _run_or_reuse_experiment(
            config=config,
            project_root=project_root,
            run_id=run_id,
            required_files=("test_metrics.json",),
            skip_existing_runs=skip_existing_runs,
            progress=progress,
        )
        metrics = _metrics_with_backfilled_comparison_fields(
            run_dir=run_dir,
            metrics=load_json(run_dir / "test_metrics.json"),
        )
        _progress_message(
            (
                f"[noise run {run_position}/{len(run_specs)}] done "
                f"run={run_dir.name}, seed={seed}, "
                f"visual_gain={metrics.get('visual_velocity_gain', float('nan')):.4g}, "
                f"dark_gain={metrics.get('darkness_velocity_gain', float('nan')):.4g}, "
                f"dark_rmse={metrics.get('darkness_velocity_tracking_rmse', float('nan')):.4g}"
            ),
            enabled=progress,
        )
        rows.append(
            _metric_row(
                run_dir=run_dir,
                metrics=metrics,
                extra_values={
                    "visual_noise_std": float(noise_std),
                    "repeat_index": int(repeat_index),
                    "seed_offset": int(seed_offset),
                    "seed": int(seed),
                    **_visual_noise_metadata(config),
                },
            )
        )
    _write_csv(output_dir / "noise_sweep_summary.csv", rows)
    for stale_path in (
        output_dir / "noise_sweep_aggregate_summary.csv",
        output_dir / "noise_sweep_paired_delta_summary.csv",
        output_dir / "noise_sweep_paired_delta_aggregate_summary.csv",
        output_dir / "figures" / "noise_sweep_paired_seed_lines.png",
        output_dir / "figures" / "noise_sweep_paired_delta_metrics.png",
    ):
        _remove_file_if_exists(stale_path)
    _plot_noise_seed_lines(
        rows=rows,
        path=output_dir / "figures" / "noise_sweep_metrics.png",
    )
    return rows


def _plot_cross_mouse_tuning_moments(
    *,
    theta_aligned: np.ndarray,
    mouse_mean_curves: np.ndarray,
    mouse_std_curves: np.ndarray,
    path: Path,
    normalization_label: str,
    n_theta: int,
) -> None:
    """Plot Clark-style per-mouse moments and their cross-mouse average."""
    theta_aligned = np.asarray(theta_aligned, dtype=float)
    mouse_mean_curves = np.asarray(mouse_mean_curves, dtype=float)
    mouse_std_curves = np.asarray(mouse_std_curves, dtype=float)
    expected_shape = (mouse_mean_curves.shape[0], theta_aligned.size)
    if (
        theta_aligned.ndim != 1
        or mouse_mean_curves.ndim != 2
        or mouse_std_curves.shape != expected_shape
    ):
        raise ValueError("cross-mouse tuning moments must have shape (mouse, heading)")
    if mouse_mean_curves.shape[0] < 2:
        raise ValueError("cross-mouse tuning analysis requires at least two simulated mice")

    theta_closed = np.concatenate([theta_aligned, [theta_aligned[0] + 2.0 * np.pi]])
    means_closed = np.concatenate([mouse_mean_curves, mouse_mean_curves[:, :1]], axis=1)
    stds_closed = np.concatenate([mouse_std_curves, mouse_std_curves[:, :1]], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharex=True)
    mouse_curve_color = "#c7c7c7"
    for mouse_mean in means_closed:
        axes[0].plot(
            theta_closed,
            mouse_mean,
            color=mouse_curve_color,
            linewidth=0.9,
            alpha=0.72,
        )
    axes[0].plot(
        theta_closed,
        np.mean(means_closed, axis=0),
        color="#111111",
        linewidth=3.0,
        label=f"cross-mouse mean (K={mouse_mean_curves.shape[0]})",
    )
    axes[0].set_ylabel(f"{normalization_label} within-mouse mean")
    axes[0].set_title("Mean tuning profile")

    for mouse_std in stds_closed:
        axes[1].plot(
            theta_closed,
            mouse_std,
            color=mouse_curve_color,
            linewidth=0.9,
            alpha=0.72,
        )
    axes[1].plot(
        theta_closed,
        np.mean(stds_closed, axis=0),
        color="#111111",
        linewidth=3.0,
        label=f"cross-mouse mean std (K={mouse_std_curves.shape[0]})",
    )
    axes[1].set_ylabel(f"{normalization_label} within-mouse neuron std")
    axes[1].set_title("Tuning heterogeneity")

    radian_ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    radian_labels = ["-pi", "-pi/2", "0", "pi/2", "pi"]
    for axis in axes:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks(radian_ticks)
        axis.set_xticklabels(radian_labels)
        axis.set_xlabel("heading relative to circular COM [rad]")
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Simulated-mouse COM-aligned HD tuning (N={n_theta} cells per mouse)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_converged_tuning_archives(
    *,
    rows: list[dict[str, float | int | str]],
    minimum_converged_fraction: float | None,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, np.ndarray]],
]:
    """Load seed-ordered tuning archives that pass the convergence filter."""
    available_rows: list[dict[str, float | int | str]] = []
    tuning_archives: list[dict[str, np.ndarray]] = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        convergence_fraction = float(
            row.get("hd_tuning_curve_converged_fraction", float("nan"))
        )
        if (
            minimum_converged_fraction is not None
            and np.isfinite(convergence_fraction)
            and convergence_fraction < minimum_converged_fraction
        ):
            continue
        archive_path = Path(str(row["run_dir"])) / "hd_tuning_com_aligned.npz"
        if not archive_path.exists():
            continue
        available_rows.append(row)
        tuning_archives.append(load_npz(archive_path))
    return available_rows, tuning_archives


def _write_cross_mouse_tuning_summaries(
    *,
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    normalizations: list[str] | None = None,
    minimum_converged_fraction: float | None = 0.99,
) -> list[dict[str, float | int | str]]:
    """Aggregate equal-N seed replicates as independent simulated mice."""
    grouped_rows: dict[int, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped_rows.setdefault(int(row["n_theta"]), []).append(row)
    summary_rows: list[dict[str, float | int | str]] = []
    normalization_specs_by_label = {
        "per_neuron_peak": (
            "r_hd_peak_normalized_com_aligned_mean",
            "r_hd_peak_normalized_com_aligned_std",
        ),
        "unit_mean_clark": (
            "r_hd_unit_mean_com_aligned_mean",
            "r_hd_unit_mean_com_aligned_std",
        ),
    }
    selected_normalizations = (
        list(normalization_specs_by_label)
        if normalizations is None
        else [str(label) for label in normalizations]
    )
    if not selected_normalizations:
        raise ValueError("cross-mouse tuning normalizations must not be empty")
    unknown_normalizations = sorted(
        set(selected_normalizations).difference(normalization_specs_by_label)
    )
    if unknown_normalizations:
        raise ValueError(
            "unknown cross-mouse tuning normalizations: "
            + ", ".join(unknown_normalizations)
    )
    for n_theta, group_rows in sorted(grouped_rows.items()):
        available_rows, tuning_archives = _load_converged_tuning_archives(
            rows=group_rows,
            minimum_converged_fraction=minimum_converged_fraction,
        )
        if len(tuning_archives) < 2:
            continue
        theta_aligned = np.asarray(tuning_archives[0]["theta_aligned"], dtype=float)
        for archive in tuning_archives[1:]:
            if not np.allclose(archive["theta_aligned"], theta_aligned):
                raise ValueError("simulated-mouse tuning archives must share one angle grid")
        for normalization_label in selected_normalizations:
            mean_key, std_key = normalization_specs_by_label[normalization_label]
            mouse_mean_curves = np.vstack([archive[mean_key] for archive in tuning_archives])
            mouse_std_curves = np.vstack([archive[std_key] for archive in tuning_archives])
            output_stem = (
                f"heterogeneous_cross_mouse_tuning_n{n_theta}_{normalization_label}"
            )
            save_npz(
                output_dir / f"{output_stem}.npz",
                theta_aligned=theta_aligned,
                mouse_mean_curves=mouse_mean_curves,
                mouse_std_curves=mouse_std_curves,
                cross_mouse_mean_curve=np.mean(mouse_mean_curves, axis=0),
                cross_mouse_mean_std_curve=np.mean(mouse_std_curves, axis=0),
                seeds=np.asarray([int(row["seed"]) for row in available_rows], dtype=int),
                run_dirs=np.asarray([str(row["run_dir"]) for row in available_rows]),
                converged_fractions=np.asarray(
                    [
                        float(
                            row.get(
                                "hd_tuning_curve_converged_fraction",
                                float("nan"),
                            )
                        )
                        for row in available_rows
                    ],
                    dtype=float,
                ),
            )
            _plot_cross_mouse_tuning_moments(
                theta_aligned=theta_aligned,
                mouse_mean_curves=mouse_mean_curves,
                mouse_std_curves=mouse_std_curves,
                path=output_dir / "figures" / f"{output_stem}.png",
                normalization_label=normalization_label,
                n_theta=n_theta,
            )
            summary_rows.append(
                {
                    "n_theta": n_theta,
                    "n_simulated_mice": len(tuning_archives),
                    "normalization": normalization_label,
                    "npz_path": str(output_dir / f"{output_stem}.npz"),
                    "figure_path": str(output_dir / "figures" / f"{output_stem}.png"),
                }
            )
    return summary_rows


def _restore_unaligned_unit_mean_tuning(
    tuning_archive: dict[str, np.ndarray],
) -> np.ndarray:
    """Undo integer COM shifts and retain Clark-valid unit-mean curves."""
    aligned_tuning = np.asarray(
        tuning_archive["r_hd_unit_mean_com_aligned"],
        dtype=float,
    )
    alignment_shifts = np.asarray(
        tuning_archive["com_alignment_shift_bins"],
        dtype=int,
    )
    valid_mask = np.asarray(
        tuning_archive["r_hd_tuning_valid_mask"],
        dtype=bool,
    )
    if (
        aligned_tuning.ndim != 2
        or alignment_shifts.shape != (aligned_tuning.shape[0],)
        or valid_mask.shape != (aligned_tuning.shape[0],)
    ):
        raise ValueError("invalid COM-aligned tuning archive shapes")
    unaligned_tuning = np.vstack(
        [
            np.roll(neuron_tuning, -int(shift_bins))
            for neuron_tuning, shift_bins in zip(
                aligned_tuning,
                alignment_shifts,
                strict=True,
            )
        ]
    )
    valid_mask &= np.all(np.isfinite(unaligned_tuning), axis=1)
    return unaligned_tuning[valid_mask]


def _plot_clark_figure4_abc(
    *,
    empirical_com_by_mouse: np.ndarray,
    uniform_com_by_mouse: np.ndarray,
    correlation_matrices: np.ndarray,
    circulant_errors: np.ndarray,
    example_mouse_indices: np.ndarray,
    example_mouse_seeds: np.ndarray,
    subset_sizes: np.ndarray,
    kuiper_p_values: np.ndarray,
    kuiper_bh_adjusted_p_values: np.ndarray,
    uniformity_alpha: float,
    path: Path,
) -> None:
    """Plot Clark Figure 4A-C for a simulated-mouse cohort."""
    n_mice = empirical_com_by_mouse.shape[0]
    n_examples = example_mouse_indices.size
    n_subset_sizes = subset_sizes.size
    figure_height = max(5.4, 1.25 * n_examples + 0.8)
    fig = plt.figure(figsize=(14.8, figure_height))
    outer_grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.15, 1.15, 5.4),
        left=0.055,
        right=0.94,
        bottom=0.12,
        top=0.88,
        wspace=0.16,
    )
    empirical_axis = fig.add_subplot(outer_grid[0, 0])
    uniform_axis = fig.add_subplot(outer_grid[0, 1], sharey=empirical_axis)
    for axis, com_values, title in (
        (empirical_axis, empirical_com_by_mouse, "post-training data"),
        (uniform_axis, uniform_com_by_mouse, "random uniform"),
    ):
        mouse_grid = np.broadcast_to(
            np.arange(1, n_mice + 1, dtype=float)[:, None],
            com_values.shape,
        )
        finite_mask = np.isfinite(com_values)
        axis.scatter(
            com_values[finite_mask],
            mouse_grid[finite_mask],
            s=4.0,
            c="black",
            linewidths=0.0,
            rasterized=True,
        )
        axis.set_xlim(-np.pi, np.pi)
        axis.set_ylim(0.25, n_mice + 0.75)
        axis.set_xticks(
            [-np.pi, 0.0, np.pi],
            [r"$-\pi$", "0", r"$\pi$"],
        )
        axis.set_yticks(np.arange(5, n_mice + 1, 5))
        axis.set_xlabel("tuning-curve COM [rad]")
        axis.set_title(title, fontsize=10)
        axis.tick_params(length=2.5, labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    empirical_axis.set_ylabel("simulated mouse index")
    raw_rejection_count = int(np.count_nonzero(kuiper_p_values < uniformity_alpha))
    bh_rejection_count = int(
        np.count_nonzero(kuiper_bh_adjusted_p_values < uniformity_alpha)
    )
    empirical_axis.set_title(
        "post-training data\n"
        f"Kuiper p<{uniformity_alpha:g}: {raw_rejection_count}/{n_mice}; "
        f"BH q<{uniformity_alpha:g}: {bh_rejection_count}/{n_mice}",
        fontsize=8,
    )
    uniform_axis.tick_params(labelleft=False)

    correlation_grid = outer_grid[0, 2].subgridspec(
        n_examples,
        n_subset_sizes,
        wspace=0.10,
        hspace=0.20,
    )
    finite_correlations = correlation_matrices[np.isfinite(correlation_matrices)]
    color_max = (
        max(float(np.percentile(finite_correlations, 99.5)), 1e-12)
        if finite_correlations.size
        else 1.0
    )
    image = None
    for mouse_row in range(n_examples):
        for subset_column in range(n_subset_sizes):
            axis = fig.add_subplot(correlation_grid[mouse_row, subset_column])
            image = axis.imshow(
                correlation_matrices[mouse_row, subset_column],
                origin="lower",
                extent=(-np.pi, np.pi, -np.pi, np.pi),
                cmap="viridis",
                vmin=0.0,
                vmax=color_max,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xticks([-np.pi, np.pi], [r"$-\pi$", r"$\pi$"])
            axis.set_yticks(
                [-np.pi, np.pi],
                [r"$-\pi$", r"$\pi$"] if subset_column == 0 else [],
            )
            axis.tick_params(length=1.8, labelsize=6, pad=1)
            if mouse_row == 0:
                axis.set_title(
                    rf"$N_{{sub}}={int(subset_sizes[subset_column])}$",
                    fontsize=8,
                    pad=3,
                )
            if subset_column == 0:
                axis.set_ylabel(
                    f"mouse {int(example_mouse_indices[mouse_row])}\n"
                    f"seed {int(example_mouse_seeds[mouse_row])}\n"
                    + r"$\theta'$",
                    fontsize=7,
                    labelpad=2,
                )
            if mouse_row == n_examples - 1:
                axis.set_xlabel(r"$\theta$", fontsize=7, labelpad=-2)
            axis.text(
                0.03,
                0.96,
                rf"$\epsilon_{{circ}}={circulant_errors[mouse_row, subset_column]:.2f}$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.42, "pad": 0.8, "edgecolor": "none"},
            )
    if image is None:
        raise ValueError("Clark Figure 4C requires correlation matrices")
    colorbar_axis = fig.add_axes([0.951, 0.17, 0.011, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label(r"$C(\theta,\theta')$", fontsize=8)
    fig.text(0.045, 0.91, "A", fontsize=13, fontweight="bold")
    fig.text(0.181, 0.91, "B", fontsize=13, fontweight="bold")
    fig.text(0.315, 0.91, "C", fontsize=13, fontweight="bold")
    fig.suptitle("Clark-style COM coverage and two-point correlation convergence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_clark_figure4_summary(
    *,
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    example_mouse_indices: list[int],
    subset_sizes: list[int],
    random_seed: int,
    minimum_converged_fraction: float | None = 0.99,
    uniformity_alpha: float = 0.05,
) -> dict[str, float | int | str]:
    """Build Clark Figure 4A-C statistics from independent simulated mice."""
    if not example_mouse_indices or min(example_mouse_indices) < 1:
        raise ValueError("Clark Figure 4 example mouse indices must be positive")
    if len(set(example_mouse_indices)) != len(example_mouse_indices):
        raise ValueError("Clark Figure 4 example mouse indices must be unique")
    if not subset_sizes or min(subset_sizes) < 1:
        raise ValueError("Clark Figure 4 subset sizes must be positive")
    if not 0.0 < uniformity_alpha < 1.0:
        raise ValueError("Clark Figure 4 uniformity alpha must lie in (0, 1)")
    subset_sizes = sorted(set(int(size) for size in subset_sizes))

    available_rows, tuning_archives = _load_converged_tuning_archives(
        rows=rows,
        minimum_converged_fraction=minimum_converged_fraction,
    )
    if len(tuning_archives) < max(example_mouse_indices):
        raise ValueError(
            "Clark Figure 4 requested mouse index "
            f"{max(example_mouse_indices)}, but only {len(tuning_archives)} archives are available"
        )

    theta_grid = np.asarray(tuning_archives[0]["theta_aligned"], dtype=float)
    if theta_grid.ndim != 1:
        raise ValueError("Clark Figure 4 requires a one-dimensional heading grid")
    for archive in tuning_archives[1:]:
        if not np.allclose(archive["theta_aligned"], theta_grid):
            raise ValueError("Clark Figure 4 tuning archives must share one angle grid")

    valid_com_values: list[np.ndarray] = []
    for archive in tuning_archives:
        preferred_direction = np.asarray(
            archive["empirical_preferred_direction"],
            dtype=float,
        )
        valid_mask = np.asarray(archive["r_hd_tuning_valid_mask"], dtype=bool)
        if preferred_direction.shape != valid_mask.shape:
            raise ValueError("Clark Figure 4 COM and validity arrays must share shape")
        wrapped_com = (preferred_direction[valid_mask] + np.pi) % (2.0 * np.pi) - np.pi
        valid_com_values.append(wrapped_com[np.isfinite(wrapped_com)])

    maximum_neurons = max(com_values.size for com_values in valid_com_values)
    empirical_com_by_mouse = np.full(
        (len(valid_com_values), maximum_neurons),
        np.nan,
        dtype=float,
    )
    uniform_com_by_mouse = np.full_like(empirical_com_by_mouse, np.nan)
    rng = np.random.default_rng(int(random_seed))
    for mouse_row, com_values in enumerate(valid_com_values):
        empirical_com_by_mouse[mouse_row, : com_values.size] = com_values
        uniform_com_by_mouse[mouse_row, : com_values.size] = rng.uniform(
            -np.pi,
            np.pi,
            size=com_values.size,
        )

    kuiper_statistics = np.empty(len(valid_com_values), dtype=float)
    kuiper_p_values = np.empty(len(valid_com_values), dtype=float)
    for mouse_row, com_values in enumerate(valid_com_values):
        statistic, p_value = kuiper_uniformity_test_asymptotic(com_values)
        kuiper_statistics[mouse_row] = statistic
        kuiper_p_values[mouse_row] = p_value
    kuiper_bh_adjusted_p_values = benjamini_hochberg_adjusted_p_values(
        kuiper_p_values
    )
    uniformity_rows: list[dict[str, float | int | str]] = []
    for mouse_row, row in enumerate(available_rows):
        uniformity_rows.append(
            {
                "mouse_index": mouse_row + 1,
                "seed": int(row["seed"]),
                "n_valid_neurons": int(valid_com_values[mouse_row].size),
                "kuiper_statistic": float(kuiper_statistics[mouse_row]),
                "kuiper_p_value": float(kuiper_p_values[mouse_row]),
                "kuiper_bh_adjusted_p_value": float(
                    kuiper_bh_adjusted_p_values[mouse_row]
                ),
                "uniformity_alpha": float(uniformity_alpha),
                "reject_uniformity_raw": int(
                    kuiper_p_values[mouse_row] < uniformity_alpha
                ),
                "reject_uniformity_bh": int(
                    kuiper_bh_adjusted_p_values[mouse_row] < uniformity_alpha
                ),
            }
        )

    correlation_matrices = np.empty(
        (
            len(example_mouse_indices),
            len(subset_sizes),
            theta_grid.size,
            theta_grid.size,
        ),
        dtype=float,
    )
    circulant_errors = np.empty(
        (len(example_mouse_indices), len(subset_sizes)),
        dtype=float,
    )
    error_rows: list[dict[str, float | int | str]] = []
    example_seeds: list[int] = []
    for example_row, mouse_index in enumerate(example_mouse_indices):
        tuning = _restore_unaligned_unit_mean_tuning(tuning_archives[mouse_index - 1])
        if tuning.shape[0] < max(subset_sizes):
            raise ValueError(
                f"simulated mouse {mouse_index} has {tuning.shape[0]} valid neurons, "
                f"fewer than N_subset={max(subset_sizes)}"
            )
        permutation = rng.permutation(tuning.shape[0])
        mouse_seed = int(available_rows[mouse_index - 1]["seed"])
        example_seeds.append(mouse_seed)
        for subset_column, subset_size in enumerate(subset_sizes):
            correlation = empirical_two_point_correlation(
                tuning[permutation[:subset_size]]
            )
            circulant_error = relative_circulant_error(correlation)
            correlation_matrices[example_row, subset_column] = correlation
            circulant_errors[example_row, subset_column] = circulant_error
            error_rows.append(
                {
                    "mouse_index": int(mouse_index),
                    "seed": mouse_seed,
                    "n_subset": int(subset_size),
                    "relative_circulant_error": circulant_error,
                }
            )

    figure_path = output_dir / "figures" / "heterogeneous_clark_figure4_abc.png"
    statistics_path = output_dir / "heterogeneous_clark_figure4_statistics.npz"
    error_table_path = output_dir / "heterogeneous_clark_figure4_circulant_error.csv"
    uniformity_table_path = output_dir / "heterogeneous_clark_figure4_com_uniformity.csv"
    save_npz(
        statistics_path,
        theta_grid=theta_grid,
        mouse_indices=np.arange(1, len(tuning_archives) + 1, dtype=int),
        seeds=np.asarray([int(row["seed"]) for row in available_rows], dtype=int),
        empirical_com_by_mouse=empirical_com_by_mouse,
        uniform_com_by_mouse=uniform_com_by_mouse,
        valid_neuron_counts=np.asarray(
            [values.size for values in valid_com_values],
            dtype=int,
        ),
        kuiper_statistics=kuiper_statistics,
        kuiper_p_values=kuiper_p_values,
        kuiper_bh_adjusted_p_values=kuiper_bh_adjusted_p_values,
        uniformity_alpha=np.asarray(uniformity_alpha, dtype=float),
        example_mouse_indices=np.asarray(example_mouse_indices, dtype=int),
        example_mouse_seeds=np.asarray(example_seeds, dtype=int),
        subset_sizes=np.asarray(subset_sizes, dtype=int),
        correlation_matrices=correlation_matrices.astype(np.float32),
        relative_circulant_error=circulant_errors,
    )
    _write_csv(error_table_path, error_rows)
    _write_csv(uniformity_table_path, uniformity_rows)
    _plot_clark_figure4_abc(
        empirical_com_by_mouse=empirical_com_by_mouse,
        uniform_com_by_mouse=uniform_com_by_mouse,
        correlation_matrices=correlation_matrices,
        circulant_errors=circulant_errors,
        example_mouse_indices=np.asarray(example_mouse_indices, dtype=int),
        example_mouse_seeds=np.asarray(example_seeds, dtype=int),
        subset_sizes=np.asarray(subset_sizes, dtype=int),
        kuiper_p_values=kuiper_p_values,
        kuiper_bh_adjusted_p_values=kuiper_bh_adjusted_p_values,
        uniformity_alpha=uniformity_alpha,
        path=figure_path,
    )
    return {
        "n_simulated_mice": len(tuning_archives),
        "n_example_mice": len(example_mouse_indices),
        "max_subset_size": max(subset_sizes),
        "random_seed": int(random_seed),
        "uniformity_alpha": float(uniformity_alpha),
        "kuiper_raw_rejection_count": int(
            np.count_nonzero(kuiper_p_values < uniformity_alpha)
        ),
        "kuiper_bh_rejection_count": int(
            np.count_nonzero(kuiper_bh_adjusted_p_values < uniformity_alpha)
        ),
        "figure_path": str(figure_path),
        "statistics_path": str(statistics_path),
        "circulant_error_table_path": str(error_table_path),
        "com_uniformity_table_path": str(uniformity_table_path),
    }


def run_neuron_count_sweep(
    *,
    base_config: ExperimentConfig,
    project_root: Path,
    output_dir: Path,
    neuron_counts: list[int],
    run_id_prefix: str,
    train_duration: float | None,
    save_interval_duration: float | None,
    training_progress: bool,
    progress: bool,
    skip_existing_runs: bool,
    seed_offsets: list[int],
    min_visual_sigma_bins: float | None = None,
    plot_sweep_metrics: bool = True,
    cross_mouse_tuning_enabled: bool = True,
    cross_mouse_tuning_normalizations: list[str] | None = None,
    cross_mouse_tuning_minimum_converged_fraction: float | None = 0.99,
    clark_figure4_enabled: bool = False,
    clark_figure4_example_mouse_indices: list[int] | None = None,
    clark_figure4_subset_sizes: list[int] | None = None,
    clark_figure4_random_seed: int = 20_251_026,
    clark_figure4_minimum_converged_fraction: float | None = 0.99,
    clark_figure4_uniformity_alpha: float = 0.05,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    seed_offsets = _normalize_seed_offsets(seed_offsets)
    run_specs = [
        (neuron_index, n_theta, repeat_index, seed_offset)
        for neuron_index, n_theta in enumerate(neuron_counts, start=1)
        for repeat_index, seed_offset in enumerate(seed_offsets, start=1)
    ]
    indexed_run_specs = list(enumerate(run_specs, start=1))
    for run_position, (neuron_index, n_theta, repeat_index, seed_offset) in _progress_iter(
        indexed_run_specs,
        desc="neuron-count sweep",
        enabled=progress,
    ):
        if n_theta <= 0 or n_theta % 2 != 0:
            raise ValueError("Neuron-count sweep expects positive even model sizes")
        seed = int(base_config.simulation.seed) + int(seed_offset)
        _progress_message(
            (
                f"[neurons run {run_position}/{len(run_specs)}; "
                f"value {neuron_index}/{len(neuron_counts)}; repeat {repeat_index}/{len(seed_offsets)}] "
                f"start n_theta=n_hr={n_theta}, seed={seed}"
            ),
            enabled=progress,
        )
        config = _clone_config_with_runtime_overrides(
            base_config,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
        )
        config.simulation.seed = seed
        config.model.n_theta = int(n_theta)
        config.model.n_hr = int(n_theta)
        _retune_visual_width_for_neuron_count(
            config=config,
            n_theta=int(n_theta),
            min_visual_sigma_bins=min_visual_sigma_bins,
        )
        run_id = f"{run_id_prefix}_neurons_{n_theta}"
        if seed_offset != 0:
            run_id = f"{run_id}_seed_{seed}"
        run_dir = _run_or_reuse_experiment(
            config=config,
            project_root=project_root,
            run_id=run_id,
            required_files=("test_metrics.json",),
            skip_existing_runs=skip_existing_runs,
            progress=progress,
        )
        metrics = _metrics_with_backfilled_comparison_fields(
            run_dir=run_dir,
            metrics=load_json(run_dir / "test_metrics.json"),
        )
        _progress_message(
            (
                f"[neurons run {run_position}/{len(run_specs)}] done "
                f"run={run_dir.name}, seed={seed}, "
                f"visual_gain={metrics.get('visual_velocity_gain', float('nan')):.4g}, "
                f"dark_gain={metrics.get('darkness_velocity_gain', float('nan')):.4g}, "
                f"dark_rmse={metrics.get('darkness_velocity_tracking_rmse', float('nan')):.4g}"
            ),
            enabled=progress,
        )
        rows.append(
            _metric_row(
                run_dir=run_dir,
                metrics=metrics,
                extra_values={
                    "n_theta": int(n_theta),
                    "n_hr": int(n_theta),
                    "repeat_index": int(repeat_index),
                    "seed_offset": int(seed_offset),
                    "seed": int(seed),
                    "visual_kappa": float(config.visual.kappa),
                    "visual_sigma_rad": _visual_sigma_from_kappa(float(config.visual.kappa)),
                },
            )
        )
    aggregate_rows = _aggregate_sweep_rows(rows, x_key="n_theta")
    _write_csv(output_dir / "neuron_count_sweep_summary.csv", rows)
    _write_csv(output_dir / "neuron_count_sweep_aggregate_summary.csv", aggregate_rows)
    if plot_sweep_metrics:
        _plot_sweep_summary(
            rows=rows,
            x_key="n_theta",
            path=output_dir / "figures" / "neuron_count_sweep_metrics.png",
            title="Neuron-count robustness sweep",
        )
    cross_mouse_summary = (
        _write_cross_mouse_tuning_summaries(
            rows=rows,
            output_dir=output_dir,
            normalizations=cross_mouse_tuning_normalizations,
            minimum_converged_fraction=cross_mouse_tuning_minimum_converged_fraction,
        )
        if (
            cross_mouse_tuning_enabled
            and base_config.visual.profile.lower() == "heterogeneous_gaussian_process"
        )
        else []
    )
    if cross_mouse_summary:
        save_json(
            output_dir / "heterogeneous_cross_mouse_tuning_summary.json",
            cross_mouse_summary,
        )
    if clark_figure4_enabled:
        clark_figure4_summary = _write_clark_figure4_summary(
            rows=rows,
            output_dir=output_dir,
            example_mouse_indices=(
                [1, 2, 3, 4]
                if clark_figure4_example_mouse_indices is None
                else clark_figure4_example_mouse_indices
            ),
            subset_sizes=(
                [5, 10, 20, 40, 80]
                if clark_figure4_subset_sizes is None
                else clark_figure4_subset_sizes
            ),
            random_seed=clark_figure4_random_seed,
            minimum_converged_fraction=clark_figure4_minimum_converged_fraction,
            uniformity_alpha=clark_figure4_uniformity_alpha,
        )
        save_json(
            output_dir / "heterogeneous_clark_figure4_summary.json",
            clark_figure4_summary,
        )
    return rows


def run_noise_by_neuron_count_sweep(
    *,
    base_config: ExperimentConfig,
    project_root: Path,
    output_dir: Path,
    noise_stds: list[float],
    neuron_counts: list[int],
    run_id_prefix: str,
    train_duration: float | None,
    save_interval_duration: float | None,
    training_progress: bool,
    progress: bool,
    skip_existing_runs: bool,
    seed_offsets: list[int],
    min_visual_sigma_bins: float | None = None,
    apply_noise_during_training: bool | None = None,
    apply_noise_during_visual_test: bool | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    seed_offsets = _normalize_seed_offsets(seed_offsets)
    run_specs = [
        (neuron_index, n_theta, noise_index, noise_std, repeat_index, seed_offset)
        for neuron_index, n_theta in enumerate(neuron_counts, start=1)
        for noise_index, noise_std in enumerate(noise_stds, start=1)
        for repeat_index, seed_offset in enumerate(seed_offsets, start=1)
    ]
    indexed_run_specs = list(enumerate(run_specs, start=1))
    for (
        run_position,
        (neuron_index, n_theta, noise_index, noise_std, repeat_index, seed_offset),
    ) in _progress_iter(
        indexed_run_specs,
        desc="noise x neurons sweep",
        enabled=progress,
    ):
        if n_theta <= 0 or n_theta % 2 != 0:
            raise ValueError("Noise x neuron-count sweep expects positive even model sizes")
        seed = int(base_config.simulation.seed) + int(seed_offset)
        _progress_message(
            (
                f"[grid run {run_position}/{len(run_specs)}; "
                f"neurons {neuron_index}/{len(neuron_counts)}; "
                f"noise {noise_index}/{len(noise_stds)}; "
                f"repeat {repeat_index}/{len(seed_offsets)}] "
                f"start n_theta=n_hr={n_theta}, visual.noise_std={noise_std}, seed={seed}"
            ),
            enabled=progress,
        )
        config = _clone_config_with_runtime_overrides(
            base_config,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
        )
        config.simulation.seed = seed
        if apply_noise_during_training is not None:
            config.visual.apply_noise_during_training = bool(apply_noise_during_training)
        if apply_noise_during_visual_test is not None:
            config.visual.apply_noise_during_visual_test = bool(apply_noise_during_visual_test)
        config.model.n_theta = int(n_theta)
        config.model.n_hr = int(n_theta)
        _retune_visual_width_for_neuron_count(
            config=config,
            n_theta=int(n_theta),
            min_visual_sigma_bins=min_visual_sigma_bins,
        )
        config.visual.noise_std = float(noise_std)
        run_id = f"{run_id_prefix}_neurons_{n_theta}_{_visual_noise_run_label(config)}"
        if seed_offset != 0:
            run_id = f"{run_id}_seed_{seed}"
        run_dir = _run_or_reuse_experiment(
            config=config,
            project_root=project_root,
            run_id=run_id,
            required_files=("test_metrics.json",),
            skip_existing_runs=skip_existing_runs,
            progress=progress,
        )
        metrics = _metrics_with_backfilled_comparison_fields(
            run_dir=run_dir,
            metrics=load_json(run_dir / "test_metrics.json"),
        )
        _progress_message(
            (
                f"[grid run {run_position}/{len(run_specs)}] done "
                f"run={run_dir.name}, seed={seed}, "
                f"visual_gain={metrics.get('visual_velocity_gain', float('nan')):.4g}, "
                f"dark_gain={metrics.get('darkness_velocity_gain', float('nan')):.4g}, "
                f"dark_rmse={metrics.get('darkness_velocity_tracking_rmse', float('nan')):.4g}"
            ),
            enabled=progress,
        )
        rows.append(
            _metric_row(
                run_dir=run_dir,
                metrics=metrics,
                extra_values={
                    "visual_noise_std": float(noise_std),
                    "n_theta": int(n_theta),
                    "n_hr": int(n_theta),
                    "repeat_index": int(repeat_index),
                    "seed_offset": int(seed_offset),
                    "seed": int(seed),
                    "visual_kappa": float(config.visual.kappa),
                    "visual_sigma_rad": _visual_sigma_from_kappa(float(config.visual.kappa)),
                    **_visual_noise_metadata(config),
                },
            )
        )

    aggregate_rows = _aggregate_sweep_grid_rows(
        rows,
        x_key="n_theta",
        y_key="visual_noise_std",
    )
    _write_csv(output_dir / "noise_by_neuron_count_sweep_summary.csv", rows)
    _write_csv(output_dir / "noise_by_neuron_count_sweep_aggregate_summary.csv", aggregate_rows)
    _plot_sweep_metric_meshgrid(
        aggregate_rows=aggregate_rows,
        x_key="n_theta",
        y_key="visual_noise_std",
        path=output_dir / "figures" / "noise_by_neuron_count_metric_meshgrid.png",
        title="Visual current-noise std x neuron-count robustness (mean; +/- SEM across repeated seeds)",
    )
    _plot_sweep_metric_meshgrid(
        aggregate_rows=aggregate_rows,
        x_key="n_theta",
        y_key="visual_noise_std",
        path=output_dir / "figures" / "noise_by_neuron_count_hd_decode_error_heatmap.png",
        title="HD decode error comparison",
        metric_names=["visual_cue_hd_decode_rms_error", "darkness_hd_decode_rms_error"],
    )
    _plot_sweep_metric_meshgrid(
        aggregate_rows=aggregate_rows,
        x_key="n_theta",
        y_key="visual_noise_std",
        path=output_dir / "figures" / "noise_by_neuron_count_visual_cue_hd_decode_error_heatmap.png",
        title="Visual-cue HD decode error heatmap",
        metric_names=["visual_cue_hd_decode_rms_error"],
    )
    all_paired_delta_rows: list[dict[str, float | int | str]] = []
    all_paired_delta_aggregate_rows: list[dict[str, float | int | str]] = []
    for n_theta in sorted({int(row["n_theta"]) for row in rows}):
        neuron_rows = [row for row in rows if int(row["n_theta"]) == n_theta]
        _plot_noise_seed_lines(
            rows=neuron_rows,
            path=(
                output_dir
                / "figures"
                / f"noise_by_neuron_count_n{n_theta}_noise_response.png"
            ),
            title=f"n={n_theta} visual-noise response across paired seeds",
        )
        neuron_delta_rows = _compute_noise_paired_delta_rows(neuron_rows)
        if neuron_delta_rows:
            for row in neuron_delta_rows:
                row["n_theta"] = n_theta
                row["n_hr"] = n_theta
            neuron_delta_aggregate_rows = _aggregate_noise_delta_rows(neuron_delta_rows)
            for row in neuron_delta_aggregate_rows:
                row["n_theta"] = n_theta
                row["n_hr"] = n_theta
            all_paired_delta_rows.extend(neuron_delta_rows)
            all_paired_delta_aggregate_rows.extend(neuron_delta_aggregate_rows)
            _write_csv(
                output_dir / f"noise_by_neuron_count_n{n_theta}_paired_delta_summary.csv",
                neuron_delta_rows,
            )
            _write_csv(
                output_dir
                / f"noise_by_neuron_count_n{n_theta}_paired_delta_aggregate_summary.csv",
                neuron_delta_aggregate_rows,
            )
            _plot_noise_delta_summary(
                delta_rows=neuron_delta_rows,
                path=(
                    output_dir
                    / "figures"
                    / f"noise_by_neuron_count_n{n_theta}_paired_delta_metrics.png"
                ),
                title=f"n={n_theta} paired change from sigma=0 baseline",
            )
    _write_csv(
        output_dir / "noise_by_neuron_count_paired_delta_summary.csv",
        all_paired_delta_rows,
    )
    _write_csv(
        output_dir / "noise_by_neuron_count_paired_delta_aggregate_summary.csv",
        all_paired_delta_aggregate_rows,
    )
    return rows, aggregate_rows


def run_extended_recue_analysis(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    output_dir: Path,
    recue_duration: float,
) -> dict[str, float]:
    history = run_constant_velocity_visual_dark_visual_protocol(
        config=config,
        trained_state=trained_state,
        theta_true=config.simulation.theta0,
        darkness_duration=config.simulation.darkness_test_duration,
        angular_velocity=config.tests.darkness_angular_velocity,
        cue_duration=get_pi_cue_duration(config),
        recue_duration=recue_duration,
    )
    dark_mask = phase_mask(history, DARKNESS_PHASE_ID)
    recue_mask = phase_mask(history, VISUAL_RECUE_PHASE_ID)
    pi_error = circular_error_trace(history["theta_hd_decoded"], history["theta_true"])
    metrics = {
        "recue_duration": float(recue_duration),
        "darkness_final_abs_pi_error": final_abs_circular_error(
            history["theta_hd_decoded"][dark_mask],
            theta_reference=history["theta_true"][dark_mask][-1],
        ),
        "recue_initial_abs_pi_error": final_abs_circular_error(
            history["theta_hd_decoded"][dark_mask],
            theta_reference=history["theta_true"][dark_mask][-1],
        ),
        "recue_final_abs_pi_error": final_abs_circular_error(
            history["theta_hd_decoded"][recue_mask],
            theta_reference=history["theta_true"][recue_mask][-1],
        ),
        "recue_min_abs_pi_error": float(np.nanmin(np.abs(pi_error[recue_mask])))
        if np.any(recue_mask)
        else float("nan"),
    }
    save_npz(output_dir / "extended_recue_history.npz", **history)
    save_json(output_dir / "extended_recue_metrics.json", metrics)
    figures_dir = output_dir / "figures"
    plot_true_vs_decoded_heading(
        time=history["time"],
        theta_true=history["theta_true"],
        theta_hd_decoded=history["theta_hd_decoded"],
        theta_hd_decoded_peak=history.get("theta_hd_decoded_peak"),
        phase_id=history.get("phase_id"),
        path=figures_dir / "extended_recue_true_vs_decoded_heading.png",
        title=f"Extended re-visual cue ({recue_duration:.1f} s)",
    )
    plot_pi_error(
        time=history["time"],
        pi_error=pi_error,
        phase_id=history.get("phase_id"),
        path=figures_dir / "extended_recue_pi_error.png",
        title=f"PI error with {recue_duration:.1f} s re-visual input",
    )
    return metrics


def _state_to_manifold_point(state: VafidisToyState) -> np.ndarray:
    return state.r_hd.copy()


def _settle_to_manifold_state(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
    settle_duration: float,
) -> VafidisToyState:
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
        theta_true=theta_true,
    )
    state.w_hd_to_hd = trained_state.w_hd_to_hd.copy()
    state.w_hr_to_hd = trained_state.w_hr_to_hd.copy()
    state.w_hd_to_hr = trained_state.w_hd_to_hr.copy()
    settle_steps = int(round(max(settle_duration, 0.0) / params.dt))
    for _step_index in range(settle_steps):
        state = step_vafidis_toy(
            state=state,
            params=params,
            angular_velocity=0.0,
            visual_teacher=True,
            training=False,
        )
    return state


def build_discrete_target_manifold(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    n_points: int,
    settle_duration: float,
) -> tuple[np.ndarray, np.ndarray, list[VafidisToyState]]:
    theta_grid = np.linspace(-np.pi, np.pi, n_points, endpoint=False, dtype=float)
    manifold_states = [
        _settle_to_manifold_state(
            config=config,
            trained_state=trained_state,
            theta_true=float(theta_true),
            settle_duration=settle_duration,
        )
        for theta_true in theta_grid
    ]
    manifold_points = np.asarray([_state_to_manifold_point(state) for state in manifold_states])
    return theta_grid, manifold_points, manifold_states


def _nearest_manifold_distance(point: np.ndarray, manifold_points: np.ndarray) -> float:
    distances = np.linalg.norm(manifold_points - point[None, :], axis=1)
    return float(np.min(distances) / np.sqrt(point.size))


def _perturb_state(
    *,
    state: VafidisToyState,
    rng: np.random.Generator,
    perturbation_scale: float,
    max_rate: float,
) -> VafidisToyState:
    perturbed_state = state.copy()
    perturbation = rng.normal(loc=0.0, scale=perturbation_scale, size=state.r_hd.shape)
    perturbed_state.r_hd = np.clip(state.r_hd + perturbation, 0.0, max_rate)
    perturbed_state.r_hd_to_hr_lp = np.clip(
        state.r_hd_to_hr_lp + perturbation,
        0.0,
        max_rate,
    )
    return perturbed_state


def _simulate_perturbation_trajectories(
    *,
    config: ExperimentConfig,
    manifold_states: list[VafidisToyState],
    manifold_points: np.ndarray,
    protocol_name: str,
    visual_teacher: bool,
    duration: float,
    perturbation_count: int,
    perturbation_scale: float,
) -> dict[str, np.ndarray]:
    params = VafidisToyParams.from_config(config)
    rng = make_rng(config.simulation.seed + 30_000 + (0 if visual_teacher else 1_000))
    trajectory_steps = int(round(max(duration, 0.0) / params.dt))
    anchor_indices = np.linspace(
        0,
        len(manifold_states) - 1,
        min(perturbation_count, len(manifold_states)),
        dtype=int,
    )
    trajectories: list[np.ndarray] = []
    distance_traces: list[np.ndarray] = []
    time = np.arange(trajectory_steps + 1, dtype=float) * params.dt
    for anchor_index in anchor_indices:
        state = _perturb_state(
            state=manifold_states[int(anchor_index)],
            rng=rng,
            perturbation_scale=perturbation_scale,
            max_rate=params.activation_max_rate,
        )
        trajectory_points = [_state_to_manifold_point(state)]
        distance_trace = [_nearest_manifold_distance(trajectory_points[-1], manifold_points)]
        for _step_index in range(trajectory_steps):
            state = step_vafidis_toy(
                state=state,
                params=params,
                angular_velocity=0.0,
                visual_teacher=visual_teacher,
                training=False,
            )
            trajectory_points.append(_state_to_manifold_point(state))
            distance_trace.append(_nearest_manifold_distance(trajectory_points[-1], manifold_points))
        trajectories.append(np.asarray(trajectory_points))
        distance_traces.append(np.asarray(distance_trace))
    return {
        "time": time,
        "anchor_indices": anchor_indices,
        "trajectories": np.asarray(trajectories),
        "distance_to_manifold": np.asarray(distance_traces),
        "protocol_name": np.asarray([protocol_name]),
    }


def _pca_project(points: np.ndarray, *, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_point = np.mean(points, axis=0)
    centered = points - mean_point[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    coordinates = centered @ components.T
    return coordinates, mean_point, components


def _plot_manifold_pca(
    *,
    manifold_points: np.ndarray,
    trajectories: np.ndarray,
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    all_points = np.vstack([manifold_points, flat_trajectories])
    coordinates, mean_point, components = _pca_project(all_points, n_components=3)
    manifold_coordinates = (manifold_points - mean_point[None, :]) @ components.T
    trajectory_coordinates = (flat_trajectories - mean_point[None, :]) @ components.T
    trajectory_coordinates = trajectory_coordinates.reshape(
        trajectories.shape[0],
        trajectories.shape[1],
        3,
    )

    fig = plt.figure(figsize=(6.8, 5.8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(
        manifold_coordinates[:, 0],
        manifold_coordinates[:, 1],
        manifold_coordinates[:, 2],
        color="black",
        linewidth=1.2,
        alpha=0.75,
        label="discrete target manifold",
    )
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, trajectories.shape[0]))
    for trajectory_index, color in enumerate(colors):
        coords = trajectory_coordinates[trajectory_index]
        axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=1.0)
        axis.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=18, marker="o")
        axis.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, s=18, marker="x")
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_distance_traces(
    *,
    time: np.ndarray,
    distance_to_manifold: np.ndarray,
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 3.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.86)
    for distance_trace in distance_to_manifold:
        axis.plot(time, distance_trace, linewidth=1.2, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("nearest manifold distance / sqrt(n)")
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_manifold_perturbation_analysis(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    output_dir: Path,
    n_manifold_points: int,
    manifold_settle_duration: float,
    perturbation_count: int,
    perturbation_duration: float,
    perturbation_scale: float,
) -> dict[str, float]:
    theta_grid, manifold_points, manifold_states = build_discrete_target_manifold(
        config=config,
        trained_state=trained_state,
        n_points=n_manifold_points,
        settle_duration=manifold_settle_duration,
    )
    visual_trajectories = _simulate_perturbation_trajectories(
        config=config,
        manifold_states=manifold_states,
        manifold_points=manifold_points,
        protocol_name="visual_teacher",
        visual_teacher=True,
        duration=perturbation_duration,
        perturbation_count=perturbation_count,
        perturbation_scale=perturbation_scale,
    )
    darkness_trajectories = _simulate_perturbation_trajectories(
        config=config,
        manifold_states=manifold_states,
        manifold_points=manifold_points,
        protocol_name="darkness",
        visual_teacher=False,
        duration=perturbation_duration,
        perturbation_count=perturbation_count,
        perturbation_scale=perturbation_scale,
    )
    save_npz(
        output_dir / "target_manifold.npz",
        theta_grid=theta_grid,
        r_hd_manifold=manifold_points,
    )
    save_npz(output_dir / "perturbation_visual_teacher.npz", **visual_trajectories)
    save_npz(output_dir / "perturbation_darkness.npz", **darkness_trajectories)
    figures_dir = output_dir / "figures"
    _plot_manifold_pca(
        manifold_points=manifold_points,
        trajectories=visual_trajectories["trajectories"],
        path=figures_dir / "perturbation_visual_teacher_pca3.png",
        title="Perturbed states near target manifold: visual teacher",
    )
    _plot_manifold_pca(
        manifold_points=manifold_points,
        trajectories=darkness_trajectories["trajectories"],
        path=figures_dir / "perturbation_darkness_pca3.png",
        title="Perturbed states near target manifold: darkness",
    )
    _plot_distance_traces(
        time=visual_trajectories["time"],
        distance_to_manifold=visual_trajectories["distance_to_manifold"],
        path=figures_dir / "perturbation_visual_teacher_distance.png",
        title="Distance to target manifold under visual teacher",
    )
    _plot_distance_traces(
        time=darkness_trajectories["time"],
        distance_to_manifold=darkness_trajectories["distance_to_manifold"],
        path=figures_dir / "perturbation_darkness_distance.png",
        title="Distance to target manifold in darkness",
    )
    metrics = {
        "visual_teacher_initial_distance_mean": float(np.mean(visual_trajectories["distance_to_manifold"][:, 0])),
        "visual_teacher_final_distance_mean": float(np.mean(visual_trajectories["distance_to_manifold"][:, -1])),
        "darkness_initial_distance_mean": float(np.mean(darkness_trajectories["distance_to_manifold"][:, 0])),
        "darkness_final_distance_mean": float(np.mean(darkness_trajectories["distance_to_manifold"][:, -1])),
        **summarize_weight_structure(trained_state.w_hd_to_hd, trained_state.w_hr_to_hd),
    }
    save_json(output_dir / "perturbation_manifold_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the base experiment YAML.")
    parser.add_argument(
        "--robustness-config",
        "--probe-config",
        dest="robustness_config",
        default=None,
        help="Optional YAML config for robustness probes and sweeps.",
    )
    parser.add_argument("--run-dir", default=None, help="Optional existing run directory for trained weights.")
    parser.add_argument("--output-dir", default=None, help="Directory for robustness-analysis outputs.")
    parser.add_argument("--run-id-prefix", default=None, help="Prefix for generated run ids.")
    parser.add_argument("--noise-stds", default=None, help="Comma-separated visual noise std values.")
    parser.add_argument("--neuron-counts", default=None, help="Comma-separated even neuron counts.")
    parser.add_argument(
        "--seed-offsets",
        default=None,
        help="Comma-separated offsets added to simulation.seed for repeated sweep runs.",
    )
    parser.add_argument("--train-duration", type=float, default=None, help="Override train duration for sweeps.")
    parser.add_argument(
        "--save-interval-duration",
        type=float,
        default=None,
        help="Override saved-history interval in seconds for sweeps.",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--training-progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-existing-runs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-sweeps", action="store_true", help="Only run re-cue and perturbation analyses.")
    parser.add_argument("--skip-perturbation", action="store_true", help="Skip target-manifold perturbation analysis.")
    parser.add_argument("--extended-recue-duration", type=float, default=8.0)
    parser.add_argument("--manifold-points", type=int, default=180)
    parser.add_argument("--manifold-settle-duration", type=float, default=1.0)
    parser.add_argument("--perturbation-count", type=int, default=6)
    parser.add_argument("--perturbation-duration", type=float, default=2.0)
    parser.add_argument("--perturbation-scale", type=float, default=0.05)
    args = parser.parse_args()

    robustness_config_path = (
        _resolve_existing_cli_path(args.robustness_config)
        if args.robustness_config
        else None
    )
    robustness_config = load_yaml(robustness_config_path) if robustness_config_path else {}
    project_root = find_project_root(robustness_config_path or args.config or Path.cwd())

    config_arg = args.config or _optional_str(
        _nested_get(robustness_config, ("experiment", "config"), robustness_config.get("base_config"))
    )
    if config_arg is None:
        raise ValueError("Provide --config or set experiment.config in the robustness config.")
    config_path = _resolve_path_argument(
        config_arg,
        project_root=project_root,
        relative_to=robustness_config_path.parent if robustness_config_path else None,
    )
    base_config = load_experiment_config(config_path)

    run_dir_arg = args.run_dir or _optional_str(
        _nested_get(robustness_config, ("experiment", "run_dir"), robustness_config.get("run_dir"))
    )
    output_dir_arg = args.output_dir or _optional_str(
        _nested_get(robustness_config, ("output", "directory"), robustness_config.get("output_dir"))
    )
    run_id_prefix = args.run_id_prefix or _optional_str(
        _nested_get(robustness_config, ("output", "run_id_prefix"), robustness_config.get("run_id_prefix"))
    )
    if run_id_prefix is None:
        run_id_prefix = "attractor_robustness"
    train_duration = (
        args.train_duration
        if args.train_duration is not None
        else _optional_float(
            _nested_get(robustness_config, ("runtime", "train_duration"), robustness_config.get("train_duration"))
        )
    )
    save_interval_duration = (
        float(args.save_interval_duration)
        if args.save_interval_duration is not None
        else _optional_float(
            _nested_get(
                robustness_config,
                ("runtime", "save_interval_duration"),
                robustness_config.get("save_interval_duration"),
            )
        )
    )
    progress = (
        bool(args.progress)
        if args.progress is not None
        else bool(_nested_get(robustness_config, ("runtime", "progress"), True))
    )
    training_progress = (
        bool(args.training_progress)
        if args.training_progress is not None
        else bool(_nested_get(robustness_config, ("runtime", "training_progress"), True))
    )
    skip_existing_runs = (
        bool(args.skip_existing_runs)
        if args.skip_existing_runs is not None
        else bool(_nested_get(robustness_config, ("runtime", "skip_existing_runs"), True))
    )
    noise_stds = (
        _parse_float_list(args.noise_stds)
        if args.noise_stds is not None
        else _coerce_float_list(
            _nested_get(
                robustness_config,
                ("sweeps", "visual_noise", "stds"),
                robustness_config.get("noise_stds", [0.0, 0.05, 0.1]),
            )
        )
    )
    neuron_counts = (
        _parse_int_list(args.neuron_counts)
        if args.neuron_counts is not None
        else _coerce_int_list(
            _nested_get(
                robustness_config,
                ("sweeps", "neuron_count", "counts"),
                robustness_config.get("neuron_counts", [16, 32, 60]),
            )
        )
    )
    neuron_count_min_visual_sigma_bins = _optional_float(
        _nested_get(
            robustness_config,
            ("sweeps", "neuron_count", "min_visual_sigma_bins"),
            None,
        )
    )
    plot_neuron_count_metrics = bool(
        _nested_get(
            robustness_config,
            ("plots", "neuron_count_metrics", "enabled"),
            True,
        )
    )
    cross_mouse_tuning_enabled = bool(
        _nested_get(
            robustness_config,
            ("plots", "cross_mouse_tuning", "enabled"),
            True,
        )
    )
    cross_mouse_tuning_normalizations_raw = _nested_get(
        robustness_config,
        ("plots", "cross_mouse_tuning", "normalizations"),
        None,
    )
    cross_mouse_tuning_normalizations = (
        None
        if cross_mouse_tuning_normalizations_raw is None
        else _coerce_str_list(cross_mouse_tuning_normalizations_raw)
    )
    cross_mouse_tuning_minimum_converged_fraction = _optional_float(
        _nested_get(
            robustness_config,
            ("plots", "cross_mouse_tuning", "minimum_converged_fraction"),
            0.99,
        )
    )
    clark_figure4_enabled = bool(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "enabled"),
            False,
        )
    )
    clark_figure4_example_mouse_indices = _coerce_int_list(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "example_mouse_indices"),
            [1, 2, 3, 4],
        )
    )
    clark_figure4_subset_sizes = _coerce_int_list(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "subset_sizes"),
            [5, 10, 20, 40, 80],
        )
    )
    clark_figure4_random_seed = int(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "random_seed"),
            20_251_026,
        )
    )
    clark_figure4_minimum_converged_fraction = _optional_float(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "minimum_converged_fraction"),
            0.99,
        )
    )
    clark_figure4_uniformity_alpha = float(
        _nested_get(
            robustness_config,
            ("plots", "clark_figure4", "uniformity_alpha"),
            0.05,
        )
    )
    noise_by_neuron_count_noise_stds = (
        _parse_float_list(args.noise_stds)
        if args.noise_stds is not None
        else _coerce_float_list(
            _nested_get(
                robustness_config,
                ("sweeps", "noise_by_neuron_count", "stds"),
                noise_stds,
            )
        )
    )
    noise_by_neuron_count_neuron_counts = (
        _parse_int_list(args.neuron_counts)
        if args.neuron_counts is not None
        else _coerce_int_list(
            _nested_get(
                robustness_config,
                ("sweeps", "noise_by_neuron_count", "counts"),
                neuron_counts,
            )
        )
    )
    noise_by_neuron_count_min_visual_sigma_bins = _optional_float(
        _nested_get(
            robustness_config,
            ("sweeps", "noise_by_neuron_count", "min_visual_sigma_bins"),
            neuron_count_min_visual_sigma_bins,
        )
    )
    grid_apply_noise_during_training_raw = _nested_get(
        robustness_config,
        ("sweeps", "noise_by_neuron_count", "apply_noise_during_training"),
        None,
    )
    grid_apply_noise_during_visual_test_raw = _nested_get(
        robustness_config,
        ("sweeps", "noise_by_neuron_count", "apply_noise_during_visual_test"),
        None,
    )
    grid_apply_noise_during_training = (
        None
        if grid_apply_noise_during_training_raw is None
        else bool(grid_apply_noise_during_training_raw)
    )
    grid_apply_noise_during_visual_test = (
        None
        if grid_apply_noise_during_visual_test_raw is None
        else bool(grid_apply_noise_during_visual_test_raw)
    )
    seed_offsets = _normalize_seed_offsets(
        _parse_int_list(args.seed_offsets)
        if args.seed_offsets is not None
        else _coerce_int_list(
            _nested_get(
                robustness_config,
                ("runtime", "seed_offsets"),
                robustness_config.get("seed_offsets", [0]),
            )
        )
    )
    run_extended_recue = bool(
        _nested_get(robustness_config, ("probes", "extended_recue", "enabled"), True)
    )
    run_perturbation = bool(
        _nested_get(robustness_config, ("probes", "manifold_perturbation", "enabled"), True)
    ) and not args.skip_perturbation
    run_noise_sweep_enabled = bool(
        _nested_get(robustness_config, ("sweeps", "visual_noise", "enabled"), True)
    )
    run_neuron_count_sweep_enabled = bool(
        _nested_get(robustness_config, ("sweeps", "neuron_count", "enabled"), True)
    )
    run_noise_by_neuron_count_sweep_enabled = bool(
        _nested_get(robustness_config, ("sweeps", "noise_by_neuron_count", "enabled"), False)
    )
    if args.skip_sweeps:
        run_noise_sweep_enabled = False
        run_neuron_count_sweep_enabled = False
        run_noise_by_neuron_count_sweep_enabled = False

    extended_recue_duration = float(
        _nested_get(
            robustness_config,
            ("probes", "extended_recue", "duration"),
            args.extended_recue_duration,
        )
    )
    manifold_points = int(
        _nested_get(
            robustness_config,
            ("probes", "manifold_perturbation", "manifold_points"),
            args.manifold_points,
        )
    )
    manifold_settle_duration = float(
        _nested_get(
            robustness_config,
            ("probes", "manifold_perturbation", "settle_duration"),
            args.manifold_settle_duration,
        )
    )
    perturbation_count = int(
        _nested_get(
            robustness_config,
            ("probes", "manifold_perturbation", "perturbation_count"),
            args.perturbation_count,
        )
    )
    perturbation_duration = float(
        _nested_get(
            robustness_config,
            ("probes", "manifold_perturbation", "perturbation_duration"),
            args.perturbation_duration,
        )
    )
    perturbation_scale = float(
        _nested_get(
            robustness_config,
            ("probes", "manifold_perturbation", "perturbation_scale"),
            args.perturbation_scale,
        )
    )
    output_root = (
        _resolve_path_argument(output_dir_arg, project_root=project_root)
        if output_dir_arg is not None
        else project_root / base_config.paths.reports_root / "attractor_robustness"
    )
    output_dir = _create_timestamped_report_dir(
        output_root=output_root,
        label=run_id_prefix,
    )
    _progress_message(f"[setup] output_root={output_root}", enabled=progress)
    _progress_message(f"[setup] output_dir={output_dir}", enabled=progress)

    analysis_config = _clone_config_with_runtime_overrides(
        base_config,
        train_duration=train_duration,
        save_interval_duration=save_interval_duration,
        training_progress=training_progress,
    )
    save_yaml(output_dir / "base_analysis_config.yaml", analysis_config.to_dict())
    if robustness_config:
        save_yaml(output_dir / "robustness_config_resolved.yaml", robustness_config)

    baseline_run_dir: Path | None = None
    trained_config: ExperimentConfig | None = None
    trained_state: VafidisToyState | None = None
    needs_baseline_state = run_extended_recue or run_perturbation
    if needs_baseline_state:
        if run_dir_arg is None:
            _progress_message("[baseline] prepare baseline run", enabled=progress)
            baseline_run_dir = _run_or_reuse_experiment(
                config=analysis_config,
                project_root=project_root,
                run_id=f"{run_id_prefix}_baseline",
                required_files=("config_resolved.yaml", "trained_weights.npz", "test_metrics.json"),
                skip_existing_runs=skip_existing_runs,
                progress=progress,
            )
            _progress_message(f"[baseline] ready run={baseline_run_dir.name}", enabled=progress)
        else:
            baseline_run_dir = _resolve_path_argument(run_dir_arg, project_root=project_root)
            _progress_message(f"[baseline] reuse run={baseline_run_dir}", enabled=progress)
        trained_config, trained_state = load_trained_state_from_run(baseline_run_dir)

    summary: dict[str, object] = {
        "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir is not None else None,
        "robustness_config_path": str(robustness_config_path) if robustness_config_path else None,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "report_id": output_dir.name,
        "base_config": asdict(base_config),
        "analysis_config": asdict(analysis_config),
        "seed_offsets": seed_offsets,
    }
    if run_extended_recue:
        if trained_config is None or trained_state is None:
            raise RuntimeError("Extended re-cue analysis requires a trained baseline state")
        _progress_message("[probe] start extended re-visual cue analysis", enabled=progress)
        summary["extended_recue"] = run_extended_recue_analysis(
            config=trained_config,
            trained_state=trained_state,
            output_dir=output_dir,
            recue_duration=extended_recue_duration,
        )
        _progress_message("[probe] done extended re-visual cue analysis", enabled=progress)
    if run_perturbation:
        if trained_config is None or trained_state is None:
            raise RuntimeError("Manifold perturbation analysis requires a trained baseline state")
        _progress_message("[probe] start manifold perturbation analysis", enabled=progress)
        summary["perturbation_manifold"] = run_manifold_perturbation_analysis(
            config=trained_config,
            trained_state=trained_state,
            output_dir=output_dir,
            n_manifold_points=manifold_points,
            manifold_settle_duration=manifold_settle_duration,
            perturbation_count=perturbation_count,
            perturbation_duration=perturbation_duration,
            perturbation_scale=perturbation_scale,
        )
        _progress_message("[probe] done manifold perturbation analysis", enabled=progress)
    if run_noise_sweep_enabled:
        noise_sweep_rows = run_noise_sweep(
            base_config=base_config,
            project_root=project_root,
            output_dir=output_dir,
            noise_stds=noise_stds,
            run_id_prefix=run_id_prefix,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
            progress=progress,
            skip_existing_runs=skip_existing_runs,
            seed_offsets=seed_offsets,
            min_visual_sigma_bins=neuron_count_min_visual_sigma_bins,
        )
        summary["noise_sweep"] = noise_sweep_rows
    if run_neuron_count_sweep_enabled:
        neuron_count_sweep_rows = run_neuron_count_sweep(
            base_config=base_config,
            project_root=project_root,
            output_dir=output_dir,
            neuron_counts=neuron_counts,
            run_id_prefix=run_id_prefix,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
            progress=progress,
            skip_existing_runs=skip_existing_runs,
            seed_offsets=seed_offsets,
            min_visual_sigma_bins=neuron_count_min_visual_sigma_bins,
            plot_sweep_metrics=plot_neuron_count_metrics,
            cross_mouse_tuning_enabled=cross_mouse_tuning_enabled,
            cross_mouse_tuning_normalizations=cross_mouse_tuning_normalizations,
            cross_mouse_tuning_minimum_converged_fraction=(
                cross_mouse_tuning_minimum_converged_fraction
            ),
            clark_figure4_enabled=clark_figure4_enabled,
            clark_figure4_example_mouse_indices=clark_figure4_example_mouse_indices,
            clark_figure4_subset_sizes=clark_figure4_subset_sizes,
            clark_figure4_random_seed=clark_figure4_random_seed,
            clark_figure4_minimum_converged_fraction=(
                clark_figure4_minimum_converged_fraction
            ),
            clark_figure4_uniformity_alpha=clark_figure4_uniformity_alpha,
        )
        summary["neuron_count_sweep"] = neuron_count_sweep_rows
        summary["neuron_count_sweep_aggregate"] = _aggregate_sweep_rows(neuron_count_sweep_rows, x_key="n_theta")
    if run_noise_by_neuron_count_sweep_enabled:
        grid_rows, grid_aggregate_rows = run_noise_by_neuron_count_sweep(
            base_config=base_config,
            project_root=project_root,
            output_dir=output_dir,
            noise_stds=noise_by_neuron_count_noise_stds,
            neuron_counts=noise_by_neuron_count_neuron_counts,
            run_id_prefix=run_id_prefix,
            train_duration=train_duration,
            save_interval_duration=save_interval_duration,
            training_progress=training_progress,
            progress=progress,
            skip_existing_runs=skip_existing_runs,
            seed_offsets=seed_offsets,
            min_visual_sigma_bins=noise_by_neuron_count_min_visual_sigma_bins,
            apply_noise_during_training=grid_apply_noise_during_training,
            apply_noise_during_visual_test=grid_apply_noise_during_visual_test,
        )
        summary["noise_by_neuron_count_sweep"] = grid_rows
        summary["noise_by_neuron_count_sweep_aggregate"] = grid_aggregate_rows
    save_json(output_dir / "attractor_robustness_summary.json", summary)
    _progress_message("[done] wrote attractor_robustness_summary.json", enabled=progress)
    print(f"Saved attractor robustness outputs to {output_dir}")


if __name__ == "__main__":
    main()
