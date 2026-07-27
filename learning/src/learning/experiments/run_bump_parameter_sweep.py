"""Search frozen circular kernels for population-mean bump maintenance."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from learning.common.angles import collapse_activity_by_theta, circular_difference
from learning.common.random import make_rng
from learning.config.load_config import (
    find_project_root,
    load_experiment_config,
    load_yaml,
    save_yaml,
)
from learning.config.schema import ExperimentConfig, experiment_config_from_dict
from learning.experiments.run_vafidis_toy import (
    DARKNESS_PHASE_ID,
    VISUAL_CUE_PHASE_ID,
    resolve_config_path,
    run_bump_maintenance_test,
)
from learning.io.save_load import save_json, save_npz
from learning.models.vafidis_toy import initialize_vafidis_toy_state
from learning.plotting.activity import plot_activity_heatmap
from learning.plotting.backend import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt


def make_gaussian_minus_uniform_kernel(
    *,
    theta_preference: np.ndarray,
    excitatory_scale: float,
    sigma_radians: float,
    inhibitory_ratio: float,
    include_self_connections: bool,
) -> np.ndarray:
    """Return an intensive circulant local-excitation/global-inhibition kernel."""
    theta_preference = np.asarray(theta_preference, dtype=float)
    if theta_preference.ndim != 1 or theta_preference.size == 0:
        raise ValueError("theta_preference must be a non-empty vector")
    if excitatory_scale <= 0.0:
        raise ValueError("excitatory_scale must be positive")
    if sigma_radians <= 0.0:
        raise ValueError("sigma_radians must be positive")
    if inhibitory_ratio < 0.0:
        raise ValueError("inhibitory_ratio must be non-negative")
    angular_offset = circular_difference(
        theta_preference[:, None], theta_preference[None, :]
    )
    kernel = float(excitatory_scale) * (
        np.exp(-0.5 * np.square(angular_offset / float(sigma_radians)))
        - float(inhibitory_ratio)
    )
    if not include_self_connections:
        np.fill_diagonal(kernel, 0.0)
    return kernel


def _recursive_update(mapping: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(mapping.get(key), dict):
            _recursive_update(mapping[key], value)
        else:
            mapping[key] = deepcopy(value)


def _condition_config(
    *,
    base_config_dict: dict[str, Any],
    visual_overrides: dict[str, Any],
    n_hd: int,
    seed: int,
    theta0: float,
    cue_duration: float,
    darkness_duration: float,
) -> ExperimentConfig:
    config_dict = deepcopy(base_config_dict)
    model = config_dict.setdefault("model", {})
    simulation = config_dict.setdefault("simulation", {})
    visual = config_dict.setdefault("visual", {})
    model["n_theta"] = int(n_hd)
    model["n_hr"] = int(n_hd)
    model["hd_distal_normalization"] = "presynaptic_population_mean"
    model.setdefault("init", {})["w_hd_to_hd_mode"] = "zeros"
    model["init"]["w_hr_to_hd_mode"] = "zeros"
    model["init"]["random_jitter"] = 0.0
    simulation["seed"] = int(seed)
    simulation["plasticity_enabled"] = False
    simulation["train_duration"] = 0.0
    simulation["theta0"] = float(theta0)
    simulation["cue_duration"] = float(cue_duration)
    simulation["bump_test_duration"] = float(darkness_duration)
    simulation["recue_duration"] = 0.0
    simulation["progress"] = False
    _recursive_update(visual, visual_overrides)
    return experiment_config_from_dict(config_dict)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def _half_max_width_degrees(
    *, theta_preference: np.ndarray, rates: np.ndarray
) -> tuple[float, int]:
    """Return total angular support above the baseline-corrected half maximum."""
    theta, collapsed_rates = collapse_activity_by_theta(theta_preference, rates)
    if theta.size < 2 or not np.isfinite(collapsed_rates).all():
        return float("nan"), 0
    rate_min = float(np.min(collapsed_rates))
    rate_max = float(np.max(collapsed_rates))
    threshold = rate_min + 0.5 * (rate_max - rate_min)
    half_max_bins = int(np.count_nonzero(collapsed_rates >= threshold))
    theta_step = float(2.0 * np.pi / theta.size)
    return float(np.rad2deg(theta_step * half_max_bins)), half_max_bins


def _optional_upper_criterion(
    criteria: dict[str, bool],
    acceptance: dict[str, float],
    *,
    name: str,
    value: float,
    config_key: str,
) -> None:
    if config_key in acceptance:
        criteria[name] = value <= float(acceptance[config_key])


def _optional_lower_criterion(
    criteria: dict[str, bool],
    acceptance: dict[str, float],
    *,
    name: str,
    value: float,
    config_key: str,
) -> None:
    if config_key in acceptance:
        criteria[name] = value >= float(acceptance[config_key])


def _trial_row(
    *,
    config: ExperimentConfig,
    cue_condition: str,
    seed_offset: int,
    theta0: float,
    excitatory_scale: float,
    sigma_radians: float,
    inhibitory_ratio: float,
    include_self_connections: bool,
    acceptance: dict[str, float],
    return_history: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, np.ndarray]:
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    kernel = make_gaussian_minus_uniform_kernel(
        theta_preference=state.theta_hd_pref,
        excitatory_scale=excitatory_scale,
        sigma_radians=sigma_radians,
        inhibitory_ratio=inhibitory_ratio,
        include_self_connections=include_self_connections,
    )
    state.w_hd_to_hd = kernel
    state.w_hr_to_hd.fill(0.0)
    history, metrics = run_bump_maintenance_test(config=config, trained_state=state)
    cue_indices = np.flatnonzero(history["phase_id"] == VISUAL_CUE_PHASE_ID)
    dark_indices = np.flatnonzero(history["phase_id"] == DARKNESS_PHASE_ID)
    if cue_indices.size == 0 or dark_indices.size == 0:
        raise ValueError("bump protocol must contain cue and darkness samples")
    cue_index = int(cue_indices[-1])
    dark_final_index = int(dark_indices[-1])
    cue_pva = float(history["pva_strength_hd"][cue_index])
    cue_contrast = float(history["bump_contrast_hd"][cue_index])
    final_pva = float(history["pva_strength_hd"][dark_final_index])
    final_contrast = float(history["bump_contrast_hd"][dark_final_index])
    minimum_dark_pva = float(np.min(history["pva_strength_hd"][dark_indices]))
    minimum_dark_contrast = float(np.min(history["bump_contrast_hd"][dark_indices]))
    final_rates = np.asarray(history["r_hd"][dark_final_index], dtype=float)
    dark_rates = np.asarray(history["r_hd"][dark_indices], dtype=float)
    final_max_rate = float(np.max(final_rates))
    final_heading_error_degrees = float(
        abs(
            np.rad2deg(
                circular_difference(
                    history["theta_hd_decoded"][dark_final_index], theta0
                )
            )
        )
    )
    cue_heading_error_degrees = float(
        abs(
            np.rad2deg(
                circular_difference(history["theta_hd_decoded"][cue_index], theta0)
            )
        )
    )
    half_max_width_degrees, half_max_bin_count = _half_max_width_degrees(
        theta_preference=state.theta_hd_pref,
        rates=final_rates,
    )
    contrast_retention = _safe_ratio(final_contrast, cue_contrast)
    max_saturated_fraction = float(np.max(np.mean(dark_rates >= 0.99, axis=1)))
    release_shift_degrees = float(metrics["bump_abs_release_shift_deg"])
    criteria = {
        "cue_pva": cue_pva >= float(acceptance["minimum_cue_pva_strength"]),
        "dark_pva": minimum_dark_pva
        >= float(acceptance["minimum_dark_pva_strength"]),
        "final_pva": final_pva >= float(acceptance["minimum_final_pva_strength"]),
        "dark_contrast": minimum_dark_contrast
        >= float(acceptance["minimum_dark_contrast"]),
        "final_contrast": final_contrast >= float(acceptance["minimum_final_contrast"]),
        "contrast_retention": contrast_retention
        >= float(acceptance["minimum_contrast_retention"]),
        "release_shift": release_shift_degrees
        <= float(acceptance["maximum_abs_release_shift_degrees"]),
        "saturation": max_saturated_fraction
        <= float(acceptance["maximum_saturated_hd_fraction"]),
        "mean_rate": float(np.mean(final_rates))
        <= float(acceptance["maximum_final_mean_rate"]),
    }
    _optional_upper_criterion(
        criteria,
        acceptance,
        name="max_rate",
        value=final_max_rate,
        config_key="maximum_final_max_rate",
    )
    _optional_upper_criterion(
        criteria,
        acceptance,
        name="heading_error",
        value=final_heading_error_degrees,
        config_key="maximum_final_heading_error_degrees",
    )
    _optional_lower_criterion(
        criteria,
        acceptance,
        name="half_max_width_min",
        value=half_max_width_degrees,
        config_key="minimum_half_max_width_degrees",
    )
    _optional_upper_criterion(
        criteria,
        acceptance,
        name="half_max_width_max",
        value=half_max_width_degrees,
        config_key="maximum_half_max_width_degrees",
    )
    row: dict[str, Any] = {
        "cue_condition": cue_condition,
        "seed": int(config.simulation.seed),
        "seed_offset": int(seed_offset),
        "theta0": float(theta0),
        "theta0_degrees": float(np.rad2deg(theta0)),
        "excitatory_scale": float(excitatory_scale),
        "sigma_radians": float(sigma_radians),
        "inhibitory_ratio": float(inhibitory_ratio),
        "kernel_min": float(np.min(kernel)),
        "kernel_max": float(np.max(kernel)),
        "kernel_mean": float(np.mean(kernel)),
        "cue_final_pva_strength": cue_pva,
        "cue_final_contrast": cue_contrast,
        "dark_final_pva_strength": final_pva,
        "dark_minimum_pva_strength": minimum_dark_pva,
        "dark_final_contrast": final_contrast,
        "dark_minimum_contrast": minimum_dark_contrast,
        "contrast_retention": contrast_retention,
        "dark_final_mean_rate": float(np.mean(final_rates)),
        "dark_final_max_rate": final_max_rate,
        "cue_final_heading_error_degrees": cue_heading_error_degrees,
        "dark_final_heading_error_degrees": final_heading_error_degrees,
        "dark_final_half_max_width_degrees": half_max_width_degrees,
        "dark_final_half_max_bin_count": half_max_bin_count,
        "dark_max_saturated_fraction": max_saturated_fraction,
        "dark_final_hd_current_mean": float(history["mean_i_hd_from_hd"][dark_final_index]),
        "dark_final_hd_current_rms": float(history["rms_i_hd_from_hd"][dark_final_index]),
        "abs_release_shift_degrees": release_shift_degrees,
        "max_abs_release_displacement_degrees": float(
            metrics["bump_max_abs_release_displacement_deg"]
        ),
        "final_local_peak_count_25pct": float(
            metrics["bump_dark_final_local_peak_count_25pct"]
        ),
        **{f"passes_{name}": bool(value) for name, value in criteria.items()},
        "passes_all": bool(all(criteria.values())),
    }
    return row, history if return_history else None, kernel


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["cue_condition"]),
            float(row["excitatory_scale"]),
            float(row["sigma_radians"]),
            float(row["inhibitory_ratio"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (condition, scale, sigma, inhibition), group in grouped.items():
        success_fraction = _mean(group, "passes_all")
        minimum_pva = min(float(row["dark_final_pva_strength"]) for row in group)
        minimum_contrast = min(float(row["dark_final_contrast"]) for row in group)
        maximum_shift = max(float(row["abs_release_shift_degrees"]) for row in group)
        maximum_saturation = max(float(row["dark_max_saturated_fraction"]) for row in group)
        maximum_final_rate = max(float(row["dark_final_max_rate"]) for row in group)
        maximum_heading_error = max(
            float(row["dark_final_heading_error_degrees"]) for row in group
        )
        rank_score = (
            4.0 * success_fraction
            + minimum_pva
            + minimum_contrast
            - maximum_shift / 180.0
            - maximum_saturation
            - 0.5 * maximum_final_rate
            - 0.002 * scale
        )
        aggregates.append(
            {
                "cue_condition": condition,
                "excitatory_scale": scale,
                "sigma_radians": sigma,
                "inhibitory_ratio": inhibition,
                "trial_count": len(group),
                "success_fraction": success_fraction,
                "minimum_final_pva_strength": minimum_pva,
                "minimum_dark_pva_strength": min(
                    float(row["dark_minimum_pva_strength"]) for row in group
                ),
                "mean_final_pva_strength": _mean(group, "dark_final_pva_strength"),
                "minimum_final_contrast": minimum_contrast,
                "minimum_dark_contrast": min(
                    float(row["dark_minimum_contrast"]) for row in group
                ),
                "mean_final_contrast": _mean(group, "dark_final_contrast"),
                "minimum_contrast_retention": min(
                    float(row["contrast_retention"]) for row in group
                ),
                "maximum_abs_release_shift_degrees": maximum_shift,
                "maximum_saturated_hd_fraction": maximum_saturation,
                "maximum_final_max_rate": maximum_final_rate,
                "maximum_final_heading_error_degrees": maximum_heading_error,
                "minimum_half_max_width_degrees": min(
                    float(row["dark_final_half_max_width_degrees"]) for row in group
                ),
                "maximum_half_max_width_degrees": max(
                    float(row["dark_final_half_max_width_degrees"]) for row in group
                ),
                "maximum_final_mean_rate": max(
                    float(row["dark_final_mean_rate"]) for row in group
                ),
                "rank_score": float(rank_score),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            str(row["cue_condition"]),
            -float(row["success_fraction"]),
            -float(row["rank_score"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary_grids(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    metric_specs = [
        ("success_fraction", "success fraction", 0.0, 1.0),
        ("minimum_final_pva_strength", "minimum final PVA", 0.0, 1.0),
        ("minimum_final_contrast", "minimum final contrast", 0.0, 1.0),
        ("maximum_abs_release_shift_degrees", "maximum release shift [deg]", 0.0, None),
        ("maximum_final_max_rate", "maximum final firing rate", 0.0, 1.0),
        ("maximum_final_heading_error_degrees", "maximum final heading error [deg]", 0.0, None),
        ("maximum_half_max_width_degrees", "maximum half-max width [deg]", 0.0, None),
    ]
    conditions = sorted({str(row["cue_condition"]) for row in rows})
    sigmas = sorted({float(row["sigma_radians"]) for row in rows})
    scales = sorted({float(row["excitatory_scale"]) for row in rows})
    inhibitions = sorted({float(row["inhibitory_ratio"]) for row in rows})
    for condition in conditions:
        condition_rows = [row for row in rows if row["cue_condition"] == condition]
        for metric, label, vmin, vmax in metric_specs:
            fig, axes = plt.subplots(
                1, len(sigmas), figsize=(4.2 * len(sigmas), 3.8), squeeze=False
            )
            for axis, sigma in zip(axes[0], sigmas):
                grid = np.full((len(scales), len(inhibitions)), np.nan)
                for row in condition_rows:
                    if np.isclose(float(row["sigma_radians"]), sigma):
                        i = scales.index(float(row["excitatory_scale"]))
                        j = inhibitions.index(float(row["inhibitory_ratio"]))
                        grid[i, j] = float(row[metric])
                image = axis.imshow(
                    grid,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(f"sigma={sigma:g} rad")
                axis.set_xticks(range(len(inhibitions)), [f"{x:g}" for x in inhibitions])
                axis.set_yticks(range(len(scales)), [f"{x:g}" for x in scales])
                axis.set_xlabel("inhibitory ratio")
                axis.set_ylabel("excitatory scale")
                fig.colorbar(image, ax=axis, label=label)
            fig.suptitle(f"{condition}: {label}")
            fig.tight_layout()
            figure_path = output_dir / "figures" / f"{condition}_{metric}.png"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)


def _timestamped_output_dir(*, root: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{timestamp}_{prefix}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{prefix}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _as_float_list(value: Iterable[Any], *, name: str) -> list[float]:
    result = [float(item) for item in value]
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def run_sweep(*, sweep_config_path: Path, dry_run: bool = False) -> Path | None:
    sweep = load_yaml(sweep_config_path)
    project_root = find_project_root(sweep_config_path)
    experiment_path = resolve_config_path(str(sweep["experiment"]["config"]))
    base_config_dict = load_yaml(experiment_path)
    base_config = load_experiment_config(experiment_path)
    protocol = sweep["protocol"]
    kernel_config = sweep["kernel"]
    runtime = sweep["runtime"]
    acceptance = {key: float(value) for key, value in sweep["acceptance"].items()}
    n_hd = int(protocol["n_hd"])
    if n_hd <= 0 or n_hd % 2 != 0:
        raise ValueError("protocol.n_hd must be a positive even number")
    if str(kernel_config["form"]) != "gaussian_minus_uniform":
        raise ValueError("only kernel.form=gaussian_minus_uniform is supported")
    scales = _as_float_list(kernel_config["excitatory_scales"], name="excitatory_scales")
    sigmas = _as_float_list(kernel_config["sigma_radians"], name="sigma_radians")
    inhibitions = _as_float_list(kernel_config["inhibitory_ratios"], name="inhibitory_ratios")
    headings = _as_float_list(protocol["initial_headings"], name="initial_headings")
    default_seed_offsets = [int(value) for value in runtime["seed_offsets"]]
    cue_conditions = list(protocol["cue_conditions"])
    trial_specs: list[tuple[dict[str, Any], int, float, float, float, float]] = []
    for condition in cue_conditions:
        condition_seed_offsets = [
            int(value)
            for value in condition.get("seed_offsets", default_seed_offsets)
        ]
        if not condition_seed_offsets:
            raise ValueError(f"cue condition {condition['name']} has no seed offsets")
        trial_specs.extend(
            product(
                [condition],
                condition_seed_offsets,
                headings,
                scales,
                sigmas,
                inhibitions,
            )
        )
    if dry_run:
        print(f"validated {sweep_config_path}")
        print(f"base experiment: {experiment_path}")
        print(f"trials: {len(trial_specs)}")
        print(f"kernel candidates per cue condition: {len(scales) * len(sigmas) * len(inhibitions)}")
        return None

    output_root = project_root / str(sweep["output"]["directory"])
    output_dir = _timestamped_output_dir(
        root=output_root,
        prefix=str(sweep["output"]["run_id_prefix"]),
    )
    save_yaml(output_dir / "sweep_config_resolved.yaml", sweep)
    rows: list[dict[str, Any]] = []
    iterator = tqdm(
        trial_specs,
        disable=not bool(runtime.get("progress", True)),
        desc="frozen bump sweep",
    )
    for condition, seed_offset, theta0, scale, sigma, inhibition in iterator:
        condition_name = str(condition["name"])
        config = _condition_config(
            base_config_dict=base_config_dict,
            visual_overrides=dict(condition.get("visual_overrides", {})),
            n_hd=n_hd,
            seed=int(base_config.simulation.seed) + seed_offset,
            theta0=theta0,
            cue_duration=float(protocol["cue_duration"]),
            darkness_duration=float(protocol["darkness_duration"]),
        )
        row, _history, _kernel = _trial_row(
            config=config,
            cue_condition=condition_name,
            seed_offset=seed_offset,
            theta0=theta0,
            excitatory_scale=scale,
            sigma_radians=sigma,
            inhibitory_ratio=inhibition,
            include_self_connections=bool(kernel_config["include_self_connections"]),
            acceptance=acceptance,
        )
        rows.append(row)

    aggregates = aggregate_trial_rows(rows)
    _write_csv(output_dir / "bump_parameter_trials.csv", rows)
    _write_csv(output_dir / "bump_parameter_aggregate.csv", aggregates)
    _plot_summary_grids(rows=aggregates, output_dir=output_dir)

    top_k = int(sweep["output"].get("top_k_per_cue_condition", 0))
    top_candidates: list[dict[str, Any]] = []
    representative_theta = min(headings, key=abs)
    for condition in cue_conditions:
        condition_name = str(condition["name"])
        condition_seed_offsets = [
            int(value)
            for value in condition.get("seed_offsets", default_seed_offsets)
        ]
        ranked = [row for row in aggregates if row["cue_condition"] == condition_name]
        selected = sorted(ranked, key=lambda row: float(row["rank_score"]), reverse=True)[:top_k]
        top_candidates.extend(selected)
        for rank, candidate in enumerate(selected, start=1):
            config = _condition_config(
                base_config_dict=base_config_dict,
                visual_overrides=dict(condition.get("visual_overrides", {})),
                n_hd=n_hd,
                seed=int(base_config.simulation.seed) + condition_seed_offsets[0],
                theta0=representative_theta,
                cue_duration=float(protocol["cue_duration"]),
                darkness_duration=float(protocol["darkness_duration"]),
            )
            _row, history, frozen_kernel = _trial_row(
                config=config,
                cue_condition=condition_name,
                seed_offset=condition_seed_offsets[0],
                theta0=representative_theta,
                excitatory_scale=float(candidate["excitatory_scale"]),
                sigma_radians=float(candidate["sigma_radians"]),
                inhibitory_ratio=float(candidate["inhibitory_ratio"]),
                include_self_connections=bool(kernel_config["include_self_connections"]),
                acceptance=acceptance,
                return_history=True,
            )
            assert history is not None
            candidate_dir = output_dir / "top_candidates" / f"{condition_name}_rank{rank}"
            save_npz(candidate_dir / "bump_history.npz", **history)
            save_npz(candidate_dir / "frozen_kernel.npz", w_hd_to_hd=frozen_kernel)
            plot_activity_heatmap(
                r_hd_history=history["r_hd"],
                time=history["time"],
                path=candidate_dir / "bump_activity_heatmap.png",
                title=(
                    f"{condition_name} rank {rank}: J={candidate['excitatory_scale']:g}, "
                    f"sigma={candidate['sigma_radians']:g}, rho={candidate['inhibitory_ratio']:g}"
                ),
                theta_hd_pref=initialize_vafidis_toy_state(
                    config=config, rng=make_rng(config.simulation.seed)
                ).theta_hd_pref,
                theta_hd_decoded=history["theta_hd_decoded"],
                theta_hd_decoded_peak=history["theta_hd_decoded_peak"],
                phase_id=history["phase_id"],
            )

    summary = {
        "base_experiment_config": str(experiment_path),
        "n_hd": n_hd,
        "trial_count": len(rows),
        "candidate_count_per_cue_condition": len(scales) * len(sigmas) * len(inhibitions),
        "acceptance": acceptance,
        "top_candidates": top_candidates,
    }
    save_json(output_dir / "bump_parameter_sweep_summary.json", summary)
    print(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", required=True, help="Phase 1A sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count trials only")
    args = parser.parse_args()
    run_sweep(sweep_config_path=resolve_config_path(args.sweep_config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
