"""Calibrate frozen antisymmetric HR-to-HD kernels for path integration."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from learning.common.angles import circular_difference
from learning.common.random import make_rng
from learning.analysis.metrics import (
    estimate_decoded_velocity,
    summarize_velocity_gain,
    summarize_velocity_tracking,
)
from learning.config.load_config import find_project_root, load_yaml, save_yaml
from learning.experiments.run_bump_parameter_sweep import (
    _condition_config,
    make_gaussian_minus_uniform_kernel,
)
from learning.experiments.run_vafidis_toy import (
    DARKNESS_PHASE_ID,
    phase_mask,
    resolve_config_path,
    run_bump_maintenance_test,
    run_visual_dark_visual_protocol,
    summarize_zero_velocity_drive,
)
from learning.io.save_load import save_json, save_npz
from learning.models.vafidis_toy import initialize_vafidis_toy_state
from learning.plotting.activity import plot_activity_heatmap
from learning.plotting.backend import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt


def make_antisymmetric_shifted_hr_kernel(
    *,
    theta_hd_preference: np.ndarray,
    scale: float,
    sigma_radians: float,
    shift_radians: float,
    orientation: float,
) -> np.ndarray:
    """Return concatenated L/R kernels with exact W_R=-W_L cancellation."""
    theta_hd_preference = np.asarray(theta_hd_preference, dtype=float)
    if theta_hd_preference.ndim != 1 or theta_hd_preference.size % 2 != 0:
        raise ValueError("theta_hd_preference must contain paired HD cells")
    if scale <= 0.0 or sigma_radians <= 0.0 or shift_radians <= 0.0:
        raise ValueError("HR scale, sigma and shift must be positive")
    if orientation not in {-1.0, 1.0}:
        raise ValueError("orientation must be -1 or +1")
    theta_lhr = theta_hd_preference[0::2]
    theta_rhr = theta_hd_preference[1::2]
    if not np.allclose(theta_lhr, theta_rhr):
        raise ValueError("paired LHR/RHR source preferences must coincide")
    angular_offset = circular_difference(
        theta_hd_preference[:, None], theta_lhr[None, :]
    )
    shifted_positive = circular_difference(angular_offset, float(shift_radians))
    shifted_negative = circular_difference(angular_offset, -float(shift_radians))
    differential = float(orientation) * float(scale) * (
        np.exp(-0.5 * np.square(shifted_positive / float(sigma_radians)))
        - np.exp(-0.5 * np.square(shifted_negative / float(sigma_radians)))
    )
    return np.concatenate([differential, -differential], axis=1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _timestamped_output_dir(*, root: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{timestamp}_{prefix}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{prefix}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_stationary_cue_velocity_gain_test(
    *, config, trained_state
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Measure darkness gain after a stationary cue, avoiding visual-path bias."""
    commanded = np.asarray(config.tests.gain_velocities, dtype=float)
    decoded_values: list[float] = []
    for commanded_velocity in commanded:
        elapsed = 0.0
        cue_duration = float(config.simulation.pi_cue_duration)

        def phase_velocity_step(dt: float) -> float:
            nonlocal elapsed
            elapsed += float(dt)
            return 0.0 if elapsed <= cue_duration + 1e-12 else float(commanded_velocity)

        history = run_visual_dark_visual_protocol(
            config=config,
            trained_state=trained_state,
            theta_true=config.simulation.theta0,
            darkness_duration=config.simulation.darkness_test_duration,
            angular_velocity_step=phase_velocity_step,
            cue_duration=cue_duration,
            recue_duration=0.0,
        )
        dark = phase_mask(history, DARKNESS_PHASE_ID)
        dark_time = history["time"][dark]
        decoded_values.append(
            estimate_decoded_velocity(
                time=dark_time - dark_time[0],
                theta_decoded=history["theta_hd_decoded"][dark],
                start_fraction=0.25,
            )
        )
    decoded = np.asarray(decoded_values, dtype=float)
    gain = summarize_velocity_gain(
        commanded_velocity=commanded,
        decoded_velocity=decoded,
    )
    tracking = summarize_velocity_tracking(
        commanded_velocity=commanded,
        decoded_velocity=decoded,
    )
    return {
        "commanded_velocity": commanded,
        "decoded_velocity": decoded,
    }, {
        "velocity_gain": float(gain["gain"]),
        "velocity_gain_abs_error": float(abs(gain["gain"] - 1.0)),
        "velocity_gain_intercept": float(gain["intercept"]),
        "velocity_gain_intercept_abs": float(abs(gain["intercept"])),
        "velocity_gain_r_squared": float(gain["r_squared"]),
        "velocity_tracking_rmse": float(tracking["velocity_tracking_rmse"]),
        "velocity_direction_match_fraction": float(
            tracking["velocity_direction_match_fraction"]
        ),
    }


def _trial_row(
    *,
    config,
    hd_kernel_config: dict[str, Any],
    hr_scale: float,
    hr_sigma: float,
    hr_shift: float,
    orientation: float,
    seed_offset: int,
    theta0: float,
    acceptance: dict[str, float],
    cue_condition: str = "unspecified",
    return_histories: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, np.ndarray, np.ndarray]:
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    hd_kernel = make_gaussian_minus_uniform_kernel(
        theta_preference=state.theta_hd_pref,
        excitatory_scale=float(hd_kernel_config["excitatory_scale"]),
        sigma_radians=float(hd_kernel_config["sigma_radians"]),
        inhibitory_ratio=float(hd_kernel_config["inhibitory_ratio"]),
        include_self_connections=False,
    )
    hr_kernel = make_antisymmetric_shifted_hr_kernel(
        theta_hd_preference=state.theta_hd_pref,
        scale=hr_scale,
        sigma_radians=hr_sigma,
        shift_radians=hr_shift,
        orientation=orientation,
    )
    state.w_hd_to_hd = hd_kernel
    state.w_hr_to_hd = hr_kernel
    bump_history, bump_metrics = run_bump_maintenance_test(
        config=config, trained_state=state
    )
    gain_history, gain_metrics = run_stationary_cue_velocity_gain_test(
        config=config, trained_state=state
    )
    zero_drive = summarize_zero_velocity_drive(
        config=config,
        trained_state=state,
        bump_history=bump_history,
    )
    criteria = {
        "velocity_gain": float(gain_metrics["velocity_gain_abs_error"])
        <= float(acceptance["maximum_velocity_gain_abs_error"]),
        "velocity_intercept": float(gain_metrics["velocity_gain_intercept_abs"])
        <= float(acceptance["maximum_velocity_gain_intercept_abs"]),
        "velocity_r_squared": float(gain_metrics["velocity_gain_r_squared"])
        >= float(acceptance["minimum_velocity_gain_r_squared"]),
        "velocity_direction": float(gain_metrics["velocity_direction_match_fraction"])
        >= float(acceptance["minimum_velocity_direction_match_fraction"]),
        "velocity_rmse": float(gain_metrics["velocity_tracking_rmse"])
        <= float(acceptance["maximum_velocity_tracking_rmse"]),
        "bump_pva": float(bump_metrics["bump_final_pva_strength"])
        >= float(acceptance["minimum_bump_final_pva_strength"]),
        "bump_contrast": float(bump_metrics["bump_final_contrast"])
        >= float(acceptance["minimum_bump_final_contrast"]),
        "zero_velocity_shift": float(bump_metrics["bump_abs_release_shift_deg"])
        <= float(acceptance["maximum_zero_velocity_release_shift_degrees"]),
        "bump_saturation": float(bump_metrics["bump_max_saturated_hd_bins"])
        / float(config.model.n_theta)
        <= float(acceptance["maximum_bump_saturated_hd_fraction"]),
    }
    row: dict[str, Any] = {
        "cue_condition": str(cue_condition),
        "hd_kernel_name": str(hd_kernel_config["name"]),
        "hd_excitatory_scale": float(hd_kernel_config["excitatory_scale"]),
        "hd_sigma_radians": float(hd_kernel_config["sigma_radians"]),
        "hd_inhibitory_ratio": float(hd_kernel_config["inhibitory_ratio"]),
        "seed": int(config.simulation.seed),
        "seed_offset": int(seed_offset),
        "theta0": float(theta0),
        "theta0_degrees": float(np.rad2deg(theta0)),
        "hr_scale": float(hr_scale),
        "hr_sigma_radians": float(hr_sigma),
        "hr_shift_radians": float(hr_shift),
        "hr_orientation": float(orientation),
        "hr_kernel_min": float(np.min(hr_kernel)),
        "hr_kernel_max": float(np.max(hr_kernel)),
        "velocity_gain": float(gain_metrics["velocity_gain"]),
        "velocity_gain_abs_error": float(gain_metrics["velocity_gain_abs_error"]),
        "velocity_gain_intercept": float(gain_metrics["velocity_gain_intercept"]),
        "velocity_gain_intercept_abs": float(gain_metrics["velocity_gain_intercept_abs"]),
        "velocity_gain_r_squared": float(gain_metrics["velocity_gain_r_squared"]),
        "velocity_tracking_rmse": float(gain_metrics["velocity_tracking_rmse"]),
        "velocity_direction_match_fraction": float(
            gain_metrics["velocity_direction_match_fraction"]
        ),
        "bump_final_pva_strength": float(bump_metrics["bump_final_pva_strength"]),
        "bump_final_contrast": float(bump_metrics["bump_final_contrast"]),
        "bump_abs_release_shift_degrees": float(
            bump_metrics["bump_abs_release_shift_deg"]
        ),
        "bump_max_saturated_hd_fraction": float(
            bump_metrics["bump_max_saturated_hd_bins"] / config.model.n_theta
        ),
        **zero_drive,
        **{f"passes_{name}": bool(value) for name, value in criteria.items()},
        "passes_all": bool(all(criteria.values())),
    }
    return (
        row,
        bump_history if return_histories else None,
        gain_history if return_histories else None,
        hd_kernel,
        hr_kernel,
    )


def aggregate_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parameter_names = (
        "cue_condition",
        "hd_kernel_name",
        "hr_scale",
        "hr_sigma_radians",
        "hr_shift_radians",
        "hr_orientation",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[name] for name in parameter_names), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        success_fraction = float(np.mean([float(row["passes_all"]) for row in group]))
        mean_gain_error = float(np.mean([float(row["velocity_gain_abs_error"]) for row in group]))
        max_rmse = max(float(row["velocity_tracking_rmse"]) for row in group)
        min_pva = min(float(row["bump_final_pva_strength"]) for row in group)
        max_shift = max(float(row["bump_abs_release_shift_degrees"]) for row in group)
        rank_score = (
            5.0 * success_fraction
            - mean_gain_error
            - max_rmse
            + min_pva
            - max_shift / 180.0
            - 0.001 * float(key[2])
        )
        aggregates.append(
            {
                **dict(zip(parameter_names, key)),
                "trial_count": len(group),
                "success_fraction": success_fraction,
                "mean_velocity_gain": float(
                    np.mean([float(row["velocity_gain"]) for row in group])
                ),
                "maximum_velocity_gain_abs_error": max(
                    float(row["velocity_gain_abs_error"]) for row in group
                ),
                "maximum_velocity_gain_intercept_abs": max(
                    float(row["velocity_gain_intercept_abs"]) for row in group
                ),
                "minimum_velocity_gain_r_squared": min(
                    float(row["velocity_gain_r_squared"]) for row in group
                ),
                "maximum_velocity_tracking_rmse": max_rmse,
                "minimum_bump_final_pva_strength": min_pva,
                "minimum_bump_final_contrast": min(
                    float(row["bump_final_contrast"]) for row in group
                ),
                "maximum_zero_velocity_release_shift_degrees": max_shift,
                "rank_score": float(rank_score),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (-float(row["success_fraction"]), -float(row["rank_score"])),
    )


def _plot_gain_curve(*, history: dict[str, np.ndarray], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    commanded = np.asarray(history["commanded_velocity"], dtype=float)
    decoded = np.asarray(history["decoded_velocity"], dtype=float)
    fig, axis = plt.subplots(figsize=(4.5, 4.0))
    axis.plot(commanded, commanded, color="black", linestyle="--", label="unit gain")
    axis.plot(commanded, decoded, marker="o", color="#2a6fbb", label="darkness decode")
    axis.set_xlabel("commanded angular velocity [rad/s]")
    axis.set_ylabel("decoded angular velocity [rad/s]")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_summary_grids(*, rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot HR scale x shift grids separately by HD background and orientation."""
    metric_specs = [
        ("success_fraction", "success fraction", 0.0, 1.0),
        ("mean_velocity_gain", "mean velocity gain", None, None),
        ("maximum_velocity_gain_abs_error", "maximum gain error", 0.0, None),
        ("maximum_velocity_tracking_rmse", "maximum tracking RMSE", 0.0, None),
    ]
    conditions = sorted({str(row["cue_condition"]) for row in rows})
    hd_names = sorted({str(row["hd_kernel_name"]) for row in rows})
    orientations = sorted({float(row["hr_orientation"]) for row in rows})
    sigmas = sorted({float(row["hr_sigma_radians"]) for row in rows})
    scales = sorted({float(row["hr_scale"]) for row in rows})
    shifts = sorted({float(row["hr_shift_radians"]) for row in rows})
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        for hd_name in hd_names:
            for orientation in orientations:
                selected = [
                    row
                    for row in rows
                    if row["cue_condition"] == condition
                    and row["hd_kernel_name"] == hd_name
                    and np.isclose(float(row["hr_orientation"]), orientation)
                ]
                for metric, label, vmin, vmax in metric_specs:
                    fig, axes = plt.subplots(
                        1,
                        len(sigmas),
                        figsize=(4.2 * len(sigmas), 3.8),
                        squeeze=False,
                    )
                    for axis, sigma in zip(axes[0], sigmas):
                        grid = np.full((len(scales), len(shifts)), np.nan)
                        for row in selected:
                            if np.isclose(float(row["hr_sigma_radians"]), sigma):
                                i = scales.index(float(row["hr_scale"]))
                                j = shifts.index(float(row["hr_shift_radians"]))
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
                        axis.set_xticks(range(len(shifts)), [f"{x:g}" for x in shifts])
                        axis.set_yticks(range(len(scales)), [f"{x:g}" for x in scales])
                        axis.set_xlabel("HR angular shift [rad]")
                        axis.set_ylabel("HR scale")
                        fig.colorbar(image, ax=axis, label=label)
                    fig.suptitle(
                        f"{condition}, {hd_name}, orientation={orientation:+g}: {label}"
                    )
                    fig.tight_layout()
                    fig.savefig(
                        figure_dir
                        / f"{condition}_{hd_name}_orientation_{orientation:+g}_{metric}.png",
                        dpi=160,
                    )
                    plt.close(fig)


def run_sweep(*, sweep_config_path: Path, dry_run: bool = False) -> Path | None:
    sweep = load_yaml(sweep_config_path)
    project_root = find_project_root(sweep_config_path)
    experiment_path = resolve_config_path(str(sweep["experiment"]["config"]))
    base_config_dict = load_yaml(experiment_path)
    protocol = sweep["protocol"]
    runtime = sweep["runtime"]
    hr_config = sweep["hr_kernel"]
    if str(hr_config["form"]) != "antisymmetric_shifted_gaussian":
        raise ValueError("only antisymmetric_shifted_gaussian HR kernels are supported")
    hd_kernels = list(sweep["hd_kernels"])
    scales = [float(value) for value in hr_config["scales"]]
    sigmas = [float(value) for value in hr_config["sigma_radians"]]
    shifts = [float(value) for value in hr_config["shifts_radians"]]
    orientations = [float(value) for value in hr_config["orientations"]]
    default_seed_offsets = [int(value) for value in runtime["seed_offsets"]]
    headings = [float(value) for value in protocol["initial_headings"]]
    cue_conditions = list(protocol["cue_conditions"])
    trial_specs = []
    for condition in cue_conditions:
        condition_seed_offsets = [
            int(value)
            for value in condition.get("seed_offsets", default_seed_offsets)
        ]
        trial_specs.extend(
            product(
                [condition],
                hd_kernels,
                condition_seed_offsets,
                headings,
                scales,
                sigmas,
                shifts,
                orientations,
            )
        )
    if dry_run:
        print(f"validated {sweep_config_path}")
        print(f"base experiment: {experiment_path}")
        print(f"trials: {len(trial_specs)}")
        print(f"HR candidates per HD kernel and cue: {len(scales) * len(sigmas) * len(shifts) * len(orientations)}")
        return None

    output_dir = _timestamped_output_dir(
        root=project_root / str(sweep["output"]["directory"]),
        prefix=str(sweep["output"]["run_id_prefix"]),
    )
    save_yaml(output_dir / "sweep_config_resolved.yaml", sweep)
    acceptance = {key: float(value) for key, value in sweep["acceptance"].items()}
    rows: list[dict[str, Any]] = []
    iterator = tqdm(
        trial_specs,
        disable=not bool(runtime.get("progress", True)),
        desc="frozen HR velocity sweep",
    )
    for condition, hd_kernel, seed_offset, theta0, scale, sigma, shift, orientation in iterator:
        config = _condition_config(
            base_config_dict=base_config_dict,
            visual_overrides=dict(condition.get("visual_overrides", {})),
            n_hd=int(protocol["n_hd"]),
            seed=int(base_config_dict["simulation"]["seed"]) + seed_offset,
            theta0=theta0,
            cue_duration=float(protocol["cue_duration"]),
            darkness_duration=float(protocol["bump_darkness_duration"]),
        )
        config.simulation.pi_cue_duration = float(protocol["cue_duration"])
        config.simulation.darkness_test_duration = float(
            protocol["velocity_darkness_duration"]
        )
        config.tests.gain_velocities = [
            float(value) for value in protocol["gain_velocities"]
        ]
        row, *_ = _trial_row(
            config=config,
            hd_kernel_config=hd_kernel,
            hr_scale=scale,
            hr_sigma=sigma,
            hr_shift=shift,
            orientation=orientation,
            seed_offset=seed_offset,
            theta0=theta0,
            acceptance=acceptance,
            cue_condition=str(condition["name"]),
        )
        rows.append(row)

    aggregates = aggregate_trial_rows(rows)
    _write_csv(output_dir / "hr_parameter_trials.csv", rows)
    _write_csv(output_dir / "hr_parameter_aggregate.csv", aggregates)
    _plot_summary_grids(rows=aggregates, output_dir=output_dir)
    top_k = int(sweep["output"].get("top_k_per_cue_condition", 0))
    top_candidates: list[dict[str, Any]] = []
    ranked_with_context: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for condition in cue_conditions:
        condition_name = str(condition["name"])
        selected = [
            row for row in aggregates if row["cue_condition"] == condition_name
        ][:top_k]
        top_candidates.extend(selected)
        ranked_with_context.extend(
            (rank, candidate, condition)
            for rank, candidate in enumerate(selected, start=1)
        )
    for rank, candidate, condition in ranked_with_context:
        condition_name = str(condition["name"])
        hd_kernel = next(
            item for item in hd_kernels if item["name"] == candidate["hd_kernel_name"]
        )
        theta0 = headings[0]
        condition_seed_offsets = [
            int(value)
            for value in condition.get("seed_offsets", default_seed_offsets)
        ]
        seed_offset = condition_seed_offsets[0]
        config = _condition_config(
            base_config_dict=base_config_dict,
            visual_overrides=dict(condition.get("visual_overrides", {})),
            n_hd=int(protocol["n_hd"]),
            seed=int(base_config_dict["simulation"]["seed"]) + seed_offset,
            theta0=theta0,
            cue_duration=float(protocol["cue_duration"]),
            darkness_duration=float(protocol["bump_darkness_duration"]),
        )
        config.simulation.pi_cue_duration = float(protocol["cue_duration"])
        config.simulation.darkness_test_duration = float(
            protocol["velocity_darkness_duration"]
        )
        config.tests.gain_velocities = [float(value) for value in protocol["gain_velocities"]]
        _row, bump_history, gain_history, hd_weights, hr_weights = _trial_row(
            config=config,
            hd_kernel_config=hd_kernel,
            hr_scale=float(candidate["hr_scale"]),
            hr_sigma=float(candidate["hr_sigma_radians"]),
            hr_shift=float(candidate["hr_shift_radians"]),
            orientation=float(candidate["hr_orientation"]),
            seed_offset=seed_offset,
            theta0=theta0,
            acceptance=acceptance,
            cue_condition=condition_name,
            return_histories=True,
        )
        assert bump_history is not None and gain_history is not None
        candidate_dir = output_dir / "top_candidates" / f"{condition_name}_rank{rank}"
        save_npz(candidate_dir / "bump_history.npz", **bump_history)
        save_npz(candidate_dir / "velocity_gain_history.npz", **gain_history)
        save_npz(
            candidate_dir / "frozen_weights.npz",
            w_hd_to_hd=hd_weights,
            w_hr_to_hd=hr_weights,
        )
        plot_activity_heatmap(
            r_hd_history=bump_history["r_hd"],
            time=bump_history["time"],
            path=candidate_dir / "bump_activity_heatmap.png",
            title=f"HR sweep rank {rank}: {candidate['hd_kernel_name']}",
            theta_hd_pref=initialize_vafidis_toy_state(
                config=config, rng=make_rng(config.simulation.seed)
            ).theta_hd_pref,
            theta_hd_decoded=bump_history["theta_hd_decoded"],
            theta_hd_decoded_peak=bump_history["theta_hd_decoded_peak"],
            phase_id=bump_history["phase_id"],
        )
        _plot_gain_curve(
            history=gain_history,
            path=candidate_dir / "velocity_gain_curve.png",
            title=f"HR sweep rank {rank}",
        )
    save_json(
        output_dir / "hr_parameter_sweep_summary.json",
        {
            "base_experiment_config": str(experiment_path),
            "trial_count": len(rows),
            "acceptance": acceptance,
            "top_candidates": top_candidates,
        },
    )
    print(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", required=True, help="Phase 1B HR sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count trials only")
    args = parser.parse_args()
    run_sweep(sweep_config_path=resolve_config_path(args.sweep_config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
