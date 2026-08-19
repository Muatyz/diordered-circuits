"""Run frozen-weight tests for a Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.common.random import make_rng
from learning.config.diagnostics import selected_diagnostics
from learning.config.load_config import (
    find_project_root,
    load_experiment_config,
    save_yaml,
)
from learning.experiments.run_vafidis_toy import (
    IncrementalDiagnosticRecorder,
    resolve_config_path,
    resolve_weight_states,
    run_all_tests,
    run_weight_performance_selection,
    save_run_outputs,
    state_from_weight_snapshot,
)
from learning.io.run_dir import create_run_dir
from learning.io.save_load import load_npz, save_json, save_npz
from learning.dynamics.hd_dynamics import effective_hd_distal_weight_matrices
from learning.models.vafidis_toy import (
    VafidisToyParams,
    initialize_vafidis_toy_state,
    validate_vafidis_toy_state,
)


def load_trained_state_from_run(
    *,
    run_dir: Path,
    diagnostics_config_path: Path,
):
    base_config_path = run_dir / "config_resolved.yaml"
    config = load_experiment_config(
        base_config_path,
        diagnostics_path=diagnostics_config_path,
    )
    rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(config=config, rng=rng)
    source = str(config.diagnostics.weight_source).lower()
    source_filename = {
        "best": "best_weights.npz",
        "final": "final_weights.npz",
        "training_selected": "training_selected_weights.npz",
    }.get(source)
    source_path = run_dir / source_filename if source_filename is not None else None
    trained_weights = load_npz(
        source_path
        if source_path is not None and source_path.exists()
        else run_dir / "trained_weights.npz"
    )
    state.w_hd_to_hd = trained_weights["w_hd_to_hd"].copy()
    state.w_hr_to_hd = trained_weights["w_hr_to_hd"].copy()
    state.w_hd_to_hr = trained_weights["w_hd_to_hr"].copy()
    if "weight_snapshot_time" in trained_weights:
        state.time = float(trained_weights["weight_snapshot_time"])
    saved_visual_profiles = trained_weights.get("visual_tuning_profiles")
    if saved_visual_profiles is not None and saved_visual_profiles.size > 0:
        state.visual_tuning_profiles = saved_visual_profiles.copy()
    weight_history_path = run_dir / "weight_history.npz"
    weight_history = (
        load_npz(weight_history_path) if weight_history_path.exists() else {}
    )
    if source == "snapshot":
        if config.diagnostics.weight_snapshot_index is None:
            raise ValueError(
                "diagnostics.weight_snapshot_index is required for snapshot source"
            )
        state = state_from_weight_snapshot(
            template_state=state,
            weight_history=weight_history,
            snapshot_index=int(config.diagnostics.weight_snapshot_index),
        )
    elif source == "final" and (source_path is None or not source_path.exists()):
        state = state_from_weight_snapshot(
            template_state=state,
            weight_history=weight_history,
            snapshot_index=-1,
        )
    elif source == "best" and (source_path is None or not source_path.exists()):
        selection_history, _selection_metrics = run_weight_performance_selection(
            config=config,
            trained_state=state,
            weight_history=weight_history,
        )
        _final_state, best_state, _diagnostic_state, metadata = (
            resolve_weight_states(
                config=config,
                training_selected_state=state,
                weight_history=weight_history,
                weight_selection_history=selection_history,
            )
        )
        state = best_state
        save_diagnostic_weight_state(
            run_dir=run_dir,
            config=config,
            state=best_state,
            filename="best_weights.npz",
            weight_source="best",
            source_index=int(metadata["best_snapshot_source_index"]),
        )
        save_npz(
            run_dir / "weight_selection_history.npz",
            **{**selection_history, **metadata},
        )
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    return config, state


def save_diagnostic_weight_state(
    *,
    run_dir: Path,
    config,
    state,
    filename: str = "diagnostic_weights.npz",
    weight_source: str | None = None,
    source_index: int | None = None,
) -> None:
    """Persist the weights actually used by an independent diagnostic run."""

    params = VafidisToyParams.from_config(config)
    effective_hd, effective_hr = effective_hd_distal_weight_matrices(
        w_hd_to_hd=state.w_hd_to_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        normalization=params.hd_distal_normalization,
    )
    save_npz(
        run_dir / filename,
        theta_hd_pref=state.theta_hd_pref,
        w_hd_to_hd=state.w_hd_to_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        effective_w_hd_to_hd=effective_hd,
        effective_w_hr_to_hd=effective_hr,
        w_lhr_to_hd=state.w_lhr_to_hd,
        w_rhr_to_hd=state.w_rhr_to_hd,
        w_hd_to_hr=state.w_hd_to_hr,
        visual_tuning_profiles=(
            np.empty((0, 0), dtype=float)
            if state.visual_tuning_profiles is None
            else state.visual_tuning_profiles
        ),
        weight_source=np.asarray(
            str(config.diagnostics.weight_source)
            if weight_source is None
            else str(weight_source)
        ),
        weight_snapshot_source_index=np.asarray(
            (
                -1
                if config.diagnostics.weight_snapshot_index is None
                else int(config.diagnostics.weight_snapshot_index)
            )
            if source_index is None
            else int(source_index),
            dtype=int,
        ),
        weight_snapshot_time=np.asarray(float(state.time)),
    )


def run_tests_for_existing_run(
    *,
    run_dir: Path,
    make_figures: bool,
    diagnostics_config_path: Path,
) -> None:
    config, trained_state = load_trained_state_from_run(
        run_dir=run_dir,
        diagnostics_config_path=diagnostics_config_path,
    )
    cached_histories = {}
    previous_diagnostic_weights_path = run_dir / "diagnostic_weights.npz"
    cached_weights_match = False
    if previous_diagnostic_weights_path.exists():
        previous_diagnostic_weights = load_npz(previous_diagnostic_weights_path)
        cached_weights_match = bool(
            np.array_equal(
                previous_diagnostic_weights.get("w_hd_to_hd", np.empty(0)),
                trained_state.w_hd_to_hd,
            )
            and np.array_equal(
                previous_diagnostic_weights.get("w_hr_to_hd", np.empty(0)),
                trained_state.w_hr_to_hd,
            )
        )
    weight_history_path = run_dir / "weight_history.npz"
    weight_history = (
        load_npz(weight_history_path) if weight_history_path.exists() else {}
    )
    cached_hd_tuning_path = run_dir / "hd_tuning_history.npz"
    if cached_weights_match and cached_hd_tuning_path.exists():
        cached_histories["hd_tuning"] = load_npz(cached_hd_tuning_path)
    cached_trajectory_path = run_dir / "bump_attractor_trajectory_history.npz"
    if cached_weights_match and cached_trajectory_path.exists():
        cached_histories["bump_attractor_trajectories"] = load_npz(
            cached_trajectory_path
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
        trained_state=trained_state,
        weight_history=weight_history,
        cached_histories=cached_histories,
        diagnostic_recorder=diagnostic_recorder,
        continue_on_error=True,
    )
    diagnostic_source = str(config.diagnostics.weight_source).lower()
    test_metrics.update(
        {
            "diagnostic_weight_source_is_best": float(
                diagnostic_source == "best"
            ),
            "diagnostic_weight_source_is_final": float(
                diagnostic_source == "final"
            ),
            "diagnostic_weight_snapshot_time": float(trained_state.time),
        }
    )
    diagnostic_recorder.finalize(test_metrics)

    enabled_diagnostic_names = selected_diagnostics(config)
    histories_to_save = {
        "bump_maintenance": ("bump_history.npz", bump_history),
        "hd_tuning": ("hd_tuning_history.npz", hd_tuning_history),
        "bump_attractor_trajectories": (
            "bump_attractor_trajectory_history.npz",
            bump_attractor_trajectory_history,
        ),
        "slow_manifold": (
            "slow_manifold_diagnostics.npz",
            slow_manifold_history,
        ),
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
    for diagnostic_name, (filename, history) in histories_to_save.items():
        if history and diagnostic_name in enabled_diagnostic_names:
            save_npz(run_dir / filename, **history)
    if (
        "hd_tuning_dependency_computed" in test_metrics
        and "hd_tuning" not in enabled_diagnostic_names
    ):
        if hd_tuning_history:
            save_npz(run_dir / "hd_tuning_history.npz", **hd_tuning_history)
    if (
        "bump_attractor_dependency_computed" in test_metrics
        and "bump_attractor_trajectories" not in enabled_diagnostic_names
    ):
        if bump_attractor_trajectory_history:
            save_npz(
                run_dir / "bump_attractor_trajectory_history.npz",
                **bump_attractor_trajectory_history,
            )
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_diagnostic_weight_state(
        run_dir=run_dir,
        config=config,
        state=trained_state,
    )
    save_yaml(run_dir / "test_config_resolved.yaml", config.to_dict())
    if make_figures:
        make_vafidis_figures_for_run(run_dir=run_dir)


def run_tests_from_config(
    *,
    config_path: Path,
    make_figures: bool,
    diagnostics_config_path: Path,
    profile_paths: Sequence[Path] = (),
    config_overrides: Sequence[str] = (),
) -> Path:
    project_root = find_project_root(config_path)
    config = load_experiment_config(
        config_path,
        diagnostics_path=diagnostics_config_path,
        profile_paths=profile_paths,
        overrides=config_overrides,
    )
    rng = make_rng(config.simulation.seed)
    training_selected_state = initialize_vafidis_toy_state(config=config, rng=rng)
    weight_snapshot_pi_history, weight_snapshot_pi_metrics = (
        run_weight_performance_selection(
            config=config,
            trained_state=training_selected_state,
            weight_history={},
        )
    )
    final_state, best_state, trained_state, weight_selection_metadata = (
        resolve_weight_states(
            config=config,
            training_selected_state=training_selected_state,
            weight_history={},
            weight_selection_history=weight_snapshot_pi_history,
        )
    )
    run_dir = create_run_dir(
        project_root=project_root,
        runs_root=config.paths.runs_root,
        experiment_name=f"{config.experiment_name}_untrained_test",
        seed=config.simulation.seed,
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
        trained_state=trained_state,
        weight_history={},
        precomputed_weight_snapshot_pi=(
            weight_snapshot_pi_history,
            weight_snapshot_pi_metrics,
        ),
        diagnostic_recorder=diagnostic_recorder,
        continue_on_error=True,
    )
    diagnostic_recorder.finalize(test_metrics)
    save_run_outputs(
        run_dir=run_dir,
        config=config,
        params=VafidisToyParams.from_config(config),
        trained_state=trained_state,
        final_state=final_state,
        best_state=best_state,
        training_selected_state=training_selected_state,
        weight_selection_metadata=weight_selection_metadata,
        training_history={"time": [], "theta_true": [], "theta_hd_decoded": [], "angular_velocity": []},
        weight_history={},
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
    if make_figures:
        make_vafidis_figures_for_run(run_dir=run_dir)
    return run_dir


def main() -> None:
    """CLI driven exclusively by the grouped diagnostics hyper config."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        help="Existing run directory containing trained_weights.npz.",
    )
    source.add_argument(
        "--config",
        help="Experiment config for untrained frozen-weight tests.",
    )
    parser.add_argument(
        "--diagnostics-config",
        required=True,
        help="Single grouped diagnostics hyper config.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="PATH",
        help="Partial config for --config mode; repeat to compose.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Dotted config override for --config mode; repeat as needed.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    if args.run_dir is not None and (args.profile or args.config_overrides):
        parser.error("--profile and --set are only valid together with --config")
    diagnostics_config_path = resolve_config_path(args.diagnostics_config)
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        run_tests_for_existing_run(
            run_dir=run_dir,
            make_figures=not args.no_figures,
            diagnostics_config_path=diagnostics_config_path,
        )
        print(f"Updated tests in {run_dir}")
        return

    config_path = resolve_config_path(args.config)
    profile_paths = [
        resolve_config_path(profile_path) for profile_path in args.profile
    ]
    run_dir = run_tests_from_config(
        config_path=config_path,
        make_figures=not args.no_figures,
        diagnostics_config_path=diagnostics_config_path,
        profile_paths=profile_paths,
        config_overrides=args.config_overrides,
    )
    print(f"Saved untrained tests to {run_dir}")


if __name__ == "__main__":
    main()
