"""Run frozen-weight tests for a Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.common.random import make_rng
from learning.config.load_config import (
    find_project_root,
    load_experiment_config,
    save_yaml,
)
from learning.experiments.run_vafidis_toy import (
    resolve_config_path,
    run_all_tests,
    save_run_outputs,
)
from learning.io.run_dir import create_run_dir
from learning.io.save_load import load_npz, save_json, save_npz
from learning.models.vafidis_toy import (
    VafidisToyParams,
    initialize_vafidis_toy_state,
    validate_vafidis_toy_state,
)


def load_trained_state_from_run(*, run_dir: Path):
    test_config_path = run_dir / "test_config_resolved.yaml"
    config = load_experiment_config(
        test_config_path
        if test_config_path.exists()
        else run_dir / "config_resolved.yaml"
    )
    trained_weights = load_npz(run_dir / "trained_weights.npz")
    rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(config=config, rng=rng)
    state.w_hd_to_hd = trained_weights["w_hd_to_hd"].copy()
    state.w_hr_to_hd = trained_weights["w_hr_to_hd"].copy()
    state.w_hd_to_hr = trained_weights["w_hd_to_hr"].copy()
    saved_visual_profiles = trained_weights.get("visual_tuning_profiles")
    if saved_visual_profiles is not None and saved_visual_profiles.size > 0:
        state.visual_tuning_profiles = saved_visual_profiles.copy()
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    return config, state


def run_tests_for_existing_run(
    *,
    run_dir: Path,
    make_figures: bool,
    bump_attractor_duration: float | None = None,
    enable_timescale_separation: bool = False,
    enable_velocity_trajectory_sweep: bool = False,
    enable_velocity_phase_flow: bool = False,
    velocity_sweep_values: list[float] | None = None,
    velocity_phase_flow_values: list[float] | None = None,
    velocity_phase_flow_probe_count: int | None = None,
) -> None:
    config, trained_state = load_trained_state_from_run(run_dir=run_dir)
    if bump_attractor_duration is not None:
        if bump_attractor_duration <= 0.0:
            raise ValueError("bump_attractor_duration must be positive")
        config.tests.bump_attractor_trajectory_enabled = True
        config.tests.bump_attractor_duration = float(bump_attractor_duration)
    if enable_timescale_separation:
        config.tests.bump_attractor_trajectory_enabled = True
        config.tests.timescale_separation_enabled = True
    if enable_velocity_trajectory_sweep:
        config.tests.velocity_trajectory_sweep_enabled = True
    if enable_velocity_phase_flow:
        config.tests.velocity_trajectory_sweep_enabled = True
        config.tests.velocity_phase_flow_probe_enabled = True
    if velocity_sweep_values is not None:
        config.tests.velocity_trajectory_sweep_enabled = True
        config.tests.velocity_trajectory_sweep_velocities = velocity_sweep_values
    if velocity_phase_flow_values is not None:
        config.tests.velocity_trajectory_sweep_enabled = True
        config.tests.velocity_phase_flow_probe_enabled = True
        config.tests.velocity_phase_flow_probe_velocities = (
            velocity_phase_flow_values
        )
    if velocity_phase_flow_probe_count is not None:
        if velocity_phase_flow_probe_count < 3:
            raise ValueError("velocity_phase_flow_probe_count must be at least three")
        config.tests.velocity_trajectory_sweep_enabled = True
        config.tests.velocity_phase_flow_probe_enabled = True
        config.tests.velocity_phase_flow_initial_conditions = int(
            velocity_phase_flow_probe_count
        )
    (
        hd_tuning_history,
        bump_history,
        bump_attractor_trajectory_history,
        timescale_separation_history,
        velocity_trajectory_sweep_history,
        bump_diffusion_history,
        darkness_history,
        ou_darkness_history,
        ou_pi_ensemble_history,
        velocity_gain_history,
        test_metrics,
    ) = run_all_tests(
        config=config,
        trained_state=trained_state,
    )
    save_npz(run_dir / "bump_history.npz", **bump_history)
    save_npz(run_dir / "hd_tuning_history.npz", **hd_tuning_history)
    save_npz(
        run_dir / "bump_attractor_trajectory_history.npz",
        **bump_attractor_trajectory_history,
    )
    save_npz(
        run_dir / "timescale_separation_history.npz",
        **timescale_separation_history,
    )
    save_npz(
        run_dir / "velocity_trajectory_sweep_history.npz",
        **velocity_trajectory_sweep_history,
    )
    save_npz(run_dir / "bump_diffusion_history.npz", **bump_diffusion_history)
    save_npz(run_dir / "darkness_history.npz", **darkness_history)
    save_npz(run_dir / "ou_darkness_history.npz", **ou_darkness_history)
    save_npz(run_dir / "ou_pi_ensemble_history.npz", **ou_pi_ensemble_history)
    save_npz(run_dir / "velocity_gain_history.npz", **velocity_gain_history)
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_yaml(run_dir / "test_config_resolved.yaml", config.to_dict())
    if make_figures:
        make_vafidis_figures_for_run(run_dir=run_dir)


def run_tests_from_config(*, config_path: Path, make_figures: bool) -> Path:
    project_root = find_project_root(config_path)
    config = load_experiment_config(config_path)
    rng = make_rng(config.simulation.seed)
    trained_state = initialize_vafidis_toy_state(config=config, rng=rng)
    run_dir = create_run_dir(
        project_root=project_root,
        runs_root=config.paths.runs_root,
        experiment_name=f"{config.experiment_name}_untrained_test",
        seed=config.simulation.seed,
    )
    (
        hd_tuning_history,
        bump_history,
        bump_attractor_trajectory_history,
        timescale_separation_history,
        velocity_trajectory_sweep_history,
        bump_diffusion_history,
        darkness_history,
        ou_darkness_history,
        ou_pi_ensemble_history,
        velocity_gain_history,
        test_metrics,
    ) = run_all_tests(
        config=config,
        trained_state=trained_state,
    )
    save_run_outputs(
        run_dir=run_dir,
        config=config,
        params=VafidisToyParams.from_config(config),
        trained_state=trained_state,
        training_history={"time": [], "theta_true": [], "theta_hd_decoded": [], "angular_velocity": []},
        weight_history={},
        hd_tuning_history=hd_tuning_history,
        bump_history=bump_history,
        bump_attractor_trajectory_history=bump_attractor_trajectory_history,
        timescale_separation_history=timescale_separation_history,
        velocity_trajectory_sweep_history=velocity_trajectory_sweep_history,
        bump_diffusion_history=bump_diffusion_history,
        darkness_history=darkness_history,
        ou_darkness_history=ou_darkness_history,
        ou_pi_ensemble_history=ou_pi_ensemble_history,
        velocity_gain_history=velocity_gain_history,
        test_metrics=test_metrics,
    )
    if make_figures:
        make_vafidis_figures_for_run(run_dir=run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Existing run directory with trained_weights.npz.")
    parser.add_argument("--config", default=None, help="Config path for untrained frozen-weight tests.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument(
        "--bump-attractor-duration",
        type=float,
        default=None,
        help="Override zero-input darkness duration when retesting an existing run.",
    )
    parser.add_argument(
        "--enable-timescale-separation",
        action="store_true",
        help="Enable the Clark-style timescale assay when retesting an existing run.",
    )
    parser.add_argument(
        "--enable-velocity-trajectory-sweep",
        action="store_true",
        help="Enable the frozen-weight constant-velocity trajectory sweep.",
    )
    parser.add_argument(
        "--enable-velocity-phase-flow",
        action="store_true",
        help="Run dense rollouts for direct phase-flow, fixed-point, and basin analysis.",
    )
    parser.add_argument(
        "--velocity-sweep-values",
        type=lambda value: [float(item) for item in value.split(",")],
        default=None,
        help="Comma-separated constant velocities for an existing trained run.",
    )
    parser.add_argument(
        "--velocity-phase-flow-values",
        type=lambda value: [float(item) for item in value.split(",")],
        default=None,
        help="Comma-separated subset to receive dense phase-flow probes.",
    )
    parser.add_argument(
        "--velocity-phase-flow-probes",
        type=int,
        default=None,
        help="Number of uniformly spaced independent phase-flow probes.",
    )
    args = parser.parse_args()

    if args.run_dir is None and args.config is None:
        parser.error("Provide either --run-dir or --config")
    if args.config is not None and (
        args.bump_attractor_duration is not None
        or args.enable_timescale_separation
        or args.enable_velocity_trajectory_sweep
        or args.enable_velocity_phase_flow
        or args.velocity_sweep_values is not None
        or args.velocity_phase_flow_values is not None
        or args.velocity_phase_flow_probes is not None
    ):
        parser.error("test overrides are only valid together with --run-dir")
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        run_tests_for_existing_run(
            run_dir=run_dir,
            make_figures=not args.no_figures,
            bump_attractor_duration=args.bump_attractor_duration,
            enable_timescale_separation=args.enable_timescale_separation,
            enable_velocity_trajectory_sweep=(
                args.enable_velocity_trajectory_sweep
            ),
            enable_velocity_phase_flow=args.enable_velocity_phase_flow,
            velocity_sweep_values=args.velocity_sweep_values,
            velocity_phase_flow_values=args.velocity_phase_flow_values,
            velocity_phase_flow_probe_count=args.velocity_phase_flow_probes,
        )
        print(f"Updated tests in {run_dir}")
    else:
        config_path = resolve_config_path(args.config)
        run_dir = run_tests_from_config(config_path=config_path, make_figures=not args.no_figures)
        print(f"Saved untrained tests to {run_dir}")


if __name__ == "__main__":
    main()
