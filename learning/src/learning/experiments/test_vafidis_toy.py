"""Run frozen-weight tests for a Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path

from learning.analysis.make_vafidis_figures import make_vafidis_figures_for_run
from learning.common.random import make_rng
from learning.config.load_config import find_project_root, load_experiment_config
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
    config = load_experiment_config(run_dir / "config_resolved.yaml")
    trained_weights = load_npz(run_dir / "trained_weights.npz")
    rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(config=config, rng=rng)
    state.w_hd_to_hd = trained_weights["w_hd_to_hd"].copy()
    state.w_hr_to_hd = trained_weights["w_hr_to_hd"].copy()
    state.w_hd_to_hr = trained_weights["w_hd_to_hr"].copy()
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    return config, state


def run_tests_for_existing_run(*, run_dir: Path, make_figures: bool) -> None:
    config, trained_state = load_trained_state_from_run(run_dir=run_dir)
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
    save_npz(run_dir / "bump_history.npz", **bump_history)
    save_npz(run_dir / "darkness_history.npz", **darkness_history)
    save_npz(run_dir / "ou_darkness_history.npz", **ou_darkness_history)
    save_npz(run_dir / "velocity_gain_history.npz", **velocity_gain_history)
    save_json(run_dir / "test_metrics.json", test_metrics)
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
        params=VafidisToyParams.from_config(config),
        trained_state=trained_state,
        training_history={"time": [], "theta_true": [], "theta_hd_decoded": [], "angular_velocity": []},
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
    parser.add_argument("--run-dir", default=None, help="Existing run directory with trained_weights.npz.")
    parser.add_argument("--config", default=None, help="Config path for untrained frozen-weight tests.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    if args.run_dir is None and args.config is None:
        parser.error("Provide either --run-dir or --config")
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        run_tests_for_existing_run(run_dir=run_dir, make_figures=not args.no_figures)
        print(f"Updated tests in {run_dir}")
    else:
        config_path = resolve_config_path(args.config)
        run_dir = run_tests_from_config(config_path=config_path, make_figures=not args.no_figures)
        print(f"Saved untrained tests to {run_dir}")


if __name__ == "__main__":
    main()
