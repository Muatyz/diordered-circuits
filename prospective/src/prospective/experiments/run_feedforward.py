"""Train or probe the M0--M2 feedforward Emina-Kropff toy model."""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from prospective.analysis.decoding import learned_competitive_positions
from prospective.analysis.metrics import summarize_training
from prospective.analysis.weights import diagnose_weights
from prospective.common.random import make_rng
from prospective.config import ExperimentConfig, load_config
from prospective.io.run_dir import create_run_dir, mark_run, write_json
from prospective.io.save_load import save_npz
from prospective.models.feedforward_toy import FeedforwardState, initialize_state, step_feedforward, validate_state


@dataclass
class TrainingResult:
    """In-memory output returned to tests, sweeps, and the CLI."""

    run_dir: Path
    state: FeedforwardState
    history: dict[str, np.ndarray]
    weight_history: dict[str, np.ndarray]
    metrics: dict[str, Any]


def _append_state(history: dict[str, list[Any]], state: FeedforwardState, *, store_weights: bool) -> None:
    history["step"].append(state.step)
    history["time"].append(state.time)
    history["tutor_position"].append(state.tutor_position)
    history["tutor_rate"].append(state.tutor_rate.copy())
    history["membrane"].append(state.membrane.copy())
    history["adaptation"].append(state.adaptation.copy())
    history["rate"].append(state.rate.copy())
    history["delta_weight_norm"].append(float(np.linalg.norm(state.last_delta_weights)))
    history["clipped_count"].append(int(np.count_nonzero(state.last_clipped)))
    if store_weights:
        history["weights"].append(state.weights.copy())
        history["delta_weights"].append(state.last_delta_weights.copy())
        history["clipped"].append(state.last_clipped.copy())


def _as_arrays(history: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(values) for key, values in history.items()}


def train_feedforward(
    config: ExperimentConfig,
    *,
    output_cwd: str | Path | None = None,
    weight_mode: str = "random",
    learning_enabled: bool | None = None,
    make_figures: bool = True,
) -> TrainingResult:
    """Run a complete M1 oracle or M2 local-learning experiment.

    M1 uses `weight_mode='theory', learning_enabled=False`. M2 uses the default
    random weights and the configured local rule. No analysis modifies state.
    """

    run_dir = create_run_dir(config, base_cwd=output_cwd)
    rng = make_rng(config.experiment.seed)
    state = initialize_state(config, rng, weight_mode=weight_mode)
    initial_weights = state.weights.copy()
    state_history: dict[str, list[Any]] = {
        "step": [], "time": [], "tutor_position": [], "tutor_rate": [],
        "membrane": [], "adaptation": [], "rate": [], "delta_weight_norm": [],
        "clipped_count": [],
    }
    if config.animation.enabled:
        state_history["weights"] = []
        state_history["delta_weights"] = []
        state_history["clipped"] = []
    # Preserve the true initial condition for full-training animations and
    # diagnostics. Previously, the first saved state was one sampling interval
    # after t=0 even though weight_history already began at t=0.
    _append_state(state_history, state, store_weights=config.animation.enabled)
    weight_steps: list[int] = [0]
    weight_times: list[float] = [0.0]
    weight_snapshots: list[np.ndarray] = [initial_weights.copy()]
    weight_correlations: list[float] = [diagnose_weights(
        initial_weights,
        beta=config.feedforward_learning.beta,
        sigma_r=config.tutor.sigma,
        length=config.geometry.length,
        periodic=config.geometry.boundary_mode == "periodic_ring",
    )[0].gaussian_correlation]
    total_steps = int(round(config.simulation.duration / config.neural.dt))
    iterator = trange(total_steps, disable=not config.simulation.progress, desc=config.experiment.name)
    try:
        for _ in iterator:
            state = step_feedforward(state, config, learning_enabled=learning_enabled)
            if state.step % config.simulation.state_sample_interval_steps == 0:
                validate_state(state, config)
                _append_state(state_history, state, store_weights=config.animation.enabled)
            if state.step % config.simulation.weight_snapshot_interval_steps == 0:
                weight_steps.append(state.step)
                weight_times.append(state.time)
                weight_snapshots.append(state.weights.copy())
                diag, _ = diagnose_weights(
                    state.weights,
                    beta=config.feedforward_learning.beta,
                    sigma_r=config.tutor.sigma,
                    length=config.geometry.length,
                    periodic=config.geometry.boundary_mode == "periodic_ring",
                )
                weight_correlations.append(diag.gaussian_correlation)
        validate_state(state, config)
        if not state_history["step"] or state_history["step"][-1] != state.step:
            _append_state(state_history, state, store_weights=config.animation.enabled)
        if weight_steps[-1] != state.step:
            weight_steps.append(state.step)
            weight_times.append(state.time)
            weight_snapshots.append(state.weights.copy())
            weight_correlations.append(diagnose_weights(
                state.weights,
                beta=config.feedforward_learning.beta,
                sigma_r=config.tutor.sigma,
                length=config.geometry.length,
                periodic=config.geometry.boundary_mode == "periodic_ring",
            )[0].gaussian_correlation)
        history = _as_arrays(state_history)
        weight_history = {
            "step": np.asarray(weight_steps),
            "time": np.asarray(weight_times),
            "weights": np.asarray(weight_snapshots),
            "gaussian_correlation": np.asarray(weight_correlations),
        }
        learned_positions = learned_competitive_positions(state.weights, state.input_positions)
        metrics = summarize_training(
            weights=state.weights,
            rate=state.rate,
            learned_positions=learned_positions,
            update_norm=float(np.linalg.norm(state.last_delta_weights)),
            config=config,
        )
        metrics.update({
            "weight_mode": weight_mode,
            "learning_enabled": config.feedforward_learning.enabled if learning_enabled is None else learning_enabled,
            "completed_steps": state.step,
            "simulated_duration": state.time,
        })
        save_npz(run_dir / "training_history.npz", **history)
        save_npz(run_dir / "weight_history.npz", **weight_history)
        save_npz(
            run_dir / "final_state.npz",
            time=np.asarray(state.time),
            step=np.asarray(state.step),
            input_positions=state.input_positions,
            competitive_positions=state.competitive_positions,
            learned_positions=learned_positions,
            tutor_position=np.asarray(state.tutor_position),
            tutor_rate=state.tutor_rate,
            membrane=state.membrane,
            adaptation=state.adaptation,
            rate=state.rate,
        )
        save_npz(run_dir / "final_weights.npz", initial_weights=initial_weights, weights=state.weights)
        write_json(run_dir / "metrics.json", metrics)
        if make_figures:
            from prospective.plotting.summary import make_training_figures

            make_training_figures(run_dir, config, history, weight_history, initial_weights, state)
        mark_run(run_dir, "completed", completed_steps=state.step, simulated_duration=state.time)
        return TrainingResult(run_dir, state, history, weight_history, metrics)
    except Exception as exc:
        mark_run(run_dir, "failed", error=str(exc), traceback=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--theory-weights", action="store_true", help="use analytic Gaussian J and freeze learning (M1)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = train_feedforward(
        config,
        weight_mode="theory" if args.theory_weights else "random",
        learning_enabled=False if args.theory_weights else None,
        make_figures=not args.no_figures,
    )
    print(result.run_dir)


if __name__ == "__main__":
    main()
