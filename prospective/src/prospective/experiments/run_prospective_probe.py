"""Frozen-weight M3 probes of adaptation-induced prospective coding."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from prospective.analysis.decoding import circular_center, learned_competitive_positions, linear_center, vector_strength
from prospective.analysis.weights import signed_offset
from prospective.common.random import make_rng
from prospective.config import load_config
from prospective.io.run_dir import write_json
from prospective.io.save_load import load_npz, save_npz
from prospective.models.feedforward_toy import initialize_state, step_feedforward, validate_state
from prospective.theory.equilibrium import equilibrium_widths
from prospective.theory.hermite import prospective_gamma


def _decode(values: np.ndarray, positions: np.ndarray, *, length: float, periodic: bool) -> float:
    return circular_center(values, positions, length) if periodic else linear_center(values, positions)


def run_single_probe(
    source_config,
    weights: np.ndarray,
    *,
    adaptation_strength: float,
    speed: float,
    duration: float,
    transient_duration: float,
    sample_interval_steps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one frozen-weight condition and return valid-window shift metrics."""

    config = replace(
        source_config,
        tutor=replace(
            source_config.tutor,
            speed=speed,
            initial_position=source_config.geometry.length / 2.0 if abs(speed) <= 1e-12 else source_config.tutor.initial_position,
        ),
        neural=replace(source_config.neural, adaptation_strength=adaptation_strength),
        feedforward_learning=replace(source_config.feedforward_learning, enabled=False),
        simulation=replace(source_config.simulation, duration=duration, progress=False),
    )
    state = initialize_state(config, make_rng(config.experiment.seed), fixed_weights=weights)
    learned_positions = learned_competitive_positions(weights, state.input_positions)
    steps = int(round(duration / config.neural.dt))
    history: dict[str, list[Any]] = {key: [] for key in [
        "time", "tutor_position", "decoded_rate", "decoded_u", "decoded_v", "shift", "vector_strength",
        "membrane", "adaptation", "rate", "tutor_rate",
    ]}
    periodic = config.geometry.boundary_mode == "periodic_ring"
    margin = config.analysis.boundary_margin_sigmas * config.tutor.sigma
    for _ in range(steps):
        state = step_feedforward(state, config, learning_enabled=False)
        if state.step % sample_interval_steps:
            continue
        validate_state(state, config)
        decoded_r = _decode(state.rate, learned_positions, length=config.geometry.length, periodic=periodic)
        decoded_u = _decode(np.maximum(state.membrane, 0.0), learned_positions, length=config.geometry.length, periodic=periodic)
        decoded_v = _decode(np.maximum(state.adaptation, 0.0), learned_positions, length=config.geometry.length, periodic=periodic)
        history["time"].append(state.time)
        history["tutor_position"].append(state.tutor_position)
        history["decoded_rate"].append(decoded_r)
        history["decoded_u"].append(decoded_u)
        history["decoded_v"].append(decoded_v)
        history["shift"].append(signed_offset(decoded_r, state.tutor_position, config.geometry.length, periodic))
        history["vector_strength"].append(vector_strength(state.rate, learned_positions, config.geometry.length))
        history["membrane"].append(state.membrane.copy())
        history["adaptation"].append(state.adaptation.copy())
        history["rate"].append(state.rate.copy())
        history["tutor_rate"].append(state.tutor_rate.copy())
    arrays = {key: np.asarray(value) for key, value in history.items()}
    valid = arrays["time"] >= transient_duration
    if not periodic:
        valid &= (arrays["tutor_position"] >= margin) & (arrays["tutor_position"] <= config.geometry.length - margin)
    valid &= arrays["vector_strength"] >= config.analysis.bump_strength_min
    shifts = arrays["shift"][valid]
    _, sigma_u = equilibrium_widths(config.feedforward_learning.beta, config.tutor.sigma)
    gamma_theory, lag_theory = prospective_gamma(
        speed=speed,
        adaptation_strength=adaptation_strength,
        tau_u=config.neural.tau_u,
        tau_v=config.neural.tau_v,
        sigma_u=sigma_u,
    )
    metrics = {
        "adaptation_strength": adaptation_strength,
        "speed": speed,
        "valid_samples": int(valid.sum()),
        "mean_shift": float(np.mean(shifts)) if shifts.size else float("nan"),
        "mean_shift_along_motion": float(np.sign(speed) * np.mean(shifts)) if shifts.size and abs(speed) > 1e-12 else float("nan"),
        "median_shift": float(np.median(shifts)) if shifts.size else float("nan"),
        "shift_std": float(np.std(shifts)) if shifts.size else float("nan"),
        "anticipation_time": float(np.mean(shifts) / speed) if shifts.size and abs(speed) > 1e-12 else float("nan"),
        "mean_vector_strength": float(np.mean(arrays["vector_strength"][valid])) if np.any(valid) else 0.0,
        "theory_gamma": gamma_theory,
        "theory_shift": sigma_u * gamma_theory,
        "theory_adaptation_lag": lag_theory,
    }
    arrays["valid"] = valid
    return metrics, arrays


def run_probe_grid(source_run: str | Path, probe_config_path: str | Path) -> Path:
    """Run the configured `(m, v)` grid and generate M3 figures/tables."""

    source_run = Path(source_run)
    source_config = load_config(source_run / "config_resolved.yaml")
    weights = load_npz(source_run / "final_weights.npz")["weights"]
    values = yaml.safe_load(Path(probe_config_path).read_text(encoding="utf-8")) or {}
    m_values = [float(value) for value in values["adaptation_strengths"]]
    velocities = [float(value) for value in values["velocities"]]
    duration = float(values.get("duration", 25.0))
    transient = float(values.get("transient_duration", 10.0))
    sample_interval = int(values.get("sample_interval_steps", 10))
    output_dir = source_run / "prospective_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    traces: dict[str, np.ndarray] | None = None
    for adaptation_strength in m_values:
        for speed in velocities:
            metrics, arrays = run_single_probe(
                source_config,
                weights,
                adaptation_strength=adaptation_strength,
                speed=speed,
                duration=duration,
                transient_duration=transient,
                sample_interval_steps=sample_interval,
            )
            rows.append(metrics)
            if traces is None or (adaptation_strength == source_config.neural.adaptation_strength and speed == source_config.tutor.speed):
                traces = arrays
    with (output_dir / "prospective_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "prospective_metrics.json", {"source_run": str(source_run), "conditions": rows})
    if traces is not None:
        save_npz(output_dir / "representative_trace.npz", **traces)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        axes[0].plot(traces["time"], traces["tutor_position"], label="tutor")
        axes[0].plot(traces["time"], traces["decoded_rate"], label="decoded r")
        axes[0].legend()
        axes[0].set_ylabel("position")
        axes[1].plot(traces["time"], traces["shift"])
        axes[1].scatter(traces["time"][~traces["valid"]], traces["shift"][~traces["valid"]], s=3, color="gray", label="masked")
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set(xlabel="time (s)", ylabel="decoded - tutor shift")
        axes[1].legend()
        fig.savefig(output_dir / "prospective_shift_vs_time.png", dpi=180)
        plt.close(fig)
    shift_grid = np.full((len(m_values), len(velocities)), np.nan)
    validity_grid = np.zeros_like(shift_grid)
    for row in rows:
        i = m_values.index(row["adaptation_strength"])
        j = velocities.index(row["speed"])
        shift_grid[i, j] = row["mean_shift"]
        validity_grid[i, j] = row["mean_vector_strength"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    image = axes[0].imshow(shift_grid, origin="lower", aspect="auto", extent=[min(velocities), max(velocities), min(m_values), max(m_values)], cmap="coolwarm")
    axes[0].set(xlabel="velocity", ylabel="adaptation m", title="Mean prospective shift")
    fig.colorbar(image, ax=axes[0])
    image = axes[1].imshow(validity_grid, origin="lower", aspect="auto", extent=[min(velocities), max(velocities), min(m_values), max(m_values)], vmin=0, vmax=1, cmap="viridis")
    axes[1].set(xlabel="velocity", ylabel="adaptation m", title="Bump vector strength")
    fig.colorbar(image, ax=axes[1])
    fig.savefig(output_dir / "prospective_shift_parameter_map.png", dpi=180)
    plt.close(fig)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/experiments/prospective_shift.yaml")
    args = parser.parse_args()
    print(run_probe_grid(args.run_dir, args.config))


if __name__ == "__main__":
    main()
