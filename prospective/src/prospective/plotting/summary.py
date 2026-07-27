"""M1--M2 training figures defined by scientific object rather than paper panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prospective.analysis.decoding import learned_competitive_positions
from prospective.analysis.weights import diagnose_weights
from prospective.config.schema import ExperimentConfig
from prospective.models.feedforward_toy import FeedforwardState
from prospective.theory.equilibrium import equilibrium_weight_amplitude


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_training_figures(
    run_dir: str | Path,
    config: ExperimentConfig,
    history: dict[str, np.ndarray],
    weight_history: dict[str, np.ndarray],
    initial_weights: np.ndarray,
    state: FeedforwardState,
) -> None:
    """Generate the required M1--M2 activity, connectivity, and learning plots."""

    run_dir = Path(run_dir)
    activity_dir = run_dir / "figures" / "activity"
    connectivity_dir = run_dir / "figures" / "connectivity"
    learning_dir = run_dir / "figures" / "learning"
    time = history["time"]
    extent_input = [0.0, config.geometry.length, time[-1], time[0]]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    entries = [
        (history["tutor_rate"], "Tutor rate R"),
        (history["membrane"], "Membrane U"),
        (history["adaptation"], "Adaptation V"),
        (history["rate"], "Competitive rate r"),
    ]
    for ax, (values, title) in zip(axes, entries):
        image = ax.imshow(values, aspect="auto", extent=extent_input, cmap="viridis")
        ax.plot(history["tutor_position"], time, color="white", lw=0.7, alpha=0.8)
        ax.set_ylabel("time (s)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.025)
    axes[-1].set_xlabel("position / neuron coordinate")
    _save(fig, activity_dir / "tutor_and_activity_heatmaps.png")

    sample_indices = np.unique(np.linspace(0, len(time) - 1, 4, dtype=int))
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 2.4 * len(sample_indices)), sharex=True)
    axes = np.atleast_1d(axes)
    learned_positions = learned_competitive_positions(state.weights, state.input_positions)
    order = np.argsort(learned_positions)
    for ax, idx in zip(axes, sample_indices):
        ax.plot(state.input_positions, history["tutor_rate"][idx], label="R", lw=2)
        ax.plot(learned_positions[order], history["membrane"][idx][order], label="U")
        ax.plot(learned_positions[order], history["adaptation"][idx][order], label="V")
        ax.plot(learned_positions[order], history["rate"][idx][order], label="r")
        ax.axvline(history["tutor_position"][idx], color="black", ls="--", lw=0.8)
        ax.set_title(f"t = {time[idx]:.2f} s")
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("learned preferred position")
    _save(fig, activity_dir / "activity_profile_snapshots.png")

    snapshots = weight_history["weights"]
    indices = [0, len(snapshots) // 2, len(snapshots) - 1]
    final_peaks = np.argmax(state.weights, axis=1)
    row_order = np.argsort(final_peaks)
    vmax = float(np.max(snapshots[indices]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, idx, title in zip(axes, indices, ["initial", "middle", "final"]):
        image = ax.imshow(snapshots[idx][row_order], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{title}: t={weight_history['time'][idx]:.1f}s")
        ax.set_xlabel("input neuron (pre)")
        ax.set_ylabel("competitive neuron, final-order (post)")
    fig.colorbar(image, ax=axes, fraction=0.025, label="J[post, pre]")
    _save(fig, connectivity_dir / "feedforward_weights_initial_mid_final.png")

    diagnostics, details = diagnose_weights(
        state.weights,
        beta=config.feedforward_learning.beta,
        sigma_r=config.tutor.sigma,
        length=config.geometry.length,
        periodic=config.geometry.boundary_mode == "periodic_ring",
    )
    displacement = details["relative_displacement"]
    aligned = details["aligned_rows"]
    theory_amplitude = equilibrium_weight_amplitude(
        config.feedforward_learning.beta,
        diagnostics.theoretical_width,
        config.tutor.integrated_drive,
        config.feedforward_learning.alpha,
    )
    theory = theory_amplitude * details["theory_unit_profile"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(displacement, np.percentile(aligned, 25, axis=0), np.percentile(aligned, 75, axis=0), alpha=0.25, label="row IQR")
    ax.plot(displacement, details["mean_profile"], lw=2, label="learned aligned mean")
    ax.plot(displacement, theory, "k--", lw=1.5, label=f"theory sigma_J={diagnostics.theoretical_width:.2f}")
    ax.set(xlabel="relative displacement", ylabel="weight", title="Learned feedforward profile versus Eq. 13")
    ax.legend()
    _save(fig, connectivity_dir / "aligned_weight_profile_vs_theory.png")

    correlations = []
    widths = []
    translation_errors = []
    norms = []
    for weights in snapshots:
        diag, _ = diagnose_weights(
            weights,
            beta=config.feedforward_learning.beta,
            sigma_r=config.tutor.sigma,
            length=config.geometry.length,
            periodic=config.geometry.boundary_mode == "periodic_ring",
        )
        correlations.append(diag.gaussian_correlation)
        widths.append(diag.learned_width)
        translation_errors.append(diag.translation_invariance_error)
        norms.append(np.linalg.norm(weights))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0, 0].plot(weight_history["time"], correlations)
    axes[0, 0].set_ylabel("Gaussian correlation")
    axes[0, 1].plot(weight_history["time"], widths, label="learned")
    axes[0, 1].axhline(diagnostics.theoretical_width, color="black", ls="--", label="theory")
    axes[0, 1].set_ylabel("profile width")
    axes[0, 1].legend()
    axes[1, 0].plot(weight_history["time"], translation_errors)
    axes[1, 0].set_ylabel("translation invariance error")
    axes[1, 1].plot(weight_history["time"], norms, label="weight norm")
    axes[1, 1].set_ylabel("weight norm")
    for ax in axes[1]:
        ax.set_xlabel("training time (s)")
    _save(fig, learning_dir / "learning_diagnostics.png")
