# python ./reproduction/src/plot_figure3_ghi.py
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from reproduction_config import load_figure3_config
from plot_figure3_abcd import (
    PROCESSED,
    REPRODUCTION_ROOT,
    build_figure3_matrices,
    figure3_matrix_cache_matches,
)
from plot_figure3_ef import load_sorted_target_manifold
from utils import (
    circulant_recurrent_drive,
    dense_recurrent_drive,
    lowrank_recurrent_drive,
    optimized_recurrent_factors,
    relax_rate_network_states,
    softplus,
)


DEFAULT_MATRIX_PATH = PROCESSED / "figure3_abcd_weight_matrices.npz"
DEFAULT_DYNAMICS_PATH = PROCESSED / "figure3_ghi_activity.npz"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
FIGURE3_CONFIG = load_figure3_config()
NETWORK_CONFIG = FIGURE3_CONFIG["network"]
GHI_CONFIG = FIGURE3_CONFIG["panels_ghi"]


def make_regularly_spaced_initial_states(target_current, n_initial, noise_std, seed):
    """
    Sample regularly spaced target-manifold points and perturb their currents.

    The returned states follow the COM neuron order used by Figure 3B-D. The
    same perturbed initial conditions are supplied to all three weight
    matrices, making differences in final activity attributable to connectivity.
    """
    angle_idx = np.linspace(0, len(target_current), int(n_initial), endpoint=False, dtype=int)
    rng = np.random.default_rng(seed)
    initial_states = target_current[angle_idx].astype(np.float32, copy=True)
    initial_states += rng.normal(0.0, float(noise_std), size=initial_states.shape).astype(np.float32)
    return angle_idx, initial_states


def overlap_with_target(final_rates, target_rate):
    """
    Project final population activity onto the target firing-rate manifold.

    This is the overlap order parameter m(theta) evaluated only at the final
    simulation time. It is saved as a compact diagnostic for identifying
    whether many initial conditions collapse onto the same decoded phase.
    """
    final_rates = np.asarray(final_rates, dtype=float)
    target_rate = np.asarray(target_rate, dtype=float)
    return np.einsum("an,tn->at", target_rate, final_rates, optimize=False) / final_rates.shape[1]


def cache_matches(
    path,
    n_initial,
    duration_s,
    tau_s,
    dt_s,
    noise_std,
    inhibition_c,
    seed,
):
    """
    Check whether a saved Figure 3G-I activity checkpoint matches this run.
    """
    path = Path(path)
    if not path.exists():
        return False
    data = np.load(path)
    required = {
        "optimized_final_rates",
        "circulant_final_rates",
        "noisy_circulant_final_rates",
        "decoded_angle_idx",
    }
    return required.issubset(data.files) and (
        int(data.get("n_initial", -1)) == int(n_initial)
        and float(data.get("duration_s", np.nan)) == float(duration_s)
        and float(data.get("tau_s", np.nan)) == float(tau_s)
        and float(data.get("dt_s", np.nan)) == float(dt_s)
        and float(data.get("noise_std", np.nan)) == float(noise_std)
        and float(data.get("inhibition_c", np.nan)) == float(inhibition_c)
        and int(data.get("seed", -1)) == int(seed)
    )


def compute_figure3_ghi_activity(
    matrix_path=DEFAULT_MATRIX_PATH,
    out_path=DEFAULT_DYNAMICS_PATH,
    n_initial=int(GHI_CONFIG["n_initial"]),
    duration_s=float(GHI_CONFIG["duration_s"]),
    tau_s=float(NETWORK_CONFIG["tau_s"]),
    dt_s=float(NETWORK_CONFIG["dt_s"]),
    noise_std=float(GHI_CONFIG["noise_std"]),
    inhibition_c=float(NETWORK_CONFIG["inhibition_c"]),
    seed=int(GHI_CONFIG["seed"]),
    force=False,
):
    """
    Evolve matched initial states under the three Figure 3 weight matrices.

    The optimized matrix is applied through its low-rank factors, the
    circulant matrix through FFT convolution, and the shuffled-residual matrix
    through dense matrix multiplication. Only final states are retained,
    because panels G-I show steady-state activity after 30 seconds rather than
    full trajectories.
    """
    out_path = Path(out_path)
    if not force and cache_matches(
        out_path,
        n_initial=n_initial,
        duration_s=duration_s,
        tau_s=tau_s,
        dt_s=dt_s,
        noise_std=noise_std,
        inhibition_c=inhibition_c,
        seed=seed,
    ):
        print(f"Using cached Figure 3G-I activity: {out_path}", flush=True)
        return out_path

    matrix_path = Path(matrix_path)
    if not figure3_matrix_cache_matches(matrix_path):
        matrix_path = build_figure3_matrices(out_path=matrix_path)

    target_current, target_rate, angles_rad, regularization, activation_beta, alpha_floor = (
        load_sorted_target_manifold(matrix_path)
    )
    angle_idx, initial_states = make_regularly_spaced_initial_states(
        target_current,
        n_initial=n_initial,
        noise_std=noise_std,
        seed=seed,
    )

    matrix_data = np.load(matrix_path)
    factor_a, factor_b, factor_diagonal = optimized_recurrent_factors(
        target_rate.T,
        regularization=regularization,
        activation_beta=activation_beta,
        enforce_zero_diagonal=True,
    )
    drives = {
        "optimized": lowrank_recurrent_drive(factor_a, factor_b, factor_diagonal),
        "circulant": circulant_recurrent_drive(matrix_data["weights_circulant"][:, 0]),
        "noisy_circulant": dense_recurrent_drive(matrix_data["weights_noisy_circulant"]),
    }

    final_states = {}
    final_rates = {}
    started = time.perf_counter()
    for name, drive in drives.items():
        print(f"Relaxing {name} network for {duration_s:g}s...", flush=True)
        final_states[name] = relax_rate_network_states(
            drive,
            initial_states,
            tau_s=tau_s,
            dt_s=dt_s,
            duration_s=duration_s,
            inhibition_c=inhibition_c,
            activation_beta=activation_beta,
            current_clip=None,
            progress_label=f"Figure 3G-I {name}",
            progress_interval_wall_s=10.0,
        )
        if not np.isfinite(final_states[name]).all():
            raise FloatingPointError(f"{name} dynamics produced non-finite final states")
        final_rates[name] = softplus(final_states[name], beta=activation_beta).astype(np.float32)

    overlaps = {
        name: overlap_with_target(rates, target_rate).astype(np.float32)
        for name, rates in final_rates.items()
    }
    decoded_angle_idx = np.vstack(
        [np.argmax(overlaps[name], axis=0) for name in ("optimized", "circulant", "noisy_circulant")]
    )

    np.savez_compressed(
        out_path,
        angle_idx=angle_idx,
        initial_angles_rad=angles_rad[angle_idx],
        bin_centers_rad=angles_rad,
        initial_states=initial_states,
        target_rate=target_rate.astype(np.float32),
        optimized_final_states=final_states["optimized"],
        circulant_final_states=final_states["circulant"],
        noisy_circulant_final_states=final_states["noisy_circulant"],
        optimized_final_rates=final_rates["optimized"],
        circulant_final_rates=final_rates["circulant"],
        noisy_circulant_final_rates=final_rates["noisy_circulant"],
        optimized_overlap=overlaps["optimized"],
        circulant_overlap=overlaps["circulant"],
        noisy_circulant_overlap=overlaps["noisy_circulant"],
        decoded_angle_idx=decoded_angle_idx,
        network_names=np.asarray(["optimized", "circulant", "noisy_circulant"]),
        n_initial=int(n_initial),
        duration_s=float(duration_s),
        tau_s=float(tau_s),
        dt_s=float(dt_s),
        noise_std=float(noise_std),
        inhibition_c=float(inhibition_c),
        regularization=float(regularization),
        activation_beta=float(activation_beta),
        alpha_floor=float(alpha_floor),
        circulant_gain=float(matrix_data["circulant_gain"]),
        figure3_config_schema_version=int(FIGURE3_CONFIG["schema_version"]),
        seed=int(seed),
    )
    print(f"Saved Figure 3G-I activity: {out_path}", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    return out_path


def activity_axis_upper_limit(rates, headroom=0.08):
    """
    Return a rounded upper limit above the true maximum firing rate.

    Each activity panel receives its own scale. Using the actual maximum rather
    than a percentile prevents rare high-rate neurons from being clipped, while
    the rounded headroom keeps the upper envelope away from the panel border.
    """
    rates = np.asarray(rates, dtype=float)
    finite = rates[np.isfinite(rates)]
    if finite.size == 0:
        return 1.0

    target = max(float(np.max(finite)) * (1.0 + float(headroom)), 1.0)
    magnitude = 10.0 ** np.floor(np.log10(target))
    normalized = target / magnitude
    for step in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if normalized <= step:
            return float(step * magnitude)
    return float(10.0 * magnitude)


def plot_figure3_ghi(
    dynamics_path=DEFAULT_DYNAMICS_PATH,
    out_path=FIGURES / "figure3_ghi_reproduction.png",
):
    """
    Draw steady-state cluster activity for optimized and control weights.
    """
    dynamics_path = Path(dynamics_path)
    if not dynamics_path.exists():
        dynamics_path = compute_figure3_ghi_activity(out_path=dynamics_path)

    data = np.load(dynamics_path)
    rate_keys = [
        "optimized_final_rates",
        "circulant_final_rates",
        "noisy_circulant_final_rates",
    ]
    rates_by_panel = [data[key].astype(float) for key in rate_keys]
    initial_angles = np.mod(data["initial_angles_rad"].astype(float), 2.0 * np.pi)
    colors = plt.cm.hsv(initial_angles / (2.0 * np.pi))
    y_limits = [activity_axis_upper_limit(rates) for rates in rates_by_panel]

    panel_titles = [
        "G  optimized data-derived weights",
        "H  diagonal-averaged circulant weights",
        "I  circulant plus shuffled residuals",
    ]
    panel_notes = [
        "heterogeneous stable bumps",
        "translation-equivalent bumps",
        "collapse toward discrete stable states",
    ]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 9),
        gridspec_kw={"width_ratios": [4.5, 1.2]},
        constrained_layout=True,
    )
    neuron_index = np.arange(1, rates_by_panel[0].shape[1] + 1)
    phase_ticks = [0.0, np.pi, 2.0 * np.pi]
    phase_ticklabels = ["0", "π", "2π"]

    for panel_idx, (rates, title, note, y_max) in enumerate(
        zip(rates_by_panel, panel_titles, panel_notes, y_limits)
    ):
        activity_ax = axes[panel_idx, 0]
        phase_ax = axes[panel_idx, 1]
        decoded = data["decoded_angle_idx"][panel_idx]
        n_unique = len(np.unique(decoded))
        for trial, color in enumerate(colors):
            activity_ax.plot(neuron_index, rates[trial], color=color, linewidth=0.5)
        activity_ax.set_xlim(1, rates.shape[1])
        activity_ax.set_ylim(0, y_max)
        activity_ax.set_xlabel("Neurons (COM order)")
        activity_ax.set_ylabel("Firing rate")
        activity_ax.set_title(
            f"{title}\n{note}; {n_unique}/{len(decoded)} unique decoded final bins"
        )
        activity_ax.set_xticks([1, rates.shape[1]], ["1", "N"])

        decoded_angles = np.mod(data["bin_centers_rad"][decoded], 2.0 * np.pi)
        phase_ax.plot([0, 2.0 * np.pi], [0, 2.0 * np.pi], linestyle="--", color="0.7")
        phase_ax.scatter(
            initial_angles,
            decoded_angles,
            c=initial_angles,
            cmap="hsv",
            norm=Normalize(0.0, 2.0 * np.pi),
            edgecolors="black",
            linewidths=0.4,
        )
        phase_ax.set_xlim(0, 2.0 * np.pi)
        phase_ax.set_ylim(0, 2.0 * np.pi)
        phase_ax.set_xticks(phase_ticks, phase_ticklabels)
        phase_ax.set_yticks(phase_ticks, phase_ticklabels)
        phase_ax.set_xlabel("Initial phase")
        phase_ax.set_ylabel("Final decoded phase")
        phase_ax.set_title("Initial → final phase")

    scalar_mappable = plt.cm.ScalarMappable(
        norm=Normalize(0.0, 2.0 * np.pi),
        cmap="hsv",
    )
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.04,
    )
    colorbar.set_ticks(phase_ticks)
    colorbar.set_ticklabels(phase_ticklabels)
    colorbar.set_label("Angular location of initial condition")

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    """
    Command-line entry point for Figure 3G-I reproduction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute the 30 s dynamics")
    parser.add_argument("--duration-s", type=float, default=float(GHI_CONFIG["duration_s"]))
    parser.add_argument("--dt-s", type=float, default=float(NETWORK_CONFIG["dt_s"]))
    args = parser.parse_args()

    dynamics_path = compute_figure3_ghi_activity(
        duration_s=args.duration_s,
        dt_s=args.dt_s,
        force=args.force,
    )
    figure_path = plot_figure3_ghi(dynamics_path)
    data = np.load(dynamics_path)
    print("Saved:", figure_path)
    for panel_idx, name in enumerate(data["network_names"]):
        decoded = data["decoded_angle_idx"][panel_idx]
        print(f"{name}: {len(np.unique(decoded))}/{len(decoded)} unique decoded final bins")


if __name__ == "__main__":
    main()
