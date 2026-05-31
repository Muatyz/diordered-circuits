import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_figure3_abcd import WEIGHT_FORMULA_VERSION, build_figure3_matrices, figure3_matrix_cache_matches
from utils import (
    lowrank_recurrent_drive,
    nearest_manifold_distance,
    optimized_recurrent_factors,
    pca_basis,
    project_onto_basis,
    simulate_rate_network_with_drive,
    softplus_inverse,
)


PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def elapsed_text(seconds):
    """
    Format a wall-clock duration for terminal progress messages.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def load_sorted_target_manifold(matrix_path, activation_beta=2.0, firing_rate_floor=1e-4):
    """
    Load Figure 3A-D matrices and return the COM-ordered target manifold.

    The optimized weight matrix in panel B is sorted by each neuron's circular
    center of mass, so the target firing-rate manifold must be sorted in the
    same order before it is used as a fixed-point target for panels E and F.
    """
    if not figure3_matrix_cache_matches(
        matrix_path,
        activation_beta=activation_beta,
        firing_rate_floor=firing_rate_floor,
    ):
        print(f"Missing {matrix_path}; building Figure 3A-D matrices first.", flush=True)
        matrix_path = build_figure3_matrices(
            out_path=matrix_path,
            activation_beta=activation_beta,
            firing_rate_floor=firing_rate_floor,
        )

    data = np.load(matrix_path)
    sorted_order = data["sorted_order"]
    target_rate = data["doubly_normalized_tuning"][sorted_order].T.astype(np.float32)
    activation_beta = float(data["activation_beta"])
    target_current = softplus_inverse(target_rate, beta=activation_beta).astype(np.float32)
    angles_rad = data["bin_centers_rad"].astype(float)
    regularization = float(data["regularization"])
    firing_rate_floor = float(data["firing_rate_floor"])
    return target_current, target_rate, angles_rad, regularization, activation_beta, firing_rate_floor


def make_initial_states(target_current, angle_idx, noise_std=0.05, seed=20260531):
    """
    Create initial conditions near selected points on the target manifold.

    Each trial starts at `x*(theta)` plus small independent Gaussian current
    noise. These are simulated initial conditions, not recorded neural states.
    """
    rng = np.random.default_rng(seed)
    initial = target_current[angle_idx].copy()
    initial += rng.normal(0.0, noise_std, size=initial.shape).astype(np.float32)
    return initial


def compute_figure3_ef_dynamics(
    matrix_path=PROCESSED / "figure3_abcd_weight_matrices.npz",
    out_path=PROCESSED / "figure3_ef_dynamics.npz",
    n_initial=24,
    duration_s=10.0,
    tau_s=0.05,
    dt_s=0.001,
    record_every_s=0.5,
    noise_std=0.05,
    inhibition_c=1.0,
    activation_beta=2.0,
    firing_rate_floor=1e-4,
    seed=20260531,
    force=False,
):
    """
    Simulate optimized-network convergence to the target attractor manifold.

    The saved arrays support panel E's PCA visualization and panel F's nearest
    distance-to-manifold traces. Existing results are reused unless `force` is
    true because this simulation can take noticeably longer than plotting.
    """
    if out_path.exists() and not force:
        cached = np.load(out_path)
        cache_matches = (
            float(cached.get("duration_s", np.nan)) == float(duration_s)
            and float(cached.get("tau_s", np.nan)) == float(tau_s)
            and float(cached.get("dt_s", np.nan)) == float(dt_s)
            and float(cached.get("record_every_s", np.nan)) == float(record_every_s)
            and float(cached.get("noise_std", np.nan)) == float(noise_std)
            and float(cached.get("inhibition_c", np.nan)) == float(inhibition_c)
            and float(cached.get("activation_beta", np.nan)) == float(activation_beta)
            and float(cached.get("firing_rate_floor", np.nan)) == float(firing_rate_floor)
            and str(cached.get("distance_space", "")) == "input_current_full_state"
            and str(cached.get("distance_normalization", "")) == "l2_div_sqrt_n_neurons"
            and str(cached.get("weight_formula_version", "")) == WEIGHT_FORMULA_VERSION
            and "optimized_l2_distance_to_manifold" in cached.files
            and int(cached.get("seed", -1)) == int(seed)
            and len(cached.get("angle_idx", [])) == int(n_initial)
        )
        if cache_matches:
            print(f"Using cached dynamics: {out_path}", flush=True)
            return out_path
        print(f"Cached dynamics parameters differ; recomputing {out_path}.", flush=True)

    start = time.perf_counter()
    print("Loading Figure 3 target manifold...", flush=True)
    target_current, target_rate, angles_rad, regularization, activation_beta, firing_rate_floor = load_sorted_target_manifold(
        matrix_path,
        activation_beta=activation_beta,
        firing_rate_floor=firing_rate_floor,
    )
    angle_idx = np.linspace(0, len(angles_rad), n_initial, endpoint=False, dtype=int)
    initial_states = make_initial_states(target_current, angle_idx, noise_std=noise_std, seed=seed)

    print("Building low-rank optimized recurrent drive...", flush=True)
    factors_start = time.perf_counter()
    factor_a, factor_b, diagonal = optimized_recurrent_factors(
        target_rate.T,
        regularization=regularization,
        activation_beta=activation_beta,
        enforce_zero_diagonal=True,
    )
    recurrent_drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    print(f"Built recurrent drive in {elapsed_text(time.perf_counter() - factors_start)}.", flush=True)

    print(
        "Simulating optimized recurrent dynamics "
        f"({n_initial} initial states, {duration_s:g}s biological time, tau={tau_s:g}s, dt={dt_s:g}s)...",
        flush=True,
    )
    times, trajectory = simulate_rate_network_with_drive(
        recurrent_drive,
        initial_states,
        tau_s=tau_s,
        dt_s=dt_s,
        duration_s=duration_s,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        record_every_s=record_every_s,
        progress_label="Figure 3E/F dynamics",
        progress_interval_wall_s=10.0,
    )

    print("Projecting target manifold and trajectories onto three PCs...", flush=True)
    pca_mean, basis = pca_basis(target_current, n_components=3)
    manifold_pc = project_onto_basis(target_current, pca_mean, basis).astype(np.float32)
    trajectory_pc = np.empty((*trajectory.shape[:2], 3), dtype=np.float32)
    distance = np.empty(trajectory.shape[:2], dtype=np.float32)
    l2_distance = np.empty(trajectory.shape[:2], dtype=np.float32)
    nearest_idx = np.empty(trajectory.shape[:2], dtype=np.int64)

    for t_idx in range(len(times)):
        trajectory_pc[t_idx] = project_onto_basis(trajectory[t_idx], pca_mean, basis)
        distance[t_idx], nearest_idx[t_idx], l2_distance[t_idx] = nearest_manifold_distance(
            trajectory[t_idx],
            target_current,
            return_l2=True,
        )

    np.savez_compressed(
        out_path,
        times=times,
        angle_idx=angle_idx,
        bin_centers_rad=angles_rad,
        target_current=target_current,
        target_rate=target_rate,
        initial_states=initial_states,
        optimized_trajectory=trajectory,
        pca_mean=pca_mean,
        pca_basis=basis,
        manifold_pc=manifold_pc,
        optimized_pc=trajectory_pc,
        optimized_distance_to_manifold=distance,
        optimized_l2_distance_to_manifold=l2_distance,
        optimized_nearest_angle_idx=nearest_idx,
        distance_space="input_current_full_state",
        distance_normalization="l2_div_sqrt_n_neurons",
        weight_formula_version=WEIGHT_FORMULA_VERSION,
        initial_condition_source="simulated_target_manifold_points_plus_iid_gaussian_current_noise",
        duration_s=duration_s,
        tau_s=tau_s,
        dt_s=dt_s,
        record_every_s=record_every_s,
        noise_std=noise_std,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        firing_rate_floor=firing_rate_floor,
        regularization=regularization,
        seed=seed,
    )
    print(f"Saved dynamics: {out_path} ({elapsed_text(time.perf_counter() - start)})", flush=True)
    return out_path


def plot_figure3_ef(dynamics_path=PROCESSED / "figure3_ef_dynamics.npz"):
    """
    Plot Figure 3E-F from the cached or newly computed dynamics.
    """
    if not dynamics_path.exists():
        dynamics_path = compute_figure3_ef_dynamics(out_path=dynamics_path)

    data = np.load(dynamics_path)
    angles = data["bin_centers_rad"]
    angle_idx = data["angle_idx"]
    colors = plt.cm.hsv((angles[angle_idx] % (2 * np.pi)) / (2 * np.pi))
    initial_pc = project_onto_basis(data["initial_states"], data["pca_mean"], data["pca_basis"])
    final_pc = data["optimized_pc"][-1]

    fig = plt.figure(figsize=(10.6, 4.6), constrained_layout=True)
    ax_e = fig.add_subplot(1, 2, 1, projection="3d")
    ax_f = fig.add_subplot(1, 2, 2)

    manifold_pc = data["manifold_pc"]
    ax_e.scatter(
        manifold_pc[:, 0],
        manifold_pc[:, 1],
        manifold_pc[:, 2],
        c="0.18",
        s=12,
        linewidths=0,
        alpha=0.9,
    )
    ax_e.scatter(
        initial_pc[:, 0],
        initial_pc[:, 1],
        initial_pc[:, 2],
        c="black",
        marker="x",
        s=52,
        linewidths=1.4,
    )
    ax_e.scatter(
        final_pc[:, 0],
        final_pc[:, 1],
        final_pc[:, 2],
        c="black",
        marker="o",
        s=26,
        linewidths=0,
        edgecolors="none",
    )
    for trial in range(len(angle_idx)):
        trace = data["optimized_pc"][:, trial]
        ax_e.plot(trace[:, 0], trace[:, 1], trace[:, 2], color=colors[trial], linewidth=0.8, alpha=0.45)

    ax_e.set_title("E  convergence to target manifold")
    ax_e.set_xlabel("PC1")
    ax_e.set_ylabel("PC2")
    ax_e.set_zlabel("PC3")
    ax_e.view_init(elev=24, azim=-58)

    for trial, color in enumerate(colors):
        ax_f.plot(data["times"], data["optimized_distance_to_manifold"][:, trial], color=color, linewidth=1.4)
    ax_f.axhline(1e-3, color="0.2", linestyle="--", linewidth=1.0)
    ax_f.set_yscale("log")
    ax_f.set_title("F  distance to nearest manifold point")
    ax_f.set_xlabel("time (s)")
    ax_f.set_ylabel(r"$\min_\theta ||x(t)-x^*(\theta)||_2 / \sqrt{N}$")
    ax_f.grid(True, which="both", color="0.9", linewidth=0.8)

    out = FIGURES / "figure3_ef_reproduction.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out, dynamics_path


def main():
    """
    Command-line entry point for Figure 3E-F reproduction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute dynamics even if the cache exists")
    args = parser.parse_args()

    dynamics_path = compute_figure3_ef_dynamics(force=args.force)
    figure_path, _ = plot_figure3_ef(dynamics_path)
    data = np.load(dynamics_path)
    final_max = float(np.max(data["optimized_distance_to_manifold"][-1]))
    print("Saved:", figure_path)
    print("Saved:", dynamics_path)
    print("Final max distance / sqrt(N):", f"{final_max:.3e}")


if __name__ == "__main__":
    main()
