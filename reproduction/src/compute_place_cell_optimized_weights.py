import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    gaussian_process_place_fields_1d,
    linear_center_of_mass_positions,
    optimized_recurrent_weights,
    sinkhorn_normalize,
    softplus_inverse,
)


PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def load_rate_map(path, key=None):
    """
    Load an experimental firing-rate map shaped `(n_cells, n_positions)`.

    NumPy `.npy` files are loaded directly. For `.npz` files, `key` selects
    the array; when omitted the archive must contain exactly one array.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        rate_map = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        if key is None:
            if len(archive.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; pass --rate-map-key")
            key = archive.files[0]
        rate_map = archive[key]
    else:
        raise ValueError("Rate maps must be stored as .npy or .npz")

    rate_map = np.asarray(rate_map, dtype=float)
    if rate_map.ndim != 2:
        raise ValueError("Expected a rate map shaped (n_cells, n_positions)")
    if not np.all(np.isfinite(rate_map)):
        raise ValueError("Rate map contains non-finite values")
    return np.maximum(rate_map, 0.0)


def count_active_fields(rate_map):
    """
    Count contiguous positive place fields in each row of a 1D rate map.
    """
    active = np.asarray(rate_map) > 0
    starts = active[:, :1].astype(int)
    transitions = (~active[:, :-1]) & active[:, 1:]
    return starts[:, 0] + np.sum(transitions, axis=1)


def compute_place_cell_weights(
    rate_map=None,
    n_cells=500,
    environment_length=200.0,
    resolution=0.5,
    correlation_length=2.5,
    threshold=1.0,
    firing_rate_floor=1e-4,
    regularization=3e-4,
    activation_beta=2.0,
    seed=20260614,
    out_path=PROCESSED / "place_cell_optimized_weights.npz",
):
    """
    Construct optimized recurrent weights for a 1D place-cell manifold.

    When `rate_map` is absent, place maps are sampled from the thresholded
    Gaussian-process model of Mainali et al. The target firing-rate manifold
    is floored and doubly normalized before applying the minimum-norm weight
    solution used for the disordered continuous-attractor model.
    """
    if rate_map is None:
        n_positions = int(round(environment_length / resolution))
        gp_input, raw_rate = gaussian_process_place_fields_1d(
            n_cells=n_cells,
            n_positions=n_positions,
            correlation_length_bins=correlation_length / resolution,
            threshold=threshold,
            seed=seed,
        )
        source = "thresholded_gaussian_process"
    else:
        raw_rate = np.asarray(rate_map, dtype=float)
        n_cells, n_positions = raw_rate.shape
        resolution = environment_length / n_positions
        gp_input = np.full_like(raw_rate, np.nan)
        source = "experimental_rate_map"

    positions = (np.arange(n_positions) + 0.5) * resolution
    target_rate = sinkhorn_normalize(np.maximum(raw_rate, firing_rate_floor))
    weights = optimized_recurrent_weights(
        target_rate,
        regularization=regularization,
        activation_beta=activation_beta,
    )

    target_current = softplus_inverse(target_rate, beta=activation_beta)
    recurrent_current = weights @ target_rate
    fixed_point_residual = recurrent_current - target_current
    residual_rms_by_position = np.sqrt(np.mean(fixed_point_residual**2, axis=0))

    centers = linear_center_of_mass_positions(raw_rate, positions)
    sorted_order = np.argsort(centers)
    field_counts = count_active_fields(raw_rate)

    np.savez_compressed(
        out_path,
        gp_input=gp_input.astype(np.float32),
        raw_rate=raw_rate.astype(np.float32),
        target_rate=target_rate.astype(np.float32),
        target_current=target_current.astype(np.float32),
        weights=weights.astype(np.float32),
        positions=positions,
        centers=centers,
        sorted_order=sorted_order,
        field_counts=field_counts,
        fixed_point_residual=fixed_point_residual.astype(np.float32),
        residual_rms_by_position=residual_rms_by_position.astype(np.float32),
        source=source,
        environment_length=environment_length,
        resolution=resolution,
        correlation_length=correlation_length,
        threshold=threshold,
        firing_rate_floor=firing_rate_floor,
        regularization=regularization,
        activation_beta=activation_beta,
        seed=seed,
    )
    return out_path


def plot_place_cell_weights(
    data_path=PROCESSED / "place_cell_optimized_weights.npz",
    out_path=FIGURES / "place_cell_optimized_weights.png",
):
    """
    Plot generated place maps, target manifold, optimized weights, and errors.
    """
    data = np.load(data_path)
    order = data["sorted_order"]
    raw_rate = data["raw_rate"][order]
    target_rate = data["target_rate"][order]
    weights = data["weights"][np.ix_(order, order)]
    positions = data["positions"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    axes[0, 0].imshow(
        raw_rate,
        aspect="auto",
        origin="lower",
        extent=[positions[0], positions[-1], 0, len(raw_rate)],
        cmap="magma",
    )
    axes[0, 0].set_title("Thresholded GP place maps, COM order")
    axes[0, 0].set_xlabel("position")
    axes[0, 0].set_ylabel("place cell")

    axes[0, 1].imshow(
        target_rate,
        aspect="auto",
        origin="lower",
        extent=[positions[0], positions[-1], 0, len(target_rate)],
        cmap="magma",
    )
    axes[0, 1].set_title("Doubly normalized target manifold")
    axes[0, 1].set_xlabel("position")
    axes[0, 1].set_ylabel("place cell")

    vmax = np.percentile(np.abs(weights), 99)
    axes[1, 0].imshow(weights, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[1, 0].set_title("Optimized recurrent weights, COM order")
    axes[1, 0].set_xlabel("presynaptic cell")
    axes[1, 0].set_ylabel("postsynaptic cell")

    axes[1, 1].plot(positions, data["residual_rms_by_position"], color="black")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Fixed-point equation residual")
    axes[1, 1].set_xlabel("position")
    axes[1, 1].set_ylabel("RMS current residual")
    axes[1, 1].grid(True, which="both", color="0.9")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    """
    Command-line entry point for place-cell optimized-weight construction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-map", type=Path, help="optional experimental .npy/.npz rate map")
    parser.add_argument("--rate-map-key", help="array key when --rate-map is an .npz archive")
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--environment-length", type=float, default=200.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--correlation-length", type=float, default=2.5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rate_map = None
    if args.rate_map is not None:
        rate_map = load_rate_map(args.rate_map, key=args.rate_map_key)

    data_path = compute_place_cell_weights(
        rate_map=rate_map,
        n_cells=args.n_cells,
        environment_length=args.environment_length,
        resolution=args.resolution,
        correlation_length=args.correlation_length,
        threshold=args.threshold,
        regularization=args.regularization,
        seed=args.seed,
    )
    figure_path = plot_place_cell_weights(data_path)
    data = np.load(data_path)
    residual = data["fixed_point_residual"]
    print("Saved:", data_path)
    print("Saved:", figure_path)
    print("source:", str(data["source"]))
    print("shape:", data["raw_rate"].shape)
    print("mean fields per cell:", f"{np.mean(data['field_counts']):.3f}")
    print("fixed-point residual RMS:", f"{np.sqrt(np.mean(residual**2)):.6g}")
    print("fixed-point residual max abs:", f"{np.max(np.abs(residual)):.6g}")


if __name__ == "__main__":
    main()
