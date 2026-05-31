from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    circular_center_of_mass_angles,
    circulant_from_diagonal_means,
    finite_row_mask,
    optimized_recurrent_weights,
    scramble_residuals,
    sinkhorn_normalize,
)


PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
WEIGHT_FORMULA_VERSION = "paper_a2_discrete_kernel_b2_preprocessing_v2"


def load_population_tuning(index_path=PROCESSED / "hd_tuning_index.csv"):
    """
    Load all QC-passing HD tuning curves into one population matrix.

    Returns the pooled unit-mean-normalized tuning curves, angular bin centers,
    and metadata for tracing each row back to its subject and unit id.
    """
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    index = pd.read_csv(index_path)
    curves = []
    metadata = []
    bin_centers_rad = None

    for _, row in index.iterrows():
        data = np.load(row["tuning_path"])
        matrix = data["normalized_rate"]
        included = data["included_qc"].astype(bool)
        unit_ids = data["unit_ids"]

        valid = included & finite_row_mask(matrix, min_fraction=1.0)
        matrix = matrix[valid]
        unit_ids = unit_ids[valid]
        if len(matrix) == 0:
            continue

        curves.append(matrix)
        for unit_id in unit_ids:
            metadata.append({"subject_id": row["subject_id"], "unit_id": int(unit_id)})

        if bin_centers_rad is None:
            bin_centers_rad = data["bin_centers_rad"]

    if not curves:
        raise ValueError("No finite included HD tuning curves were found")

    return np.vstack(curves), np.asarray(bin_centers_rad), pd.DataFrame(metadata)


def build_figure3_matrices(
    regularization=1e-6,
    activation_beta=2.0,
    firing_rate_floor=1e-4,
    circulant_gain=1.275,
    seed=20260126,
    out_path=PROCESSED / "figure3_abcd_weight_matrices.npz",
):
    """
    Construct the four Figure 3A-D weight matrices from processed tuning data.

    Panel A uses a fixed random neuron order. Panel B uses circular
    center-of-mass order. Panel C is the diagonal-average circulant matrix.
    Panel D adds randomly permuted B-C residuals back to the circulant matrix.
    """
    tuning, angles_rad, metadata = load_population_tuning()
    tuning = np.maximum(tuning, firing_rate_floor)
    tuning = sinkhorn_normalize(tuning)
    com_angles = circular_center_of_mass_angles(tuning, angles_rad)

    weights = optimized_recurrent_weights(
        tuning,
        regularization=regularization,
        activation_beta=activation_beta,
    )
    sorted_order = np.argsort(com_angles)

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(weights))

    weights_random = weights[np.ix_(random_order, random_order)]
    weights_sorted = weights[np.ix_(sorted_order, sorted_order)]
    weights_circulant, diagonal_means = circulant_from_diagonal_means(weights_sorted)
    weights_noisy, scrambled_residual = scramble_residuals(weights_sorted, weights_circulant, rng)

    # The reference scales the circulant variants to ensure bump formation.
    weights_circulant_scaled = circulant_gain * weights_circulant
    weights_noisy_scaled = circulant_gain * weights_noisy

    metadata_sorted = metadata.iloc[sorted_order].reset_index(drop=True)
    metadata_random = metadata.iloc[random_order].reset_index(drop=True)
    metadata_sorted.to_csv(PROCESSED / "figure3_sorted_units.csv", index=False)
    metadata_random.to_csv(PROCESSED / "figure3_random_units.csv", index=False)

    np.savez_compressed(
        out_path,
        weights_random=weights_random.astype(np.float32),
        weights_sorted=weights_sorted.astype(np.float32),
        weights_circulant=weights_circulant_scaled.astype(np.float32),
        weights_noisy_circulant=weights_noisy_scaled.astype(np.float32),
        sorted_order=sorted_order,
        random_order=random_order,
        center_of_mass_rad=com_angles,
        bin_centers_rad=angles_rad,
        doubly_normalized_tuning=tuning.astype(np.float32),
        diagonal_means=diagonal_means.astype(np.float32),
        scrambled_residual=scrambled_residual.astype(np.float32),
        regularization=regularization,
        activation_beta=activation_beta,
        firing_rate_floor=firing_rate_floor,
        circulant_gain=circulant_gain,
        weight_formula_version=WEIGHT_FORMULA_VERSION,
        seed=seed,
    )
    return out_path


def figure3_matrix_cache_matches(
    matrix_path,
    regularization=1e-6,
    activation_beta=2.0,
    firing_rate_floor=1e-4,
    circulant_gain=1.275,
):
    """
    Check whether a cached Figure 3A-D matrix file matches current settings.

    This protects downstream dynamics from silently reusing old weights built
    with a different softplus beta or optimization regularization.
    """
    if not matrix_path.exists():
        return False

    data = np.load(matrix_path)
    return (
        float(data.get("regularization", np.nan)) == float(regularization)
        and float(data.get("activation_beta", np.nan)) == float(activation_beta)
        and float(data.get("firing_rate_floor", np.nan)) == float(firing_rate_floor)
        and float(data.get("circulant_gain", np.nan)) == float(circulant_gain)
        and str(data.get("weight_formula_version", "")) == WEIGHT_FORMULA_VERSION
    )


def symmetric_color_limits(matrix, percentile=95.0):
    """
    Choose color limits from robust percentiles of matrix magnitudes.

    The full matrices are saved without clipping. For display, each panel uses
    its own symmetric clipped scale so the weak circulant component is visible
    alongside the much larger disordered residuals.
    """
    vmax = float(np.nanpercentile(np.abs(matrix), percentile))
    return -vmax, vmax


def plot_matrix_panel(ax, matrix, title):
    """
    Draw one weight matrix panel without axis clutter.
    """
    vmin, vmax = symmetric_color_limits(matrix)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Presynaptic neuron")
    ax.set_ylabel("Postsynaptic neuron")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_figure3_abcd(matrix_path=PROCESSED / "figure3_abcd_weight_matrices.npz"):
    """
    Generate the Figure 3A-D weight-matrix reproduction.

    If the processed matrix file is absent, it is constructed first from the
    pooled DANDI 000939 HD tuning curves.
    """
    if not figure3_matrix_cache_matches(matrix_path):
        matrix_path = build_figure3_matrices(out_path=matrix_path)

    data = np.load(matrix_path)
    matrices = [
        data["weights_random"],
        data["weights_sorted"],
        data["weights_circulant"],
        data["weights_noisy_circulant"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8), constrained_layout=True)
    titles = [
        "A  optimized weights, random order",
        "B  optimized weights, COM order",
        "C  diagonal-averaged circulant",
        "D  circulant plus shuffled residuals",
    ]

    images = []
    for ax, matrix, title in zip(axes.ravel(), matrices, titles):
        images.append(plot_matrix_panel(ax, matrix, title))

    for ax, im in zip(axes.ravel(), images):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    out = FIGURES / "figure3_abcd_reproduction.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out, matrix_path


def main():
    """
    Command-line entry point for Figure 3A-D reproduction.
    """
    matrix_path = build_figure3_matrices()
    figure_path, _ = plot_figure3_abcd(matrix_path)
    data = np.load(matrix_path)
    print("Saved:", figure_path)
    print("Saved:", matrix_path)
    print("N:", data["weights_sorted"].shape[0])
    print("regularization:", float(data["regularization"]))
    print("activation_beta:", float(data["activation_beta"]))
    print("firing_rate_floor:", float(data["firing_rate_floor"]))
    print("circulant_gain:", float(data["circulant_gain"]))


if __name__ == "__main__":
    main()
