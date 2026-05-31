from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    align_rows_to_peak,
    circular_bin_centers_deg,
    circular_smooth,
    close_circular_trace,
    finite_row_mask,
    population_mean_and_std,
)


PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def classic_ring_attractor_curves(n_bins=100, n_shifts=9, width_deg=32.0):
    """
    Generate ideal ring-attractor tuning curves by circularly shifting one bump.

    Every curve has the same shape and differs only by preferred direction,
    illustrating translational invariance on the angular ring.
    """
    theta = circular_bin_centers_deg(n_bins, centered=False)
    distance = ((theta + 180.0) % 360.0) - 180.0
    base = np.exp(-0.5 * (distance / width_deg) ** 2)
    base = circular_smooth(base, sigma_bins=1.0)
    shifts = np.linspace(0, n_bins, n_shifts, endpoint=False).astype(int)
    curves = np.vstack([np.roll(base, shift) for shift in shifts])
    return theta, curves


def load_aligned_subject_tuning(index_path=PROCESSED / "hd_tuning_index.csv"):
    """
    Load all processed sessions and peak-align included HD-cell tuning curves.

    The function returns one row per subject containing the full aligned
    `(n_units, n_bins)` matrix, plus per-subject population mean and standard
    deviation. Curves are unit-mean-normalized before alignment.
    """
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    index = pd.read_csv(index_path)
    rows = []
    for _, row in index.iterrows():
        data = np.load(row["tuning_path"])
        matrix = data["normalized_rate"]
        included = data["included_qc"].astype(bool)
        matrix = matrix[included]

        valid = finite_row_mask(matrix, min_fraction=1.0)
        matrix = matrix[valid]
        if len(matrix) == 0:
            continue

        aligned = align_rows_to_peak(matrix)
        mean_curve, std_curve = population_mean_and_std(aligned)
        rows.append(
            {
                "subject_id": row["subject_id"],
                "n_units": int(len(aligned)),
                "aligned": aligned,
                "mean": mean_curve,
                "std": std_curve,
            }
        )

    if not rows:
        raise ValueError("No finite included HD tuning curves were found")
    return rows


def plot_ring_panel(ax):
    """
    Draw Figure 2C-style ideal ring-attractor curves.

    Curves are plotted across the full 0 to 360 degree direction range.
    """
    theta, curves = classic_ring_attractor_curves()
    x, curves_closed = close_circular_trace(theta, curves)
    colors = plt.cm.hsv(np.linspace(0, 1, len(curves), endpoint=False))
    for curve, color in zip(curves_closed, colors):
        ax.plot(x, curve, color=color, linewidth=1.8, alpha=0.9)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Head direction (deg)")
    ax.set_ylabel("Normalized rate")
    ax.set_title("C  Ideal ring attractor")


def plot_aligned_unit_panel(ax, subject_rows, max_units=160):
    """
    Draw Figure 2D-style aligned postsubicular unit curves.

    Units are sampled evenly from the pooled aligned matrix only for visual
    readability; the plotted angular axis still includes every bin from
    -180 to 180 degrees.
    """
    pooled = np.vstack([row["aligned"] for row in subject_rows])
    if len(pooled) > max_units:
        pick = np.linspace(0, len(pooled) - 1, max_units).astype(int)
        pooled = pooled[pick]

    x = circular_bin_centers_deg(pooled.shape[1], centered=True)
    x_closed, pooled_closed = close_circular_trace(x, pooled)
    for curve in pooled_closed:
        ax.plot(x_closed, curve, color="#3f78b5", linewidth=0.55, alpha=0.16)

    mean_curve = np.nanmean(np.vstack([row["mean"] for row in subject_rows]), axis=0)
    _, mean_closed = close_circular_trace(x, mean_curve)
    ax.plot(x_closed, mean_closed, color="#111111", linewidth=2.1)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Aligned head direction (deg)")
    ax.set_ylabel("Unit-mean normalized rate")
    ax.set_title("D  Postsubicular HD cells")


def plot_subject_mean_panel(ax, subject_rows):
    """
    Draw Figure 2E-style subject mean curves and their across-subject average.

    Light lines show each mouse's mean across units. The dark line is the
    average of those subject means, matching the TODO definition.
    """
    means = np.vstack([row["mean"] for row in subject_rows])
    x = circular_bin_centers_deg(means.shape[1], centered=True)
    x_closed, means_closed = close_circular_trace(x, means)
    for curve in means_closed:
        ax.plot(x_closed, curve, color="#7aa6cf", linewidth=0.8, alpha=0.42)

    grand = np.nanmean(means, axis=0)
    _, grand_closed = close_circular_trace(x, grand)
    ax.plot(x_closed, grand_closed, color="#111111", linewidth=2.4)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Aligned head direction (deg)")
    ax.set_ylabel("Mean normalized rate")
    ax.set_title("E  Mean across units")


def plot_subject_std_panel(ax, subject_rows):
    """
    Draw Figure 2F-style subject standard-deviation curves.

    Light lines show each mouse's per-angle standard deviation across units.
    The dark line is the average across mice.
    """
    stds = np.vstack([row["std"] for row in subject_rows])
    x = circular_bin_centers_deg(stds.shape[1], centered=True)
    x_closed, stds_closed = close_circular_trace(x, stds)
    for curve in stds_closed:
        ax.plot(x_closed, curve, color="#cf8a54", linewidth=0.8, alpha=0.42)

    grand = np.nanmean(stds, axis=0)
    _, grand_closed = close_circular_trace(x, grand)
    ax.plot(x_closed, grand_closed, color="#111111", linewidth=2.4)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Aligned head direction (deg)")
    ax.set_ylabel("SD across units")
    ax.set_title("F  Heterogeneity across units")


def save_population_summary(subject_rows):
    """
    Save the angle-wise Figure 2E/F statistics used for plotting.

    The CSV makes it easy to inspect that every aligned angle bin is present
    rather than only one side of the circular tuning curves.
    """
    means = np.vstack([row["mean"] for row in subject_rows])
    stds = np.vstack([row["std"] for row in subject_rows])
    x = circular_bin_centers_deg(means.shape[1], centered=True)
    summary = pd.DataFrame(
        {
            "aligned_angle_deg": x,
            "mean_rate_grand": np.nanmean(means, axis=0),
            "std_rate_grand": np.nanmean(stds, axis=0),
            "n_subjects": len(subject_rows),
            "n_units_total": sum(row["n_units"] for row in subject_rows),
        }
    )
    out = PROCESSED / "figure2_ef_aligned_population_summary.csv"
    summary.to_csv(out, index=False)
    return out


def plot_figure2_cdef():
    """
    Generate a four-panel reproduction of Figure 2C-F.

    Panel C is an idealized ring-attractor control. Panels D-F use the
    processed DANDI 000939 postsubicular HD-cell tuning curves.
    """
    subject_rows = load_aligned_subject_tuning()
    summary_path = save_population_summary(subject_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    plot_ring_panel(axes[0, 0])
    plot_aligned_unit_panel(axes[0, 1], subject_rows)
    plot_subject_mean_panel(axes[1, 0], subject_rows)
    plot_subject_std_panel(axes[1, 1], subject_rows)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)

    out = FIGURES / "figure2_cdef_reproduction.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out, summary_path, subject_rows


def main():
    """
    Command-line entry point for Figure 2C-F reproduction.
    """
    figure_path, summary_path, subject_rows = plot_figure2_cdef()
    print("Saved:", figure_path)
    print("Saved:", summary_path)
    print("Subjects:", len(subject_rows))
    print("Included units:", sum(row["n_units"] for row in subject_rows))


if __name__ == "__main__":
    main()
