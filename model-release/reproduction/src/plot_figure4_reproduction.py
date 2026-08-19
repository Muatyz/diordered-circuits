from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    benjamini_hochberg,
    circular_center_of_mass_angles,
    empirical_two_point_correlation,
    finite_row_mask,
    kuiper_uniformity_test_asymptotic,
    relative_circulant_error,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
PROCESSED = WORKSPACE_ROOT / "data/processed"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

EXAMPLE_SUBJECTS = ("A3707", "A3716", "A3711", "A3706")
SUBSET_SIZES = (5, 10, 20, 40, 80)
RANDOM_SEED = 20251026


def resolve_tuning_path(path):
    """
    Resolve a processed tuning path saved with either Windows or POSIX syntax.
    """
    path = Path(str(path).replace("\\", "/"))
    candidates = [path, WORKSPACE_ROOT / path, PROCESSED / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find processed tuning file: {path}")


def load_subject_tuning(index_path=PROCESSED / "hd_tuning_index.csv"):
    """
    Load QC-passing, finite, unit-mean-normalized tuning curves by mouse.

    Each returned dictionary contains the subject id, tuning matrix, angular
    bin centers, and circular center-of-mass angle for every included neuron.
    """
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    sessions = []
    for _, row in pd.read_csv(index_path).iterrows():
        data = np.load(resolve_tuning_path(row["tuning_path"]))
        tuning = np.asarray(data["normalized_rate"], dtype=float)
        included = np.asarray(data["included_qc"], dtype=bool)
        valid = included & finite_row_mask(tuning, min_fraction=1.0)
        tuning = tuning[valid]
        if tuning.shape[0] == 0:
            continue

        angles_rad = np.asarray(data["bin_centers_rad"], dtype=float)
        sessions.append(
            {
                "subject_id": str(row["subject_id"]),
                "tuning": tuning,
                "angles_rad": angles_rad,
                "center_of_mass_rad": circular_center_of_mass_angles(tuning, angles_rad),
            }
        )

    if not sessions:
        raise ValueError("No finite QC-passing tuning curves were found")
    return sessions


def build_figure4_statistics(
    sessions,
    seed=RANDOM_SEED,
    subset_sizes=SUBSET_SIZES,
    example_subjects=EXAMPLE_SUBJECTS,
):
    """
    Build the experimental, uniform-null, and correlation data for Figure 4.

    Uniform-null angles are sampled independently while preserving the exact
    neuron count of every mouse. For panel C, one random neuron permutation is
    drawn per example mouse and nested prefixes form the increasing subsets.
    """
    rng = np.random.default_rng(seed)
    com_rows = []
    correlation_rows = []
    correlation_arrays = {}
    session_lookup = {session["subject_id"]: session for session in sessions}

    for mouse_index, session in enumerate(sessions, start=1):
        experimental = session["center_of_mass_rad"]
        random_uniform = rng.uniform(0.0, 2.0 * np.pi, size=experimental.size)
        statistic, p_value = kuiper_uniformity_test_asymptotic(experimental)
        for source, angles in (("data", experimental), ("random_uniform", random_uniform)):
            for unit_index, angle in enumerate(angles):
                com_rows.append(
                    {
                        "subject_id": session["subject_id"],
                        "mouse_index": mouse_index,
                        "unit_index": unit_index,
                        "source": source,
                        "center_of_mass_rad": float(angle),
                    }
                )
        session["kuiper_statistic"] = statistic
        session["kuiper_p_value"] = p_value

    adjusted = benjamini_hochberg(
        np.asarray([session["kuiper_p_value"] for session in sessions], dtype=float)
    )
    for session, adjusted_p in zip(sessions, adjusted):
        session["kuiper_p_adjusted_bh"] = float(adjusted_p)

    for subject_id in example_subjects:
        if subject_id not in session_lookup:
            raise ValueError(f"Figure 4C subject is missing: {subject_id}")
        session = session_lookup[subject_id]
        tuning = session["tuning"]
        if tuning.shape[0] < max(subset_sizes):
            raise ValueError(
                f"{subject_id} has {tuning.shape[0]} neurons, fewer than "
                f"N_subset={max(subset_sizes)}"
            )

        permutation = rng.permutation(tuning.shape[0])
        for subset_size in subset_sizes:
            subset = tuning[permutation[:subset_size]]
            correlation = empirical_two_point_correlation(subset)
            key = f"{subject_id}_n{subset_size}"
            correlation_arrays[key] = correlation.astype(np.float32)
            correlation_rows.append(
                {
                    "subject_id": subject_id,
                    "n_subset": int(subset_size),
                    "relative_circulant_error": relative_circulant_error(correlation),
                    "correlation_min": float(np.min(correlation)),
                    "correlation_max": float(np.max(correlation)),
                }
            )

    return pd.DataFrame(com_rows), pd.DataFrame(correlation_rows), correlation_arrays


def save_figure4_statistics(sessions, com_table, correlation_table, correlation_arrays):
    """
    Save Figure 4 source values and circular-uniformity diagnostics.
    """
    com_path = PROCESSED / "figure4_com_angles.csv"
    correlation_path = PROCESSED / "figure4_correlation_convergence.csv"
    matrices_path = PROCESSED / "figure4_correlation_matrices.npz"
    kuiper_path = PROCESSED / "figure4_kuiper_uniformity.csv"

    com_table.to_csv(com_path, index=False)
    correlation_table.to_csv(correlation_path, index=False)
    np.savez_compressed(matrices_path, **correlation_arrays)
    pd.DataFrame(
        [
            {
                "subject_id": session["subject_id"],
                "n_neurons": session["tuning"].shape[0],
                "kuiper_statistic": session["kuiper_statistic"],
                "kuiper_p_value_asymptotic": session["kuiper_p_value"],
                "kuiper_p_adjusted_bh": session["kuiper_p_adjusted_bh"],
            }
            for session in sessions
        ]
    ).to_csv(kuiper_path, index=False)
    return com_path, correlation_path, matrices_path, kuiper_path


def plot_com_panel(ax, com_table, source, title):
    """
    Plot one dot raster of center-of-mass angles across mice.
    """
    subset = com_table.loc[com_table["source"] == source]
    ax.scatter(
        subset["center_of_mass_rad"],
        subset["mouse_index"],
        s=3.0,
        c="black",
        linewidths=0.0,
        rasterized=True,
    )
    n_mice = int(com_table["mouse_index"].max())
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.25, n_mice + 0.75)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks(np.arange(5, n_mice + 1, 5))
    ax.set_xlabel("center of mass")
    ax.set_title(title, fontsize=10)
    ax.tick_params(length=2.5, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_correlation_panel(fig, subgrid, correlation_arrays):
    """
    Plot the 4-by-5 Figure 4C grid of uncentered two-point functions.
    """
    image = None
    for row, subject_id in enumerate(EXAMPLE_SUBJECTS):
        for col, subset_size in enumerate(SUBSET_SIZES):
            ax = fig.add_subplot(subgrid[row, col])
            image = ax.imshow(
                correlation_arrays[f"{subject_id}_n{subset_size}"],
                origin="lower",
                extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
                cmap="viridis",
                vmin=0.0,
                vmax=5.0,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([0.0, 2.0 * np.pi], ["0", r"$2\pi$"])
            ax.set_yticks([0.0, 2.0 * np.pi], ["0", r"$2\pi$"] if col == 0 else [])
            ax.tick_params(length=1.8, labelsize=6, pad=1)
            if row == 0:
                ax.set_title(rf"$N_{{sub}}={subset_size}$", fontsize=8, pad=3)
            if col == 0:
                ax.set_ylabel(
                    f"mouse\n{subject_id}\n" + r"$\theta^\prime$",
                    fontsize=7,
                    rotation=90,
                    labelpad=2,
                )
            if row == len(EXAMPLE_SUBJECTS) - 1:
                ax.set_xlabel(r"$\theta$", fontsize=7, labelpad=-2)

    return image


def plot_figure4():
    """
    Generate the Figure 4A-C reproduction and save its numerical source data.
    """
    sessions = load_subject_tuning()
    com_table, correlation_table, correlation_arrays = build_figure4_statistics(sessions)
    statistic_paths = save_figure4_statistics(
        sessions,
        com_table,
        correlation_table,
        correlation_arrays,
    )

    fig = plt.figure(figsize=(12.2, 5.35))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.25, 1.25, 4.9),
        left=0.06,
        right=0.93,
        bottom=0.12,
        top=0.88,
        wspace=0.14,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1], sharey=ax_a)
    plot_com_panel(ax_a, com_table, "data", "data")
    plot_com_panel(ax_b, com_table, "random_uniform", "random uniform")
    ax_a.set_ylabel("mouse index")
    ax_b.set_ylabel("")
    ax_b.tick_params(labelleft=False)

    correlation_grid = outer[0, 2].subgridspec(4, 5, wspace=0.08, hspace=0.12)
    image = plot_correlation_panel(fig, correlation_grid, correlation_arrays)
    colorbar_ax = fig.add_axes([0.945, 0.16, 0.012, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=[0.0, 2.5, 5.0])
    colorbar.ax.tick_params(labelsize=7, length=2)

    fig.text(0.052, 0.91, "A", fontsize=13, fontweight="bold")
    fig.text(0.196, 0.91, "B", fontsize=13, fontweight="bold")
    fig.text(0.335, 0.91, "C", fontsize=13, fontweight="bold")
    out = FIGURES / "figure4_abc_reproduction.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out, statistic_paths, sessions, correlation_table


def main():
    """
    Command-line entry point for the Figure 4A-C reproduction.
    """
    figure_path, statistic_paths, sessions, correlation_table = plot_figure4()
    print("Saved:", figure_path)
    for path in statistic_paths:
        print("Saved:", path)
    print("Mice:", len(sessions))
    print("Included neurons:", sum(session["tuning"].shape[0] for session in sessions))
    print(
        "Asymptotic Kuiper p<0.05 before/after BH:",
        sum(session["kuiper_p_value"] < 0.05 for session in sessions),
        "/",
        sum(session["kuiper_p_adjusted_bh"] < 0.05 for session in sessions),
    )
    print(correlation_table.to_string(index=False))


if __name__ == "__main__":
    main()
