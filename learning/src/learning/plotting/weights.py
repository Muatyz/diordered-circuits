"""Weight matrix plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np


def plot_weight_matrix(
    *,
    weight_matrix: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "coolwarm",
    x_label: str = "source neuron index [unitless]",
    y_label: str = "target HD neuron index [unitless]",
    colorbar_label: str = "synaptic weight [a.u.]",
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.14, top=0.88)
    max_abs_weight = float(np.nanmax(np.abs(weight_matrix))) if weight_matrix.size else 1.0
    if max_abs_weight <= 0.0:
        max_abs_weight = 1.0
    mesh = axis.imshow(
        weight_matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-max_abs_weight if np.min(weight_matrix) < 0.0 else 0.0,
        vmax=max_abs_weight,
        extent=extent,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    fig.colorbar(mesh, ax=axis, label=colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_weight_matrices_side_by_side(
    *,
    w_hd_to_hd: np.ndarray,
    w_hr_to_hd: np.ndarray,
    path: str | Path,
    title: str = "Trained weight matrices",
) -> None:
    """Plot recurrent and HR-to-HD weights in one shared-color-scale figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    finite_abs_weights = np.concatenate([np.ravel(np.abs(w_hd_to_hd)), np.ravel(np.abs(w_hr_to_hd))])
    max_abs_weight = float(np.nanmax(finite_abs_weights)) if finite_abs_weights.size else 1.0
    if max_abs_weight <= 0.0:
        max_abs_weight = 1.0

    hd_mesh = axes[0].imshow(
        w_hd_to_hd,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-max_abs_weight,
        vmax=max_abs_weight,
        extent=(-np.pi, np.pi, -np.pi, np.pi),
    )
    axes[0].set_title("HD-to-HD")
    axes[0].set_xlabel("source HD theta [rad]")
    axes[0].set_ylabel("target HD theta [rad]")
    axes[0].set_xticks([-np.pi, 0.0, np.pi])
    axes[0].set_xticklabels(["-pi", "0", "pi"])
    axes[0].set_yticks([-np.pi, 0.0, np.pi])
    axes[0].set_yticklabels(["-pi", "0", "pi"])

    axes[1].imshow(
        w_hr_to_hd,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-max_abs_weight,
        vmax=max_abs_weight,
        extent=(0.0, float(w_hr_to_hd.shape[1]), -np.pi, np.pi),
    )
    axes[1].set_title("HR-to-HD")
    axes[1].set_xlabel("source HR cell index [unitless]")
    axes[1].set_ylabel("target HD theta [rad]")
    axes[1].set_yticks([-np.pi, 0.0, np.pi])
    axes[1].set_yticklabels(["-pi", "0", "pi"])

    fig.suptitle(title)
    fig.colorbar(hd_mesh, ax=axes, label="synaptic weight [a.u.]", shrink=0.88)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sorted_real_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    return np.sort(np.real(np.asarray(eigenvalues, dtype=complex)))[::-1]


def plot_weight_eigen_spectrum(
    *,
    hd_to_hd_eigenvalues: np.ndarray,
    hr_to_hd_eigenvalues: np.ndarray,
    path: str | Path,
    title: str = "Weight eigenvalue spectrum",
    diagnostics: dict[str, dict[str, float]] | None = None,
) -> None:
    """Plot complex eigenvalues and sorted real-value curves for two weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spectra = {
        "HD-to-HD": np.asarray(hd_to_hd_eigenvalues, dtype=complex),
        "HR-to-HD": np.asarray(hr_to_hd_eigenvalues, dtype=complex),
    }
    colors = {"HD-to-HD": "#2f6f9f", "HR-to-HD": "#a05a2c"}
    diagnostic_keys = {"HD-to-HD": "hd_to_hd", "HR-to-HD": "hr_to_hd"}

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.90, hspace=0.34, wspace=0.30)
    for column_index, (label, eigenvalues) in enumerate(spectra.items()):
        scatter_axis = axes[0, column_index]
        curve_axis = axes[1, column_index]
        scatter_axis.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=18, color=colors[label], alpha=0.82)
        scatter_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        scatter_axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.35)
        scatter_axis.set_title(f"{label}: complex plane")
        scatter_axis.set_xlabel("Re(lambda)")
        scatter_axis.set_ylabel("Im(lambda)")

        sorted_real = _sorted_real_eigenvalues(eigenvalues)
        curve_axis.plot(np.arange(sorted_real.size), sorted_real, color=colors[label], linewidth=1.4)
        curve_axis.set_title(f"{label}: sorted Re(lambda)")
        curve_axis.set_xlabel("rank")
        curve_axis.set_ylabel("Re(lambda)")
        if diagnostics is not None:
            matrix_diagnostics = diagnostics.get(diagnostic_keys[label], {})
            pair_fraction = matrix_diagnostics.get("nonconstant_pair_fraction_le_2pct")
            first_gap = matrix_diagnostics.get("first_nonconstant_pair_gap_norm")
            if pair_fraction is not None and first_gap is not None:
                curve_axis.text(
                    0.04,
                    0.96,
                    f"pair frac<=2%: {pair_fraction:.2f}\nfirst pair gap: {first_gap:.3g}",
                    transform=curve_axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
                )
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
