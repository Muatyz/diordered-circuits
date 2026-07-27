"""Weight matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import TWO_PI, circular_difference, make_theta_hd_pref

TITLE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 8
COLORBAR_LABEL_FONTSIZE = 9


def _style_axis_text(axis: plt.Axes) -> None:
    axis.title.set_fontsize(PANEL_TITLE_FONTSIZE)
    axis.xaxis.label.set_fontsize(LABEL_FONTSIZE)
    axis.yaxis.label.set_fontsize(LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _style_colorbar(colorbar) -> None:
    colorbar.ax.yaxis.label.set_fontsize(COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)


def _set_pi_ticks(axis: plt.Axes, *, which: str) -> None:
    ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
    labels = ["-pi", "-pi/2", "0", "pi/2", "pi"]
    if which == "x":
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
    elif which == "y":
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
    else:
        raise ValueError(f"Unknown axis selector: {which}")


def _set_pi_ticks_for_extent(axis: plt.Axes, extent: tuple[float, float, float, float] | None) -> None:
    if extent is None:
        return
    x_min, x_max, y_min, y_max = extent
    if np.isclose(x_min, -np.pi) and np.isclose(x_max, np.pi):
        _set_pi_ticks(axis, which="x")
    if np.isclose(y_min, -np.pi) and np.isclose(y_max, np.pi):
        _set_pi_ticks(axis, which="y")


def _coerce_theta_pref(theta_pref: np.ndarray, expected_size: int, *, population_name: str) -> np.ndarray:
    theta_pref = np.asarray(theta_pref, dtype=float)
    if theta_pref.ndim != 1 or theta_pref.size != expected_size:
        raise ValueError(f"{population_name} must be a 1D array with size {expected_size}")
    return theta_pref


def _offset_count(theta_target_pref: np.ndarray, theta_source_pref: np.ndarray) -> int:
    target_count = np.unique(theta_target_pref).size
    source_count = np.unique(theta_source_pref).size
    return int(max(target_count, source_count, 1))


def compute_receptive_field_offset_profile(
    *,
    weight_target_by_source: np.ndarray,
    theta_target_pref: np.ndarray,
    theta_source_pref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a weight matrix in source-centered receptive-field coordinates.

    The released Vafidis plotting code rolls each presynaptic column so that
    the source neuron sits at the center, then averages over columns.  This is
    the same operation written in angular coordinates, which also handles the
    paired-HD representation used by this toy model.
    """
    weight_target_by_source = np.asarray(weight_target_by_source, dtype=float)
    if weight_target_by_source.ndim != 2:
        raise ValueError("weight_target_by_source must be a 2D array")
    theta_target_pref = _coerce_theta_pref(
        theta_target_pref,
        weight_target_by_source.shape[0],
        population_name="theta_target_pref",
    )
    theta_source_pref = _coerce_theta_pref(
        theta_source_pref,
        weight_target_by_source.shape[1],
        population_name="theta_source_pref",
    )
    n_offsets = _offset_count(theta_target_pref, theta_source_pref)
    offset_grid = np.linspace(-np.pi, np.pi, n_offsets, endpoint=False, dtype=float)
    if weight_target_by_source.size == 0:
        return offset_grid, np.full(n_offsets, np.nan, dtype=float)

    offset_matrix = circular_difference(theta_target_pref[:, None], theta_source_pref[None, :])
    offset_index = np.floor(((offset_matrix + np.pi) % TWO_PI) / (TWO_PI / n_offsets) + 0.5).astype(int)
    offset_index %= n_offsets
    finite_mask = np.isfinite(weight_target_by_source) & np.isfinite(offset_matrix)
    profile_sum = np.zeros(n_offsets, dtype=float)
    profile_count = np.zeros(n_offsets, dtype=float)
    np.add.at(profile_sum, offset_index[finite_mask], weight_target_by_source[finite_mask])
    np.add.at(profile_count, offset_index[finite_mask], 1.0)
    profile = np.full(n_offsets, np.nan, dtype=float)
    valid_offsets = profile_count > 0.0
    profile[valid_offsets] = profile_sum[valid_offsets] / profile_count[valid_offsets]
    return offset_grid, profile


def compute_receptive_field_offset_profile_history(
    *,
    weight_history: np.ndarray,
    theta_target_pref: np.ndarray,
    theta_source_pref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return receptive-field profiles for each saved weight snapshot."""
    weight_history = np.asarray(weight_history, dtype=float)
    if weight_history.ndim != 3:
        raise ValueError("weight_history must have shape (time, target, source)")
    if weight_history.shape[0] == 0:
        theta_target_pref = np.asarray(theta_target_pref, dtype=float)
        theta_source_pref = np.asarray(theta_source_pref, dtype=float)
        n_offsets = _offset_count(theta_target_pref, theta_source_pref)
        return np.linspace(-np.pi, np.pi, n_offsets, endpoint=False, dtype=float), np.empty((n_offsets, 0))

    offset_grid, first_profile = compute_receptive_field_offset_profile(
        weight_target_by_source=weight_history[0],
        theta_target_pref=theta_target_pref,
        theta_source_pref=theta_source_pref,
    )
    profile_history = np.empty((offset_grid.size, weight_history.shape[0]), dtype=float)
    profile_history[:, 0] = first_profile
    for snapshot_index in range(1, weight_history.shape[0]):
        _offset_grid, profile = compute_receptive_field_offset_profile(
            weight_target_by_source=weight_history[snapshot_index],
            theta_target_pref=theta_target_pref,
            theta_source_pref=theta_source_pref,
        )
        profile_history[:, snapshot_index] = profile
    return offset_grid, profile_history


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
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    fig.subplots_adjust(left=0.15, right=0.84, bottom=0.14, top=0.88)
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
    axis.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    _set_pi_ticks_for_extent(axis, extent)
    _style_axis_text(axis)
    colorbar = fig.colorbar(mesh, ax=axis, label=colorbar_label, pad=0.04)
    _style_colorbar(colorbar)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.84, wspace=0.24)
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
    )
    axes[0].set_title("HD-to-HD", pad=8)
    axes[0].set_xlabel("source HD neuron ID (COM-sorted)")
    axes[0].set_ylabel("target HD neuron ID (COM-sorted)")
    _style_axis_text(axes[0])

    axes[1].imshow(
        w_hr_to_hd,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-max_abs_weight,
        vmax=max_abs_weight,
    )
    axes[1].set_title("HR-to-HD", pad=8)
    axes[1].set_xlabel("source HR neuron ID (L/R, COM-sorted)")
    axes[1].set_ylabel("target HD neuron ID (COM-sorted)")
    _style_axis_text(axes[1])

    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    colorbar = fig.colorbar(hd_mesh, ax=axes, label="synaptic weight [a.u.]", shrink=0.88, pad=0.035)
    _style_colorbar(colorbar)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_weight_snapshot_grid(
    *,
    weight_history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
    title: str,
    max_snapshots: int = 6,
    cmap: str = "coolwarm",
    extent: tuple[float, float, float, float] | None = None,
    x_label: str = "source index",
    y_label: str = "target index",
) -> None:
    """Plot representative weight matrices across training time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_history = np.asarray(weight_history, dtype=float)
    time = np.asarray(time, dtype=float)
    if weight_history.ndim != 3 or time.size != weight_history.shape[0] or time.size == 0:
        fig, axis = plt.subplots(figsize=(6.0, 3.0))
        axis.text(0.5, 0.5, "No weight snapshots", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    snapshot_count = min(max_snapshots, time.size)
    snapshot_indices = np.linspace(0, time.size - 1, snapshot_count, dtype=int)
    column_count = min(3, snapshot_count)
    row_count = int(np.ceil(snapshot_count / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.0 * column_count + 1.0, 3.45 * row_count + 0.4),
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.12, hspace=0.24)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    max_abs_weight = float(np.nanmax(np.abs(weight_history))) if weight_history.size else 1.0
    if max_abs_weight <= 0.0:
        max_abs_weight = 1.0
    last_mesh = None
    for axis_index, axis in enumerate(axes_array):
        if axis_index >= snapshot_count:
            axis.set_axis_off()
            continue
        row_index = axis_index // column_count
        column_index = axis_index % column_count
        snapshot_index = int(snapshot_indices[axis_index])
        last_mesh = axis.imshow(
            weight_history[snapshot_index],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-max_abs_weight,
            vmax=max_abs_weight,
            extent=extent,
        )
        axis.set_title(f"t={time[snapshot_index]:.1f} s", pad=6)
        axis.set_xlabel(x_label if row_index == row_count - 1 else "")
        axis.set_ylabel(y_label if column_index == 0 else "")
        _set_pi_ticks_for_extent(axis, extent)
        axis.tick_params(labelbottom=row_index == row_count - 1, labelleft=column_index == 0)
        _style_axis_text(axis)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    if last_mesh is not None:
        colorbar = fig.colorbar(
            last_mesh,
            ax=axes_array.tolist(),
            label="synaptic weight [a.u.]",
            shrink=0.82,
            pad=0.035,
        )
        _style_colorbar(colorbar)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _profile_color_norm(profile_history: np.ndarray) -> mcolors.Normalize:
    finite_values = np.asarray(profile_history, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 0.05
        vmin -= delta
        vmax += delta
    if vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _filter_and_sort_profile_time(
    *,
    time: np.ndarray,
    profile_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    profile_history = np.asarray(profile_history, dtype=float)
    if profile_history.ndim != 2 or profile_history.shape[1] != time.size:
        return np.empty(0), np.empty((profile_history.shape[0] if profile_history.ndim == 2 else 0, 0))
    finite_time_mask = np.isfinite(time)
    time = time[finite_time_mask]
    profile_history = profile_history[:, finite_time_mask]
    if time.size == 0:
        return time, profile_history
    sort_index = np.argsort(time)
    time = time[sort_index]
    profile_history = profile_history[:, sort_index]
    unique_time, unique_index = np.unique(time, return_index=True)
    return unique_time, profile_history[:, unique_index]


def _training_time_scale(time: np.ndarray) -> tuple[float, str]:
    max_time = float(np.nanmax(np.abs(time))) if time.size else 0.0
    if max_time >= 3600.0:
        return 1.0 / 3600.0, "h"
    if max_time >= 120.0:
        return 1.0 / 60.0, "min"
    return 1.0, "s"


def _linear_time_edges(time: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if time.size == 1:
        if time[0] >= 0.0:
            return np.asarray([0.0, max(float(time[0]), 1.0)], dtype=float)
        half_width = max(abs(float(time[0])) * 0.05, 1.0)
        return np.asarray([time[0] - half_width, time[0] + half_width], dtype=float)
    midpoints = 0.5 * (time[:-1] + time[1:])
    first_edge = time[0] - (midpoints[0] - time[0])
    last_edge = float(time[-1])
    if time[0] >= 0.0:
        first_edge = max(0.0, float(first_edge))
    return np.concatenate([[first_edge], midpoints, [last_edge]]).astype(float)


def _plot_empty_weight_profile_history(*, path: Path, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    axis.set_title(title, fontsize=TITLE_FONTSIZE)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_profile_history_panel(
    *,
    axis: plt.Axes,
    time: np.ndarray,
    profile_history: np.ndarray,
    norm: mcolors.Normalize,
    title: str,
    y_label: str,
    cmap: str,
) -> tuple[object, str]:
    time_scale, time_unit = _training_time_scale(time)
    x_edges = _linear_time_edges(time) * time_scale
    y_edges = np.linspace(-np.pi, np.pi, profile_history.shape[0] + 1, dtype=float)
    mesh = axis.pcolormesh(x_edges, y_edges, profile_history, shading="auto", cmap=cmap, norm=norm)
    axis.set_title(title, pad=6)
    axis.set_ylabel(y_label)
    axis.set_ylim(-np.pi, np.pi)
    _set_pi_ticks(axis, which="y")
    _style_axis_text(axis)
    axis.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return mesh, f"training time [{time_unit}]"


def plot_hd_to_hd_weight_profile_history(
    *,
    weight_history: np.ndarray,
    time: np.ndarray,
    theta_hd_pref: np.ndarray,
    path: str | Path,
    title: str = "HD-to-HD weight profile development",
    cmap: str = "coolwarm",
) -> None:
    """Plot recurrent weight development after source-centered profile averaging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_history = np.asarray(weight_history, dtype=float)
    time = np.asarray(time, dtype=float)
    if weight_history.ndim != 3 or time.size != weight_history.shape[0] or time.size == 0:
        _plot_empty_weight_profile_history(path=path, title=title, message="No weight snapshots")
        return
    theta_hd_pref = _coerce_theta_pref(theta_hd_pref, weight_history.shape[1], population_name="theta_hd_pref")
    if weight_history.shape[2] != theta_hd_pref.size:
        raise ValueError("HD-to-HD weight history must have matching target and source HD dimensions")

    _offset_grid, profile_history = compute_receptive_field_offset_profile_history(
        weight_history=weight_history,
        theta_target_pref=theta_hd_pref,
        theta_source_pref=theta_hd_pref,
    )
    time, profile_history = _filter_and_sort_profile_time(time=time, profile_history=profile_history)
    if time.size == 0 or profile_history.shape[1] == 0:
        _plot_empty_weight_profile_history(path=path, title=title, message="No finite weight snapshots")
        return

    fig, axis = plt.subplots(figsize=(7.3, 4.0))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.16, top=0.86)
    mesh, x_label = _plot_profile_history_panel(
        axis=axis,
        time=time,
        profile_history=profile_history,
        norm=_profile_color_norm(profile_history),
        title=title,
        y_label="target-source RF offset [rad]",
        cmap=cmap,
    )
    axis.set_xlabel(x_label)
    _style_axis_text(axis)
    colorbar = fig.colorbar(mesh, ax=axis, label="mean synaptic weight [a.u.]", pad=0.04)
    _style_colorbar(colorbar)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hr_to_hd_weight_profile_history(
    *,
    weight_history: np.ndarray,
    time: np.ndarray,
    theta_hd_pref: np.ndarray,
    path: str | Path,
    title: str = "HR-to-HD weight profile development",
    cmap: str = "coolwarm",
) -> None:
    """Plot left/right HR-to-HD profile development in receptive-field coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_history = np.asarray(weight_history, dtype=float)
    time_values = np.asarray(time, dtype=float)
    if weight_history.ndim != 3 or time_values.size != weight_history.shape[0] or time_values.size == 0:
        _plot_empty_weight_profile_history(path=path, title=title, message="No weight snapshots")
        return
    theta_hd_pref = _coerce_theta_pref(theta_hd_pref, weight_history.shape[1], population_name="theta_hd_pref")
    n_hr = weight_history.shape[2]
    if n_hr % 2 != 0:
        raise ValueError("HR-to-HD weight history must contain equal left/right HR wings")
    n_hr_per_wing = n_hr // 2
    if n_hr == theta_hd_pref.size:
        theta_lhr_pref = theta_hd_pref[0::2]
        theta_rhr_pref = theta_hd_pref[1::2]
    else:
        theta_lhr_pref = make_theta_hd_pref(n_hr_per_wing)
        theta_rhr_pref = make_theta_hd_pref(n_hr_per_wing)

    _offset_grid, lhr_profile_history = compute_receptive_field_offset_profile_history(
        weight_history=weight_history[:, :, :n_hr_per_wing],
        theta_target_pref=theta_hd_pref,
        theta_source_pref=theta_lhr_pref,
    )
    _offset_grid, rhr_profile_history = compute_receptive_field_offset_profile_history(
        weight_history=weight_history[:, :, n_hr_per_wing:],
        theta_target_pref=theta_hd_pref,
        theta_source_pref=theta_rhr_pref,
    )
    time_values, lhr_profile_history = _filter_and_sort_profile_time(
        time=time_values,
        profile_history=lhr_profile_history,
    )
    _time_values, rhr_profile_history = _filter_and_sort_profile_time(
        time=np.asarray(time, dtype=float),
        profile_history=rhr_profile_history,
    )
    if time_values.size == 0 or lhr_profile_history.shape[1] == 0 or rhr_profile_history.shape[1] == 0:
        _plot_empty_weight_profile_history(path=path, title=title, message="No finite weight snapshots")
        return

    norm = _profile_color_norm(np.concatenate([lhr_profile_history, rhr_profile_history], axis=None))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.86, hspace=0.34)
    mesh, _x_label = _plot_profile_history_panel(
        axis=axes[0],
        time=time_values,
        profile_history=lhr_profile_history,
        norm=norm,
        title="L-HR to HD",
        y_label="target-source RF offset [rad]",
        cmap=cmap,
    )
    _mesh, x_label = _plot_profile_history_panel(
        axis=axes[1],
        time=time_values,
        profile_history=rhr_profile_history,
        norm=norm,
        title="R-HR to HD",
        y_label="target-source RF offset [rad]",
        cmap=cmap,
    )
    axes[1].set_xlabel(x_label)
    _style_axis_text(axes[0])
    _style_axis_text(axes[1])
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    colorbar = fig.colorbar(mesh, ax=axes, label="mean synaptic weight [a.u.]", shrink=0.92, pad=0.035)
    _style_colorbar(colorbar)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_weight_norm_trace(
    *,
    time: np.ndarray,
    weight_norm_hd_to_hd: np.ndarray,
    weight_norm_hr_to_hd: np.ndarray,
    path: str | Path,
    title: str = "Weight norms across training",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.86)
    if time.size == 0:
        axis.text(0.5, 0.5, "No weight snapshots", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.plot(time, weight_norm_hd_to_hd, label="||W_HD->HD||", linewidth=1.5)
        axis.plot(time, weight_norm_hr_to_hd, label="||W_HR->HD||", linewidth=1.5)
        axis.set_xlabel("training time [s]")
        axis.set_ylabel("Frobenius norm")
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    axis.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    _style_axis_text(axis)
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

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.90, hspace=0.40, wspace=0.28)
    for column_index, (label, eigenvalues) in enumerate(spectra.items()):
        scatter_axis = axes[0, column_index]
        curve_axis = axes[1, column_index]
        scatter_axis.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=18, color=colors[label], alpha=0.82)
        scatter_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        scatter_axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.35)
        scatter_axis.set_title(f"{label}: complex plane", pad=8)
        scatter_axis.set_xlabel("Re(lambda)")
        scatter_axis.set_ylabel("Im(lambda)")
        _style_axis_text(scatter_axis)

        sorted_real = _sorted_real_eigenvalues(eigenvalues)
        curve_axis.plot(np.arange(sorted_real.size), sorted_real, color=colors[label], linewidth=1.4)
        curve_axis.set_title(f"{label}: sorted Re(lambda)", pad=8)
        curve_axis.set_xlabel("rank")
        curve_axis.set_ylabel("Re(lambda)")
        _style_axis_text(curve_axis)
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
                    fontsize=TICK_FONTSIZE,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
                )
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    fig.savefig(path, dpi=160)
    plt.close(fig)
