"""Activity plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import collapse_activity_by_theta, peak_decode, pva_decode, pva_vector_strength


def _set_radian_ticks(axis: plt.Axes, *, which: str) -> None:
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


def _wrapped_trace_for_axis(theta_trace: np.ndarray) -> np.ndarray:
    theta_wrapped = (np.asarray(theta_trace, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if theta_wrapped.size < 2:
        return theta_wrapped
    plot_trace = theta_wrapped.copy()
    jump_mask = np.abs(np.diff(theta_wrapped)) > np.pi
    plot_trace[np.flatnonzero(jump_mask) + 1] = np.nan
    return plot_trace


def _plot_decoded_overlay(
    *,
    axis: plt.Axes,
    time: np.ndarray,
    theta_trace: np.ndarray | None,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
) -> bool:
    if theta_trace is None:
        return False
    theta_trace = np.asarray(theta_trace, dtype=float)
    if time.size != theta_trace.size or time.size == 0:
        return False
    axis.plot(
        time,
        _wrapped_trace_for_axis(theta_trace),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.88,
        label=label,
    )
    return True


def _select_slice_indices(time: np.ndarray, slice_times: np.ndarray | None) -> np.ndarray:
    if time.size == 0:
        return np.empty(0, dtype=int)
    if slice_times is None or slice_times.size == 0:
        candidate_times = np.linspace(float(time[0]), float(time[-1]), min(5, time.size))
    else:
        candidate_times = np.asarray(slice_times, dtype=float)
    slice_indices = [int(np.argmin(np.abs(time - candidate_time))) for candidate_time in candidate_times]
    return np.asarray(sorted(set(slice_indices)), dtype=int)


def plot_activity_heatmap(
    *,
    r_hd_history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
    title: str = "HD activity",
    theta_hd_pref: np.ndarray | None = None,
    theta_hd_decoded: np.ndarray | None = None,
    theta_hd_decoded_peak: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.88)
    if r_hd_history.size == 0:
        image = np.zeros((1, 1))
        extent = None
        y_label = "HD neuron index [unitless]"
    else:
        image = r_hd_history.T
        if theta_hd_pref is None or theta_hd_pref.size != r_hd_history.shape[1]:
            extent = [float(time[0]), float(time[-1]), 0, r_hd_history.shape[1] - 1] if time.size else None
            y_label = "HD neuron index [unitless]"
        else:
            extent = [float(time[0]), float(time[-1]), -np.pi, np.pi] if time.size else None
            y_label = "HD preferred direction theta_HD [rad]"
    mesh = axis.imshow(image, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    if theta_hd_pref is not None and time.size > 0:
        plotted_pva = _plot_decoded_overlay(
            axis=axis,
            time=time,
            theta_trace=theta_hd_decoded,
            label="PVA decode",
            color="white",
            linestyle="-",
            linewidth=0.85,
        )
        plotted_peak = _plot_decoded_overlay(
            axis=axis,
            time=time,
            theta_trace=theta_hd_decoded_peak,
            label="peak decode",
            color="#ffb000",
            linestyle="--",
            linewidth=0.9,
        )
        if plotted_pva or plotted_peak:
            axis.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    if theta_hd_pref is not None and theta_hd_pref.size == r_hd_history.shape[1]:
        axis.set_ylim(-np.pi, np.pi)
        _set_radian_ticks(axis, which="y")
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(y_label)
    fig.colorbar(mesh, ax=axis, label="normalized HD firing rate [a.u.]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_activity_tuning_slices(
    *,
    r_hd_history: np.ndarray,
    time: np.ndarray,
    theta_hd_pref: np.ndarray,
    path: str | Path,
    title: str = "HD activity tuning slices",
    slice_times: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    if r_hd_history.size == 0 or time.size == 0:
        axis.text(0.5, 0.5, "No activity history", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    slice_indices = _select_slice_indices(time, slice_times)
    color_values = plt.cm.viridis(np.linspace(0.0, 1.0, max(slice_indices.size, 1)))
    for color_index, slice_index in enumerate(slice_indices):
        r_hd = np.asarray(r_hd_history[slice_index], dtype=float)
        theta_plot, r_hd_plot = collapse_activity_by_theta(theta_hd_pref, r_hd)
        theta_bump = pva_decode(theta_hd_pref, r_hd)
        theta_peak = peak_decode(theta_hd_pref, r_hd)
        bump_strength = pva_vector_strength(theta_hd_pref, r_hd)
        if np.isfinite(theta_bump):
            label = (
                f"t={time[slice_index]:.2f} s, PVA={theta_bump:.2f} rad, "
                f"peak={theta_peak:.2f} rad, |PVA|={bump_strength:.2f}"
            )
            axis.axvline(theta_bump, color=color_values[color_index], linewidth=0.9, alpha=0.45)
            if np.isfinite(theta_peak):
                axis.axvline(
                    theta_peak,
                    color=color_values[color_index],
                    linewidth=0.9,
                    alpha=0.65,
                    linestyle="--",
                )
        else:
            label = f"t={time[slice_index]:.2f} s, PVA=nan"
        axis.plot(theta_plot, r_hd_plot, color=color_values[color_index], linewidth=1.4, label=label)
    axis.set_title(title)
    axis.set_xlabel("HD preferred direction theta_HD [rad]")
    axis.set_ylabel("normalized HD firing rate [a.u.]")
    axis.set_xlim(-np.pi, np.pi)
    _set_radian_ticks(axis, which="x")
    axis.legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)
