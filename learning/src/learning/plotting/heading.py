"""Heading and error plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import circular_difference, unwrap_heading_trace
from learning.analysis.metrics import linear_fit_slope_intercept


def _shade_dark_phase(axis, *, time: np.ndarray, phase_id: np.ndarray | None) -> None:
    if phase_id is None or np.asarray(phase_id).size != time.size or time.size == 0:
        return
    dark_mask = np.asarray(phase_id, dtype=float) == 1.0
    if not np.any(dark_mask):
        return
    dark_indices = np.flatnonzero(dark_mask)
    start_index = int(dark_indices[0])
    end_index = int(dark_indices[-1])
    axis.axvspan(
        float(time[start_index]),
        float(time[end_index]),
        color="#2f2f66",
        alpha=0.10,
        linewidth=0.0,
    )


def plot_true_vs_decoded_heading(
    *,
    time: np.ndarray,
    theta_true: np.ndarray,
    theta_hd_decoded: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_decoded_peak: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
    angle_unit: str = "deg",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
    if angle_unit == "pi":
        angle_scale = 1.0 / np.pi
        y_label = "unwrapped heading [pi rad]"
    elif angle_unit == "rad":
        angle_scale = 1.0
        y_label = "unwrapped heading angle [rad]"
    elif angle_unit == "deg":
        angle_scale = 180.0 / np.pi
        y_label = "unwrapped heading angle [deg]"
    else:
        raise ValueError(f"Unknown angle_unit: {angle_unit}")
    axis.plot(time, angle_scale * unwrap_heading_trace(theta_true), label="true heading", linewidth=1.8)
    axis.plot(
        time,
        angle_scale * unwrap_heading_trace(theta_hd_decoded),
        label="PVA decode",
        linewidth=1.4,
    )
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        axis.plot(
            time,
            angle_scale * unwrap_heading_trace(theta_hd_decoded_peak),
            label="peak decode",
            linewidth=1.1,
            linestyle="--",
        )
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(y_label)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decoded_vs_true_heading_panels(
    *,
    time: np.ndarray,
    theta_true: np.ndarray,
    theta_hd_decoded: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_decoded_peak: np.ndarray | None = None,
) -> None:
    """Plot heading and circular error in stacked panels for a short window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if time.size == 0:
        fig, axis = plt.subplots(figsize=(7.0, 3.5))
        axis.text(0.5, 0.5, "No heading history", ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    source_start_time = float(time[0])
    source_end_time = float(time[-1])
    plot_time = time - float(time[0])
    true_unwrapped = unwrap_heading_trace(theta_true)
    pva_error = circular_difference(theta_hd_decoded, theta_true)
    pva_aligned = true_unwrapped + pva_error
    peak_error = None
    peak_aligned = None
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        peak_error = circular_difference(theta_hd_decoded_peak, theta_true)
        peak_aligned = true_unwrapped + peak_error

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90, hspace=0.18)
    axes[0].plot(plot_time, true_unwrapped / np.pi, label="true heading", linewidth=1.6)
    axes[0].plot(plot_time, pva_aligned / np.pi, label="PVA decode", linewidth=1.3)
    if peak_aligned is not None:
        axes[0].plot(
            plot_time,
            peak_aligned / np.pi,
            label="peak decode",
            linewidth=1.1,
            linestyle="--",
        )
    axes[1].plot(plot_time, pva_error / np.pi, label="PVA error", color="#3b6ea8", linewidth=1.4)
    if peak_error is not None:
        axes[1].plot(
            plot_time,
            peak_error / np.pi,
            label="peak error",
            color="#d18f00",
            linewidth=1.1,
            linestyle="--",
        )
    axes[0].set_title(f"{title}\nsource heatmap t={source_start_time:.2f}-{source_end_time:.2f} s")
    axes[0].set_ylabel("heading [pi rad]")
    axes[1].set_ylabel("decoded - true [pi rad]")
    axes[1].set_xlabel("time from window start [s]")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axes[1].set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pi_error(
    *,
    time: np.ndarray,
    pi_error: np.ndarray,
    path: str | Path,
    title: str = "PI error",
    phase_id: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86)
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
    axis.plot(time, pi_error, color="#3b6ea8", linewidth=1.5)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylim(-np.pi, np.pi)
    axis.set_yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    axis.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("decoded - true heading error [rad]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_gain_curve(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    path: str | Path,
    title: str = "Velocity gain",
    decoded_velocity_peak: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.5, 4.0))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.88)
    axis.scatter(commanded_velocity, decoded_velocity, color="#7c4d79", label="PVA decode")
    annotation_lines: list[str] = []
    if commanded_velocity.size >= 2 and np.all(np.isfinite(decoded_velocity)):
        slope, intercept = linear_fit_slope_intercept(commanded_velocity, decoded_velocity)
        velocity_grid = np.linspace(float(np.min(commanded_velocity)), float(np.max(commanded_velocity)), 100)
        axis.plot(velocity_grid, slope * velocity_grid + intercept, color="#7c4d79", linewidth=1.2)
        annotation_lines.append(f"PVA gain={slope:.3f}")
        annotation_lines.append(f"PVA intercept={intercept:.3f}")
    if decoded_velocity_peak is not None:
        decoded_velocity_peak = np.asarray(decoded_velocity_peak, dtype=float)
        if decoded_velocity_peak.size == commanded_velocity.size:
            axis.scatter(
                commanded_velocity,
                decoded_velocity_peak,
                color="#d18f00",
                marker="x",
                label="peak decode",
            )
            if commanded_velocity.size >= 2 and np.all(np.isfinite(decoded_velocity_peak)):
                slope, intercept = linear_fit_slope_intercept(commanded_velocity, decoded_velocity_peak)
                velocity_grid = np.linspace(float(np.min(commanded_velocity)), float(np.max(commanded_velocity)), 100)
                axis.plot(
                    velocity_grid,
                    slope * velocity_grid + intercept,
                    color="#d18f00",
                    linewidth=1.2,
                    linestyle="--",
                )
                annotation_lines.append(f"peak gain={slope:.3f}")
    axis.axline((0.0, 0.0), slope=1.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("commanded angular velocity [rad/s]")
    axis.set_ylabel("decoded bump angular velocity [rad/s]")
    if annotation_lines:
        axis.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
