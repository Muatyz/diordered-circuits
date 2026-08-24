"""Heading and error plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from learning.common.angles import circular_difference, wrap_angle
from learning.analysis.metrics import (
    classify_endpoint_map_fixed_points,
    linear_fit_slope_intercept,
    summarize_velocity_gain,
)


DARKNESS_PHASE_ID = 1.0


def _shade_endpoint_angle_band(
    axis: plt.Axes,
    *,
    angle_deg: float,
    half_width_deg: float,
    color: str,
) -> None:
    """Shade a periodic horizontal FP marker band without crossing bounds."""
    lower = float(angle_deg - half_width_deg)
    upper = float(angle_deg + half_width_deg)
    intervals = [(lower, upper)]
    if lower < -180.0:
        intervals = [(-180.0, upper), (lower + 360.0, 180.0)]
    elif upper > 180.0:
        intervals = [(lower, 180.0), (-180.0, upper - 360.0)]
    for interval_lower, interval_upper in intervals:
        axis.axhspan(
            interval_lower,
            interval_upper,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=0,
        )


def _plot_endpoint_angle_line(
    axis: plt.Axes,
    *,
    angle_deg: float,
    color: str,
    linestyle: str,
) -> None:
    """Mark an inferred circular angle without implying a finite-width region."""
    axis.axhline(
        float(angle_deg),
        color=color,
        linewidth=0.55,
        linestyle=linestyle,
        alpha=0.55,
        zorder=0.2,
    )


def _plot_horizontal_reference(
    axis: plt.Axes,
    *,
    x_values: np.ndarray,
    y_value: float = 0.0,
    color: str = "black",
    linewidth: float = 0.7,
    alpha: float = 0.35,
) -> None:
    if np.asarray(x_values).size == 0:
        return
    axis.plot(
        [float(np.nanmin(x_values)), float(np.nanmax(x_values))],
        [y_value, y_value],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def _set_pi_y_axis(axis: plt.Axes) -> None:
    axis.set_ylim(-np.pi, np.pi)
    axis.set_yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    axis.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])


def _wrapped_trace_for_axis(theta_trace: np.ndarray) -> np.ndarray:
    theta_wrapped = np.asarray(wrap_angle(theta_trace), dtype=float)
    if theta_wrapped.size < 2:
        return theta_wrapped
    plot_trace = theta_wrapped.copy()
    jump_mask = np.abs(np.diff(theta_wrapped)) > np.pi
    plot_trace[np.flatnonzero(jump_mask) + 1] = np.nan
    return plot_trace


def _plot_wrapped_heading_trace(
    axis: plt.Axes,
    *,
    time: np.ndarray,
    theta_trace: np.ndarray,
    label: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    axis.plot(
        time,
        _wrapped_trace_for_axis(theta_trace),
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def _dark_phase_intervals(time: np.ndarray, phase_id: np.ndarray | None) -> list[tuple[float, float]]:
    if phase_id is None:
        return []
    time = np.asarray(time, dtype=float)
    phase_id = np.asarray(phase_id, dtype=float)
    if time.ndim != 1 or phase_id.ndim != 1 or time.size != phase_id.size or time.size == 0:
        return []
    dark_mask = np.isfinite(time) & np.isfinite(phase_id) & np.isclose(phase_id, DARKNESS_PHASE_ID)
    dark_indices = np.flatnonzero(dark_mask)
    if dark_indices.size == 0:
        return []

    intervals: list[tuple[float, float]] = []
    split_points = np.flatnonzero(np.diff(dark_indices) > 1) + 1
    for segment_indices in np.split(dark_indices, split_points):
        start_index = int(segment_indices[0])
        end_index = int(segment_indices[-1])
        start_time = float(time[start_index])
        end_time = float(time[end_index])
        if start_index > 0 and np.isfinite(time[start_index - 1]):
            start_time = 0.5 * (float(time[start_index - 1]) + start_time)
        if end_index + 1 < time.size and np.isfinite(time[end_index + 1]):
            end_time = 0.5 * (end_time + float(time[end_index + 1]))
        if end_time <= start_time:
            end_time = float(time[end_index])
        if end_time > start_time:
            intervals.append((start_time, end_time))
    return intervals


def _shade_dark_phase(axis, *, time: np.ndarray, phase_id: np.ndarray | None) -> None:
    intervals = _dark_phase_intervals(time, phase_id)
    if not intervals:
        return
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    y_min, y_max = map(float, y_limits)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if y_max <= y_min:
        y_min -= 0.5
        y_max += 0.5
    for start_time, end_time in intervals:
        shade_rgba = np.ones((2, 2, 4), dtype=float)
        shade_rgba[..., :3] = 0.88
        shade_rgba[..., 3] = 0.65
        axis.imshow(
            shade_rgba,
            extent=(float(start_time), float(end_time), y_min, y_max),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            zorder=0.1,
        )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)


def plot_true_vs_decoded_heading(
    *,
    time: np.ndarray,
    theta_true: np.ndarray,
    theta_hd_decoded: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_decoded_peak: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    _plot_wrapped_heading_trace(axis, time=time, theta_trace=theta_true, label="true heading", linewidth=1.8)
    _plot_wrapped_heading_trace(
        axis,
        time=time,
        theta_trace=theta_hd_decoded,
        label="PVA decode",
        linewidth=1.4,
    )
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        _plot_wrapped_heading_trace(
            axis,
            time=time,
            theta_trace=theta_hd_decoded_peak,
            label="peak decode",
            linewidth=1.1,
            linestyle="--",
        )
    _set_pi_y_axis(axis)
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("heading angle [rad]")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_cartesian_angular_ring(
    axis: plt.Axes,
    *,
    start_angle: float,
    end_angle: float,
    color: str,
    linewidth: float = 15.0,
    alpha: float = 1.0,
) -> None:
    """Draw one counter-clockwise circular arc without a polar projection."""
    end_unwrapped = float(end_angle)
    while end_unwrapped <= start_angle:
        end_unwrapped += 2.0 * np.pi
    angle = np.linspace(start_angle, end_unwrapped, 160)
    axis.plot(
        np.cos(angle),
        np.sin(angle),
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        alpha=alpha,
    )


def _format_cartesian_angular_ring(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-1.22, 1.22)
    axis.set_ylim(-1.22, 1.22)
    axis.axis("off")
    for angle, label in (
        (0.0, r"$0$"),
        (0.5 * np.pi, r"$\pi/2$"),
        (np.pi, r"$\pm\pi$"),
        (-0.5 * np.pi, r"$-\pi/2$"),
    ):
        axis.text(
            1.14 * np.cos(angle),
            1.14 * np.sin(angle),
            label,
            ha="center",
            va="center",
            fontsize=8,
        )


def _ring_axes_for_velocity_count(
    velocity_count: int,
    *,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    column_count = velocity_count if velocity_count <= 3 else 2
    row_count = int(np.ceil(velocity_count / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.5 * column_count, 4.5 * row_count + 0.6),
        squeeze=False,
    )
    for axis in axes.ravel()[velocity_count:]:
        axis.axis("off")
    fig.suptitle(title, y=0.98)
    return fig, axes.ravel()[:velocity_count]


def plot_actual_fp_basin_rings(
    *,
    summary: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Plot actual phase-flow roots and forward stable basins."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(summary["commanded_velocity"], dtype=float)
    fig, axes = _ring_axes_for_velocity_count(
        velocity.size,
        title="Actual fixed points and forward-time stable basins",
    )
    stable_color = "#20a7a0"
    neutral_color = "#b8b8b8"
    fixed_slot = np.asarray(summary["fixed_point_velocity_slot"], dtype=int)
    fixed_angle = np.asarray(summary["fixed_point_angle"], dtype=float)
    fixed_stability = np.asarray(summary["fixed_point_stability"], dtype=np.int8)
    basin_slot = np.asarray(summary["basin_velocity_slot"], dtype=int)
    for slot, axis in enumerate(axes):
        _draw_cartesian_angular_ring(
            axis,
            start_angle=-np.pi,
            end_angle=np.pi,
            color=neutral_color,
        )
        for basin_index in np.flatnonzero(basin_slot == slot):
            left = float(summary["basin_left_boundary"][basin_index])
            stable = float(summary["basin_stable_angle"][basin_index])
            right = float(summary["basin_right_boundary"][basin_index])
            while stable < left:
                stable += 2.0 * np.pi
            while right < stable:
                right += 2.0 * np.pi
            _draw_cartesian_angular_ring(
                axis,
                start_angle=left,
                end_angle=right,
                color=stable_color,
            )
        current = fixed_slot == slot
        current_angle = fixed_angle[current]
        current_stability = fixed_stability[current]
        for stability, marker, size in ((-1, "o", 34), (1, "x", 42)):
            selected = current_stability == stability
            if not np.any(selected):
                continue
            axis.scatter(
                np.cos(current_angle[selected]),
                np.sin(current_angle[selected]),
                c="black",
                marker=marker,
                s=size,
                linewidths=1.3,
                zorder=5,
            )
        _format_cartesian_angular_ring(axis)
        stable_count = int(np.sum(current_stability == -1))
        unstable_count = int(np.sum(current_stability == 1))
        axis.set_title(
            f"v={velocity[slot]:.2f} rad/s | S={stable_count}, U={unstable_count}\n"
            f"dense discrete field, support={100.0 * summary['sample_support_fraction'][slot]:.0f}%",
            fontsize=9,
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color=stable_color, linewidth=8, label="stable basin"),
            Line2D([0], [0], color="black", marker="o", linestyle="none", label="stable FP"),
            Line2D([0], [0], color="black", marker="x", linestyle="none", label="unstable FP / separatrix"),
        ],
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="lower center",
    )
    fig.subplots_adjust(bottom=0.12, top=0.88, hspace=0.34, wspace=0.24)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_dense_probe_trajectories(
    *,
    summary: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Plot every independent dense-probe trajectory without model fitting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(summary["commanded_velocity"], dtype=float)
    time = np.asarray(summary["probe_time"], dtype=float)
    theta = np.asarray(summary["probe_theta_pva"], dtype=float)
    column_count = 2 if velocity.size > 1 else 1
    row_count = int(np.ceil(velocity.size / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.0 * column_count, 2.7 * row_count),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for slot, axis in enumerate(axes.ravel()):
        if slot >= velocity.size:
            axis.axis("off")
            continue
        unwrapped = np.unwrap(theta[slot], axis=1)
        aligned = unwrapped + (
            np.asarray(wrap_angle(theta[slot, :, 0])) - unwrapped[:, 0]
        )[:, None]
        axis.plot(time, aligned.T, color="#4c78a8", linewidth=0.45, alpha=0.12)
        axis.set_title(
            f"v={velocity[slot]:.2f} rad/s | {theta.shape[1]} independent probes",
            fontsize=9,
        )
        axis.set_ylabel(r"unwrapped phase $\theta(t)$ [rad]")
        axis.grid(alpha=0.15)
    for axis in axes[-1]:
        if axis.axison:
            axis.set_xlabel("time in darkness [s]")
    fig.suptitle("Dense frozen-weight phase-flow probe trajectories")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_phase_flow_diagnostics(
    *,
    summary: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Plot direct binned flow and ddot(theta)/dot(theta) cross-check."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(summary["commanded_velocity"], dtype=float)
    angle = np.asarray(summary["grid_angle"], dtype=float)
    fig, axes = plt.subplots(
        velocity.size,
        2,
        figsize=(12.0, 2.6 * velocity.size + 0.4),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    fixed_slot = np.asarray(summary["fixed_point_velocity_slot"], dtype=int)
    for slot, (flow_axis, lambda_axis) in enumerate(axes):
        flow = np.asarray(summary["phase_flow"][slot], dtype=float)
        raw_flow = np.asarray(summary["phase_flow_raw"][slot], dtype=float)
        observed = np.isfinite(raw_flow)
        flow_axis.scatter(
            angle[observed],
            raw_flow[observed],
            color="#4c78a8",
            s=7,
            alpha=0.35,
            linewidths=0.0,
            label="bin median",
        )
        flow_axis.plot(
            angle,
            flow,
            color="#1f5f99",
            linewidth=1.5,
            label="circular moving average",
        )
        flow_axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
        current_indices = np.flatnonzero(fixed_slot == slot)
        for fixed_index in current_indices:
            stability = int(summary["fixed_point_stability"][fixed_index])
            marker = "o" if stability == -1 else "x"
            fp_angle = float(summary["fixed_point_angle"][fixed_index])
            flow_axis.scatter(
                [fp_angle],
                [0.0],
                c="black",
                marker=marker,
                s=28,
                zorder=4,
            )
        flow_axis.set_ylabel(r"$F_v(\theta)$ [rad/s]")
        flow_axis.set_title(
            f"v={velocity[slot]:.2f} | support="
            f"{100.0 * summary['sample_support_fraction'][slot]:.0f}%, "
            f"within-bin RMS={summary['residual_rms'][slot]:.3g}",
            fontsize=9,
        )
        flow_axis.grid(alpha=0.18)

        field_derivative = np.asarray(
            summary["phase_flow_derivative"][slot], dtype=float
        )
        lambda_axis.plot(
            angle,
            field_derivative,
            color="#1f5f99",
            linewidth=1.4,
            label=r"$dF_v/d\theta$",
        )
        if not np.isclose(velocity[slot], 0.0):
            empirical_lambda = np.asarray(
                summary["empirical_lambda"][slot], dtype=float
            )
            raw_lambda = np.asarray(
                summary["empirical_lambda_raw"][slot], dtype=float
            )
            raw_observed = np.isfinite(raw_lambda)
            lambda_axis.scatter(
                angle[raw_observed],
                raw_lambda[raw_observed],
                color="#ed8b2c",
                s=6,
                alpha=0.25,
                linewidths=0.0,
            )
            lambda_axis.plot(
                angle,
                empirical_lambda,
                color="#ed8b2c",
                linewidth=1.2,
                label=r"median $\ddot\theta/\dot\theta$",
            )
        else:
            lambda_axis.text(
                0.5,
                0.88,
                r"$\ddot\theta/\dot\theta$ omitted at $v=0$",
                transform=lambda_axis.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
        lambda_axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
        lambda_axis.set_ylabel(r"local rate [s$^{-1}$]")
        lambda_axis.set_title("One-dimensional identity cross-check", fontsize=9)
        lambda_axis.grid(alpha=0.18)
        if slot == 0:
            flow_axis.legend(frameon=False, fontsize=7, loc="upper right")
            lambda_axis.legend(frameon=False, fontsize=7, loc="upper right")

    for axis in axes[-1]:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
        axis.set_xticklabels(
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        )
        axis.set_xlabel(r"bump phase $\theta$")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heading_and_pi_error_panels(
    *,
    time: np.ndarray,
    theta_true: np.ndarray,
    theta_hd_decoded: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_decoded_peak: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
    pi_error: np.ndarray | None = None,
    peak_pi_error: np.ndarray | None = None,
    circular_error_axis: bool = True,
) -> None:
    """Plot heading traces and either absolute or accumulated PI error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.90, hspace=0.16)

    _plot_wrapped_heading_trace(
        axes[0],
        time=time,
        theta_trace=theta_true,
        label="true heading",
        linewidth=1.7,
    )
    _plot_wrapped_heading_trace(
        axes[0],
        time=time,
        theta_trace=theta_hd_decoded,
        label="PVA decode",
        linewidth=1.3,
    )
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        _plot_wrapped_heading_trace(
            axes[0],
            time=time,
            theta_trace=theta_hd_decoded_peak,
            label="peak decode",
            linewidth=1.0,
            linestyle="--",
        )

    pva_error = (
        circular_difference(theta_hd_decoded, theta_true)
        if pi_error is None
        else np.asarray(pi_error, dtype=float)
    )
    if pva_error.shape != np.asarray(time).shape:
        raise ValueError("pi_error must match time")
    axes[1].plot(time, pva_error, label="PVA error", color="#3b6ea8", linewidth=1.3)
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        peak_error = (
            circular_difference(theta_hd_decoded_peak, theta_true)
            if peak_pi_error is None
            else np.asarray(peak_pi_error, dtype=float)
        )
        axes[1].plot(
            time,
            peak_error,
            label="peak error",
            color="#d18f00",
            linewidth=1.0,
            linestyle="--",
        )

    axes[0].set_title(title, fontsize=10)
    axes[0].set_ylabel("heading angle [rad]")
    axes[1].set_ylabel(
        "decoded - true [rad]"
        if circular_error_axis
        else "release-relative accumulated error [rad]"
    )
    axes[1].set_xlabel("time [s]")
    axes[0].legend(
        frameon=True,
        framealpha=0.82,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.02),
        ncol=2,
    )
    axes[1].legend(
        frameon=True,
        framealpha=0.82,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.02),
        ncol=2,
    )
    for axis in axes:
        _plot_horizontal_reference(axis, x_values=time)
    _set_pi_y_axis(axes[0])
    if circular_error_axis:
        _set_pi_y_axis(axes[1])
    for axis in axes:
        _shade_dark_phase(axis, time=time, phase_id=phase_id)
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
    pva_error = circular_difference(theta_hd_decoded, theta_true)
    peak_error = None
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        peak_error = circular_difference(theta_hd_decoded_peak, theta_true)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90, hspace=0.18)
    _plot_wrapped_heading_trace(
        axes[0],
        time=plot_time,
        theta_trace=theta_true,
        label="true heading",
        linewidth=1.6,
    )
    _plot_wrapped_heading_trace(
        axes[0],
        time=plot_time,
        theta_trace=theta_hd_decoded,
        label="PVA decode",
        linewidth=1.3,
    )
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        _plot_wrapped_heading_trace(
            axes[0],
            time=plot_time,
            theta_trace=theta_hd_decoded_peak,
            label="peak decode",
            linewidth=1.1,
            linestyle="--",
        )
    axes[1].plot(plot_time, pva_error, label="PVA error", color="#3b6ea8", linewidth=1.4)
    if peak_error is not None:
        axes[1].plot(
            plot_time,
            peak_error,
            label="peak error",
            color="#d18f00",
            linewidth=1.1,
            linestyle="--",
        )
    axes[0].set_title(f"{title}\nsource heatmap t={source_start_time:.2f}-{source_end_time:.2f} s")
    axes[0].set_ylabel("heading angle [rad]")
    axes[1].set_ylabel("decoded - true [rad]")
    axes[1].set_xlabel("time from window start [s]")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        _plot_horizontal_reference(axis, x_values=plot_time)
        _set_pi_y_axis(axis)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pi_error(
    *,
    time: np.ndarray,
    pi_error: np.ndarray,
    path: str | Path,
    title: str = "PI error",
    phase_id: np.ndarray | None = None,
    circular_axis: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86)
    axis.plot(time, pi_error, color="#3b6ea8", linewidth=1.5)
    _plot_horizontal_reference(axis, x_values=time, linewidth=0.8, alpha=0.5)
    if circular_axis:
        _set_pi_y_axis(axis)
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(
        "decoded - true heading error [rad]"
        if circular_axis
        else "release-relative accumulated PI error [rad]"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_constant_velocity_pi_error_grid(
    *,
    time: np.ndarray,
    pi_error: np.ndarray,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    path: str | Path,
    phase_id: np.ndarray | None = None,
    title: str = "Constant-velocity PI error across speeds",
) -> None:
    """Plot matched PI-error traces in a compact two-column comparison."""
    time = np.asarray(time, dtype=float)
    pi_error = np.asarray(pi_error, dtype=float)
    commanded_velocity = np.asarray(commanded_velocity, dtype=float)
    decoded_velocity = np.asarray(decoded_velocity, dtype=float)
    if time.ndim != 1:
        raise ValueError("constant PI time must be one-dimensional")
    if pi_error.ndim != 2 or pi_error.shape[1] != time.size:
        raise ValueError("constant PI error must have shape (velocity, time)")
    velocity_count = pi_error.shape[0]
    if (
        commanded_velocity.shape != (velocity_count,)
        or decoded_velocity.shape != (velocity_count,)
    ):
        raise ValueError("constant PI velocity arrays must match the trace count")
    if velocity_count == 0:
        raise ValueError("at least one constant PI velocity trace is required")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    column_count = 2
    row_count = int(np.ceil(velocity_count / column_count))
    # ``sharey`` is deliberately disabled: the release-relative accumulated
    # error is unbounded and grows linearly in time whenever the decoded
    # velocity gain differs from one.  Low-speed pinning traces can reach
    # hundreds of degrees over a long darkness window, and sharing one y-axis
    # across all speeds flattens the informative mid-speed traces into a line
    # near zero.  Each subplot therefore auto-scales to its own error range.
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(9.0, 3.0 * row_count),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.11,
        top=0.88,
        hspace=0.35,
        wspace=0.18,
    )
    axes_flat = axes.ravel()
    for velocity_index, axis in enumerate(axes_flat):
        if velocity_index >= velocity_count:
            axis.set_axis_off()
            continue
        command = float(commanded_velocity[velocity_index])
        decoded = float(decoded_velocity[velocity_index])
        gain = decoded / command if not np.isclose(command, 0.0) else float("nan")
        axis.plot(
            time,
            np.rad2deg(pi_error[velocity_index]),
            color="#3b6ea8",
            linewidth=1.35,
        )
        _plot_horizontal_reference(axis, x_values=time, linewidth=0.8, alpha=0.5)
        _shade_dark_phase(axis, time=time, phase_id=phase_id)
        axis.set_title(
            f"command {np.rad2deg(command):+.0f} deg/s | "
            f"PVA {np.rad2deg(decoded):+.1f} deg/s | gain {gain:.3f}"
        )
        axis.grid(alpha=0.15, linewidth=0.5)
    fig.suptitle(title)
    fig.text(0.5, 0.025, "time [s]", ha="center")
    fig.text(
        0.018,
        0.5,
        "release-relative accumulated PI error [deg]",
        va="center",
        rotation="vertical",
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_gain_curve(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    path: str | Path,
    title: str = "Velocity gain",
    decoded_velocity_peak: np.ndarray | None = None,
    decoded_velocity_visual: np.ndarray | None = None,
    decoded_velocity_visual_peak: np.ndarray | None = None,
) -> None:
    """Plot sampled velocity responses and only show defensible linear fits.

    Raw samples are sorted and connected so saturation, locking, and other
    nonlinear regimes remain visible. A global fit is drawn only when its
    coefficient of determination reaches ``0.95``.
    """
    commanded_velocity = np.asarray(commanded_velocity, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.5, 4.0))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.88)
    annotation_lines: list[str] = []

    def plot_decoded_velocity_series(
        *,
        decoded_values: np.ndarray,
        color: str,
        label: str,
        marker: str = "o",
        linestyle: str = "-",
        annotation_prefix: str,
        markersize: float = 3.5,
    ) -> None:
        nonlocal annotation_lines
        decoded_values = np.asarray(decoded_values, dtype=float)
        if decoded_values.size != commanded_velocity.size:
            return
        finite_mask = np.isfinite(commanded_velocity) & np.isfinite(decoded_values)
        if not np.any(finite_mask):
            return
        order = np.argsort(commanded_velocity[finite_mask])
        x_finite = commanded_velocity[finite_mask][order]
        y_finite = decoded_values[finite_mask][order]
        axis.plot(
            x_finite,
            y_finite,
            color=color,
            linewidth=1.0,
            marker=marker,
            markersize=markersize,
            markeredgewidth=0.7,
            linestyle="-",
            label=label,
        )
        if x_finite.size < 2:
            return
        summary = summarize_velocity_gain(
            commanded_velocity=x_finite,
            decoded_velocity=y_finite,
        )
        slope = summary["gain"]
        intercept = summary["intercept"]
        r_squared = summary["r_squared"]
        annotation_lines.append(f"{annotation_prefix} global g={slope:.3f}, R2={r_squared:.3f}")
        if np.isfinite(r_squared) and r_squared >= 0.95:
            velocity_grid = np.linspace(float(np.min(x_finite)), float(np.max(x_finite)), 100)
            axis.plot(
                velocity_grid,
                slope * velocity_grid + intercept,
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
                label=f"{label} linear fit",
            )
        else:
            annotation_lines.append(f"{annotation_prefix} nonlinear: fit hidden")

    plot_decoded_velocity_series(
        decoded_values=decoded_velocity,
        color="#7c4d79",
        label="darkness PVA",
        annotation_prefix="dark",
    )
    if decoded_velocity_visual is not None:
        plot_decoded_velocity_series(
            decoded_values=decoded_velocity_visual,
            color="#2f8f6f",
            label="visual PVA",
            marker="s",
            annotation_prefix="visual",
        )
    if decoded_velocity_peak is not None:
        plot_decoded_velocity_series(
            decoded_values=decoded_velocity_peak,
            color="#d18f00",
            label="darkness peak",
            marker="x",
            linestyle="--",
            annotation_prefix="dark peak",
        )
    if decoded_velocity_visual_peak is not None:
        plot_decoded_velocity_series(
            decoded_values=decoded_velocity_visual_peak,
            color="#64b5a0",
            label="visual peak",
            marker="+",
            linestyle=":",
            annotation_prefix="visual peak",
        )
    if commanded_velocity.size:
        ideal_min = float(np.nanmin(commanded_velocity))
        ideal_max = float(np.nanmax(commanded_velocity))
        axis.plot(
            [ideal_min, ideal_max],
            [ideal_min, ideal_max],
            color="gray",
            linewidth=1.0,
            linestyle="--",
        )
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


def plot_effective_diffusion_msd(
    *,
    time: np.ndarray,
    theta_decoded: np.ndarray,
    theta_reference: float,
    diffusion_coefficient: float,
    path: str | Path,
    title: str = "Effective angular diffusion diagnostic",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    theta_decoded = np.asarray(theta_decoded, dtype=float)
    finite_mask = np.isfinite(time) & np.isfinite(theta_decoded)
    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.86)
    if np.count_nonzero(finite_mask) < 3:
        axis.text(0.5, 0.5, "Not enough finite heading samples", ha="center", va="center")
        axis.set_axis_off()
    else:
        plot_time = time[finite_mask] - float(time[finite_mask][0])
        displacement = circular_difference(theta_decoded[finite_mask], theta_reference)
        squared_displacement = displacement**2
        axis.plot(plot_time, squared_displacement, color="#3b6ea8", linewidth=1.3, label="single-trace MSD")
        if np.isfinite(diffusion_coefficient):
            axis.plot(
                plot_time,
                2.0 * diffusion_coefficient * plot_time,
                color="#d18f00",
                linewidth=1.2,
                linestyle="--",
                label=f"2Dt, D={diffusion_coefficient:.3g} rad^2/s",
            )
        axis.set_xlabel("time in darkness [s]")
        axis.set_ylabel("wrapped displacement^2 [rad^2]")
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    axis.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diffusion_noise_sweep(
    *,
    noise_std: np.ndarray,
    diffusion_coefficient: np.ndarray,
    path: str | Path,
    title: str = "Original-protocol bump diffusion versus test noise",
) -> None:
    """Plot the endpoint ``Var[Delta theta] / T_dark`` noise sweep."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    noise_std = np.asarray(noise_std, dtype=float)
    diffusion_coefficient = np.asarray(diffusion_coefficient, dtype=float)
    valid = (
        noise_std.ndim == diffusion_coefficient.ndim == 1
        and noise_std.shape == diffusion_coefficient.shape
        and noise_std.size > 0
    )
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.87)
    if not valid:
        axis.text(0.5, 0.5, "No valid diffusion sweep", ha="center", va="center")
        axis.set_axis_off()
    else:
        diffusion_deg2_s = np.rad2deg(1.0) ** 2 * diffusion_coefficient
        finite = np.isfinite(noise_std) & np.isfinite(diffusion_deg2_s)
        axis.plot(
            noise_std[finite],
            diffusion_deg2_s[finite],
            color="#3b6ea8",
            marker="o",
            markersize=4.0,
            linewidth=1.3,
        )
        axis.set_xlabel("test input-noise std")
        axis.set_ylabel(r"endpoint $D$ [deg$^2$/s]")
        axis.grid(alpha=0.25)
    axis.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ensemble_diffusion_variance(
    *,
    time: np.ndarray,
    displacement_mean: np.ndarray,
    displacement_variance: np.ndarray,
    diffusion_coefficient: float,
    systematic_drift_velocity: float,
    anomalous_diffusion_fit: np.ndarray | None = None,
    anomalous_diffusion_exponent: float = float("nan"),
    generalized_diffusion_coefficient: float = float("nan"),
    anomalous_diffusion_log_r_squared: float = float("nan"),
    anomalous_diffusion_fit_start_time: float = float("nan"),
    anomalous_diffusion_fit_end_time: float = float("nan"),
    n_trials: int | None = None,
    path: str | Path,
    title: str = "Zero-velocity bump ensemble: drift and diffusion",
) -> None:
    """Plot the two ensemble moments used by the Vafidis diffusion estimate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    displacement_mean = np.asarray(displacement_mean, dtype=float)
    displacement_variance = np.asarray(displacement_variance, dtype=float)
    valid = (
        time.ndim == displacement_mean.ndim == displacement_variance.ndim == 1
        and time.size == displacement_mean.size == displacement_variance.size
        and time.size >= 2
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.90, hspace=0.52)
    if not valid:
        axes[0].text(0.5, 0.5, "No valid ensemble history", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    else:
        diffusion_coefficient_deg2_s = float("nan")
        generalized_coefficient_deg2 = float("nan")
        if np.isfinite(generalized_diffusion_coefficient):
            generalized_coefficient_deg2 = float(
                np.rad2deg(1.0) ** 2 * generalized_diffusion_coefficient
            )
        axes[0].plot(time, displacement_variance, color="#3b6ea8", label="Var[Delta theta]")
        if np.isfinite(diffusion_coefficient):
            diffusion_coefficient_deg2_s = float(
                np.rad2deg(1.0) ** 2 * diffusion_coefficient
            )
            axes[0].plot(
                time,
                diffusion_coefficient * time,
                color="#d18f00",
                linestyle="--",
                label="linear reference: D t",
            )
        if anomalous_diffusion_fit is not None:
            anomalous_fit = np.asarray(anomalous_diffusion_fit, dtype=float)
            if anomalous_fit.shape == time.shape:
                fit_mask = np.isfinite(anomalous_fit)
                if np.isfinite(anomalous_diffusion_fit_start_time):
                    fit_mask &= time >= anomalous_diffusion_fit_start_time
                if np.isfinite(anomalous_diffusion_fit_end_time):
                    fit_mask &= time <= anomalous_diffusion_fit_end_time
                if np.any(fit_mask):
                    axes[0].plot(
                        time[fit_mask],
                        anomalous_fit[fit_mask],
                        color="#c44e52",
                        linewidth=1.6,
                        linestyle=":",
                        label=r"power-law fit: $D_\alpha t^\alpha$",
                    )
        annotation_lines: list[str] = []
        if np.isfinite(diffusion_coefficient):
            annotation_lines.append(
                (
                    f"D={diffusion_coefficient:.3g} rad^2/s "
                    f"= {diffusion_coefficient_deg2_s:.3g} deg^2/s"
                )
            )
        if np.isfinite(anomalous_diffusion_exponent):
            behavior = (
                "subdiffusive"
                if anomalous_diffusion_exponent < 0.9
                else "superdiffusive"
                if anomalous_diffusion_exponent > 1.1
                else "approximately normal"
            )
            annotation_lines.extend(
                [
                    f"alpha={anomalous_diffusion_exponent:.3f} ({behavior})",
                    f"log-log R^2={anomalous_diffusion_log_r_squared:.3f}",
                ]
            )
        if np.isfinite(generalized_diffusion_coefficient):
            annotation_lines.append(
                (
                    f"D_alpha={generalized_diffusion_coefficient:.3g} rad^2/s^alpha "
                    f"= {generalized_coefficient_deg2:.3g} deg^2/s^alpha"
                )
            )
        if annotation_lines:
            axes[0].text(
                0.03,
                0.97,
                "\n".join(annotation_lines),
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                },
            )
        axes[0].set_ylabel("ensemble variance [rad^2]")
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        axes[1].plot(time, displacement_mean, color="#7c4d79", label="mean displacement")
        if np.isfinite(systematic_drift_velocity):
            axes[1].plot(
                time,
                systematic_drift_velocity * time,
                color="#2f8f6f",
                linestyle="--",
                label=f"drift fit={systematic_drift_velocity:.3g} rad/s",
            )
        axes[1].set_xlabel("time in darkness [s]")
        axes[1].set_ylabel("ensemble mean [rad]")
        axes[1].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25)
    figure_title = title if n_trials is None else f"{title} (n={n_trials} trials)"
    fig.suptitle(figure_title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pi_error_ensemble(
    *,
    time: np.ndarray,
    pi_error_mean: np.ndarray,
    pi_error_sem: np.ndarray,
    systematic_drift_velocity: float,
    drift_intercept: float,
    n_trials: int,
    path: str | Path,
    title: str = "OU path-integration error ensemble (PVA/COM decode)",
) -> None:
    """Plot ensemble-mean PI error with SEM and its systematic drift fit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    mean_error = np.asarray(pi_error_mean, dtype=float)
    sem = np.asarray(pi_error_sem, dtype=float)
    fig, axis = plt.subplots(figsize=(7.0, 3.6))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.84)
    axis.plot(time, mean_error, color="#3b6ea8", linewidth=1.5, label="ensemble mean")
    if sem.shape == mean_error.shape:
        axis.fill_between(
            time,
            mean_error - sem,
            mean_error + sem,
            color="#3b6ea8",
            alpha=0.22,
            linewidth=0.0,
            label="SEM",
        )
    if np.isfinite(systematic_drift_velocity) and np.isfinite(drift_intercept):
        axis.plot(
            time,
            drift_intercept + systematic_drift_velocity * time,
            color="#c44e52",
            linewidth=1.2,
            linestyle="--",
            label=f"mean drift={systematic_drift_velocity:.3g} rad/s",
        )
    _plot_horizontal_reference(axis, x_values=time, linewidth=0.8, alpha=0.45)
    axis.set_title(f"{title} (n={int(n_trials)})")
    axis.set_xlabel("time in darkness [s]")
    axis.set_ylabel("release-relative accumulated PI error [rad]")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bump_attractor_decoder_trajectories(
    *,
    time: np.ndarray,
    theta_initial: np.ndarray,
    theta_pva: np.ndarray,
    theta_peak: np.ndarray,
    theta_overlap: np.ndarray,
    path: str | Path,
    title: str = "Zero-input bump attractor landscape",
    cue_time: np.ndarray | None = None,
    cue_theta_pva: np.ndarray | None = None,
    cue_theta_peak: np.ndarray | None = None,
    cue_theta_overlap: np.ndarray | None = None,
    endpoint_probe_theta_initial: np.ndarray | None = None,
    endpoint_probe_refinement_level: np.ndarray | None = None,
    endpoint_probe_theta_release_pva: np.ndarray | None = None,
    endpoint_probe_theta_release_peak: np.ndarray | None = None,
    endpoint_probe_theta_release_overlap: np.ndarray | None = None,
    endpoint_probe_theta_final_pva: np.ndarray | None = None,
    endpoint_probe_theta_final_peak: np.ndarray | None = None,
    endpoint_probe_theta_final_overlap: np.ndarray | None = None,
    decoder_names: tuple[str, ...] | None = None,
    dpi: int = 160,
) -> None:
    """Compare decoder trajectories through visual cue and darkness."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    theta_initial = np.asarray(theta_initial, dtype=float)
    decoder_trace_by_name = {
        "pva": ("PVA", np.asarray(theta_pva, dtype=float)),
        "peak": ("peak neuron", np.asarray(theta_peak, dtype=float)),
        "overlap": (
            "Clark overlap (Eq. 6)",
            np.asarray(theta_overlap, dtype=float),
        ),
    }
    selected_decoder_names = (
        ("pva", "peak", "overlap")
        if decoder_names is None
        else tuple(name.lower() for name in decoder_names)
    )
    unknown_decoder_names = set(selected_decoder_names) - set(
        decoder_trace_by_name
    )
    if not selected_decoder_names or unknown_decoder_names:
        raise ValueError(
            "decoder_names must contain pva, peak, and/or overlap"
        )
    if len(set(selected_decoder_names)) != len(selected_decoder_names):
        raise ValueError("decoder_names must not contain duplicates")
    if dpi <= 0:
        raise ValueError("trajectory plot dpi must be positive")
    decoder_traces = [
        (name, *decoder_trace_by_name[name]) for name in selected_decoder_names
    ]
    cue_trace_by_name = {
        "pva": cue_theta_pva,
        "peak": cue_theta_peak,
        "overlap": cue_theta_overlap,
    }
    selected_cue_values = tuple(
        cue_trace_by_name[name] for name in selected_decoder_names
    )
    cue_values = (cue_time, *selected_cue_values)
    cue_history_supplied = all(value is not None for value in cue_values)
    if any(value is not None for value in cue_values) and not cue_history_supplied:
        raise ValueError("cue trajectory plot inputs must be supplied together")
    if cue_history_supplied:
        cue_time = np.asarray(cue_time, dtype=float)
        cue_decoder_traces = [
            np.asarray(value, dtype=float) for value in selected_cue_values
        ]
    else:
        cue_decoder_traces = [None] * len(decoder_traces)
    endpoint_probe_values = (
        endpoint_probe_theta_initial,
        endpoint_probe_refinement_level,
        endpoint_probe_theta_release_pva,
        endpoint_probe_theta_release_peak,
        endpoint_probe_theta_release_overlap,
        endpoint_probe_theta_final_pva,
        endpoint_probe_theta_final_peak,
        endpoint_probe_theta_final_overlap,
    )
    endpoint_probes_supplied = all(
        value is not None for value in endpoint_probe_values
    )
    if any(value is not None for value in endpoint_probe_values) and not (
        endpoint_probes_supplied
    ):
        raise ValueError("refined endpoint probe inputs must be supplied together")
    if endpoint_probes_supplied:
        endpoint_probe_theta_initial = np.asarray(
            endpoint_probe_theta_initial, dtype=float
        )
        endpoint_probe_refinement_level = np.asarray(
            endpoint_probe_refinement_level, dtype=int
        )
        endpoint_probe_release_by_name = {
            "pva": np.asarray(endpoint_probe_theta_release_pva, dtype=float),
            "peak": np.asarray(endpoint_probe_theta_release_peak, dtype=float),
            "overlap": np.asarray(
                endpoint_probe_theta_release_overlap, dtype=float
            ),
        }
        endpoint_probe_final_by_name = {
            "pva": np.asarray(endpoint_probe_theta_final_pva, dtype=float),
            "peak": np.asarray(endpoint_probe_theta_final_peak, dtype=float),
            "overlap": np.asarray(endpoint_probe_theta_final_overlap, dtype=float),
        }
        endpoint_probe_count = endpoint_probe_theta_initial.size
        if (
            endpoint_probe_theta_initial.ndim != 1
            or endpoint_probe_refinement_level.shape != (endpoint_probe_count,)
            or any(
                value.shape != (endpoint_probe_count,)
                for value in (
                    *endpoint_probe_release_by_name.values(),
                    *endpoint_probe_final_by_name.values(),
                )
            )
        ):
            raise ValueError("refined endpoint probe arrays must be matching 1D arrays")
    else:
        endpoint_probe_theta_initial = theta_initial
        endpoint_probe_refinement_level = np.zeros(theta_initial.size, dtype=int)
        endpoint_probe_release_by_name = {
            name: trace[:, 0] for name, _label, trace in decoder_traces
        }
        endpoint_probe_final_by_name = {
            name: trace[:, -1] for name, _label, trace in decoder_traces
        }
    expected_shape = (theta_initial.size, time.size)
    if time.ndim != 1 or theta_initial.ndim != 1 or time.size == 0:
        raise ValueError("attractor trajectory plot requires non-empty time and starts")
    if any(trace.shape != expected_shape for _name, _label, trace in decoder_traces):
        raise ValueError("each decoder trace must have shape (start, time)")
    if cue_history_supplied:
        expected_cue_shape = (theta_initial.size, cue_time.size)
        if cue_time.ndim != 1 or cue_time.size == 0:
            raise ValueError("cue trajectory plot requires non-empty cue_time")
        if not np.isclose(cue_time[-1], 0.0):
            raise ValueError("cue_time must end at cue release time zero")
        if any(trace.shape != expected_cue_shape for trace in cue_decoder_traces):
            raise ValueError("each cue decoder trace must have shape (start, cue_time)")

    decoder_row_count = len(decoder_traces)
    fig, axes = plt.subplots(
        decoder_row_count,
        3,
        figsize=(18.0, 5.4) if decoder_row_count == 1 else (16.5, 10.0),
        sharex="col",
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (
        (decoder_name, decoder_label, theta_trace),
        cue_theta_trace,
    ) in enumerate(
        zip(decoder_traces, cue_decoder_traces, strict=True)
    ):
        trajectory_axis = axes[row_index, 0]
        endpoint_axis = axes[row_index, 1]
        release_endpoint_axis = axes[row_index, 2]
        if cue_theta_trace is not None:
            full_time = np.concatenate([cue_time[:-1], time])
            full_theta_trace = np.concatenate(
                [cue_theta_trace[:, :-1], theta_trace],
                axis=1,
            )
            cue_relative_trace = circular_difference(
                full_theta_trace,
                theta_initial[:, None],
            )
            absolute_trajectory = (
                theta_initial[:, None]
                + np.unwrap(cue_relative_trace, axis=1)
            )
        else:
            full_time = time
            initial_decoded = theta_initial + circular_difference(
                theta_trace[:, 0],
                theta_initial,
            )
            displacement = (
                np.unwrap(theta_trace, axis=1)
                - np.unwrap(theta_trace, axis=1)[:, :1]
            )
            absolute_trajectory = initial_decoded[:, None] + displacement
        for trajectory in absolute_trajectory:
            trajectory_axis.plot(
                full_time,
                np.rad2deg(_wrapped_trace_for_axis(trajectory)),
                color="#365f8d",
                linewidth=0.45,
                alpha=0.18,
                rasterized=True,
            )
        trajectory_axis.set_ylabel(f"{decoder_label}\ndecoded angle [deg]")
        angle_ticks_deg = np.arange(-180.0, 180.1, 30.0)
        minor_angle_ticks_deg = np.arange(-180.0, 180.1, 10.0)
        trajectory_axis.set_ylim(-180.0, 180.0)
        trajectory_axis.set_yticks(angle_ticks_deg)
        trajectory_axis.set_yticks(minor_angle_ticks_deg, minor=True)
        trajectory_axis.grid(which="major", alpha=0.22)
        trajectory_axis.grid(which="minor", alpha=0.07)
        if cue_theta_trace is not None:
            x_limits = (float(full_time[0]), float(full_time[-1]))
            y_limits = (-180.0, 180.0)
            trajectory_axis.plot(
                [0.0, 0.0],
                [float(y_limits[0]), float(y_limits[1])],
                color="#555555",
                linewidth=0.8,
                linestyle="--",
                alpha=0.8,
                zorder=1,
            )
            darkness_rgba = np.ones((2, 2, 4), dtype=float)
            darkness_rgba[..., :3] = 0.85
            darkness_rgba[..., 3] = 0.55
            trajectory_axis.imshow(
                darkness_rgba,
                extent=(0.0, float(time[-1]), *y_limits),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                zorder=0.1,
            )
            trajectory_axis.set_xlim(x_limits)
            trajectory_axis.set_ylim(y_limits)

        initial_deg = np.rad2deg(theta_initial)
        final_decoded_deg = np.rad2deg(wrap_angle(theta_trace[:, -1]))
        probe_initial = np.asarray(endpoint_probe_theta_initial, dtype=float)
        probe_release = endpoint_probe_release_by_name[decoder_name]
        probe_final = endpoint_probe_final_by_name[decoder_name]
        probe_initial_deg = np.rad2deg(wrap_angle(probe_initial))
        probe_release_deg = np.rad2deg(wrap_angle(probe_release))
        probe_final_deg = np.rad2deg(wrap_angle(probe_final))
        coarse_probe = endpoint_probe_refinement_level == 0
        refined_probe = endpoint_probe_refinement_level > 0
        if cue_theta_trace is not None:
            cue_onset_deg = np.rad2deg(wrap_angle(cue_theta_trace[:, 0]))
            cue_release_deg = np.rad2deg(wrap_angle(cue_theta_trace[:, -1]))
            endpoint_axis.scatter(
                initial_deg,
                cue_onset_deg,
                s=5.0,
                color="#9e9e9e",
                alpha=0.35,
                linewidths=0.0,
                rasterized=True,
                label="cue onset",
            )
            endpoint_axis.scatter(
                initial_deg,
                cue_release_deg,
                s=6.0,
                color="#2f6f9f",
                alpha=0.55,
                linewidths=0.0,
                rasterized=True,
                label="cue off",
            )
        endpoint_fixed_points = classify_endpoint_map_fixed_points(
            theta_initial=probe_initial,
            theta_release=probe_release,
            theta_final=probe_final,
        )
        fixed_point_theta = np.asarray(
            endpoint_fixed_points["fixed_point_theta"],
            dtype=float,
        )
        fixed_point_stability = np.asarray(
            endpoint_fixed_points["fixed_point_stability"],
            dtype=np.int8,
        )
        attracting_theta_deg = np.rad2deg(
            fixed_point_theta[fixed_point_stability == -1]
        )
        basin_boundary_theta_deg = np.rad2deg(
            np.asarray(
                endpoint_fixed_points["basin_boundary_theta"],
                dtype=float,
            )
        )
        basin_boundary_initial_theta_deg = np.rad2deg(
            np.asarray(
                endpoint_fixed_points["basin_boundary_initial_theta"],
                dtype=float,
            )
        )
        basin_boundary_resolution_limited = np.asarray(
            endpoint_fixed_points[
                "basin_boundary_release_resolution_limited"
            ],
            dtype=bool,
        )
        unresolved_boundary_theta_deg = np.rad2deg(
            np.asarray(
                endpoint_fixed_points["unresolved_boundary_theta"],
                dtype=float,
            )
        )
        nonmonotonic_transition_count = np.asarray(
            endpoint_fixed_points["nonmonotonic_transition_theta"],
            dtype=float,
        ).size
        missing_boundary_interval_count = np.asarray(
            endpoint_fixed_points["missing_boundary_interval_theta"],
            dtype=float,
        ).size
        # A fixed 4-degree display band remains legible without encoding a
        # basin/governing-region width or changing with probe density.
        band_half_width_deg = 2.0
        for attracting_angle_deg in attracting_theta_deg:
            for axis in (
                trajectory_axis,
                endpoint_axis,
                release_endpoint_axis,
            ):
                _shade_endpoint_angle_band(
                    axis,
                    angle_deg=float(attracting_angle_deg),
                    half_width_deg=band_half_width_deg,
                    color="#17becf",
                )
        for (
            boundary_angle_deg,
            boundary_initial_deg,
            resolution_limited,
        ) in zip(
            basin_boundary_theta_deg,
            basin_boundary_initial_theta_deg,
            basin_boundary_resolution_limited,
            strict=True,
        ):
            boundary_linestyle = ":" if resolution_limited else "--"
            _plot_endpoint_angle_line(
                trajectory_axis,
                angle_deg=float(boundary_angle_deg),
                color="#f28e2b",
                linestyle=boundary_linestyle,
            )
            # The cue-to-endpoint panel is parameterized by cue angle, so its
            # inferred basin boundary belongs on the x axis.  This is
            # especially important when a quantized decoder can only localize
            # the release phase to a bin but cue bisection resolves the bracket.
            endpoint_axis.axvline(
                float(boundary_initial_deg),
                color="#f28e2b",
                linewidth=0.8,
                linestyle=boundary_linestyle,
                alpha=0.7,
                zorder=1,
            )
            release_endpoint_axis.axvline(
                float(boundary_angle_deg),
                color="#f28e2b",
                linewidth=0.8,
                linestyle=boundary_linestyle,
                alpha=0.7,
                zorder=1,
            )
        for boundary_angle_deg in unresolved_boundary_theta_deg:
            for axis in (trajectory_axis, endpoint_axis):
                _plot_endpoint_angle_line(
                    axis,
                    angle_deg=float(boundary_angle_deg),
                    color="#9467bd",
                    linestyle=":",
                )
            release_endpoint_axis.axvline(
                float(boundary_angle_deg),
                color="#9467bd",
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
                zorder=1,
            )
        endpoint_axis.scatter(
            probe_initial_deg[coarse_probe],
            probe_final_deg[coarse_probe],
            s=7.0,
            color="#9c4f45",
            alpha=0.7,
            linewidths=0.0,
            rasterized=True,
            label="darkness final",
        )
        if np.any(refined_probe):
            endpoint_axis.scatter(
                probe_initial_deg[refined_probe],
                probe_final_deg[refined_probe],
                s=10.0,
                color="#f28e2b",
                alpha=0.9,
                linewidths=0.0,
                rasterized=True,
                label="bisected endpoint probe",
            )
        endpoint_axis.plot(
            [-180.0, 180.0],
            [-180.0, 180.0],
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            label="ideal identity",
        )
        endpoint_axis.set_xlim(-180.0, 180.0)
        endpoint_axis.set_ylim(-180.0, 180.0)
        endpoint_axis.set_yticks(angle_ticks_deg)
        endpoint_axis.set_yticks(minor_angle_ticks_deg, minor=True)
        endpoint_axis.tick_params(axis="y", labelleft=True)
        endpoint_axis.set_ylabel(
            "decoded angle [deg]"
            if cue_theta_trace is not None
            else "final decoded [deg]"
        )
        endpoint_axis.grid(which="major", alpha=0.22)
        endpoint_axis.grid(which="minor", alpha=0.07)
        release_endpoint_axis.scatter(
            probe_release_deg[coarse_probe],
            probe_final_deg[coarse_probe],
            s=7.0,
            color="#9c4f45",
            alpha=0.7,
            linewidths=0.0,
            rasterized=True,
            label="coarse endpoint probe",
        )
        if np.any(refined_probe):
            release_endpoint_axis.scatter(
                probe_release_deg[refined_probe],
                probe_final_deg[refined_probe],
                s=10.0,
                color="#f28e2b",
                alpha=0.9,
                linewidths=0.0,
                rasterized=True,
                label="bisected endpoint probe",
            )
        release_endpoint_axis.plot(
            [-180.0, 180.0],
            [-180.0, 180.0],
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
        )
        release_endpoint_axis.set_xlim(-180.0, 180.0)
        release_endpoint_axis.set_ylim(-180.0, 180.0)
        release_endpoint_axis.set_yticks(angle_ticks_deg)
        release_endpoint_axis.set_yticks(minor_angle_ticks_deg, minor=True)
        release_endpoint_axis.tick_params(axis="y", labelleft=True)
        release_endpoint_axis.set_ylabel("final decoded angle [deg]")
        release_endpoint_axis.grid(which="major", alpha=0.22)
        release_endpoint_axis.grid(which="minor", alpha=0.07)
        maximum_refinement_level = int(
            np.max(endpoint_probe_refinement_level)
            if endpoint_probe_refinement_level.size
            else 0
        )
        release_endpoint_axis.text(
            0.98,
            0.03,
            (
                f"coarse={np.count_nonzero(coarse_probe)}, "
                f"refined={np.count_nonzero(refined_probe)}\n"
                f"bisection depth={maximum_refinement_level}"
            ),
            transform=release_endpoint_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
        endpoint_axis.text(
            0.98,
            0.03,
            (
                f"attractors={attracting_theta_deg.size}, "
                f"unstable={basin_boundary_theta_deg.size}\n"
                f"sub-bin={np.count_nonzero(basin_boundary_resolution_limited)}, "
                f"nonmonotonic={nonmonotonic_transition_count}, "
                f"unresolved={unresolved_boundary_theta_deg.size}, "
                f"missing={missing_boundary_interval_count}"
            ),
            transform=endpoint_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
        if cue_theta_trace is not None:
            release_aligned = theta_initial + circular_difference(
                cue_theta_trace[:, -1],
                theta_initial,
            )
            finite_release = np.isfinite(release_aligned)
            if np.count_nonzero(finite_release) >= 2:
                cue_finite = theta_initial[finite_release]
                release_finite = release_aligned[finite_release]
                slope, intercept = linear_fit_slope_intercept(
                    cue_finite,
                    release_finite,
                )
                fitted_release = slope * cue_finite + intercept
                total_variation = np.sum(
                    np.square(release_finite - np.mean(release_finite))
                )
                residual_variation = np.sum(
                    np.square(release_finite - fitted_release)
                )
                release_r_squared = (
                    1.0 - residual_variation / total_variation
                    if total_variation > 1e-12
                    else float("nan")
                )
                release_rmse_deg = float(
                    np.rad2deg(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    circular_difference(
                                        cue_theta_trace[finite_release, -1],
                                        cue_finite,
                                    )
                                )
                            )
                        )
                    )
                )
                endpoint_axis.text(
                    0.02,
                    0.03,
                    (
                        f"cue-off slope={slope:.3f}, "
                        f"R²={release_r_squared:.3f}\n"
                        f"circular RMSE={release_rmse_deg:.2f}°"
                    ),
                    transform=endpoint_axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )
        if row_index == 0:
            mapping_handles = []
            if cue_theta_trace is not None:
                mapping_handles.extend(
                    [
                        Line2D(
                            [0.0],
                            [0.0],
                            color="#9e9e9e",
                            marker="o",
                            linestyle="none",
                            markersize=3.5,
                            alpha=0.55,
                            label="cue onset",
                        ),
                        Line2D(
                            [0.0],
                            [0.0],
                            color="#2f6f9f",
                            marker="o",
                            linestyle="none",
                            markersize=3.5,
                            alpha=0.75,
                            label="cue off",
                        ),
                    ]
                )
            mapping_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color="#9c4f45",
                    marker="o",
                    linestyle="none",
                    markersize=3.5,
                    alpha=0.8,
                    label="darkness final",
                )
            )
            endpoint_axis.legend(
                handles=[
                    Line2D(
                        [0.0],
                        [0.0],
                        color="black",
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.55,
                        label="ideal identity",
                    ),
                    *mapping_handles,
                    Patch(
                        facecolor="#17becf",
                        alpha=0.18,
                        label="attracting endpoint (display band)",
                    ),
                    Line2D(
                        [0.0],
                        [0.0],
                        color="#f28e2b",
                        linewidth=0.8,
                        linestyle="--",
                        label="trajectory-inferred unstable FP",
                    ),
                    Line2D(
                        [0.0],
                        [0.0],
                        color="#9467bd",
                        linewidth=0.8,
                        linestyle=":",
                        label="unresolved cluster transition",
                    ),
                ],
                frameon=False,
                fontsize=8,
                loc="upper left",
            )
            release_endpoint_axis.legend(
                frameon=False,
                fontsize=8,
                loc="upper left",
            )

    axes[-1, 0].set_xlabel(
        "time relative to cue off [s]"
        if cue_history_supplied
        else "time in darkness [s]"
    )
    axes[-1, 1].set_xlabel("initial cue angle [deg]")
    axes[-1, 2].set_xlabel("decoded bump angle at visual cue off [deg]")
    axes[0, 0].set_title(
        "visual cue (white) to darkness (gray) trajectories"
        if cue_history_supplied
        else "all zero-input trajectories (ideal: horizontal)"
    )
    axes[0, 1].set_title(
        "cue transfer and endpoint map"
        if cue_history_supplied
        else "endpoint map"
    )
    axes[0, 2].set_title("autonomous release-to-endpoint map")
    fig.suptitle(
        f"{title} (n={theta_initial.size} uniform cue starts, "
        + (
            f"T_cue={-cue_time[0]:g} s, T_dark={time[-1]:g} s)"
            if cue_history_supplied
            else f"T_dark={time[-1]:g} s)"
        )
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_bump_attractor_pva_trajectories(
    *,
    time: np.ndarray,
    theta_initial: np.ndarray,
    theta_pva: np.ndarray,
    path: str | Path,
    cue_time: np.ndarray | None = None,
    cue_theta_pva: np.ndarray | None = None,
    endpoint_probe_theta_initial: np.ndarray | None = None,
    endpoint_probe_refinement_level: np.ndarray | None = None,
    endpoint_probe_theta_release_pva: np.ndarray | None = None,
    endpoint_probe_theta_final_pva: np.ndarray | None = None,
    endpoint_probe_theta_pva_trajectory: np.ndarray | None = None,
    title: str = "PVA-decoded bump-attractor landscape",
    dpi: int = 300,
    endpoint_map_coordinate: str = "release",
) -> None:
    """Plot trajectories beside one explicitly selected endpoint map.

    ``endpoint_map_coordinate`` must be either ``"initial_cue"`` or
    ``"release"``.  Keeping the two coordinate choices in separate figures
    prevents cue-settling error from being visually conflated with autonomous
    darkness dynamics.  Fixed points are always classified in the release
    coordinate; the initial-cue view displays their corresponding preimages
    under the measured cue-transfer map.

    When ``cue_time`` and ``cue_theta_pva`` are provided, the trajectory panel
    also draws the visual-cue phase on the negative-time half of the axis
    (cue_time runs from -T_cue to 0), with a dashed vertical line at ``t=0``
    marking cue release.

    When ``endpoint_probe_theta_pva_trajectory`` is provided (full darkness
    PVA trajectories of the bisection probes, shape ``(n_probe, n_time)``),
    these refined basin-boundary trajectories are overlaid on the trajectory
    panel with a distinct color, complementing the 1-degree coarse grid.
    """

    del endpoint_probe_refinement_level
    if endpoint_map_coordinate not in {"initial_cue", "release"}:
        raise ValueError(
            "endpoint_map_coordinate must be 'initial_cue' or 'release'"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    theta_initial = np.asarray(theta_initial, dtype=float)
    theta_pva = np.asarray(theta_pva, dtype=float)
    if (
        time.ndim != 1
        or time.size == 0
        or theta_initial.ndim != 1
        or theta_pva.shape != (theta_initial.size, time.size)
    ):
        raise ValueError(
            "PVA attractor trajectories must have shape (initial, time)"
        )
    theta_release = (
        np.asarray(cue_theta_pva, dtype=float)[:, -1]
        if cue_theta_pva is not None
        else theta_pva[:, 0]
    )
    if theta_release.shape != theta_initial.shape:
        raise ValueError("cue_theta_pva must match the initial-condition count")

    probe_initial = (
        theta_initial
        if endpoint_probe_theta_initial is None
        else np.asarray(endpoint_probe_theta_initial, dtype=float)
    )
    probe_release = (
        theta_release
        if endpoint_probe_theta_release_pva is None
        else np.asarray(endpoint_probe_theta_release_pva, dtype=float)
    )
    probe_final = (
        theta_pva[:, -1]
        if endpoint_probe_theta_final_pva is None
        else np.asarray(endpoint_probe_theta_final_pva, dtype=float)
    )
    if not (
        probe_initial.ndim == 1
        and probe_release.shape == probe_initial.shape
        and probe_final.shape == probe_initial.shape
    ):
        raise ValueError("endpoint probe arrays must be matching 1D arrays")

    landscape = classify_endpoint_map_fixed_points(
        theta_initial=probe_initial,
        theta_release=probe_release,
        theta_final=probe_final,
    )
    fixed_theta = np.asarray(landscape["fixed_point_theta"], dtype=float)
    fixed_initial = np.asarray(
        landscape.get("fixed_point_initial_theta", fixed_theta),
        dtype=float,
    )
    fixed_stability = np.asarray(
        landscape["fixed_point_stability"],
        dtype=np.int8,
    )
    fixed_resolution_limited = np.asarray(
        landscape["fixed_point_release_resolution_limited"],
        dtype=bool,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.2),
        constrained_layout=True,
    )
    trajectory_axis = axes[0]
    endpoint_axis = axes[1]
    # Build the full cue -> darkness time axis.  ``cue_time`` runs from
    # -T_cue to 0 (cue release), ``time`` from 0 onward (darkness).  The
    # release sample is shared by both arrays, so the cue trace keeps only
    # its samples before release ([:, :-1]) and the darkness trace keeps all
    # of its samples.
    has_cue = (
        cue_time is not None
        and cue_theta_pva is not None
        and np.asarray(cue_time).size > 1
        and np.asarray(cue_theta_pva).shape
        == (theta_initial.size, np.asarray(cue_time).size)
    )
    if has_cue:
        cue_time = np.asarray(cue_time, dtype=float)
        cue_theta_pva = np.asarray(cue_theta_pva, dtype=float)
        full_time = np.concatenate([cue_time[:-1], time])
        full_trace = np.concatenate([cue_theta_pva[:, :-1], theta_pva], axis=1)
        x_limits = (float(full_time[0]), float(full_time[-1]))
    else:
        full_time = time
        full_trace = theta_pva
        x_limits = (float(time[0]), float(time[-1]))
    for trace in full_trace:
        trajectory_axis.plot(
            full_time,
            np.rad2deg(_wrapped_trace_for_axis(trace)),
            color="#365f8d",
            linewidth=0.45,
            alpha=0.20,
            rasterized=True,
        )
    # Overlay the refined bisection-probe trajectories (same darkness time
    # axis as ``time``; they have no cue phase).  Drawn with a brighter
    # distinct color and higher alpha so the basin-boundary refinement is
    # visible despite the dense coarse grid.
    if (
        endpoint_probe_theta_pva_trajectory is not None
        and np.asarray(endpoint_probe_theta_pva_trajectory).ndim == 2
        and np.asarray(endpoint_probe_theta_pva_trajectory).shape[1] == time.size
    ):
        for trace in np.asarray(endpoint_probe_theta_pva_trajectory):
            trajectory_axis.plot(
                time,
                np.rad2deg(_wrapped_trace_for_axis(trace)),
                color="#c44e52",
                linewidth=0.65,
                alpha=0.55,
                rasterized=True,
            )
    if has_cue:
        # Mark the cue-release boundary at t=0.
        trajectory_axis.plot(
            [0.0, 0.0],
            [-180.0, 180.0],
            color="#555555",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )
        trajectory_axis.axvspan(
            float(full_time[0]),
            0.0,
            color="#f0c060",
            alpha=0.08,
            linewidth=0.0,
            zorder=0,
        )
        trajectory_axis.set_xlabel("time [s] (cue < 0 < darkness)")
    else:
        trajectory_axis.set_xlabel("time after cue release [s]")
    trajectory_axis.set_xlim(*x_limits)
    trajectory_axis.set_ylabel("PVA angle [deg]")
    n_bisection = (
        np.asarray(endpoint_probe_theta_pva_trajectory).shape[0]
        if endpoint_probe_theta_pva_trajectory is not None
        and np.asarray(endpoint_probe_theta_pva_trajectory).ndim == 2
        else 0
    )
    title_suffix = f" + {n_bisection} bisection" if n_bisection > 0 else ""
    trajectory_axis.set_title(
        f"bump trajectories (cue then darkness){title_suffix}\n"
        "thin lines: coarse grid, red: refined basin boundaries",
        fontsize=10,
    )

    identity = np.asarray([-180.0, 180.0])
    if endpoint_map_coordinate == "initial_cue":
        endpoint_x = probe_initial
        fixed_x = fixed_initial
        endpoint_color = "#9c4f45"
        endpoint_label = "endpoint vs initial cue"
        endpoint_xlabel = "initial cue angle [deg]"
        endpoint_title = "endpoint map from initial cue"
        coordinate_title = "initial-cue coordinate"
    else:
        endpoint_x = probe_release
        fixed_x = fixed_theta
        endpoint_color = "#3b7dd8"
        endpoint_label = "endpoint vs release angle"
        endpoint_xlabel = "cue-release PVA angle [deg]"
        endpoint_title = "autonomous endpoint map and fixed points"
        coordinate_title = "release-angle coordinate"
    endpoint_axis.scatter(
        np.rad2deg(wrap_angle(endpoint_x)),
        np.rad2deg(wrap_angle(probe_final)),
        s=6.0,
        color=endpoint_color,
        alpha=0.62,
        linewidths=0.0,
        rasterized=True,
        label=endpoint_label,
    )
    endpoint_axis.plot(
        identity,
        identity,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.6,
        label="identity",
    )
    # The endpoint-map fixed points sit on the identity line E(phi)=phi.
    # Stable and unstable roots alternate and, for a near-continuous
    # attractor, can be only a fraction of a degree apart (quasi saddle-node
    # pairs).  Large scatter markers at their exact (initial, final) positions
    # therefore overlap completely.  Instead, mark every root with a narrow
    # vertical line straddling the identity: stable as a solid blue line,
    # unstable as a dashed orange line.  The line is drawn at
    # x = initial angle, spanning identity +/- half_marker_deg, so even
    # closely spaced pairs remain visually distinguishable.  In the
    # initial-cue view, x is the measured cue-coordinate preimage of the
    # autonomous root; in the release view, x is the root itself.
    half_marker_deg = 4.0
    marker_spec = {
        -1: ("-", "#17becf", "stable FP"),
        1: ("--", "#f28e2b", "unstable FP"),
    }
    for stability, (linestyle, color, label) in marker_spec.items():
        mask = fixed_stability == stability
        if np.any(mask):
            x_deg = np.rad2deg(wrap_angle(fixed_x[mask]))
            y_deg = np.rad2deg(wrap_angle(fixed_theta[mask]))
            for x_pos, y_pos in zip(x_deg, y_deg):
                endpoint_axis.plot(
                    [x_pos, x_pos],
                    [y_pos - half_marker_deg, y_pos + half_marker_deg],
                    color=color,
                    linewidth=1.3,
                    linestyle=linestyle,
                    alpha=0.85,
                    zorder=4,
                )
            # Add one legend entry per class via an invisible proxy line.
            endpoint_axis.plot(
                [],
                [],
                color=color,
                linewidth=1.3,
                linestyle=linestyle,
                label=label,
            )
    cue_valid = bool(np.asarray(landscape["cue_transfer_valid"]))
    endpoint_axis.text(
        0.02,
        0.03,
        (
            f"stable={np.count_nonzero(fixed_stability == -1)}, "
            f"unstable={np.count_nonzero(fixed_stability == 1)}\n"
            f"sub-bin={np.count_nonzero((fixed_stability == 1) & fixed_resolution_limited)}, "
            f"cue map={'valid' if cue_valid else 'distorted'}, "
            f"unresolved={np.asarray(landscape['unresolved_boundary_theta']).size}"
        ),
        transform=endpoint_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.72, ec="#cccccc", lw=0.5),
    )
    endpoint_axis.set_xlabel(endpoint_xlabel)
    endpoint_axis.set_ylabel("darkness endpoint angle [deg]")
    endpoint_axis.set_title(endpoint_title, fontsize=10)

    major_ticks = np.arange(-180.0, 180.1, 60.0)
    minor_ticks = np.arange(-180.0, 180.1, 30.0)
    for axis in axes:
        axis.set_ylim(-180.0, 180.0)
        axis.set_yticks(major_ticks)
        axis.set_yticks(minor_ticks, minor=True)
        axis.grid(which="major", alpha=0.22)
        axis.grid(which="minor", alpha=0.07)
    endpoint_axis.set_xlim(-180.0, 180.0)
    endpoint_axis.set_xticks(major_ticks)
    endpoint_axis.legend(
        frameon=True,
        framealpha=0.82,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )
    fig.suptitle(
        f"{title} (n={theta_initial.size}, T_dark={time[-1]:g} s, "
        f"{coordinate_title})",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bump_attractor_cue_transfer(
    *,
    theta_initial: np.ndarray,
    theta_release: np.ndarray,
    path: str | Path,
    title: str = "Visual-cue transfer at darkness release",
    dpi: int = 300,
) -> None:
    """Plot cue transfer against identity and its circular residual.

    The residual is ``wrap(release - initial)``.  A circular difference is
    essential near the ``-pi/pi`` seam, where ordinary subtraction can create
    a spurious error close to 360 degrees.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    theta_initial = np.asarray(theta_initial, dtype=float)
    theta_release = np.asarray(theta_release, dtype=float)
    if (
        theta_initial.ndim != 1
        or theta_initial.size == 0
        or theta_release.shape != theta_initial.shape
    ):
        raise ValueError("cue-transfer angles must be matching non-empty 1D arrays")

    initial_deg = np.rad2deg(wrap_angle(theta_initial))
    release_deg = np.rad2deg(wrap_angle(theta_release))
    residual_deg = np.rad2deg(circular_difference(theta_release, theta_initial))
    identity = np.asarray([-180.0, 180.0])
    major_ticks = np.arange(-180.0, 180.1, 60.0)
    minor_ticks = np.arange(-180.0, 180.1, 30.0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.8),
        constrained_layout=True,
    )
    transfer_axis, residual_axis = axes
    transfer_axis.scatter(
        initial_deg,
        release_deg,
        s=7.0,
        color="#2f6f9f",
        alpha=0.68,
        linewidths=0.0,
        rasterized=True,
        label="measured cue transfer",
    )
    transfer_axis.plot(
        identity,
        identity,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
        label="identity (k=1)",
    )
    transfer_axis.set_xlabel("initial cue angle [deg]")
    transfer_axis.set_ylabel("PVA angle at cue release [deg]")
    transfer_axis.set_title("cue transfer vs identity")
    transfer_axis.set_xlim(-180.0, 180.0)
    transfer_axis.set_ylim(-180.0, 180.0)
    transfer_axis.set_xticks(major_ticks)
    transfer_axis.set_yticks(major_ticks)
    transfer_axis.set_xticks(minor_ticks, minor=True)
    transfer_axis.set_yticks(minor_ticks, minor=True)
    transfer_axis.legend(frameon=False, fontsize=8, loc="upper left")

    residual_axis.scatter(
        initial_deg,
        residual_deg,
        s=7.0,
        color="#9c4f45",
        alpha=0.72,
        linewidths=0.0,
        rasterized=True,
        label="release - initial (circular)",
    )
    residual_axis.axhline(
        0.0,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
        label="ideal: zero",
    )
    finite_residual = np.abs(residual_deg[np.isfinite(residual_deg)])
    max_abs_residual = (
        float(np.max(finite_residual)) if finite_residual.size else 0.0
    )
    residual_limit = min(
        180.0,
        max(5.0, 1.1 * max_abs_residual),
    )
    residual_axis.set_xlim(-180.0, 180.0)
    residual_axis.set_ylim(-residual_limit, residual_limit)
    residual_axis.set_xticks(major_ticks)
    residual_axis.set_xticks(minor_ticks, minor=True)
    residual_axis.set_xlabel("initial cue angle [deg]")
    residual_axis.set_ylabel("circular release - initial [deg]")
    residual_axis.set_title("cue-transfer residual (ideal: 0 deg)")
    residual_axis.legend(frameon=False, fontsize=8, loc="upper left")

    for axis in axes:
        axis.grid(which="major", alpha=0.22)
        axis.grid(which="minor", alpha=0.07)
    rms_residual = float(np.sqrt(np.mean(np.square(residual_deg))))
    fig.suptitle(
        f"{title} (n={theta_initial.size}, circular RMS={rms_residual:.3g} deg)",
        fontsize=12,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_numerical_convergence_diagnostics(
    *,
    time: np.ndarray,
    dt: np.ndarray,
    integration_method: np.ndarray,
    heading_error: np.ndarray,
    rate_rms_error: np.ndarray,
    max_abs_heading_error: np.ndarray,
    max_rate_rms_error: np.ndarray,
    convergence_passed: np.ndarray,
    path: str | Path,
) -> None:
    """Plot whole-step errors against the high-resolution reference."""

    time = np.asarray(time, dtype=float)
    dt = np.asarray(dt, dtype=float)
    method = np.asarray(integration_method).astype(str)
    heading_error = np.asarray(heading_error, dtype=float)
    rate_rms_error = np.asarray(rate_rms_error, dtype=float)
    max_heading = np.asarray(max_abs_heading_error, dtype=float)
    max_rate = np.asarray(max_rate_rms_error, dtype=float)
    passed = np.asarray(convergence_passed, dtype=bool)
    expected_trace_shape = (dt.size, time.size)
    if (
        method.shape != dt.shape
        or heading_error.shape != expected_trace_shape
        or rate_rms_error.shape != expected_trace_shape
        or max_heading.shape != dt.shape
        or max_rate.shape != dt.shape
        or passed.shape != dt.shape
    ):
        raise ValueError("numerical convergence arrays have inconsistent shapes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    colors = {"forward_euler": "#d65f5f", "exact_linear": "#2878b5"}
    linestyles = ["-", "--", "-.", ":"]
    for row_index, (row_dt, row_method) in enumerate(
        zip(dt, method, strict=True)
    ):
        label = f"{row_method}, dt={1e3 * row_dt:g} ms"
        method_slots = np.flatnonzero(method == row_method)
        style_index = int(np.flatnonzero(method_slots == row_index)[0])
        axes[0].plot(
            time,
            np.rad2deg(heading_error[row_index]),
            color=colors.get(row_method, "#555555"),
            linestyle=linestyles[style_index % len(linestyles)],
            linewidth=1.2,
            label=label,
        )
        axes[1].plot(
            time,
            rate_rms_error[row_index],
            color=colors.get(row_method, "#555555"),
            linestyle=linestyles[style_index % len(linestyles)],
            linewidth=1.2,
        )
    for row_method in np.unique(method):
        mask = method == row_method
        order = np.argsort(dt[mask])
        method_dt_ms = 1e3 * dt[mask][order]
        method_heading = np.rad2deg(max_heading[mask][order])
        method_passed = passed[mask][order]
        axes[2].plot(
            method_dt_ms,
            method_heading,
            marker="o",
            color=colors.get(row_method, "#555555"),
            linewidth=1.4,
            label=row_method,
        )
        axes[2].scatter(
            method_dt_ms[method_passed],
            method_heading[method_passed],
            marker="o",
            s=42,
            facecolors="none",
            edgecolors=colors.get(row_method, "#555555"),
            linewidths=1.2,
        )
    axes[0].set_xlabel("physical time [s]")
    axes[0].set_ylabel("heading error [deg]")
    axes[0].set_title("phase error vs reference")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("physical time [s]")
    axes[1].set_ylabel("HD-rate RMS error")
    axes[1].set_title("population-state error")
    axes[2].set_xlabel("dt [ms]")
    axes[2].set_ylabel("maximum heading error [deg]")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_title("step-size convergence (open = pass)")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Coupled Vafidis step convergence")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_timescale_separation_diagnostics(
    *,
    normal_time: np.ndarray,
    perturbation_scale: np.ndarray,
    normal_distance_to_manifold: np.ndarray,
    normal_control_distance_to_manifold: np.ndarray,
    normal_e_folding_time: np.ndarray,
    normal_recovery_observed: np.ndarray,
    tangential_time: np.ndarray,
    tangential_overlap_displacement: np.ndarray,
    tangential_first_passage_time: np.ndarray,
    tangential_first_passage_observed: np.ndarray,
    tangential_threshold_rad: float,
    normal_time_p90: float,
    tangential_time_p10: float,
    conservative_timescale_ratio: float,
    criterion_ratio_threshold: float,
    criterion_passed: bool,
    criterion_ratio_is_lower_bound: bool,
    path: str | Path,
) -> None:
    """Plot normal recovery and tangential drift used by the criterion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normal_time = np.asarray(normal_time, dtype=float)
    perturbation_scale = np.asarray(perturbation_scale, dtype=float)
    normal_distance = np.asarray(normal_distance_to_manifold, dtype=float)
    control_distance = np.asarray(normal_control_distance_to_manifold, dtype=float)
    e_folding_time = np.asarray(normal_e_folding_time, dtype=float)
    recovery_observed = np.asarray(normal_recovery_observed, dtype=bool)
    tangential_time = np.asarray(tangential_time, dtype=float)
    tangential_displacement = np.asarray(
        tangential_overlap_displacement,
        dtype=float,
    )
    first_passage_time = np.asarray(tangential_first_passage_time, dtype=float)
    first_passage_observed = np.asarray(
        tangential_first_passage_observed,
        dtype=bool,
    )
    if normal_distance.ndim != 4 or normal_distance.shape[0] != perturbation_scale.size:
        raise ValueError("normal distance must have shape (scale, anchor, repeat, time)")
    if normal_distance.shape[-1] != normal_time.size:
        raise ValueError("normal distance time axis does not match normal_time")
    if tangential_displacement.ndim != 2 or tangential_displacement.shape[1] != tangential_time.size:
        raise ValueError("tangential displacement must have shape (start, time)")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, perturbation_scale.size))

    normal_axis = axes[0, 0]
    if control_distance.shape[-1:] == normal_time.shape:
        normal_axis.plot(
            normal_time,
            np.nanmedian(control_distance, axis=0),
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="unperturbed control",
        )
    for scale_index, (scale, color) in enumerate(zip(perturbation_scale, colors)):
        scale_traces = normal_distance[scale_index].reshape(-1, normal_time.size)
        median = np.nanmedian(scale_traces, axis=0)
        lower = np.nanquantile(scale_traces, 0.10, axis=0)
        upper = np.nanquantile(scale_traces, 0.90, axis=0)
        normal_axis.fill_between(
            normal_time,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        normal_axis.plot(
            normal_time,
            median,
            color=color,
            linewidth=1.5,
            label=fr"perturbation RMS={scale:g}",
        )
    positive_distance = normal_distance[np.isfinite(normal_distance) & (normal_distance > 0.0)]
    if positive_distance.size:
        normal_axis.set_yscale("log")
    normal_axis.set_title("A  Fast normal recovery")
    normal_axis.set_xlabel("time after perturbation [s]")
    normal_axis.set_ylabel(r"distance to closed manifold / $\sqrt{N}$")
    normal_axis.grid(alpha=0.20)
    normal_axis.legend(frameon=False, fontsize=7)

    recovery_axis = axes[0, 1]
    rng = np.random.default_rng(0)
    for scale_index, (scale, color) in enumerate(zip(perturbation_scale, colors)):
        scale_times = e_folding_time[scale_index].reshape(-1)
        scale_observed = recovery_observed[scale_index].reshape(-1)
        finite = np.isfinite(scale_times)
        jitter = rng.uniform(-0.08, 0.08, size=np.count_nonzero(finite))
        recovery_axis.scatter(
            np.full(np.count_nonzero(finite), scale_index, dtype=float) + jitter,
            scale_times[finite],
            s=13,
            color=color,
            alpha=0.65,
            linewidths=0.0,
        )
        censored = finite & ~scale_observed
        if np.any(censored):
            recovery_axis.scatter(
                np.full(np.count_nonzero(censored), scale_index, dtype=float),
                scale_times[censored],
                s=28,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
                marker="^",
            )
    recovery_axis.set_xticks(np.arange(perturbation_scale.size))
    recovery_axis.set_xticklabels([f"{scale:g}" for scale in perturbation_scale])
    recovery_axis.set_title("B  Normal e-folding times")
    recovery_axis.set_xlabel("current perturbation RMS")
    recovery_axis.set_ylabel(r"$T_\perp$ [s]")
    recovery_axis.grid(alpha=0.20, axis="y")

    tangent_axis = axes[1, 0]
    absolute_displacement_deg = np.rad2deg(np.abs(tangential_displacement))
    tangent_colors = plt.cm.twilight(
        np.linspace(0.0, 1.0, tangential_displacement.shape[0], endpoint=False)
    )
    for trace, color in zip(absolute_displacement_deg, tangent_colors):
        tangent_axis.plot(
            tangential_time,
            trace,
            color=color,
            linewidth=0.65,
            alpha=0.48,
        )
    threshold_deg = float(np.rad2deg(tangential_threshold_rad))
    tangent_axis.axhline(
        threshold_deg,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=f"first-passage threshold={threshold_deg:g} deg",
    )
    passage_fraction = float(np.mean(first_passage_observed))
    tangent_axis.set_title(
        f"C  Slow tangential motion (passage fraction={passage_fraction:.2f})"
    )
    tangent_axis.set_xlabel("time in zero-input darkness [s]")
    tangent_axis.set_ylabel("absolute Clark-overlap displacement [deg]")
    tangent_axis.grid(alpha=0.20)
    tangent_axis.legend(frameon=False, fontsize=8)

    summary_axis = axes[1, 1]
    times = np.asarray([normal_time_p90, tangential_time_p10], dtype=float)
    summary_axis.bar(
        [0, 1],
        times,
        color=["#4c78a8", "#e45756"],
        width=0.62,
    )
    summary_axis.set_xticks([0, 1])
    summary_axis.set_xticklabels(
        [r"$T_\perp$ p90", r"$T_\parallel$ p10"],
    )
    if np.all(np.isfinite(times)) and np.all(times > 0.0):
        summary_axis.set_yscale("log")
    ratio_prefix = ">=" if criterion_ratio_is_lower_bound else "="
    status = "PASS" if criterion_passed else "FAIL"
    summary_axis.set_title("D  Conservative timescale comparison")
    summary_axis.set_ylabel("time [s]")
    summary_axis.grid(alpha=0.20, axis="y")
    summary_axis.text(
        0.04,
        0.96,
        (
            fr"$T_\parallel/T_\perp$ {ratio_prefix} "
            f"{conservative_timescale_ratio:.2g}\n"
            f"criterion: ratio >= {criterion_ratio_threshold:g} ({status})\n"
            f"tangential first-passage observed: {np.mean(first_passage_observed):.2f}"
        ),
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
    )

    fig.suptitle(
        "Clark-style operational criterion: fast normal flow, slow tangential flow"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_velocity_trajectory_sweep(
    *,
    time: np.ndarray,
    commanded_velocity: np.ndarray,
    theta_initial: np.ndarray,
    pva_angular_displacement: np.ndarray,
    overlap_angular_displacement: np.ndarray,
    decoded_velocity_pva: np.ndarray,
    decoded_velocity_overlap: np.ndarray,
    path: str | Path,
) -> None:
    """Plot constant-velocity trajectories without inferred FP labels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    commanded_velocity = np.asarray(commanded_velocity, dtype=float)
    theta_initial = np.asarray(theta_initial, dtype=float)
    displacement = np.asarray(pva_angular_displacement, dtype=float)
    overlap_displacement = np.asarray(overlap_angular_displacement, dtype=float)
    decoded_velocity = np.asarray(decoded_velocity_pva, dtype=float)
    overlap_decoded_velocity = np.asarray(decoded_velocity_overlap, dtype=float)
    expected_trajectory_shape = (
        commanded_velocity.size,
        theta_initial.size,
        time.size,
    )
    if time.ndim != 1 or commanded_velocity.ndim != 1 or theta_initial.ndim != 1:
        raise ValueError("velocity trajectory axes must be one-dimensional")
    if displacement.shape != expected_trajectory_shape:
        raise ValueError("velocity trajectories must have shape (velocity, start, time)")
    if overlap_displacement.shape != expected_trajectory_shape:
        raise ValueError(
            "overlap velocity trajectories must have shape (velocity, start, time)"
        )
    expected_summary_shape = (commanded_velocity.size, theta_initial.size)
    if decoded_velocity.shape != expected_summary_shape:
        raise ValueError("decoded velocity must have shape (velocity, start)")
    if overlap_decoded_velocity.shape != expected_summary_shape:
        raise ValueError("overlap decoded velocity must have shape (velocity, start)")

    fig, (trajectory_axis, residual_axis, velocity_axis) = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
        constrained_layout=True,
    )
    trajectory_by_velocity = np.nanmedian(displacement, axis=1)
    velocity_normalization = Normalize(
        vmin=float(commanded_velocity[0]),
        vmax=float(commanded_velocity[-1]),
    )
    velocity_colormap = plt.get_cmap("viridis")

    for velocity_index, velocity_value in enumerate(commanded_velocity):
        trajectory = trajectory_by_velocity[velocity_index]
        color = velocity_colormap(velocity_normalization(velocity_value))
        trajectory_axis.plot(
            time,
            trajectory,
            color=color,
            linewidth=1.25,
        )
        residual_axis.plot(
            time,
            trajectory - velocity_value * time,
            color=color,
            linewidth=1.25,
        )

    start_description = (
        f"shared start={theta_initial[0]:.2f} rad"
        if theta_initial.size == 1
        else f"median across {theta_initial.size} starts"
    )
    trajectory_axis.set_title(
        f"A  Constant-velocity bump trajectories ({start_description})"
    )
    trajectory_axis.set_xlabel("time in darkness [s]")
    trajectory_axis.set_ylabel(r"unwrapped bump position $x(t)-x(0)$ [rad]")
    trajectory_axis.grid(alpha=0.20)
    velocity_mappable = plt.cm.ScalarMappable(
        norm=velocity_normalization,
        cmap=velocity_colormap,
    )
    velocity_colorbar = fig.colorbar(
        velocity_mappable,
        ax=[trajectory_axis, residual_axis],
        location="bottom",
        fraction=0.08,
        pad=0.13,
        aspect=35,
    )
    velocity_colorbar.set_label("constant input velocity [rad/s]")

    residual_axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    residual_axis.set_title("B  Deviation from ideal linear integration")
    residual_axis.set_xlabel("time in darkness [s]")
    residual_axis.set_ylabel(r"$x(t)-x(0)-vt$ [rad]")
    residual_axis.grid(alpha=0.20)

    decoded_median = np.nanmedian(decoded_velocity, axis=1)
    decoded_p10 = np.nanquantile(decoded_velocity, 0.10, axis=1)
    decoded_p90 = np.nanquantile(decoded_velocity, 0.90, axis=1)
    velocity_axis.fill_between(
        commanded_velocity,
        decoded_p10,
        decoded_p90,
        color="#4c78a8",
        alpha=0.20,
        linewidth=0.0,
        label="start-angle p10--p90",
    )
    velocity_axis.plot(
        commanded_velocity,
        decoded_median,
        color="#4c78a8",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="median decoded velocity",
    )
    velocity_axis.plot(
        commanded_velocity,
        np.nanmedian(overlap_decoded_velocity, axis=1),
        color="#2a9d8f",
        marker="s",
        markersize=3.0,
        linewidth=1.1,
        linestyle="-.",
        label="median Clark-overlap velocity",
    )
    velocity_axis.plot(
        commanded_velocity,
        commanded_velocity,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="ideal",
    )
    velocity_axis.set_title("C  Input-output velocity")
    velocity_axis.set_xlabel("constant input velocity [rad/s]")
    velocity_axis.set_ylabel("decoded PVA velocity [rad/s]")
    velocity_axis.grid(alpha=0.20)
    velocity_axis.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Figure-2-style shared-origin trajectories: linear sliding versus pinning"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
