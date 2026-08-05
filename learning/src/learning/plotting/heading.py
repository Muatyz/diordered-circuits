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
) -> None:
    """Plot heading traces and wrapped PI error for one protocol."""
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

    pva_error = circular_difference(theta_hd_decoded, theta_true)
    axes[1].plot(time, pva_error, label="PVA error", color="#3b6ea8", linewidth=1.3)
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        peak_error = circular_difference(theta_hd_decoded_peak, theta_true)
        axes[1].plot(
            time,
            peak_error,
            label="peak error",
            color="#d18f00",
            linewidth=1.0,
            linestyle="--",
        )

    axes[0].set_title(title)
    axes[0].set_ylabel("heading angle [rad]")
    axes[1].set_ylabel("decoded - true [rad]")
    axes[1].set_xlabel("time [s]")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        _plot_horizontal_reference(axis, x_values=time)
        _set_pi_y_axis(axis)
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86)
    axis.plot(time, pi_error, color="#3b6ea8", linewidth=1.5)
    _plot_horizontal_reference(axis, x_values=time, linewidth=0.8, alpha=0.5)
    _set_pi_y_axis(axis)
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
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
    axis.set_ylabel("decoded - true heading [rad]")
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
) -> None:
    """Compare PVA, peak-neuron, and Clark-overlap darkness trajectories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    theta_initial = np.asarray(theta_initial, dtype=float)
    decoder_traces = [
        ("PVA", np.asarray(theta_pva, dtype=float)),
        ("peak neuron", np.asarray(theta_peak, dtype=float)),
        ("Clark overlap (Eq. 6)", np.asarray(theta_overlap, dtype=float)),
    ]
    expected_shape = (theta_initial.size, time.size)
    if time.ndim != 1 or theta_initial.ndim != 1 or time.size == 0:
        raise ValueError("attractor trajectory plot requires non-empty time and starts")
    if any(trace.shape != expected_shape for _label, trace in decoder_traces):
        raise ValueError("each decoder trace must have shape (start, time)")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11.0, 10.0),
        sharex="col",
        constrained_layout=True,
    )
    for row_index, (decoder_label, theta_trace) in enumerate(decoder_traces):
        trajectory_axis = axes[row_index, 0]
        endpoint_axis = axes[row_index, 1]
        initial_decoded = theta_initial + circular_difference(
            theta_trace[:, 0],
            theta_initial,
        )
        displacement = np.unwrap(theta_trace, axis=1) - np.unwrap(theta_trace, axis=1)[:, :1]
        absolute_trajectory_deg = np.rad2deg(initial_decoded[:, None] + displacement)
        for trajectory in absolute_trajectory_deg:
            trajectory_axis.plot(
                time,
                trajectory,
                color="#365f8d",
                linewidth=0.45,
                alpha=0.18,
                rasterized=True,
            )
        trajectory_axis.set_ylabel(f"{decoder_label}\ndecoded angle [deg]")
        trajectory_axis.grid(alpha=0.18)

        initial_deg = np.rad2deg(theta_initial)
        final_decoded_deg = np.rad2deg(wrap_angle(theta_trace[:, -1]))
        endpoint_fixed_points = classify_endpoint_map_fixed_points(
            theta_initial=theta_initial,
            theta_final=theta_trace[:, -1],
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
        repelling_theta_deg = np.rad2deg(
            fixed_point_theta[fixed_point_stability == 1]
        )
        # A fixed 4-degree display band remains legible without encoding a
        # basin/governing-region width or changing with probe density.
        band_half_width_deg = 2.0
        for attracting_angle_deg in attracting_theta_deg:
            _shade_endpoint_angle_band(
                endpoint_axis,
                angle_deg=float(attracting_angle_deg),
                half_width_deg=band_half_width_deg,
                color="#17becf",
            )
        for repelling_angle_deg in repelling_theta_deg:
            _shade_endpoint_angle_band(
                endpoint_axis,
                angle_deg=float(repelling_angle_deg),
                half_width_deg=band_half_width_deg,
                color="#f28e2b",
            )
        endpoint_axis.scatter(
            initial_deg,
            final_decoded_deg,
            s=7.0,
            color="#9c4f45",
            alpha=0.7,
            linewidths=0.0,
            rasterized=True,
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
        endpoint_axis.set_ylabel("final decoded [deg]")
        endpoint_axis.grid(alpha=0.18)
        endpoint_axis.text(
            0.98,
            0.03,
            (
                f"stable={attracting_theta_deg.size}, "
                f"inferred unstable={repelling_theta_deg.size}"
            ),
            transform=endpoint_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
        if row_index == 0:
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
                    Patch(
                        facecolor="#17becf",
                        alpha=0.18,
                        label="stable FP (display band)",
                    ),
                    Patch(
                        facecolor="#f28e2b",
                        alpha=0.18,
                        label="unstable FP (display band, inferred)",
                    ),
                ],
                frameon=False,
                fontsize=8,
                loc="upper left",
            )

    axes[-1, 0].set_xlabel("time in darkness [s]")
    axes[-1, 1].set_xlabel("initial cue angle [deg]")
    axes[0, 0].set_title("all zero-input trajectories (ideal: horizontal)")
    axes[0, 1].set_title("endpoint map")
    fig.suptitle(
        f"{title} (n={theta_initial.size} uniform cue starts, "
        f"T_dark={time[-1]:g} s)"
    )
    fig.savefig(path, dpi=160)
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
