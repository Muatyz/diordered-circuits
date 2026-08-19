"""Activity plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import collapse_activity_by_theta, peak_decode, pva_decode, pva_vector_strength
from learning.plotting.heading import _shade_dark_phase


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


def plot_heterogeneous_visual_input_profiles(
    *,
    tuning_profiles: np.ndarray,
    path: str | Path,
    sample_count: int,
    seed: int,
    amplitude: float,
    baseline: float,
    light_excitation: float,
    proximal_scale: float,
    theta_hd_pref: np.ndarray | None = None,
    title: str = "Sampled heterogeneous visual input profiles",
) -> np.ndarray:
    """Plot randomly sampled effective visual-current tuning curves.

    The stored profiles use a ``[0, 2 pi)`` angular grid.  This plot reorders
    them onto ``[-pi, pi]`` and includes the periodic right endpoint.  Sampling
    is without replacement and reproducible for a fixed ``seed``.

    Returns:
        The sampled HD-neuron indices, primarily for reproducibility and tests.
    """
    tuning_profiles = np.asarray(tuning_profiles, dtype=float)
    if tuning_profiles.ndim != 2 or tuning_profiles.shape[0] <= 0 or tuning_profiles.shape[1] <= 1:
        raise ValueError("tuning_profiles must have shape (n_theta, n_angles)")
    if not np.isfinite(tuning_profiles).all():
        raise ValueError("tuning_profiles must be finite")
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    sampled_neuron_count = min(sample_count, tuning_profiles.shape[0])
    sampled_indices = np.sort(
        np.random.default_rng(int(seed)).choice(
            tuning_profiles.shape[0],
            size=sampled_neuron_count,
            replace=False,
        )
    )
    if theta_hd_pref is not None:
        theta_hd_pref = np.asarray(theta_hd_pref, dtype=float)
        if theta_hd_pref.shape != (tuning_profiles.shape[0],):
            raise ValueError("theta_hd_pref must have shape (n_theta,)")

    effective_current = float(proximal_scale) * (
        float(amplitude) * tuning_profiles - float(baseline) + float(light_excitation)
    )
    if not np.isfinite(effective_current).all():
        raise ValueError("effective visual-current profiles must be finite")
    n_angles = tuning_profiles.shape[1]
    theta_original = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    theta_wrapped = (theta_original + np.pi) % (2.0 * np.pi) - np.pi
    angular_order = np.argsort(theta_wrapped)
    theta_plot = np.concatenate([theta_wrapped[angular_order], [np.pi]])

    n_columns = min(4, sampled_neuron_count)
    n_rows = int(np.ceil(sampled_neuron_count / n_columns))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.4 * n_columns, 2.45 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for subplot_index, neuron_index in enumerate(sampled_indices):
        axis = flat_axes[subplot_index]
        current_ordered = effective_current[neuron_index, angular_order]
        current_plot = np.concatenate([current_ordered, current_ordered[:1]])
        axis.plot(theta_plot, current_plot, color="#2a6fbb", linewidth=1.5)
        subtitle = f"HD neuron {int(neuron_index)}"
        if theta_hd_pref is not None:
            preferred_orientation = (
                float(theta_hd_pref[neuron_index]) + np.pi
            ) % (2.0 * np.pi) - np.pi
            axis.plot(
                [preferred_orientation, preferred_orientation],
                [float(np.min(current_plot)), float(np.max(current_plot))],
                color="#c44e52",
                linewidth=1.0,
                linestyle="--",
                label="preferred orientation",
            )
            subtitle += f", pref={preferred_orientation / np.pi:.2f} pi"
        axis.set_title(subtitle, fontsize=8)
        axis.set_xlim(-np.pi, np.pi)
        _set_radian_ticks(axis, which="x")
        axis.grid(alpha=0.18, linewidth=0.6)
    for unused_axis in flat_axes[sampled_neuron_count:]:
        unused_axis.set_visible(False)
    fig.suptitle(f"{title} (seed={int(seed)})")
    fig.supxlabel("true heading theta [rad]")
    fig.supylabel("effective I_vis to HD [current]")
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.90,
        hspace=0.42,
        wspace=0.16,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return sampled_indices


def plot_single_neuron_hd_tuning_curves(
    *,
    theta_true: np.ndarray,
    r_hd_by_heading: np.ndarray,
    preferred_direction: np.ndarray,
    path: str | Path,
    sample_count: int,
    seed: int,
    title: str = "Post-training single-neuron HD tuning curves",
) -> np.ndarray:
    """Plot frozen-weight firing-rate tuning curves for sampled HD neurons."""
    theta_true = np.asarray(theta_true, dtype=float)
    rates = np.asarray(r_hd_by_heading, dtype=float)
    preferred_direction = np.asarray(preferred_direction, dtype=float)
    if theta_true.ndim != 1 or rates.ndim != 2 or rates.shape[0] != theta_true.size:
        raise ValueError("r_hd_by_heading must have shape (heading, neuron)")
    if preferred_direction.shape != (rates.shape[1],):
        raise ValueError("preferred_direction must have shape (neuron,)")
    sampled_count = min(max(int(sample_count), 1), rates.shape[1])
    sampled_indices = np.sort(
        np.random.default_rng(int(seed)).choice(rates.shape[1], sampled_count, replace=False)
    )
    angle_order = np.argsort(theta_true)
    theta_ordered = theta_true[angle_order]
    theta_plot = np.concatenate([theta_ordered, [theta_ordered[0] + 2.0 * np.pi]])
    n_columns = min(2, sampled_count)
    n_rows = int(np.ceil(sampled_count / n_columns))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.2 * n_columns, 2.7 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for subplot_index, neuron_index in enumerate(sampled_indices):
        axis = flat_axes[subplot_index]
        rate_ordered = rates[angle_order, neuron_index]
        rate_plot = np.concatenate([rate_ordered, rate_ordered[:1]])
        axis.plot(theta_plot, rate_plot, color="#2a6fbb", linewidth=1.4)
        preference = float(preferred_direction[neuron_index])
        if np.isfinite(preference):
            axis.plot(
                [preference, preference],
                [float(np.nanmin(rate_plot)), float(np.nanmax(rate_plot))],
                color="#c44e52",
                linewidth=1.0,
                linestyle="--",
            )
        axis.set_title(
            f"HD neuron {int(neuron_index)}, COM pref={preference / np.pi:.2f} pi",
            fontsize=9,
        )
        axis.set_xlim(-np.pi, np.pi)
        _set_radian_ticks(axis, which="x")
        axis.grid(alpha=0.18, linewidth=0.6)
    for unused_axis in flat_axes[sampled_count:]:
        unused_axis.set_visible(False)
    fig.suptitle(title)
    fig.supxlabel("true heading theta [rad]")
    fig.supylabel("HD firing rate [a.u.]")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90, hspace=0.42, wspace=0.16)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return sampled_indices


def plot_com_aligned_hd_tuning_population(
    *,
    theta_aligned: np.ndarray,
    r_hd_peak_normalized_com_aligned: np.ndarray,
    population_mean: np.ndarray,
    population_std: np.ndarray,
    path: str | Path,
    title: str = "Post-training COM-aligned HD tuning (one simulated mouse)",
) -> None:
    """Plot all per-neuron-peak-normalized curves and within-network statistics."""
    theta_aligned = np.asarray(theta_aligned, dtype=float)
    aligned_rates = np.asarray(r_hd_peak_normalized_com_aligned, dtype=float)
    population_mean = np.asarray(population_mean, dtype=float)
    population_std = np.asarray(population_std, dtype=float)
    if (
        theta_aligned.ndim != 1
        or aligned_rates.ndim != 2
        or aligned_rates.shape[1] != theta_aligned.size
    ):
        raise ValueError(
            "r_hd_peak_normalized_com_aligned must have shape "
            "(neuron, aligned_heading)"
        )
    if aligned_rates.shape[0] == 0:
        raise ValueError("at least one HD neuron is required")
    if population_mean.shape != theta_aligned.shape or population_std.shape != theta_aligned.shape:
        raise ValueError("population mean and std must match theta_aligned")
    if not all(
        np.isfinite(values).all()
        for values in [theta_aligned, aligned_rates, population_mean, population_std]
    ):
        raise ValueError("COM-aligned tuning plot inputs must be finite")

    theta_closed = np.concatenate([theta_aligned, [theta_aligned[0] + 2.0 * np.pi]])
    aligned_rates_closed = np.concatenate([aligned_rates, aligned_rates[:, :1]], axis=1)
    mean_closed = np.concatenate([population_mean, population_mean[:1]])
    std_closed = np.concatenate([population_std, population_std[:1]])
    valid_neuron_count = int(np.count_nonzero(np.max(aligned_rates, axis=1) > 1e-12))
    silent_neuron_count = int(aligned_rates.shape[0] - valid_neuron_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharex=True)

    for neuron_tuning in aligned_rates_closed:
        axes[0].plot(
            theta_closed,
            neuron_tuning,
            color="#4c88bd",
            linewidth=0.55,
            alpha=0.16,
        )
    axes[0].plot(
        theta_closed,
        mean_closed,
        color="#111111",
        linewidth=2.6,
        label=(
            f"within-mouse mean across all N={aligned_rates.shape[0]} HD neurons "
            f"(silent={silent_neuron_count})"
        ),
    )
    axes[0].set_ylabel("per-neuron peak-normalized HD firing rate")
    axes[0].set_title(
        "All COM-aligned tuning curves (one simulated mouse; "
        f"valid={valid_neuron_count}, silent={silent_neuron_count})"
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[1].plot(
        theta_closed,
        std_closed,
        color="#111111",
        linewidth=2.6,
        label="within-mouse neuron std (ddof=0)",
    )
    axes[1].set_ylabel("across-neuron standard deviation")
    axes[1].set_title("Within-mouse tuning-curve heterogeneity")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    for axis in axes:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xlabel("heading relative to circular COM [rad]")
        _set_radian_ticks(axis, which="x")
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hd_tuning_stage_comparison(
    *,
    theta_aligned: np.ndarray,
    visual_only_mean: np.ndarray,
    visual_only_std: np.ndarray,
    post_training_mean: np.ndarray,
    post_training_std: np.ndarray,
    path: str | Path,
    title: str = "Visual-only versus post-training steady HD tuning",
) -> None:
    """Compare within-mouse COM-aligned moments before and after recurrence."""
    theta_aligned = np.asarray(theta_aligned, dtype=float)
    visual_only_mean = np.asarray(visual_only_mean, dtype=float)
    visual_only_std = np.asarray(visual_only_std, dtype=float)
    post_training_mean = np.asarray(post_training_mean, dtype=float)
    post_training_std = np.asarray(post_training_std, dtype=float)
    moment_curves = [
        visual_only_mean,
        visual_only_std,
        post_training_mean,
        post_training_std,
    ]
    if theta_aligned.ndim != 1 or any(
        curve.shape != theta_aligned.shape for curve in moment_curves
    ):
        raise ValueError("all tuning-stage moment curves must match theta_aligned")
    if not all(np.isfinite(curve).all() for curve in [theta_aligned, *moment_curves]):
        raise ValueError("tuning-stage comparison inputs must be finite")

    theta_closed = np.concatenate([theta_aligned, [theta_aligned[0] + 2.0 * np.pi]])
    closed_moments = [np.concatenate([curve, curve[:1]]) for curve in moment_curves]
    visual_mean_closed, visual_std_closed, post_mean_closed, post_std_closed = closed_moments
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharex=True)

    axes[0].plot(
        theta_closed,
        visual_mean_closed,
        color="#4c88bd",
        linewidth=2.1,
        linestyle="--",
        label="visual-only mean",
    )
    axes[0].plot(
        theta_closed,
        post_mean_closed,
        color="#111111",
        linewidth=2.6,
        label="post-training steady mean",
    )
    axes[0].set_ylabel("per-neuron peak-normalized mean rate")
    axes[0].set_title("Within-mouse mean profile")

    axes[1].plot(
        theta_closed,
        visual_std_closed,
        color="#4c88bd",
        linewidth=2.1,
        linestyle="--",
        label="visual-only std",
    )
    axes[1].plot(
        theta_closed,
        post_std_closed,
        color="#111111",
        linewidth=2.6,
        label="post-training steady std",
    )
    axes[1].set_ylabel("within-mouse neuron std (ddof=0)")
    axes[1].set_title("Within-mouse heterogeneity")

    for axis in axes:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xlabel("heading relative to circular COM [rad]")
        _set_radian_ticks(axis, which="x")
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hd_tuning_settling_diagnostics(
    *,
    theta_true: np.ndarray,
    actual_settle_duration: np.ndarray,
    final_window_max_rate_change: np.ndarray,
    settle_converged: np.ndarray,
    convergence_tolerance: float,
    path: str | Path,
) -> None:
    """Show per-heading settling time and residual change at the stopping point."""
    theta_true = np.asarray(theta_true, dtype=float)
    actual_settle_duration = np.asarray(actual_settle_duration, dtype=float)
    final_window_max_rate_change = np.asarray(final_window_max_rate_change, dtype=float)
    settle_converged = np.asarray(settle_converged, dtype=float)
    expected_shape = theta_true.shape
    if theta_true.ndim != 1 or any(
        values.shape != expected_shape
        for values in [
            actual_settle_duration,
            final_window_max_rate_change,
            settle_converged,
        ]
    ):
        raise ValueError("settling diagnostics must have one value per heading")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive")
    heading_order = np.argsort(theta_true, kind="stable")
    theta_ordered = theta_true[heading_order]
    converged_mask = settle_converged[heading_order] > 0.5
    colors = np.where(converged_mask, "#4c88bd", "#c44e52")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    axes[0].scatter(
        theta_ordered,
        actual_settle_duration[heading_order],
        c=colors,
        s=18,
    )
    axes[0].set_ylabel("actual settling time [s]")
    axes[0].set_title(
        f"Converged headings: {int(np.count_nonzero(converged_mask))}/{theta_true.size}"
    )
    axes[1].scatter(
        theta_ordered,
        final_window_max_rate_change[heading_order],
        c=colors,
        s=18,
    )
    axes[1].plot(
        [-np.pi, np.pi],
        [convergence_tolerance, convergence_tolerance],
        color="#111111",
        linewidth=1.2,
        linestyle="--",
        label=f"tolerance={convergence_tolerance:g}",
    )
    positive_changes = final_window_max_rate_change[
        np.isfinite(final_window_max_rate_change) & (final_window_max_rate_change > 0.0)
    ]
    if positive_changes.size:
        axes[1].set_yscale("log")
    axes[1].set_ylabel("max HD-rate change over final window")
    axes[1].set_xlabel("fixed true heading [rad]")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlim(-np.pi, np.pi)
        _set_radian_ticks(axis, which="x")
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Frozen-heading tuning-sweep settling diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def _collapse_history_by_theta(
    *,
    theta_hd_pref: np.ndarray,
    r_hd_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if r_hd_history.ndim != 2:
        raise ValueError("r_hd_history must be a 2D array")
    collapsed_theta: np.ndarray | None = None
    collapsed_history: list[np.ndarray] = []
    for r_hd in r_hd_history:
        theta_current, r_hd_collapsed = collapse_activity_by_theta(theta_hd_pref, r_hd)
        if collapsed_theta is None:
            collapsed_theta = theta_current
        elif not np.allclose(theta_current, collapsed_theta):
            raise ValueError("Collapsed HD theta grid changed across history")
        collapsed_history.append(r_hd_collapsed)
    if collapsed_theta is None:
        return np.empty(0, dtype=float), np.empty((0, 0), dtype=float)
    return collapsed_theta, np.asarray(collapsed_history, dtype=float)


def _theta_grid_is_uniform(theta: np.ndarray) -> bool:
    theta = np.asarray(theta, dtype=float)
    if theta.size < 3:
        return True
    steps = np.diff(theta)
    return bool(np.allclose(steps, np.median(steps), rtol=1e-5, atol=1e-8))


def _plot_theta_heatmap(
    *,
    axis,
    time: np.ndarray,
    theta: np.ndarray,
    history: np.ndarray,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Draw empirical COM rows at their actual, potentially irregular angles."""
    image = np.asarray(history, dtype=float).T
    if _theta_grid_is_uniform(theta):
        theta_step = float(np.median(np.diff(theta))) if theta.size > 1 else 2.0 * np.pi
        extent = [
            float(time[0]),
            float(time[-1]),
            float(theta[0] - 0.5 * theta_step),
            float(theta[-1] + 0.5 * theta_step),
        ]
        return axis.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    return axis.pcolormesh(
        time,
        theta,
        image,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def _decode_activity_history(
    *, theta_hd_pref: np.ndarray, r_hd_history: np.ndarray, decoder
) -> np.ndarray:
    return np.asarray(
        [decoder(theta_hd_pref, np.asarray(r_hd, dtype=float)) for r_hd in r_hd_history],
        dtype=float,
    )


def plot_activity_heatmap(
    *,
    r_hd_history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
    title: str = "HD activity",
    theta_hd_pref: np.ndarray | None = None,
    theta_hd_decoded: np.ndarray | None = None,
    theta_hd_decoded_peak: np.ndarray | None = None,
    decode_theta_hd_pref: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
    firing_rate_label: str = "normalized HD firing rate [a.u.]",
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
        if theta_hd_pref is None or theta_hd_pref.size != r_hd_history.shape[1]:
            image = r_hd_history.T
            extent = [float(time[0]), float(time[-1]), 0, r_hd_history.shape[1] - 1] if time.size else None
            y_label = "HD neuron index [unitless]"
        else:
            theta_plot, r_hd_plot = _collapse_history_by_theta(
                theta_hd_pref=theta_hd_pref,
                r_hd_history=np.asarray(r_hd_history, dtype=float),
            )
            image = r_hd_plot.T
            if time.size and theta_plot.size > 1:
                theta_step = float(np.median(np.diff(theta_plot)))
                theta_lower = float(theta_plot[0] - 0.5 * theta_step)
                theta_upper = float(theta_plot[-1] + 0.5 * theta_step)
                extent = [float(time[0]), float(time[-1]), theta_lower, theta_upper]
            elif time.size:
                extent = [float(time[0]), float(time[-1]), -np.pi, np.pi]
            else:
                extent = None
            y_label = "HD preferred direction theta_HD [rad]"
    if (
        r_hd_history.size > 0
        and theta_hd_pref is not None
        and theta_hd_pref.size == r_hd_history.shape[1]
        and time.size > 0
    ):
        mesh = _plot_theta_heatmap(
            axis=axis,
            time=np.asarray(time, dtype=float),
            theta=theta_plot,
            history=r_hd_plot,
            cmap="viridis",
        )
    else:
        mesh = axis.imshow(image, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    if theta_hd_pref is not None and time.size > 0:
        if (
            decode_theta_hd_pref is not None
            and decode_theta_hd_pref.size == r_hd_history.shape[1]
        ):
            decode_theta_hd_pref = np.asarray(decode_theta_hd_pref, dtype=float)
            theta_hd_decoded = _decode_activity_history(
                theta_hd_pref=decode_theta_hd_pref,
                r_hd_history=np.asarray(r_hd_history, dtype=float),
                decoder=pva_decode,
            )
            theta_hd_decoded_peak = _decode_activity_history(
                theta_hd_pref=decode_theta_hd_pref,
                r_hd_history=np.asarray(r_hd_history, dtype=float),
                decoder=peak_decode,
            )
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
    _shade_dark_phase(axis, time=time, phase_id=phase_id)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(y_label)
    fig.colorbar(mesh, ax=axis, label=firing_rate_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hd_current_heatmap(
    *,
    current_history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_pref: np.ndarray | None = None,
    theta_true: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
    colorbar_label: str = "HD current [a.u.]",
) -> None:
    """Plot a time-by-HD-current heatmap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.88)
    current_history = np.asarray(current_history, dtype=float)
    time = np.asarray(time, dtype=float)
    if current_history.size == 0:
        image = np.zeros((1, 1), dtype=float)
        extent = None
        y_label = "HD neuron index [unitless]"
    else:
        if theta_hd_pref is None or theta_hd_pref.size != current_history.shape[1]:
            image = current_history.T
            extent = [float(time[0]), float(time[-1]), 0, current_history.shape[1] - 1] if time.size else None
            y_label = "HD neuron index [unitless]"
        else:
            theta_plot, current_plot = _collapse_history_by_theta(
                theta_hd_pref=theta_hd_pref,
                r_hd_history=current_history,
            )
            image = current_plot.T
            if time.size and theta_plot.size > 1:
                theta_step = float(np.median(np.diff(theta_plot)))
                extent = [
                    float(time[0]),
                    float(time[-1]),
                    float(theta_plot[0] - 0.5 * theta_step),
                    float(theta_plot[-1] + 0.5 * theta_step),
                ]
            elif time.size:
                extent = [float(time[0]), float(time[-1]), -np.pi, np.pi]
            else:
                extent = None
            y_label = "HD preferred direction theta_HD [rad]"
    finite_image = image[np.isfinite(image)]
    max_abs_current = float(np.max(np.abs(finite_image))) if finite_image.size else 1.0
    if max_abs_current <= 1e-12:
        max_abs_current = 1.0
    if (
        current_history.size > 0
        and theta_hd_pref is not None
        and theta_hd_pref.size == current_history.shape[1]
        and time.size > 0
    ):
        mesh = _plot_theta_heatmap(
            axis=axis,
            time=time,
            theta=theta_plot,
            history=current_plot,
            cmap="coolwarm",
            vmin=-max_abs_current,
            vmax=max_abs_current,
        )
    else:
        mesh = axis.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-max_abs_current,
            vmax=max_abs_current,
        )
    if (
        theta_hd_pref is not None
        and current_history.ndim == 2
        and theta_hd_pref.size == current_history.shape[1]
    ):
        axis.set_ylim(-np.pi, np.pi)
        _set_radian_ticks(axis, which="y")
    if theta_true is not None and time.size:
        theta_true = np.asarray(theta_true, dtype=float)
        if theta_true.size == time.size:
            axis.plot(
                time,
                _wrapped_trace_for_axis(theta_true),
                color="black",
                linewidth=0.75,
                alpha=0.75,
                label="true heading",
            )
            axis.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    if phase_id is not None and time.size:
        phase_id = np.asarray(phase_id, dtype=float)
        if phase_id.size == time.size:
            phase_change_indices = np.flatnonzero(np.diff(phase_id) != 0.0) + 1
            for phase_change_index in phase_change_indices:
                y_min, y_max = axis.get_ylim()
                phase_time = float(time[phase_change_index])
                axis.plot(
                    [phase_time, phase_time],
                    [y_min, y_max],
                    color="black",
                    linewidth=0.8,
                    alpha=0.35,
                )
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(y_label)
    fig.colorbar(mesh, ax=axis, label=colorbar_label)
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
    time_context: str = "source heatmap",
    firing_rate_label: str = "normalized HD firing rate [a.u.]",
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

    source_start_time = float(time[0])
    source_end_time = float(time[-1])
    slice_indices = _select_slice_indices(time, slice_times)
    color_values = plt.cm.viridis(np.linspace(0.0, 1.0, max(slice_indices.size, 1)))
    for color_index, slice_index in enumerate(slice_indices):
        r_hd = np.asarray(r_hd_history[slice_index], dtype=float)
        theta_plot, r_hd_plot = collapse_activity_by_theta(theta_hd_pref, r_hd)
        theta_bump = pva_decode(theta_hd_pref, r_hd)
        theta_peak = peak_decode(theta_hd_pref, r_hd)
        bump_strength = pva_vector_strength(theta_hd_pref, r_hd)
        rate_min = float(np.nanmin(r_hd_plot))
        rate_max = float(np.nanmax(r_hd_plot))
        if np.isfinite(theta_bump):
            label = (
                f"t={time[slice_index]:.2f} s, PVA={theta_bump:.2f} rad, "
                f"peak={theta_peak:.2f} rad, |PVA|={bump_strength:.2f}"
            )
            axis.plot(
                [theta_bump, theta_bump],
                [rate_min, rate_max],
                color=color_values[color_index],
                linewidth=0.9,
                alpha=0.45,
            )
            if np.isfinite(theta_peak):
                axis.plot(
                    [theta_peak, theta_peak],
                    [rate_min, rate_max],
                    color=color_values[color_index],
                    linewidth=0.9,
                    alpha=0.65,
                    linestyle="--",
                )
        else:
            label = f"t={time[slice_index]:.2f} s, PVA=nan"
        axis.plot(theta_plot, r_hd_plot, color=color_values[color_index], linewidth=1.4, label=label)
    axis.set_title(
        f"{title}\n{time_context} t={source_start_time:.2f}-{source_end_time:.2f} s"
    )
    axis.set_xlabel("HD preferred direction theta_HD [rad]")
    axis.set_ylabel(firing_rate_label)
    axis.set_xlim(-np.pi, np.pi)
    _set_radian_ticks(axis, which="x")
    axis.legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)
