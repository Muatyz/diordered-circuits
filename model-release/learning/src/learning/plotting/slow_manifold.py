"""Plots for autonomous slow-ring and Jacobian diagnostics."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def _plot_closed_pc_ring(axis, projection: np.ndarray, **plot_kwargs) -> None:
    projection = np.asarray(projection, dtype=float)
    if projection.shape[0] == 0:
        return
    closed_projection = np.concatenate([projection, projection[:1]], axis=0)
    axis.plot(
        closed_projection[:, 0],
        closed_projection[:, 1],
        closed_projection[:, 2],
        **plot_kwargs,
    )


def _label_pc123_axis(axis) -> None:
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    axis.view_init(elev=23.0, azim=-58.0)


def plot_ramesan_pca_variance_rank(
    *,
    history,
    path: str | Path,
) -> None:
    """Plot complete PCA spectra for state and neural observables.

    New histories use the block-standardized full Markov state as the primary
    PCA input and retain separate HD+HR-rate and PVA-input spectra.  Histories
    written before the five-state-variable migration are still accepted; in
    those archives the legacy primary spectrum was the HD+HR rate observable.
    """
    primary_explained = np.asarray(
        history["ramesan_pca_explained_variance_spectrum"], dtype=float
    )
    primary_cumulative = np.asarray(
        history["ramesan_pca_cumulative_explained_variance"], dtype=float
    )
    pva_explained = np.asarray(
        history["ramesan_pva_rate_pca_explained_variance_spectrum"], dtype=float
    )
    pva_cumulative = np.asarray(
        history["ramesan_pva_rate_pca_cumulative_explained_variance"], dtype=float
    )
    full_state_primary = "ramesan_pca_feature_scale" in history
    primary_label = (
        "block-standardized full Markov state"
        if full_state_primary
        else "legacy HD+HR firing-rate observable"
    )
    spectra = [
        (
            primary_label,
            primary_explained,
            primary_cumulative,
            "#3d78a8",
            "o",
        )
    ]
    if "ramesan_firing_rate_pca_explained_variance_spectrum" in history:
        firing_rate_explained = np.asarray(
            history["ramesan_firing_rate_pca_explained_variance_spectrum"],
            dtype=float,
        )
        firing_rate_cumulative = np.asarray(
            history["ramesan_firing_rate_pca_cumulative_explained_variance"],
            dtype=float,
        )
        spectra.append(
            (
                "HD+HR firing-rate observable",
                firing_rate_explained,
                firing_rate_cumulative,
                "#55a868",
                "^",
            )
        )
    spectra.append(
        (
            "paired-HD PVA-input observable",
            pva_explained,
            pva_cumulative,
            "#d17c1f",
            "s",
        )
    )

    for label, explained, cumulative, _color, _marker in spectra:
        if explained.ndim != 1 or cumulative.shape != explained.shape or not explained.size:
            raise ValueError(f"{label} PCA spectrum must be matching non-empty 1D arrays")
        if not np.all(np.isfinite(explained)) or not np.all(np.isfinite(cumulative)):
            raise ValueError(f"{label} PCA spectrum must be finite")

    figure, (variance_axis, cumulative_axis) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), constrained_layout=True
    )
    for label, explained, cumulative, color, marker in spectra:
        rank = np.arange(1, explained.size + 1)
        positive = explained > 0.0
        variance_axis.plot(
            rank[positive],
            explained[positive],
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.1,
            label=label,
        )
        cumulative_axis.plot(
            rank,
            cumulative,
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.2,
            label=label,
        )

    variance_axis.axvline(3, color="black", linestyle="--", linewidth=0.9)
    variance_axis.set_yscale("log")
    variance_axis.set_xlabel("PCA rank")
    variance_axis.set_ylabel("individual explained-variance fraction")
    variance_axis.set_title("A  Variance–rank spectrum")
    variance_axis.grid(alpha=0.2)
    variance_axis.legend(fontsize=8)

    cumulative_axis.axvline(3, color="black", linestyle="--", linewidth=0.9)
    for threshold in (0.80, 0.90, 0.95):
        cumulative_axis.axhline(
            threshold,
            color="#777777",
            linestyle=":" if threshold < 0.95 else "--",
            linewidth=0.8,
        )
    cumulative_axis.set_ylim(0.0, 1.02)
    cumulative_axis.set_xlabel("number of retained PCs")
    cumulative_axis.set_ylabel("cumulative explained-variance fraction")
    pc3_text = ", ".join(
        f"{label.split()[0]}={100.0 * np.sum(explained[:3]):.1f}%"
        for label, explained, _cumulative, _color, _marker in spectra
    )
    cumulative_axis.set_title(
        "B  Cumulative variance\n"
        f"PC1–3: {pc3_text}"
    )
    cumulative_axis.grid(alpha=0.2)
    cumulative_axis.legend(fontsize=8, loc="lower right")

    figure.suptitle(
        "Linear PCA dimensionality of uniformly cue-sampled state and observables\n"
        "PVA is a phase readout; q and Jacobians remain in the full Markov state",
        fontsize=12,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ramesan_firing_rate_diagnostics(
    *,
    history,
    path: str | Path,
) -> None:
    """Plot state-PC geometry, q, tangent flow, and local lambda_max."""
    probe_phase = np.asarray(history["ramesan_probe_phase"], dtype=float)
    probe_pc = np.asarray(history["ramesan_probe_pc"], dtype=float)
    explained = np.asarray(
        history["ramesan_pca_explained_variance"], dtype=float
    )
    coordinate_velocity = np.asarray(
        history["ramesan_probe_coordinate_phase_velocity"], dtype=float
    )
    tangent_state_speed = np.asarray(
        history["ramesan_probe_tangent_state_speed"], dtype=float
    )
    probe_flow = np.asarray(history["ramesan_probe_flow"], dtype=float)
    candidate_pc = np.asarray(history["ramesan_candidate_pc"], dtype=float)
    candidate_q = np.asarray(history["ramesan_candidate_q"], dtype=float)
    candidate_below_threshold = np.asarray(
        history["ramesan_candidate_below_q_threshold"], dtype=bool
    )
    q_threshold = float(np.asarray(history["ramesan_q_threshold"]))
    jacobian_pc = np.asarray(history["ramesan_jacobian_anchor_pc"], dtype=float)
    lambda_max_real = np.asarray(
        history["ramesan_jacobian_lambda_max_real"], dtype=float
    )
    if probe_pc.shape != (probe_phase.size, 3) or probe_phase.size < 4:
        raise ValueError("Ramesan plot requires a non-empty PC123 probe ring")

    figure = plt.figure(figsize=(14.5, 10.5), constrained_layout=True)
    ring_axis = figure.add_subplot(2, 2, 1, projection="3d")
    ring_scatter = ring_axis.scatter(
        probe_pc[:, 0],
        probe_pc[:, 1],
        probe_pc[:, 2],
        c=probe_phase,
        cmap="twilight",
        s=13,
        alpha=0.78,
        linewidths=0.0,
        rasterized=True,
    )
    _plot_closed_pc_ring(
        ring_axis,
        probe_pc,
        color="black",
        linewidth=0.7,
        alpha=0.45,
    )
    _label_pc123_axis(ring_axis)
    explained_text = "/".join(
        f"{100.0 * value:.1f}%" for value in np.pad(explained[:3], (0, max(0, 3 - explained[:3].size)))
    )
    full_state_pca = "ramesan_pca_feature_scale" in history
    ring_kind = (
        "block-standardized canonical-state ring"
        if full_state_pca
        else "legacy firing-rate-observable ring"
    )
    ring_axis.set_title(
        f"A  Cue-defined {ring_kind}\n"
        f"PC1/2/3 variance = {explained_text}"
    )
    figure.colorbar(
        ring_scatter,
        ax=ring_axis,
        shrink=0.68,
        pad=0.08,
        label="uniform cue phase [rad]",
    )

    flow_axis = figure.add_subplot(2, 2, 2)
    flow_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    flow_axis.plot(
        probe_phase,
        coordinate_velocity,
        color="#2f6da5",
        linewidth=1.5,
        label=r"coordinate tangent velocity $\dot{\phi}$",
    )
    flow_axis.set_xlim(-np.pi, np.pi)
    flow_axis.set_xlabel("cue-defined ring phase [rad]")
    flow_axis.set_ylabel(r"$\dot{\phi}$ [rad/s]", color="#2f6da5")
    flow_axis.tick_params(axis="y", labelcolor="#2f6da5")
    flow_axis.grid(alpha=0.2)
    state_speed_axis = flow_axis.twinx()
    state_dimension = max(1, probe_flow.shape[1])
    state_speed_axis.plot(
        probe_phase,
        np.abs(tangent_state_speed) / np.sqrt(state_dimension),
        color="#d17c1f",
        linewidth=1.1,
        alpha=0.8,
        label=r"$|v_{\parallel}|/\sqrt{D}$",
    )
    state_speed_axis.set_ylabel(
        r"state-space tangent speed $|v_{\parallel}|/\sqrt{D}$",
        color="#d17c1f",
    )
    state_speed_axis.tick_params(axis="y", labelcolor="#d17c1f")
    flow_axis.set_title("B  Zero-input tangent velocity field")

    q_axis = figure.add_subplot(2, 2, 3, projection="3d")
    _plot_closed_pc_ring(
        q_axis,
        probe_pc,
        color="#8a8a8a",
        linewidth=1.0,
        alpha=0.48,
    )
    if candidate_q.size:
        positive_q = candidate_q[candidate_q > 0.0]
        q_color_floor = (
            float(np.min(positive_q))
            if positive_q.size
            else np.finfo(float).tiny
        )
        log_q = np.log10(np.maximum(candidate_q, q_color_floor))
        draw_order = np.argsort(candidate_q)[::-1]
        q_scatter = q_axis.scatter(
            candidate_pc[draw_order, 0],
            candidate_pc[draw_order, 1],
            candidate_pc[draw_order, 2],
            c=log_q[draw_order],
            cmap="viridis",
            s=13,
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )
        exact_fixed_point = candidate_q == 0.0
        if np.any(exact_fixed_point):
            q_axis.scatter(
                candidate_pc[exact_fixed_point, 0],
                candidate_pc[exact_fixed_point, 1],
                candidate_pc[exact_fixed_point, 2],
                s=34,
                marker="*",
                color="black",
                linewidths=0.0,
            )
        figure.colorbar(
            q_scatter,
            ax=q_axis,
            shrink=0.68,
            pad=0.08,
            label=r"$\log_{10} q(x)$, $q=\frac{1}{2}||F(x)||^2$",
        )
        q_axis.set_title(
            "C  Ramesan-style slow-point candidates\n"
            f"q <= {q_threshold:.1e}: "
            f"{np.count_nonzero(candidate_below_threshold)}/{candidate_q.size}"
        )
    else:
        q_axis.text2D(
            0.5,
            0.5,
            "no low-speed candidates captured",
            transform=q_axis.transAxes,
            ha="center",
            va="center",
        )
        q_axis.set_title("C  Ramesan-style slow-point candidates")
    _label_pc123_axis(q_axis)

    jacobian_axis = figure.add_subplot(2, 2, 4, projection="3d")
    _plot_closed_pc_ring(
        jacobian_axis,
        probe_pc,
        color="#8a8a8a",
        linewidth=1.0,
        alpha=0.48,
    )
    finite_lambda = np.isfinite(lambda_max_real)
    lambda_extent = max(
        float(np.max(np.abs(lambda_max_real[finite_lambda])))
        if np.any(finite_lambda)
        else 0.0,
        1e-12,
    )
    lambda_scatter = None
    if np.any(finite_lambda):
        lambda_scatter = jacobian_axis.scatter(
            jacobian_pc[finite_lambda, 0],
            jacobian_pc[finite_lambda, 1],
            jacobian_pc[finite_lambda, 2],
            c=lambda_max_real[finite_lambda],
            cmap="coolwarm",
            norm=TwoSlopeNorm(
                vmin=-lambda_extent,
                vcenter=0.0,
                vmax=lambda_extent,
            ),
            s=48,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.25,
            rasterized=True,
        )
    else:
        jacobian_axis.text2D(
            0.5,
            0.5,
            "Jacobian eigensolver did not converge",
            transform=jacobian_axis.transAxes,
            ha="center",
            va="center",
        )
    _label_pc123_axis(jacobian_axis)
    jacobian_axis.set_title(
        "D  Local full-state Jacobian at q-small points\n"
        r"color = $\lambda_{\max}=\max_k\,\mathrm{Re}\,\lambda_k$"
    )
    if lambda_scatter is not None:
        figure.colorbar(
            lambda_scatter,
            ax=jacobian_axis,
            shrink=0.68,
            pad=0.08,
            label=r"$\lambda_{\max}$ [1/s]",
        )

    figure.suptitle(
        "Ramesan-style diagnostics of the trained zero-input HD network\n"
        "PC1–3 use the standardized full Markov state; PVA remains a separate phase readout"
        if full_state_pca
        else "Legacy PC1–3 use firing rates; q, tangent flow, and Jacobian use full state",
        fontsize=13,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ramesan_phase_landscape(
    *,
    history,
    path: str | Path,
) -> None:
    """Plot trajectory slow regions and a qualified 1-D phase landscape."""
    phase = np.asarray(history["ramesan_phase_bin_center"], dtype=float)
    sample_count = np.asarray(
        history["ramesan_phase_sample_count"], dtype=float
    )
    q_min = np.asarray(history["ramesan_phase_q_min"], dtype=float)
    q_median = np.asarray(history["ramesan_phase_q_median"], dtype=float)
    slow_mask = np.asarray(history["ramesan_phase_slow_mask"], dtype=bool)
    velocity_median = np.asarray(
        history["ramesan_phase_velocity_median"], dtype=float
    )
    velocity_q25 = np.asarray(
        history["ramesan_phase_velocity_q25"], dtype=float
    )
    velocity_q75 = np.asarray(
        history["ramesan_phase_velocity_q75"], dtype=float
    )
    velocity_smoothed = np.asarray(
        history["ramesan_phase_velocity_smoothed"], dtype=float
    )
    tilted_potential = np.asarray(
        history["ramesan_phase_tilted_potential"], dtype=float
    )
    mean_drift = float(np.asarray(history["ramesan_phase_mean_drift"]))
    fixed_theta = np.asarray(
        history["ramesan_phase_fixed_point_theta"], dtype=float
    )
    fixed_stability = np.asarray(
        history["ramesan_phase_fixed_point_stability"], dtype=np.int8
    )
    q_threshold = float(np.asarray(history["ramesan_q_threshold"]))
    ambient_theta = np.asarray(
        history.get("ramesan_ambient_theta", np.empty(0)), dtype=float
    )
    ambient_q = np.asarray(
        history.get("ramesan_ambient_q", np.empty(0)), dtype=float
    )
    ambient_scale = np.asarray(
        history.get("ramesan_ambient_requested_scale", np.empty(0)), dtype=float
    )
    if phase.ndim != 1 or phase.size < 8:
        raise ValueError("Ramesan phase landscape requires at least eight phase bins")
    angular_step = 2.0 * np.pi / phase.size

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    q_axis, velocity_axis, potential_axis, ambient_axis = axes.flat

    def shade_phase_regions(axis) -> None:
        for index, current_phase in enumerate(phase):
            left = current_phase - 0.5 * angular_step
            right = current_phase + 0.5 * angular_step
            if sample_count[index] <= 0:
                axis.axvspan(left, right, color="#d9d9d9", alpha=0.36, linewidth=0)
            elif slow_mask[index]:
                axis.axvspan(left, right, color="#75d5d8", alpha=0.24, linewidth=0)

    finite_positive_q = np.concatenate(
        [
            q_min[np.isfinite(q_min) & (q_min > 0.0)],
            q_median[np.isfinite(q_median) & (q_median > 0.0)],
            np.asarray([q_threshold]),
        ]
    )
    q_floor = max(float(np.min(finite_positive_q)), np.finfo(float).tiny)
    shade_phase_regions(q_axis)
    q_axis.plot(
        phase,
        np.log10(np.maximum(q_median, q_floor)),
        color="#777777",
        linewidth=1.0,
        label="trajectory-bin median q",
    )
    q_axis.plot(
        phase,
        np.log10(np.maximum(q_min, q_floor)),
        color="black",
        linewidth=1.5,
        label="trajectory-bin minimum q",
    )
    q_axis.axhline(
        np.log10(q_threshold),
        color="#b24745",
        linestyle="--",
        linewidth=1.1,
        label=fr"slow threshold $q={q_threshold:.1e}$",
    )
    q_axis.set_title("A  Slow regions along decoded phase")
    q_axis.set_ylabel(r"$\log_{10} q$, $q=\frac{1}{2}\|F(x)\|^2$")
    q_axis.legend(fontsize=8, loc="best")

    shade_phase_regions(velocity_axis)
    finite_velocity_band = np.isfinite(velocity_q25) & np.isfinite(velocity_q75)
    velocity_axis.fill_between(
        phase,
        velocity_q25,
        velocity_q75,
        where=finite_velocity_band,
        color="#8fb6d8",
        alpha=0.35,
        label="within-bin IQR",
    )
    velocity_axis.plot(
        phase,
        velocity_median,
        color="#3d78a8",
        linewidth=0.9,
        alpha=0.75,
        label="empirical median",
    )
    velocity_axis.plot(
        phase,
        velocity_smoothed,
        color="black",
        linewidth=1.5,
        label="periodically smoothed field",
    )
    velocity_axis.axhline(0.0, color="#555555", linewidth=0.8)
    stable = fixed_stability == -1
    unstable = fixed_stability == 1
    velocity_axis.scatter(
        fixed_theta[stable],
        np.zeros(np.count_nonzero(stable)),
        color="#25aeb4",
        edgecolor="black",
        linewidth=0.3,
        s=38,
        zorder=5,
        label="1-D attracting zero",
    )
    velocity_axis.scatter(
        fixed_theta[unstable],
        np.zeros(np.count_nonzero(unstable)),
        color="#e58b35",
        edgecolor="black",
        linewidth=0.3,
        s=38,
        zorder=5,
        label="1-D repelling zero",
    )
    velocity_axis.set_title("B  Empirical zero-input phase field")
    velocity_axis.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    velocity_axis.legend(fontsize=8, loc="best", ncol=2)

    shade_phase_regions(potential_axis)
    potential_axis.plot(phase, tilted_potential, color="black", linewidth=1.7)
    if fixed_theta.size:
        fixed_potential = np.interp(fixed_theta, phase, tilted_potential)
        potential_axis.scatter(
            fixed_theta[stable],
            fixed_potential[stable],
            color="#25aeb4",
            edgecolor="black",
            linewidth=0.3,
            s=38,
            zorder=5,
            label="attracting zero",
        )
        potential_axis.scatter(
            fixed_theta[unstable],
            fixed_potential[unstable],
            color="#e58b35",
            edgecolor="black",
            linewidth=0.3,
            s=38,
            zorder=5,
            label="repelling zero",
        )
    potential_axis.set_title(
        "C  Effective tilted phase potential (1-D approximation)\n"
        + fr"$\dot{{\theta}}=-\partial_\theta W_{{eff}}$, "
        + fr"$\bar v={mean_drift:.3g}$ rad/s; cut at $\pm\pi$"
    )
    potential_axis.set_ylabel("effective potential [a.u.]")
    potential_axis.legend(fontsize=8, loc="best")

    if ambient_q.size:
        unique_scale = np.unique(ambient_scale)
        heatmap = np.full((unique_scale.size, phase.size), np.nan)
        ambient_bin = np.searchsorted(
            np.linspace(-np.pi, np.pi, phase.size + 1),
            ambient_theta,
            side="right",
        ) - 1
        ambient_bin = np.clip(ambient_bin, 0, phase.size - 1)
        for scale_index, scale in enumerate(unique_scale):
            for phase_index in range(phase.size):
                mask = (ambient_scale == scale) & (ambient_bin == phase_index)
                if np.any(mask):
                    heatmap[scale_index, phase_index] = float(
                        np.median(np.log10(np.maximum(ambient_q[mask], q_floor)))
                    )
        image = ambient_axis.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-np.pi, np.pi, -0.5, unique_scale.size - 0.5),
            cmap="viridis",
        )
        ambient_axis.set_yticks(np.arange(unique_scale.size))
        ambient_axis.set_yticklabels(
            [
                f"{scale:g} ({100.0 * np.mean(ambient_q[ambient_scale == scale] <= q_threshold):.1f}% slow)"
                for scale in unique_scale
            ]
        )
        figure.colorbar(
            image,
            ax=ambient_axis,
            shrink=0.82,
            label=r"median $\log_{10} q$",
        )
        ambient_axis.set_title("D  Off-manifold normal tube around trajectories")
        ambient_axis.set_ylabel("standardized perturbation scale\n(fraction below q threshold)")
    else:
        ambient_axis.text(
            0.5,
            0.5,
            "ambient tube probe disabled",
            ha="center",
            va="center",
            transform=ambient_axis.transAxes,
        )
        ambient_axis.set_title("D  Off-manifold normal tube around trajectories")

    for axis in axes.flat:
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
        axis.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
        axis.set_xlabel("decoded phase [rad]")
        axis.grid(alpha=0.18)

    figure.suptitle(
        "Trajectory-conditioned slow regions and zero-input effective phase landscape\n"
        "C is a projected 1-D diagnostic, not a physical energy unless phase flow is single-valued and normal flow is negligible",
        fontsize=13,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_slow_manifold_diagnostics(
    *,
    history,
    path: str | Path,
) -> None:
    """Plot full-state PCA, tangent flow, and leading Jacobian rates."""
    candidate_state = np.asarray(history["candidate_state"], dtype=float)
    candidate_theta = np.asarray(history["candidate_theta"], dtype=float)
    manifold_state = np.asarray(history["manifold_state"], dtype=float)
    manifold_theta = np.asarray(history["manifold_theta"], dtype=float)
    if manifold_theta.size == 0:
        candidate_speed = np.asarray(history["candidate_speed"], dtype=float)
        bin_sample_count = np.asarray(
            history.get("angular_bin_sample_count", np.empty(0)), dtype=float
        )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].scatter(
            candidate_theta,
            candidate_speed,
            s=10,
            alpha=0.45,
            color="tab:blue",
        )
        axes[0].set_yscale("log")
        axes[0].set_xlim(-np.pi, np.pi)
        axes[0].set_xlabel("decoded candidate angle (rad)")
        axes[0].set_ylabel(r"full-state $||f(x)||$")
        axes[0].set_title("Retained low-speed states")
        if bin_sample_count.size:
            bin_center = np.linspace(
                -np.pi + np.pi / bin_sample_count.size,
                np.pi - np.pi / bin_sample_count.size,
                bin_sample_count.size,
            )
            axes[1].bar(
                bin_center,
                bin_sample_count,
                width=2.0 * np.pi / bin_sample_count.size,
                color="tab:orange",
            )
            support = float(np.mean(bin_sample_count > 0))
            cluster_theta = np.asarray(
                history.get("low_speed_angle_cluster_theta", np.empty(0)),
                dtype=float,
            )
            for current_theta in cluster_theta:
                axes[1].axvline(
                    current_theta,
                    color="black",
                    linewidth=0.8,
                    alpha=0.5,
                )
            axes[1].set_title(
                f"Angular support={100.0 * support:.1f}%, "
                f"clusters={cluster_theta.size}"
            )
        axes[1].set_xlim(-np.pi, np.pi)
        axes[1].set_xlabel("decoded angle (rad)")
        axes[1].set_ylabel("candidate count")
        figure.suptitle(
            "Slow-ring fit rejected: low-speed candidates do not cover the ring"
        )
        figure.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    angular_flow = np.asarray(history["angular_flow"], dtype=float)
    normal_flow_norm = np.asarray(history["normal_flow_norm"], dtype=float)
    fixed_theta = np.asarray(history["fixed_point_theta"], dtype=float)
    fixed_stability = np.asarray(history["fixed_point_stability"], dtype=int)
    jacobian_theta = np.asarray(history["jacobian_anchor_theta"], dtype=float)
    eigenvalue_real = np.asarray(history["jacobian_eigenvalue_real"], dtype=float)
    alignment = np.asarray(history["slow_mode_tangent_alignment"], dtype=float)

    center = np.mean(manifold_state, axis=0)
    _left, singular_value, right = np.linalg.svd(
        manifold_state - center,
        full_matrices=False,
    )
    component_count = min(3, right.shape[0])
    basis = right[:component_count]
    manifold_projection = (manifold_state - center) @ basis.T
    candidate_projection = (candidate_state - center) @ basis.T
    if component_count < 3:
        manifold_projection = np.pad(
            manifold_projection, ((0, 0), (0, 3 - component_count))
        )
        candidate_projection = np.pad(
            candidate_projection, ((0, 0), (0, 3 - component_count))
        )
    variance = np.square(singular_value)
    explained = variance / np.sum(variance) if np.sum(variance) > 0.0 else variance

    figure = plt.figure(figsize=(16, 4.8))
    pca_axis = figure.add_subplot(1, 3, 1, projection="3d")
    candidate_scatter = pca_axis.scatter(
        candidate_projection[:, 0],
        candidate_projection[:, 1],
        candidate_projection[:, 2],
        c=candidate_theta,
        cmap="twilight",
        s=7,
        alpha=0.28,
        linewidths=0.0,
    )
    pca_axis.plot(
        manifold_projection[:, 0],
        manifold_projection[:, 1],
        manifold_projection[:, 2],
        color="black",
        linewidth=2.0,
    )
    pca_axis.set_xlabel("PC1")
    pca_axis.set_ylabel("PC2")
    pca_axis.set_zlabel("PC3")
    explained_three = np.pad(explained[:3], (0, max(0, 3 - explained[:3].size)))
    pca_axis.set_title(
        "Full-state candidates and periodic spline\n"
        + "manifold PCA variance="
        + "/".join(f"{100.0 * value:.1f}%" for value in explained_three)
    )
    figure.colorbar(candidate_scatter, ax=pca_axis, shrink=0.65, label="decoded angle (rad)")

    flow_axis = figure.add_subplot(1, 3, 2)
    flow_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    flow_axis.plot(manifold_theta, angular_flow, color="tab:blue", label=r"$\dot{\theta}$")
    for root, stability in zip(fixed_theta, fixed_stability, strict=True):
        flow_axis.scatter(
            root,
            0.0,
            s=48,
            facecolor="tab:green" if stability == -1 else "white",
            edgecolor="tab:green" if stability == -1 else "tab:red",
            zorder=4,
        )
    flow_axis.set_xlim(-np.pi, np.pi)
    flow_axis.set_xlabel("manifold angle (rad)")
    flow_axis.set_ylabel(r"projected $\dot{\theta}$ (rad/s)", color="tab:blue")
    normal_axis = flow_axis.twinx()
    normal_axis.plot(
        manifold_theta,
        normal_flow_norm,
        color="tab:orange",
        alpha=0.65,
        label="normal residual",
    )
    normal_axis.set_ylabel("normal flow norm", color="tab:orange")
    flow_axis.set_title("Autonomous flow and reversal points")

    spectrum_axis = figure.add_subplot(1, 3, 3)
    for mode_index in range(eigenvalue_real.shape[1]):
        spectrum_axis.plot(
            jacobian_theta,
            eigenvalue_real[:, mode_index],
            linewidth=2.0 if mode_index == 0 else 0.9,
            alpha=1.0 if mode_index < 2 else 0.45,
            label=f"mode {mode_index + 1}" if mode_index < 2 else None,
        )
    spectrum_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    spectrum_axis.set_xlim(-np.pi, np.pi)
    spectrum_axis.set_xlabel("manifold angle (rad)")
    spectrum_axis.set_ylabel(r"Re($\lambda$) (1/s)")
    spectrum_axis.legend(loc="best", fontsize=8)
    alignment_axis = spectrum_axis.twinx()
    alignment_axis.plot(
        jacobian_theta,
        alignment,
        color="tab:purple",
        linestyle="--",
        linewidth=1.2,
        label="slow-mode/tangent alignment",
    )
    alignment_axis.set_ylim(0.0, 1.05)
    alignment_axis.set_ylabel("tangent alignment", color="tab:purple")
    spectrum_axis.set_title("Full-dynamics Jacobian along ring")

    figure.suptitle("Ságodi-style autonomous slow-manifold diagnostic")
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
