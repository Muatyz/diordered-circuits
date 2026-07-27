"""Layered tutor/input/synapse/U/V scene backed by saved numerical arrays."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from prospective.analysis.decoding import learned_competitive_positions, linear_center
from prospective.animation.storyboard import top_update_connections
from prospective.config.schema import ExperimentConfig


def aggregate_profile_by_position(
    positions: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average neurons sharing one learned position without spatial smoothing."""

    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if positions.shape != values.shape:
        raise ValueError("positions and values must have matching shapes")
    unique, inverse = np.unique(positions, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    return unique, sums / counts


def baseline_subtracted_center(positions: np.ndarray, values: np.ndarray) -> float:
    """Decode profile shape after removing its spatially uniform minimum."""

    values = np.asarray(values, dtype=float)
    return linear_center(np.maximum(values - np.min(values), 0.0), np.asarray(positions, dtype=float))


@dataclass
class SceneArtists:
    """Mutable artists updated from one exact saved sample at a time."""

    tutor_line: object
    tutor_center: object
    u_profile: object
    u_center: object
    v_profile: object
    v_center: object
    input_scatter: object
    u_scatter: object
    v_scatter: object
    rate_halo: object
    base_connections: LineCollection
    update_connections: LineCollection
    matrix_image: object
    time_text: object
    equation_text: object


class FeedforwardScene:
    """Render continuous profiles together with discrete neurons and synapses."""

    def __init__(
        self,
        history: dict[str, np.ndarray],
        config: ExperimentConfig,
        *,
        title: str,
        mode: str = "neural",
        playback_acceleration: float | None = None,
    ) -> None:
        if "weights" not in history or "delta_weights" not in history:
            raise ValueError("animation requires a run made with animation.enabled=true")
        self.history = history
        self.config = config
        self.mode = mode
        self.playback_acceleration = playback_acceleration
        self.x_input = np.linspace(0.0, config.geometry.length, config.geometry.n_input, endpoint=False)
        # Output neurons are permutation-symmetric before learning. For an
        # interpretable spatial profile, place each one at the peak of its final
        # learned feedforward row and retain that layout for the whole movie.
        self.x_comp = learned_competitive_positions(history["weights"][-1], self.x_input)
        self.comp_order = np.argsort(self.x_comp, kind="stable")

        self.fig = plt.figure(figsize=(14.4, 9.0), facecolor="#10131a")
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.07)
        grid = self.fig.add_gridspec(
            4,
            3,
            height_ratios=[0.9, 0.66, 0.66, 1.75],
            width_ratios=[4.8, 1.65, 0.48],
            hspace=0.13,
            wspace=0.28,
        )
        self.ax_tutor = self.fig.add_subplot(grid[0, :2])
        self.ax_u = self.fig.add_subplot(grid[1, :2], sharex=self.ax_tutor)
        self.ax_v = self.fig.add_subplot(grid[2, :2], sharex=self.ax_tutor)
        self.ax_network = self.fig.add_subplot(grid[3, 0], sharex=self.ax_tutor)
        self.ax_matrix = self.fig.add_subplot(grid[3, 1])
        colorbar_grid = grid[3, 2].subgridspec(1, 3, wspace=1.25)
        self.colorbar_axes = [self.fig.add_subplot(colorbar_grid[0, index]) for index in range(3)]
        self.profile_axes = [self.ax_tutor, self.ax_u, self.ax_v]
        for ax in [*self.profile_axes, self.ax_network, self.ax_matrix]:
            ax.set_facecolor("#10131a")
            ax.tick_params(colors="#d7dce2")
            for spine in ax.spines.values():
                spine.set_color("#55606f")
        for ax in self.profile_axes:
            ax.grid(axis="x", color="#313844", lw=0.5, alpha=0.6)
            ax.tick_params(axis="x", labelbottom=False)
        self.ax_tutor.set_xlim(0, config.geometry.length)

        tutor_max = max(float(np.max(history["tutor_rate"])), 1e-12)
        u_min, u_max = float(np.min(history["membrane"])), float(np.max(history["membrane"]))
        v_min, v_max = float(np.min(history["adaptation"])), float(np.max(history["adaptation"]))
        self.ax_tutor.set_ylim(-0.02 * tutor_max, 1.12 * tutor_max)
        self.ax_u.set_ylim(min(0.0, 1.08 * u_min), max(1e-12, 1.12 * u_max))
        self.ax_v.set_ylim(min(0.0, 1.08 * v_min), max(1e-12, 1.12 * v_max))
        self.ax_tutor.set_ylabel("Tutor R", color="#ffcf56")
        self.ax_u.set_ylabel("Membrane U", color="#ff8a65")
        self.ax_v.set_ylabel("Adaptation V", color="#64d8cb")
        self.ax_tutor.set_title(title, color="white", fontsize=15)

        self.ax_network.set_xlim(-0.02 * config.geometry.length, 1.02 * config.geometry.length)
        self.ax_network.set_ylim(-0.1, 4.2)
        self.ax_network.set_yticks(
            [0.55, 1.75, 3.55],
            labels=["V nodes", "U nodes (+ r halo)", "R nodes"],
        )
        self.ax_network.set_xlabel("preferred position", color="#d7dce2")
        self.ax_network.tick_params(axis="y", labelcolor="#d7dce2")
        for y in [0.55, 1.75, 3.55]:
            self.ax_network.axhline(y, color="#313844", lw=0.6)

        self.weight_norm = Normalize(vmin=0.0, vmax=max(float(np.max(history["weights"])), 1e-12))
        max_delta = max(float(np.max(np.abs(history["delta_weights"]))), 1e-12)
        self.delta_norm = TwoSlopeNorm(vmin=-max_delta, vcenter=0.0, vmax=max_delta)
        u_abs = max(float(np.max(np.abs(history["membrane"]))), 1e-12)
        self.u_norm = TwoSlopeNorm(vmin=-u_abs, vcenter=0.0, vmax=u_abs)
        self.v_norm = Normalize(vmin=min(0.0, v_min), vmax=max(v_max, 1e-12))
        self.r_max = max(float(np.max(history["rate"])), 1e-12)

        tutor_line, = self.ax_tutor.plot([], [], color="#ffcf56", lw=3, label="continuous tutor profile")
        tutor_center = self.ax_tutor.axvline(0.0, color="white", ls="--", lw=1, label="tutor center")
        u_profile, = self.ax_u.plot([], [], color="#ff8a65", lw=2.3, marker="o", ms=3, label="U over learned positions")
        u_center = self.ax_u.axvline(0.0, color="#ffd0c2", ls="--", lw=1, label="U shape center")
        v_profile, = self.ax_v.plot([], [], color="#64d8cb", lw=2.3, marker="o", ms=3, label="V over learned positions")
        v_center = self.ax_v.axvline(0.0, color="#c7fff7", ls="--", lw=1, label="V shape center")
        self.ax_tutor.legend(loc="upper right", fontsize=7, framealpha=0.2)
        self.ax_u.legend(loc="upper right", fontsize=7, framealpha=0.2)
        self.ax_v.legend(loc="upper right", fontsize=7, framealpha=0.2)

        input_scatter = self.ax_network.scatter(
            self.x_input, np.full_like(self.x_input, 3.55), c=np.zeros_like(self.x_input),
            cmap="YlOrBr", vmin=0, vmax=tutor_max, s=70, edgecolor="white", lw=0.3, zorder=4,
        )
        u_scatter = self.ax_network.scatter(
            self.x_comp, np.full_like(self.x_comp, 1.75), c=np.zeros_like(self.x_comp),
            cmap="coolwarm", norm=self.u_norm, s=70, edgecolor="white", lw=0.3, zorder=4,
        )
        rate_halo = self.ax_network.scatter(
            self.x_comp, np.full_like(self.x_comp, 1.75), facecolors="none", edgecolors="#78e3ff",
            s=np.full_like(self.x_comp, 75), lw=1.2, zorder=5,
        )
        v_scatter = self.ax_network.scatter(
            self.x_comp, np.full_like(self.x_comp, 0.55), c=np.zeros_like(self.x_comp),
            cmap="viridis", norm=self.v_norm, s=70, edgecolor="white", lw=0.3, zorder=4,
        )
        base_connections = LineCollection([], cmap="magma", norm=self.weight_norm, alpha=0.6, zorder=1)
        update_connections = LineCollection([], cmap="coolwarm", norm=self.delta_norm, alpha=0.9, zorder=2)
        self.ax_network.add_collection(base_connections)
        self.ax_network.add_collection(update_connections)

        initial_matrix = history["weights"][0][self.comp_order] if mode == "global" else history["weights"][0]
        matrix_image = self.ax_matrix.imshow(
            initial_matrix,
            origin="lower",
            aspect="auto",
            cmap="magma",
            norm=self.weight_norm,
        )
        matrix_title = "J(t): rows in final learned order" if mode == "global" else "J matrix"
        self.ax_matrix.set(title=matrix_title, xlabel="input / pre", ylabel="competitive / post")
        self.ax_matrix.title.set_fontsize(8)
        self.ax_matrix.title.set_color("white")
        self.ax_matrix.xaxis.label.set_color("#d7dce2")
        self.ax_matrix.yaxis.label.set_color("#d7dce2")
        self.ax_matrix.tick_params(labelsize=7, colors="#d7dce2")
        time_text = self.ax_network.text(
            0.01,
            0.98,
            "",
            transform=self.ax_network.transAxes,
            va="top",
            color="white",
            fontsize=8.5,
            clip_on=True,
        )
        equation_text = self.ax_network.text(
            0.01,
            0.03,
            "",
            transform=self.ax_network.transAxes,
            va="bottom",
            color="#d7dce2",
            fontsize=7.5,
            clip_on=True,
        )
        self.artists = SceneArtists(
            tutor_line, tutor_center, u_profile, u_center, v_profile, v_center,
            input_scatter, u_scatter, v_scatter, rate_halo, base_connections,
            update_connections, matrix_image, time_text, equation_text,
        )
        for colorbar, label in [
            (self.fig.colorbar(input_scatter, cax=self.colorbar_axes[0]), "R"),
            (self.fig.colorbar(u_scatter, cax=self.colorbar_axes[1]), "U"),
            (self.fig.colorbar(v_scatter, cax=self.colorbar_axes[2]), "V"),
        ]:
            colorbar.set_label(label, color="#d7dce2")
            colorbar.ax.tick_params(colors="#d7dce2", labelsize=6)

    def update(self, frame_index: int):
        """Update every visual variable from one exact saved sample index."""

        h, a = self.history, self.artists
        tutor = h["tutor_rate"][frame_index]
        weights = h["weights"][frame_index]
        delta = h["delta_weights"][frame_index]
        membrane = h["membrane"][frame_index]
        adaptation = h["adaptation"][frame_index]
        rate = h["rate"][frame_index]
        center = float(h["tutor_position"][frame_index])

        a.tutor_line.set_data(self.x_input, tutor)
        a.tutor_center.set_xdata([center, center])
        profile_x, profile_u = aggregate_profile_by_position(self.x_comp, membrane)
        a.u_profile.set_data(profile_x, profile_u)
        decoded_u = baseline_subtracted_center(self.x_comp, membrane)
        a.u_center.set_xdata([decoded_u, decoded_u])
        _, profile_v = aggregate_profile_by_position(self.x_comp, adaptation)
        a.v_profile.set_data(profile_x, profile_v)
        decoded_v = baseline_subtracted_center(self.x_comp, adaptation)
        a.v_center.set_xdata([decoded_v, decoded_v])
        a.input_scatter.set_array(tutor)
        a.u_scatter.set_array(membrane)
        a.v_scatter.set_array(adaptation)
        a.rate_halo.set_sizes(75.0 + 500.0 * rate / self.r_max)

        if float(np.max(np.abs(delta))) > 0.0:
            post, pre = top_update_connections(delta, self.config.animation.display_top_k_connections)
        else:
            post = np.asarray([], dtype=np.int64)
            pre = np.asarray([], dtype=np.int64)
        segments = [((self.x_input[j], 3.47), (self.x_comp[i], 1.83)) for i, j in zip(post, pre)]
        selected_weights = weights[post, pre]
        selected_delta = delta[post, pre]
        a.base_connections.set_segments(segments)
        a.base_connections.set_array(selected_weights)
        a.base_connections.set_linewidths(0.35 + 2.0 * selected_weights / max(self.weight_norm.vmax, 1e-12))
        a.update_connections.set_segments(segments)
        a.update_connections.set_array(selected_delta)
        a.update_connections.set_linewidths(0.4 + 3.0 * np.abs(selected_delta) / max(abs(self.delta_norm.vmax), 1e-12))
        a.matrix_image.set_data(weights[self.comp_order] if self.mode == "global" else weights)

        time = float(h["time"][frame_index])
        cycle = abs(self.config.tutor.speed) * time / self.config.geometry.length
        time_start = float(h["time"][0])
        time_end = float(h["time"][-1])
        progress = (time - time_start) / max(time_end - time_start, 1e-12)
        speed_note = (
            f"   avg playback={self.playback_acceleration:.1f}x"
            if self.playback_acceleration is not None
            else ""
        )
        margin = self.config.analysis.boundary_margin_sigmas * self.config.tutor.sigma
        boundary_note = "   BOUNDARY/RESET WINDOW" if self.config.geometry.boundary_mode == "paper_reset" and (center < margin or center > self.config.geometry.length - margin) else ""
        lag_note = decoded_u - decoded_v if np.isfinite(decoded_u) and np.isfinite(decoded_v) else float("nan")
        a.time_text.set_text(
            f"t={time:7.3f} s   training={100.0 * progress:5.1f}%   tutor z={center:6.2f}\n"
            f"cycle={cycle:6.2f}{speed_note}   U-V offset={lag_note:+.2f}{boundary_note}"
        )
        if selected_delta.size:
            strongest = int(np.argmax(np.abs(selected_delta)))
            i, j = int(post[strongest]), int(pre[strongest])
            clipped_note = " [clipped at J>=0]" if "clipped" in h and bool(h["clipped"][frame_index, i, j]) else ""
            a.equation_text.set_text(
                f"highlight: largest |delta J|;  J_after[{i},{j}]={weights[i,j]:.3g}, "
                f"R[{j}]={tutor[j]:.3g}, r[{i}]={rate[i]:.3g}, delta J={delta[i,j]:+.3g}{clipped_note}"
            )
        else:
            a.equation_text.set_text("initial/frozen state: delta J = 0; no plastic connection is highlighted")
        return tuple(vars(a).values())
