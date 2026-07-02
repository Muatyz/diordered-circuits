"""Heading and error plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import unwrap_heading_trace
from learning.analysis.metrics import linear_fit_slope_intercept


def plot_true_vs_decoded_heading(
    *,
    time: np.ndarray,
    theta_true: np.ndarray,
    theta_hd_decoded: np.ndarray,
    path: str | Path,
    title: str,
    theta_hd_decoded_peak: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    axis.plot(time, np.rad2deg(unwrap_heading_trace(theta_true)), label="true heading", linewidth=1.8)
    axis.plot(
        time,
        np.rad2deg(unwrap_heading_trace(theta_hd_decoded)),
        label="PVA decode",
        linewidth=1.4,
    )
    if theta_hd_decoded_peak is not None and np.asarray(theta_hd_decoded_peak).size == time.size:
        axis.plot(
            time,
            np.rad2deg(unwrap_heading_trace(theta_hd_decoded_peak)),
            label="peak decode",
            linewidth=1.1,
            linestyle="--",
        )
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("unwrapped heading angle [deg]")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pi_error(
    *,
    time: np.ndarray,
    pi_error: np.ndarray,
    path: str | Path,
    title: str = "PI error",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86)
    axis.plot(time, np.rad2deg(pi_error), color="#3b6ea8", linewidth=1.5)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("decoded - true heading error [deg]")
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
    if commanded_velocity.size >= 2 and np.all(np.isfinite(decoded_velocity)):
        slope, intercept = linear_fit_slope_intercept(commanded_velocity, decoded_velocity)
        velocity_grid = np.linspace(float(np.min(commanded_velocity)), float(np.max(commanded_velocity)), 100)
        axis.plot(velocity_grid, slope * velocity_grid + intercept, color="#7c4d79", linewidth=1.2)
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
    axis.axline((0.0, 0.0), slope=1.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("commanded angular velocity [rad/s]")
    axis.set_ylabel("decoded bump angular velocity [rad/s]")
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
