import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    sample_circular_gaussian_process,
    softplus,
    wrapped_gaussian_correlation,
    wrapped_gaussian_fourier_coefficients,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure5.json"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
PROCESSED = REPRODUCTION_ROOT.parent / "data/processed"
FIGURES.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_figure5_config(path=CONFIG_PATH):
    """
    Load and validate the parameters used for the Figure 5A-C reproduction.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    panels = config["panels_abc"]
    if any(float(value) <= 0.0 for value in panels["sigma_values"]):
        raise ValueError("all Figure 5 sigma values must be positive")
    if int(panels["n_angles"]) <= 1 or int(panels["n_samples_per_sigma"]) <= 0:
        raise ValueError("Figure 5 sampling dimensions must be positive")
    return config


def build_figure5_abc_data(config):
    """
    Compute the paper covariance curves, Fourier GP samples, and panel-C curves.

    A single white-noise realization is reused across sigma values so panel B
    isolates the effect of changing the covariance bandwidth.
    """
    panels = config["panels_abc"]
    sigma_values = np.asarray(panels["sigma_values"], dtype=float)
    n_angles = int(panels["n_angles"])
    n_samples = int(panels["n_samples_per_sigma"])
    rng = np.random.default_rng(int(panels["seed"]))
    white_noise = rng.normal(size=(n_samples, n_angles))

    delta_theta = np.linspace(-np.pi, np.pi, n_angles + 1)
    covariance = np.vstack(
        [wrapped_gaussian_correlation(delta_theta, sigma) for sigma in sigma_values]
    )

    theta = None
    samples = []
    for sigma in sigma_values:
        theta, sigma_samples = sample_circular_gaussian_process(
            n_samples=n_samples,
            n_angles=n_angles,
            sigma=sigma,
            white_noise=white_noise,
        )
        samples.append(sigma_samples)

    x_axis = np.linspace(-4.0, 4.0, 600)
    beta_values = np.asarray(panels["nonlinearity_beta_values"], dtype=float)
    bias_values = np.asarray(panels["nonlinearity_bias_values"], dtype=float)
    display_scale = float(panels["nonlinearity_display_scale"])
    nonlinearity = np.empty((len(bias_values), len(beta_values), len(x_axis)))
    for bias_index, bias in enumerate(bias_values):
        for beta_index, beta in enumerate(beta_values):
            nonlinearity[bias_index, beta_index] = (
                display_scale * softplus(x_axis - bias, beta=beta)
            )

    gaussian_density = np.exp(-0.5 * x_axis * x_axis) / np.sqrt(2.0 * np.pi)
    frequencies = np.arange(0, n_angles // 2 + 1)
    fourier_coefficients = np.vstack(
        [
            wrapped_gaussian_fourier_coefficients(frequencies, sigma)
            for sigma in sigma_values
        ]
    )
    return {
        "sigma_values": sigma_values,
        "delta_theta": delta_theta,
        "covariance": covariance,
        "theta": theta,
        "samples": np.asarray(samples),
        "white_noise": white_noise,
        "x_axis": x_axis,
        "beta_values": beta_values,
        "bias_values": bias_values,
        "nonlinearity": nonlinearity,
        "gaussian_density": gaussian_density,
        "frequencies": frequencies,
        "fourier_coefficients": fourier_coefficients,
    }


def save_figure5_abc_data(data):
    """
    Save the numerical values underlying Figure 5A-C for later reuse.
    """
    output = PROCESSED / "figure5_abc_generative_process.npz"
    np.savez_compressed(output, **data)
    return output


def _style_axis(ax):
    """
    Apply the compact, open-axis style used by the reference figure.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, pad=2)


def plot_figure5_abc(data, config):
    """
    Render the three introductory panels of the Gaussian generative process.
    """
    panels = config["panels_abc"]
    sigma_values = data["sigma_values"]
    colors = plt.cm.Blues(np.linspace(0.22, 0.95, len(sigma_values)))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.25))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.83, wspace=0.42)
    ax_a, ax_b, ax_c = axes

    for covariance, color in zip(data["covariance"], colors):
        ax_a.plot(data["delta_theta"], covariance, color=color, linewidth=1.45)
    ax_a.annotate(
        "",
        xy=(-1.7, 0.40),
        xytext=(1.7, 0.40),
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 1.1},
    )
    ax_a.text(0.0, 0.31, r"$1/\sigma$", ha="center", va="center", fontsize=12)
    ax_a.set(
        xlim=(-np.pi, np.pi),
        ylim=(0.0, 1.02),
        xlabel=r"$\Delta\theta$",
        ylabel=r"$\Gamma^x(\Delta\theta)$",
        title="covariance",
    )
    ax_a.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax_a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _style_axis(ax_a)

    display_scale = float(panels["sample_display_scale"])
    group_centers = np.linspace(0.88, 0.12, len(sigma_values))
    sample_offsets = np.linspace(-0.045, 0.045, data["samples"].shape[1])
    for sigma_index, (sigma, color, center) in enumerate(
        zip(sigma_values, colors, group_centers)
    ):
        for sample, offset in zip(data["samples"][sigma_index], sample_offsets):
            centered_sample = sample - np.mean(sample)
            ax_b.plot(
                data["theta"],
                center + offset + display_scale * centered_sample,
                color=color,
                linewidth=1.0,
                alpha=0.92,
            )
        ax_b.text(
            2.0 * np.pi + 0.08,
            center,
            rf"$\sigma={sigma:.2f}$",
            color=color,
            va="center",
            fontsize=9,
            clip_on=False,
        )
    ax_b.set(
        xlim=(0.0, 2.0 * np.pi),
        ylim=(0.02, 0.98),
        xlabel=r"$\theta$",
        title="samples",
    )
    ax_b.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax_b.set_yticks([])
    _style_axis(ax_b)

    ax_c.fill_between(
        data["x_axis"],
        data["gaussian_density"],
        color="0.72",
        alpha=0.42,
        linewidth=0.0,
    )
    ax_c.plot(data["x_axis"], data["gaussian_density"], color="0.28", linewidth=1.25)
    bias_colors = ("#e41a1c", "#ff7f0e", "#31a354")
    beta_alphas = np.linspace(0.4, 1.0, len(data["beta_values"]))
    for bias_index, color in enumerate(bias_colors):
        for beta_index, alpha in enumerate(beta_alphas):
            ax_c.plot(
                data["x_axis"],
                data["nonlinearity"][bias_index, beta_index],
                color=color,
                alpha=alpha,
                linewidth=1.1,
            )
    ax_c.annotate(
        r"$\beta$",
        xy=(0.1, 0.0),
        xytext=(-1.25, 0.045),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=12,
    )
    ax_c.annotate(
        r"$b$",
        xy=(3.65, 0.045),
        xytext=(1.65, 0.045),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=12,
        va="center",
    )
    ax_c.set(
        xlim=(-4.0, 4.0),
        ylim=(0.0, 0.42),
        xlabel=r"$x(\theta)$",
        ylabel=r"$\phi(\theta)$",
        title="nonlinearity",
    )
    ax_c.set_xticks([-4, -2, 0, 2, 4])
    ax_c.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    _style_axis(ax_c)

    for label, ax in zip("ABC", axes):
        ax.text(
            -0.13,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    output = FIGURES / "figure5_abc_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    Generate Figure 5A-C and save both the figure and its numerical source.
    """
    config = load_figure5_config()
    data = build_figure5_abc_data(config)
    data_path = save_figure5_abc_data(data)
    figure_path = plot_figure5_abc(data, config)
    print("Saved:", figure_path)
    print("Saved:", data_path)
    print("sigma values:", data["sigma_values"])
    print("sample variances:", np.var(data["samples"], axis=(1, 2)))


if __name__ == "__main__":
    main()
