import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generate_heterogeneous_tuning_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure5.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure5_g_settings():
    """
    读取 Figure 5G 的绘图设置和 Figure 5D-E 得到的最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_g"], fit


def build_figure5_g_data():
    """
    按最优 Gaussian generative process 参数生成 Figure 5G 微观样本。

    原图纵轴为 firing rate，因此绘图使用 normalized-softplus 输出
    ``phi_star(theta)``；同时保存对应 latent currents ``x_star(theta)``，
    便于检查相同统计参数下不同神经元的微观异质性。

    Returns:
        ``HeterogeneousTuningDataset`` 和 panel 配置。
    """
    panel, fit = load_figure5_g_settings()
    dataset = generate_heterogeneous_tuning_dataset(
        n_neurons=int(panel["n_samples"]),
        n_angles=int(panel["n_angles"]),
        sigma=float(fit["best_sigma"]),
        beta=float(fit["best_beta"]),
        bias=float(fit["best_bias"]),
        seed=int(panel["seed"]),
        dtype=np.float64,
    )
    return dataset, panel


def save_figure5_g_data(dataset):
    """
    保存 Figure 5G 展示的 input-current 和 firing-rate 微观样本。

    Args:
        dataset: ``build_figure5_g_data`` 生成的数据集。

    Returns:
        输出 NPZ 路径。
    """
    return dataset.save_npz(
        PROCESSED / "figure5_g_optimized_generative_samples.npz"
    )


def plot_figure5_g(dataset, panel):
    """
    绘制与原文一致的 2×8 随机 firing-rate 调谐曲线小面板。

    Args:
        dataset: 包含 16 条最优生成过程样本的数据集。
        panel: Figure 5G 配置。

    Returns:
        输出 PNG 路径。
    """
    if dataset.n_neurons != 16:
        raise ValueError("Figure 5G layout requires exactly 16 samples")

    fig, axes = plt.subplots(
        2,
        8,
        figsize=(10.8, 2.9),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.22,
        top=0.84,
        wspace=0.18,
        hspace=0.38,
    )
    theta_closed = np.append(dataset.theta_rad, 2.0 * np.pi)
    y_max = float(panel["y_max"])
    for sample_index, ax in enumerate(axes.flat):
        rate_closed = np.append(
            dataset.firing_rates[sample_index],
            dataset.firing_rates[sample_index, 0],
        )
        ax.plot(theta_closed, rate_closed, color="black", linewidth=1.15)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(0.0, y_max)
        ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.set_yticks([0.0, 5.0, 10.0])
        ax.tick_params(direction="out", length=2.2, width=0.8, labelsize=7, pad=1)
        ax.spines[["top", "right"]].set_visible(False)
        if sample_index % 8 != 0:
            ax.tick_params(labelleft=False)

    fig.supxlabel(r"$\theta$", fontsize=10, y=0.06)
    fig.supylabel("firing rate\n(a.u.)", fontsize=10, x=0.015)
    fig.text(0.018, 0.88, "G", fontsize=14, fontweight="bold")

    output = FIGURES / "figure5_g_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 5G，并保存图中使用的全部微观样本。
    """
    dataset, panel = build_figure5_g_data()
    data_path = save_figure5_g_data(dataset)
    figure_path = plot_figure5_g(dataset, panel)
    peak_counts = np.sum(
        (dataset.firing_rates > np.roll(dataset.firing_rates, 1, axis=1))
        & (dataset.firing_rates > np.roll(dataset.firing_rates, -1, axis=1)),
        axis=1,
    )
    print("Saved:", figure_path)
    print("Saved:", data_path)
    print(
        "Parameters:",
        f"sigma={dataset.sigma:.4g},",
        f"beta={dataset.beta:.4g},",
        f"b={dataset.bias:.4g}",
    )
    print("Peak-count range:", int(np.min(peak_counts)), "to", int(np.max(peak_counts)))
    print(
        "Maximum firing-rate range:",
        f"{np.min(np.max(dataset.firing_rates, axis=1)):.4g}",
        "to",
        f"{np.max(np.max(dataset.firing_rates, axis=1)):.4g}",
    )


if __name__ == "__main__":
    main()
