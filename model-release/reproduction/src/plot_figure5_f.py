import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import (
    circular_two_point_correlation,
    generated_two_point_correlation,
)
from plot_figure5_de import load_pooled_experimental_tuning


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure5.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure5_f_settings():
    """
    读取 Figure 5F 的 Monte Carlo 设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_f"], fit


def centered_circular_curve(values):
    """
    将 lag=0 的环形相关函数移动到横轴中央并闭合端点。

    Args:
        values: 按 ``[0, 2π)`` lag 顺序排列的相关函数。

    Returns:
        ``(delta_theta, shifted_values)``，横轴范围为 ``[-π, π]``。
    """
    values = np.asarray(values, dtype=float)
    shifted = np.fft.fftshift(values)
    delta_theta = np.linspace(-np.pi, np.pi, len(values), endpoint=False)
    return (
        np.append(delta_theta, np.pi),
        np.append(shifted, shifted[0]),
    )


def build_figure5_f_data():
    """
    计算实验数据和最优 Gaussian generative process 的二点相关函数。

    Returns:
        包含 Figure 5F 数值源数据和拟合参数的字典。
    """
    panel, fit = load_figure5_f_settings()
    experimental_tuning, subject_ids = load_pooled_experimental_tuning()
    experimental = circular_two_point_correlation(experimental_tuning)

    theta_lag, generated = generated_two_point_correlation(
        total_neurons=int(panel["n_generated_neurons"]),
        batch_size=int(panel["batch_size"]),
        n_angles=experimental_tuning.shape[1],
        sigma=float(fit["best_sigma"]),
        beta=float(fit["best_beta"]),
        bias=float(fit["best_bias"]),
        seed=int(panel["seed"]),
    )
    delta_theta, experimental_centered = centered_circular_curve(experimental)
    _, generated_centered = centered_circular_curve(generated)
    return {
        "delta_theta": delta_theta,
        "experimental_correlation": experimental_centered,
        "generated_correlation": generated_centered,
        "theta_lag": theta_lag,
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(experimental_tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
    }


def save_figure5_f_data(data):
    """
    保存 Figure 5F 的相关函数及其生成参数。

    Args:
        data: ``build_figure5_f_data`` 返回的字典。

    Returns:
        输出 NPZ 路径。
    """
    output = PROCESSED / "figure5_f_two_point_correlation.npz"
    np.savez_compressed(output, **data)
    return output


def plot_figure5_f(data):
    """
    绘制实验数据与最优生成过程的 Figure 5F 相关函数比较。

    Args:
        data: ``build_figure5_f_data`` 返回的字典。

    Returns:
        输出 PNG 路径。
    """
    fig, ax = plt.subplots(figsize=(3.25, 3.1))
    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.2, top=0.85)
    ax.plot(
        data["delta_theta"],
        data["experimental_correlation"],
        color="0.68",
        linewidth=3.2,
        label="data",
    )
    ax.plot(
        data["delta_theta"],
        data["generated_correlation"],
        color="black",
        linewidth=1.25,
        linestyle="--",
        label="gen.\nprocess",
    )
    ax.set(
        xlim=(-np.pi, np.pi),
        ylim=(0.0, 4.0),
        xlabel=r"$\Delta\theta$",
        ylabel=r"$\Gamma^\phi(\Delta\theta)$",
    )
    ax.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(
        -0.16,
        1.05,
        "F",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    output = FIGURES / "figure5_f_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 5F 并保存可复用的相关函数源数据。
    """
    data = build_figure5_f_data()
    data_path = save_figure5_f_data(data)
    figure_path = plot_figure5_f(data)
    residual = data["generated_correlation"] - data["experimental_correlation"]
    print("Saved:", figure_path)
    print("Saved:", data_path)
    print(
        "Parameters:",
        f"sigma={data['sigma']:.4g},",
        f"beta={data['beta']:.4g},",
        f"b={data['bias']:.4g},",
        f"modes={data['n_fourier_modes']}",
    )
    print("Correlation RMSE:", f"{np.sqrt(np.mean(residual * residual)):.6g}")


if __name__ == "__main__":
    main()
