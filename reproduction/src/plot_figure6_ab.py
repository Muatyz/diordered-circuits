import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generated_aligned_population_profile
from plot_figure2_reproduction import load_aligned_subject_tuning
from utils import close_circular_trace


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure6.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure6_ab_settings():
    """
    读取 Figure 6A-B 的 Monte Carlo 设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panels_ab"], fit


def build_figure6_ab_data():
    """
    计算实验与生成过程的 COM 对齐均值、标准差曲线。

    实验部分保留每只小鼠各自的统计曲线；生成部分按原文使用一个共享
    参数集采样 100,000 条曲线，并执行与实验数据完全相同的 COM 对齐。

    Returns:
        包含 Figure 6A-B 全部数值源数据与诊断指标的字典。
    """
    panel, fit = load_figure6_ab_settings()
    subject_rows = load_aligned_subject_tuning()
    mouse_means = np.vstack([row["mean"] for row in subject_rows])
    mouse_stds = np.vstack([row["std"] for row in subject_rows])
    n_angles = mouse_means.shape[1]

    delta_theta, generated_mean, generated_std = (
        generated_aligned_population_profile(
            total_neurons=int(panel["n_generated_neurons"]),
            batch_size=int(panel["batch_size"]),
            n_angles=n_angles,
            sigma=float(fit["best_sigma"]),
            beta=float(fit["best_beta"]),
            bias=float(fit["best_bias"]),
            seed=int(panel["seed"]),
        )
    )
    mouse_average_mean = np.mean(mouse_means, axis=0)
    mouse_average_std = np.mean(mouse_stds, axis=0)
    return {
        "delta_theta_rad": delta_theta,
        "mouse_ids": np.asarray([row["subject_id"] for row in subject_rows]),
        "mouse_unit_counts": np.asarray(
            [row["n_units"] for row in subject_rows],
            dtype=np.int64,
        ),
        "mouse_mean_profiles": mouse_means,
        "mouse_std_profiles": mouse_stds,
        "mouse_average_mean": mouse_average_mean,
        "mouse_average_std": mouse_average_std,
        "generated_mean": generated_mean,
        "generated_std": generated_std,
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(
            sum(row["n_units"] for row in subject_rows)
        ),
        "n_mice": int(len(subject_rows)),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
        "mean_rmse": float(
            np.sqrt(np.mean((generated_mean - mouse_average_mean) ** 2))
        ),
        "std_rmse": float(
            np.sqrt(np.mean((generated_std - mouse_average_std) ** 2))
        ),
    }


def save_figure6_ab_data(data):
    """
    保存 Figure 6A-B 的源数据和紧凑诊断报告。

    Args:
        data: ``build_figure6_ab_data`` 返回的字典。

    Returns:
        ``(npz_path, json_path)``。
    """
    npz_path = PROCESSED / "figure6_ab_population_profiles.npz"
    np.savez_compressed(npz_path, **data)

    report = {
        key: data[key]
        for key in (
            "sigma",
            "beta",
            "bias",
            "n_fourier_modes",
            "n_experimental_neurons",
            "n_mice",
            "n_generated_neurons",
            "seed",
            "mean_rmse",
            "std_rmse",
        )
    }
    json_path = REPORTS / "figure6_ab_diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


def _plot_profile_panel(
    ax,
    delta_theta,
    mouse_profiles,
    mouse_average,
    generated,
    ylabel,
    ylim,
):
    """
    绘制 Figure 6A 或 6B 的单鼠、跨鼠平均和生成预测曲线。
    """
    x_closed, mice_closed = close_circular_trace(delta_theta, mouse_profiles)
    _, average_closed = close_circular_trace(delta_theta, mouse_average)
    _, generated_closed = close_circular_trace(delta_theta, generated)

    for profile in mice_closed:
        ax.plot(x_closed, profile, color="0.67", linewidth=0.75, alpha=0.78)
    ax.plot(
        x_closed,
        average_closed,
        color="black",
        linewidth=2.0,
        label="mouse average",
    )
    ax.plot(
        x_closed,
        generated_closed,
        color="#ef2b2d",
        linewidth=1.6,
        label="gen. process",
    )
    ax.set(
        xlim=(-np.pi, np.pi),
        ylim=tuple(ylim),
        xlabel=r"$\Delta\theta$",
        ylabel=ylabel,
    )
    ax.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)


def plot_figure6_ab(data, panel):
    """
    绘制原文 Figure 6A-B 风格的两行定量验证图。

    Args:
        data: ``build_figure6_ab_data`` 返回的字典。
        panel: Figure 6A-B 配置。

    Returns:
        输出 PNG 路径。
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 5.55), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.11, top=0.96, hspace=0.19)
    _plot_profile_panel(
        axes[0],
        data["delta_theta_rad"],
        data["mouse_mean_profiles"],
        data["mouse_average_mean"],
        data["generated_mean"],
        "mean profile",
        panel["mean_ylim"],
    )
    _plot_profile_panel(
        axes[1],
        data["delta_theta_rad"],
        data["mouse_std_profiles"],
        data["mouse_average_std"],
        data["generated_std"],
        "std. dev. profile",
        panel["std_ylim"],
    )
    axes[0].tick_params(labelbottom=True)
    axes[0].text(
        -0.15,
        1.03,
        "A",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )
    axes[1].text(
        -0.15,
        1.03,
        "B",
        transform=axes[1].transAxes,
        fontsize=14,
        fontweight="bold",
    )
    axes[1].plot([], [], color="0.67", linewidth=1.0, label="individual mice")
    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 0, 1]
    axes[1].legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        loc="upper left",
        fontsize=8,
        handlelength=2.8,
    )

    output = FIGURES / "figure6_ab_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 6A-B，并保存数值源数据和拟合诊断。
    """
    panel, _ = load_figure6_ab_settings()
    data = build_figure6_ab_data()
    npz_path, report_path = save_figure6_ab_data(data)
    figure_path = plot_figure6_ab(data, panel)
    print("Saved:", figure_path)
    print("Saved:", npz_path)
    print("Saved:", report_path)
    print(
        "Parameters:",
        f"sigma={data['sigma']:.4g},",
        f"beta={data['beta']:.4g},",
        f"b={data['bias']:.4g},",
        f"modes={data['n_fourier_modes']}",
    )
    print(
        "Samples:",
        f"data={data['n_experimental_neurons']},",
        f"generated={data['n_generated_neurons']}",
    )
    print("Mean-profile RMSE:", f"{data['mean_rmse']:.6g}")
    print("Std-profile RMSE:", f"{data['std_rmse']:.6g}")


if __name__ == "__main__":
    main()
