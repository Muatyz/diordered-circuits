import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generated_flip_correlations
from plot_figure5_de import load_pooled_experimental_tuning
from utils import circular_flip_correlations


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure6.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure6_e_settings():
    """
    读取 Figure 6E 的相关分布设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_e"], fit


def density_on_common_bins(values, bin_edges):
    """
    在公共 bin 上计算有限相关系数的概率密度。

    Returns:
        ``(density, n_finite, n_outside)``。
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    inside = (finite >= bin_edges[0]) & (finite <= bin_edges[-1])
    density, _ = np.histogram(finite[inside], bins=bin_edges, density=True)
    return density, int(finite.size), int(np.sum(~inside))


def build_figure6_e_data():
    """
    计算实验与生成调谐曲线的 flip-symmetry correlation 分布。

    Returns:
        包含相关样本、公共密度、尾部概率和拟合参数的字典。
    """
    panel, fit = load_figure6_e_settings()
    experimental_tuning, subject_ids = load_pooled_experimental_tuning()
    angles_rad = np.linspace(
        0.0,
        2.0 * np.pi,
        experimental_tuning.shape[1],
        endpoint=False,
    )
    experimental = circular_flip_correlations(
        experimental_tuning,
        angles_rad=angles_rad,
    )
    generated = generated_flip_correlations(
        total_neurons=int(panel["n_generated_neurons"]),
        batch_size=int(panel["batch_size"]),
        n_angles=experimental_tuning.shape[1],
        sigma=float(fit["best_sigma"]),
        beta=float(fit["best_beta"]),
        bias=float(fit["best_bias"]),
        seed=int(panel["seed"]),
    )
    bin_edges = np.linspace(
        float(panel["rho_min"]),
        float(panel["rho_max"]),
        int(panel["n_histogram_bins"]) + 1,
    )
    experimental_density, n_experimental_valid, experimental_outside = (
        density_on_common_bins(experimental, bin_edges)
    )
    generated_density, n_generated_valid, generated_outside = (
        density_on_common_bins(generated, bin_edges)
    )
    bin_widths = np.diff(bin_edges)
    total_variation = 0.5 * np.sum(
        np.abs(experimental_density - generated_density) * bin_widths
    )
    tail_threshold = float(panel["tail_fraction_threshold"])
    return {
        "bin_edges": bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "experimental_correlations": experimental,
        "generated_correlations": generated,
        "experimental_density": experimental_density,
        "generated_density": generated_density,
        "total_variation_distance": float(total_variation),
        "tail_fraction_threshold": tail_threshold,
        "expanded_y_max": float(panel["expanded_y_max"]),
        "experimental_tail_fraction": float(
            np.mean(experimental < tail_threshold)
        ),
        "generated_tail_fraction": float(np.mean(generated < tail_threshold)),
        "n_experimental_valid": n_experimental_valid,
        "n_generated_valid": n_generated_valid,
        "experimental_outside_count": experimental_outside,
        "generated_outside_count": generated_outside,
        "reflection_interpolation": "periodic_linear",
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(experimental_tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
    }


def save_figure6_e_data(data):
    """
    保存 Figure 6E 相关样本、密度和诊断报告。

    Returns:
        ``(npz_path, json_path)``。
    """
    npz_path = PROCESSED / "figure6_e_flip_correlations.npz"
    np.savez_compressed(npz_path, **data)
    report_keys = (
        "sigma",
        "beta",
        "bias",
        "n_fourier_modes",
        "n_experimental_neurons",
        "n_mice",
        "n_generated_neurons",
        "n_experimental_valid",
        "n_generated_valid",
        "experimental_outside_count",
        "generated_outside_count",
        "total_variation_distance",
        "tail_fraction_threshold",
        "expanded_y_max",
        "experimental_tail_fraction",
        "generated_tail_fraction",
        "reflection_interpolation",
        "seed",
    )
    report = {key: data[key] for key in report_keys}
    json_path = REPORTS / "figure6_e_diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


def _draw_density(ax, data):
    """
    在给定坐标轴绘制实验灰柱和生成红色阶梯密度。
    """
    bin_width = float(data["bin_edges"][1] - data["bin_edges"][0])
    ax.bar(
        data["bin_centers"],
        data["experimental_density"],
        width=bin_width,
        color="0.28",
        edgecolor="none",
        label="data",
    )
    ax.stairs(
        data["generated_density"],
        data["bin_edges"],
        color="#ef2b2d",
        linewidth=1.35,
        label="gen. process",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.8, width=0.8)


def plot_figure6_e(data):
    """
    绘制 Figure 6E 的完整分布和非对称低相关尾部放大图。
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.0, 2.7),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.9, wspace=0.38)
    for ax in axes:
        _draw_density(ax, data)
        ax.set_xlabel(r"$\rho^{\mathrm{flip}}$")
    axes[0].set_xlim(-1.0, 1.0)
    axes[0].set_ylabel("density")
    axes[0].set_xticks([-1.0, 0.0, 1.0])
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_ylim(0.0, data["expanded_y_max"])
    axes[1].set_yticks([0.0, 0.5, 1.0])
    axes[1].set_xticks([-1.0, 0.0, 1.0])
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("data"), labels.index("gen. process")]
    axes[0].legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        loc="upper left",
        fontsize=8,
    )
    axes[0].text(
        -0.2,
        1.02,
        "E",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )
    output = FIGURES / "figure6_e_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 6E，并保存 flip correlation 分布和诊断。
    """
    data = build_figure6_e_data()
    npz_path, report_path = save_figure6_e_data(data)
    figure_path = plot_figure6_e(data)
    print("Saved:", figure_path)
    print("Saved:", npz_path)
    print("Saved:", report_path)
    print(
        "Parameters:",
        f"sigma={data['sigma']:.4g},",
        f"beta={data['beta']:.4g},",
        f"b={data['bias']:.4g}",
    )
    print("Total-variation distance:", f"{data['total_variation_distance']:.6g}")
    print(
        f"Fraction rho < {data['tail_fraction_threshold']:.2f} (data/generated):",
        f"{data['experimental_tail_fraction']:.6g}",
        f"{data['generated_tail_fraction']:.6g}",
    )


if __name__ == "__main__":
    main()
