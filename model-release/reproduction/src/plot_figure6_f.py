import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generated_head_direction_information
from plot_figure5_de import load_pooled_experimental_tuning
from utils import head_direction_information_content


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure6.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure6_f_settings():
    """
    读取 Figure 6F 的信息量分布设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_f"], fit


def information_density(values, bin_edges):
    """
    在公共 bin 上计算有限信息量样本的概率密度。

    Returns:
        ``(density, n_finite, n_outside)``。
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    inside = (finite >= bin_edges[0]) & (finite <= bin_edges[-1])
    density, _ = np.histogram(finite[inside], bins=bin_edges, density=True)
    return density, int(finite.size), int(np.sum(~inside))


def build_figure6_f_data():
    """
    计算实验与生成调谐曲线的头朝向信息量分布。

    Returns:
        包含 bits/spike 样本、公共密度、高信息尾部和拟合参数的字典。
    """
    panel, fit = load_figure6_f_settings()
    experimental_tuning, subject_ids = load_pooled_experimental_tuning()
    experimental = head_direction_information_content(experimental_tuning)
    generated = generated_head_direction_information(
        total_neurons=int(panel["n_generated_neurons"]),
        batch_size=int(panel["batch_size"]),
        n_angles=experimental_tuning.shape[1],
        sigma=float(fit["best_sigma"]),
        beta=float(fit["best_beta"]),
        bias=float(fit["best_bias"]),
        seed=int(panel["seed"]),
    )
    bin_edges = np.linspace(
        float(panel["information_min"]),
        float(panel["information_max"]),
        int(panel["n_histogram_bins"]) + 1,
    )
    experimental_density, n_experimental_valid, experimental_outside = (
        information_density(experimental, bin_edges)
    )
    generated_density, n_generated_valid, generated_outside = (
        information_density(generated, bin_edges)
    )
    bin_widths = np.diff(bin_edges)
    total_variation = 0.5 * np.sum(
        np.abs(experimental_density - generated_density) * bin_widths
    )
    high_threshold = float(panel["high_information_threshold"])
    return {
        "bin_edges": bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "experimental_information": experimental,
        "generated_information": generated,
        "experimental_density": experimental_density,
        "generated_density": generated_density,
        "total_variation_distance": float(total_variation),
        "high_information_threshold": high_threshold,
        "experimental_high_information_fraction": float(
            np.mean(experimental > high_threshold)
        ),
        "generated_high_information_fraction": float(
            np.mean(generated > high_threshold)
        ),
        "experimental_mean_information": float(np.nanmean(experimental)),
        "generated_mean_information": float(np.nanmean(generated)),
        "n_experimental_valid": n_experimental_valid,
        "n_generated_valid": n_generated_valid,
        "experimental_outside_count": experimental_outside,
        "generated_outside_count": generated_outside,
        "information_units": "bits/spike",
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(experimental_tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
    }


def save_figure6_f_data(data):
    """
    保存 Figure 6F 信息量样本、密度和诊断报告。

    Returns:
        ``(npz_path, json_path)``。
    """
    npz_path = PROCESSED / "figure6_f_hd_information.npz"
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
        "high_information_threshold",
        "experimental_high_information_fraction",
        "generated_high_information_fraction",
        "experimental_mean_information",
        "generated_mean_information",
        "information_units",
        "seed",
    )
    report = {key: data[key] for key in report_keys}
    json_path = REPORTS / "figure6_f_diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


def plot_figure6_f(data):
    """
    绘制 Figure 6F 的实验灰色信息量密度与生成红线。
    """
    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.22, top=0.9)
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
    ax.set(
        xlim=(data["bin_edges"][0], data["bin_edges"][-1]),
        ylim=(0.0, 0.8),
        xlabel=r"$I_{\mathrm{HD}}$ (bits/spike)",
        ylabel="density",
    )
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.8, width=0.8)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("data"), labels.index("gen. process")]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    ax.text(
        -0.18,
        1.02,
        "F",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    output = FIGURES / "figure6_f_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 6F，并保存信息量分布和诊断。
    """
    data = build_figure6_f_data()
    npz_path, report_path = save_figure6_f_data(data)
    figure_path = plot_figure6_f(data)
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
        "Mean information (data/generated):",
        f"{data['experimental_mean_information']:.6g}",
        f"{data['generated_mean_information']:.6g}",
    )
    print(
        f"Fraction I_HD > {data['high_information_threshold']:.2f}:",
        f"{data['experimental_high_information_fraction']:.6g}",
        f"{data['generated_high_information_fraction']:.6g}",
    )


if __name__ == "__main__":
    main()
