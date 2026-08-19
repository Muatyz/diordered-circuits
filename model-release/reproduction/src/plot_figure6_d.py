import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generated_ranked_peak_heights
from plot_figure5_de import load_pooled_experimental_tuning
from utils import ranked_circular_peak_heights


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure6.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure6_d_settings():
    """
    读取 Figure 6D 的峰高密度设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_d"], fit


def histogram_density(values, bin_edges):
    """
    在固定公共 bin 上计算有限样本的概率密度。

    Args:
        values: 一维峰高样本，可包含表示缺失 rank 的 NaN。
        bin_edges: 实验和生成过程共用的直方图边界。

    Returns:
        ``(density, n_finite, n_outside)``。密度对落在绘图区间内的样本
        归一化；区间外样本数量单独报告，避免静默裁切。
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    inside = (finite >= bin_edges[0]) & (finite <= bin_edges[-1])
    density, _ = np.histogram(finite[inside], bins=bin_edges, density=True)
    return density, int(finite.size), int(np.sum(~inside))


def build_figure6_d_data():
    """
    计算实验与生成过程最高三个 z-score 合格峰的峰高密度。

    Returns:
        包含 Figure 6D 峰高样本、公共直方图和诊断指标的字典。
    """
    panel, fit = load_figure6_d_settings()
    n_ranks = int(panel["n_peak_ranks"])
    z_threshold = float(panel["z_threshold"])
    experimental_tuning, subject_ids = load_pooled_experimental_tuning()
    experimental_heights = ranked_circular_peak_heights(
        experimental_tuning,
        z_threshold=z_threshold,
        n_ranks=n_ranks,
    )
    generated_heights = generated_ranked_peak_heights(
        total_neurons=int(panel["n_generated_neurons"]),
        batch_size=int(panel["batch_size"]),
        n_angles=experimental_tuning.shape[1],
        sigma=float(fit["best_sigma"]),
        beta=float(fit["best_beta"]),
        bias=float(fit["best_bias"]),
        z_threshold=z_threshold,
        n_ranks=n_ranks,
        seed=int(panel["seed"]),
    )
    bin_edges = np.linspace(
        float(panel["peak_height_min"]),
        float(panel["peak_height_max"]),
        int(panel["n_histogram_bins"]) + 1,
    )
    experimental_density = np.empty((n_ranks, len(bin_edges) - 1))
    generated_density = np.empty_like(experimental_density)
    experimental_rank_counts = np.zeros(n_ranks, dtype=np.int64)
    generated_rank_counts = np.zeros(n_ranks, dtype=np.int64)
    experimental_outside = np.zeros(n_ranks, dtype=np.int64)
    generated_outside = np.zeros(n_ranks, dtype=np.int64)
    for rank in range(n_ranks):
        (
            experimental_density[rank],
            experimental_rank_counts[rank],
            experimental_outside[rank],
        ) = histogram_density(experimental_heights[:, rank], bin_edges)
        (
            generated_density[rank],
            generated_rank_counts[rank],
            generated_outside[rank],
        ) = histogram_density(generated_heights[:, rank], bin_edges)

    bin_widths = np.diff(bin_edges)
    total_variation = 0.5 * np.sum(
        np.abs(experimental_density - generated_density)
        * bin_widths[None, :],
        axis=1,
    )
    return {
        "bin_edges": bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "experimental_peak_heights": experimental_heights,
        "generated_peak_heights": generated_heights,
        "experimental_density": experimental_density,
        "generated_density": generated_density,
        "experimental_rank_counts": experimental_rank_counts,
        "generated_rank_counts": generated_rank_counts,
        "experimental_outside_counts": experimental_outside,
        "generated_outside_counts": generated_outside,
        "total_variation_distance": total_variation,
        "z_threshold": z_threshold,
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(experimental_tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
    }


def save_figure6_d_data(data):
    """
    保存 Figure 6D 峰高样本、密度和诊断报告。

    Args:
        data: ``build_figure6_d_data`` 返回的字典。

    Returns:
        ``(npz_path, json_path)``。
    """
    npz_path = PROCESSED / "figure6_d_peak_height_distributions.npz"
    np.savez_compressed(npz_path, **data)
    report = {
        key: data[key]
        for key in (
            "sigma",
            "beta",
            "bias",
            "n_fourier_modes",
            "z_threshold",
            "n_experimental_neurons",
            "n_mice",
            "n_generated_neurons",
            "seed",
        )
    }
    for key in (
        "experimental_rank_counts",
        "generated_rank_counts",
        "experimental_outside_counts",
        "generated_outside_counts",
        "total_variation_distance",
    ):
        report[key] = data[key].tolist()
    json_path = REPORTS / "figure6_d_diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


def plot_figure6_d(data):
    """
    绘制 Figure 6D 的三行峰高密度：实验灰柱与生成红线。

    Args:
        data: ``build_figure6_d_data`` 返回的字典。

    Returns:
        输出 PNG 路径。
    """
    n_ranks = data["experimental_density"].shape[0]
    fig, axes = plt.subplots(n_ranks, 1, figsize=(3.8, 5.35), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.11, top=0.96, hspace=0.43)
    bin_edges = data["bin_edges"]
    bin_width = float(bin_edges[1] - bin_edges[0])
    for rank, ax in enumerate(np.atleast_1d(axes)):
        ax.bar(
            data["bin_centers"],
            data["experimental_density"][rank],
            width=bin_width,
            color="0.28",
            edgecolor="none",
            label="data",
        )
        ax.stairs(
            data["generated_density"][rank],
            bin_edges,
            color="#ef2b2d",
            linewidth=1.35,
            label="gen. process",
        )
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(bottom=0.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=2.8, width=0.8)
        ax.text(
            0.5,
            1.02,
            f"peak {rank + 1}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [labels.index("data"), labels.index("gen. process")]
    axes[0].legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    axes[1].set_ylabel("density")
    axes[-1].set_xlabel("peak height")
    axes[-1].set_xticks([0, 5, 10, 15])
    axes[0].text(
        -0.18,
        1.02,
        "D",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )
    output = FIGURES / "figure6_d_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 6D，并保存峰高密度及诊断指标。
    """
    data = build_figure6_d_data()
    npz_path, report_path = save_figure6_d_data(data)
    figure_path = plot_figure6_d(data)
    print("Saved:", figure_path)
    print("Saved:", npz_path)
    print("Saved:", report_path)
    print(
        "Parameters:",
        f"sigma={data['sigma']:.4g},",
        f"beta={data['beta']:.4g},",
        f"b={data['bias']:.4g},",
        f"z={data['z_threshold']:.1f}",
    )
    print(
        "Rank sample counts (data/generated):",
        data["experimental_rank_counts"].tolist(),
        data["generated_rank_counts"].tolist(),
    )
    print(
        "Total-variation distances:",
        np.array2string(data["total_variation_distance"], precision=5),
    )


if __name__ == "__main__":
    main()
