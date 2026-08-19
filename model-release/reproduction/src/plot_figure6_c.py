import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gaussian_generative_process import generated_peak_count_distribution
from plot_figure5_de import load_pooled_experimental_tuning
from utils import circular_peak_counts


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure6.json"
FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure6_c_settings():
    """
    读取 Figure 6C 的峰检测设置和 Figure 5D-E 最优参数。

    Returns:
        ``(panel_config, fitted_parameters)``。
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return config["panel_c"], fit


def peak_count_probabilities(counts, max_peak_count):
    """
    将逐神经元峰数转换为从 0 到指定最大峰数的概率分布。

    Args:
        counts: 形状为 ``(n_thresholds, n_neurons)`` 的峰数。
        max_peak_count: 输出分布显式包含的最大峰数。

    Returns:
        ``(probabilities, overflow_counts)``。
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError("counts must be shaped (n_thresholds, n_neurons)")
    max_peak_count = int(max_peak_count)
    probabilities = np.zeros(
        (counts.shape[0], max_peak_count + 1),
        dtype=float,
    )
    overflow = np.zeros(counts.shape[0], dtype=np.int64)
    for threshold_index, row in enumerate(counts):
        in_range = row <= max_peak_count
        probabilities[threshold_index] = np.bincount(
            row[in_range],
            minlength=max_peak_count + 1,
        ) / row.size
        overflow[threshold_index] = int(np.sum(~in_range))
    return probabilities, overflow


def build_figure6_c_data():
    """
    计算实验数据与生成过程在三个 z-score 阈值下的峰数分布。

    Returns:
        包含 Figure 6C 概率、样本量、拟合参数和误差指标的字典。
    """
    panel, fit = load_figure6_c_settings()
    thresholds = np.asarray(panel["z_thresholds"], dtype=float)
    max_peak_count = int(panel["max_peak_count"])
    experimental_tuning, subject_ids = load_pooled_experimental_tuning()
    experimental_counts = circular_peak_counts(
        experimental_tuning,
        thresholds,
    )
    experimental_probabilities, experimental_overflow = (
        peak_count_probabilities(experimental_counts, max_peak_count)
    )
    generated_probabilities, generated_overflow = (
        generated_peak_count_distribution(
            total_neurons=int(panel["n_generated_neurons"]),
            batch_size=int(panel["batch_size"]),
            n_angles=experimental_tuning.shape[1],
            sigma=float(fit["best_sigma"]),
            beta=float(fit["best_beta"]),
            bias=float(fit["best_bias"]),
            z_thresholds=thresholds,
            max_peak_count=max_peak_count,
            seed=int(panel["seed"]),
        )
    )
    experimental_overflow_probability = (
        experimental_overflow / experimental_tuning.shape[0]
    )
    generated_overflow_probability = (
        generated_overflow / int(panel["n_generated_neurons"])
    )
    return {
        "z_thresholds": thresholds,
        "peak_counts": np.arange(max_peak_count + 1, dtype=np.int64),
        "experimental_probabilities": experimental_probabilities,
        "generated_probabilities": generated_probabilities,
        "experimental_overflow_counts": experimental_overflow,
        "generated_overflow_counts": generated_overflow,
        "total_variation_distance": 0.5
        * (
            np.sum(
                np.abs(
                    experimental_probabilities - generated_probabilities
                ),
                axis=1,
            )
            + np.abs(
                experimental_overflow_probability
                - generated_overflow_probability
            )
        ),
        "sigma": float(fit["best_sigma"]),
        "beta": float(fit["best_beta"]),
        "bias": float(fit["best_bias"]),
        "n_fourier_modes": int(fit["n_fourier_modes"]),
        "n_experimental_neurons": int(experimental_tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_generated_neurons": int(panel["n_generated_neurons"]),
        "seed": int(panel["seed"]),
    }


def save_figure6_c_data(data):
    """
    保存 Figure 6C 数值源数据和紧凑诊断报告。

    Args:
        data: ``build_figure6_c_data`` 返回的字典。

    Returns:
        ``(npz_path, json_path)``。
    """
    npz_path = PROCESSED / "figure6_c_peak_count_distributions.npz"
    np.savez_compressed(npz_path, **data)
    report = {
        "sigma": data["sigma"],
        "beta": data["beta"],
        "bias": data["bias"],
        "n_fourier_modes": data["n_fourier_modes"],
        "z_thresholds": data["z_thresholds"].tolist(),
        "n_experimental_neurons": data["n_experimental_neurons"],
        "n_mice": data["n_mice"],
        "n_generated_neurons": data["n_generated_neurons"],
        "experimental_overflow_counts": data[
            "experimental_overflow_counts"
        ].tolist(),
        "generated_overflow_counts": data[
            "generated_overflow_counts"
        ].tolist(),
        "total_variation_distance": data[
            "total_variation_distance"
        ].tolist(),
        "seed": data["seed"],
    }
    json_path = REPORTS / "figure6_c_diagnostics.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


def plot_figure6_c(data):
    """
    绘制 Figure 6C 的三行灰色实验/红色生成峰数柱状图。

    Args:
        data: ``build_figure6_c_data`` 返回的字典。

    Returns:
        输出 PNG 路径。
    """
    fig, axes = plt.subplots(3, 1, figsize=(3.7, 5.35), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.11, top=0.96, hspace=0.43)
    displayed_counts = data["peak_counts"][1:]
    width = 0.34
    for threshold_index, (ax, threshold) in enumerate(
        zip(axes, data["z_thresholds"])
    ):
        experimental = data["experimental_probabilities"][threshold_index, 1:]
        generated = data["generated_probabilities"][threshold_index, 1:]
        ax.bar(
            displayed_counts - width / 2,
            experimental,
            width=width,
            color="0.28",
            label="data",
        )
        ax.bar(
            displayed_counts + width / 2,
            generated,
            width=width,
            color="#ef2b2d",
            label="gen. process",
        )
        ax.set_xlim(0.5, displayed_counts[-1] + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=2.8, width=0.8)
        ax.text(
            0.5,
            1.02,
            rf"$z_{{\mathrm{{thresh}}}}={threshold:.1f}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[1].set_ylabel("frac. neurons")
    axes[-1].set_xlabel("peak count")
    axes[-1].set_xticks(displayed_counts)
    axes[0].text(
        -0.18,
        1.02,
        "C",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    output = FIGURES / "figure6_c_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main():
    """
    生成 Figure 6C，并保存峰数概率与诊断指标。
    """
    data = build_figure6_c_data()
    npz_path, report_path = save_figure6_c_data(data)
    figure_path = plot_figure6_c(data)
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
        "Total-variation distances:",
        np.array2string(data["total_variation_distance"], precision=5),
    )
    print(
        "Overflow counts (data/generated):",
        data["experimental_overflow_counts"].tolist(),
        data["generated_overflow_counts"].tolist(),
    )


if __name__ == "__main__":
    main()
