from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def load_first_tuning():
    """
    读取第一个已计算好的 HD tuning 结果。

    先检查 `data/processed/hd_tuning_index.csv` 是否存在并且非空，然后取第一行
    的 `.npz` 文件路径加载结果。返回索引表中的 row 和对应的 numpy 数据对象。
    """
    index_path = PROCESSED / "hd_tuning_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    index = pd.read_csv(index_path)
    if len(index) == 0:
        raise ValueError("No tuning files found in hd_tuning_index.csv")

    row = index.iloc[0]
    return row, np.load(row["tuning_path"])


def plot_single_unit(row, data):
    """
    绘制一个代表性 unit 的头方向 firing-rate 曲线。

    优先从通过 QC 且所有 bin 有有限值的 unit 中挑选峰值 firing rate 最高者；
    如果 QC 集合为空，则退回到所有可画 unit。图片保存到 `reports/figures`，
    返回输出文件路径。
    """
    included = data["included_qc"].astype(bool)
    unit_ids = data["unit_ids"]
    rates = data["raw_rate_hz"] if "raw_rate_hz" in data.files else data["smoothed_rate_hz"]
    centers_deg = np.rad2deg(data["bin_centers_rad"])

    candidates = np.where(included & np.isfinite(rates).all(axis=1))[0]
    if len(candidates) == 0:
        candidates = np.where(np.isfinite(rates).all(axis=1))[0]
    if len(candidates) == 0:
        raise ValueError("No finite tuning curves available for plotting")

    unit_i = int(candidates[np.nanargmax(np.nanmax(rates[candidates], axis=1))])

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(centers_deg, rates[unit_i], color="black", linewidth=2)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Head direction (deg)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"{row['subject_id']} unit {unit_ids[unit_i]}")
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGURES / f"{row['subject_id']}_single_unit_tuning.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_aligned_heatmap(row, data):
    """
    绘制通过 QC 的 unit 调谐曲线对齐 heatmap。

    每个 unit 的归一化曲线按峰值方向滚动到中心位置，再按峰值强度排序。
    这样可以观察多个 HD cell 在对齐后的调谐形状是否集中、稳定。
    """
    included = data["included_qc"].astype(bool)
    raw_rates = data["raw_rate_hz"] if "raw_rate_hz" in data.files else data["smoothed_rate_hz"]
    matrix = raw_rates[included]
    unit_ids = data["unit_ids"][included]

    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    unit_ids = unit_ids[valid]
    if len(matrix) == 0:
        raise ValueError("No finite included tuning curves available for heatmap")

    unit_means = np.nanmean(matrix, axis=1)
    good = np.isfinite(unit_means) & (unit_means > 0)
    matrix = matrix.copy()
    matrix[good] = matrix[good] / unit_means[good, None]
    matrix[~good] = np.nan

    peaks = np.nanargmax(matrix, axis=1)
    aligned = np.vstack([np.roll(matrix[i], matrix.shape[1] // 2 - peaks[i]) for i in range(len(matrix))])
    order = np.argsort(np.nanmax(aligned, axis=1))[::-1]
    aligned = aligned[order]

    x = np.linspace(-180, 180, aligned.shape[1], endpoint=False)
    fig, ax = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    im = ax.imshow(
        aligned,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], 0, len(unit_ids)],
        cmap="viridis",
        vmin=0,
        vmax=np.nanpercentile(aligned, 98),
    )
    ax.set_xlabel("Aligned head direction (deg)")
    ax.set_ylabel("Unit")
    ax.set_title(f"{row['subject_id']} included HD cells")
    fig.colorbar(im, ax=ax, label="Unit-mean normalized rate")

    out = FIGURES / f"{row['subject_id']}_aligned_tuning_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    """
    命令行入口：从第一个 tuning 文件生成示例图。

    会输出一张单 unit 调谐曲线图和一张对齐后的群体 heatmap，用于快速检查
    HD tuning 计算结果是否合理。
    """
    row, data = load_first_tuning()
    single = plot_single_unit(row, data)
    heatmap = plot_aligned_heatmap(row, data)
    print("Saved:", single)
    print("Saved:", heatmap)


if __name__ == "__main__":
    main()
