from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def circular_smooth(values, sigma_bins):
    """
    对环形角度调谐曲线做高斯平滑，并正确处理 NaN 缺失值。

    `values` 的最后一维被视为 0 到 2π 的环形方向 bin。缺失 (NaN) 的 bin
    在卷积时被忽略，平滑核在每个输出位置上基于可用的有限邻域重新归一化。
    `sigma_bins <= 0` 时不平滑，只返回浮点副本。
    """
    values = np.asarray(values, dtype=float)
    if sigma_bins <= 0:
        return values.astype(float, copy=True)

    n_bins = values.shape[-1]
    offsets = np.arange(n_bins)
    circular_distance = np.minimum(offsets, n_bins - offsets)
    kernel = np.exp(-0.5 * (circular_distance / float(sigma_bins)) ** 2)
    kernel = kernel / kernel.sum()

    out = np.zeros_like(values, dtype=float)
    weight_sum = np.zeros_like(values, dtype=float)
    for offset, weight in enumerate(kernel):
        rolled = np.roll(values, offset, axis=-1)
        finite_mask = np.isfinite(rolled).astype(float)
        rolled_safe = np.where(np.isfinite(rolled), rolled, 0.0)
        out += weight * rolled_safe
        weight_sum += weight * finite_mask

    # 在有有限贡献的位置归一化；否则保持 NaN
    result = np.full_like(out, np.nan)
    mask = weight_sum > 0
    result[mask] = out[mask] / weight_sum[mask]
    return result


def behavior_dt_seconds(time_s):
    """
    根据行为时间戳估计每个采样点代表的持续时间。

    相邻时间差的中位数被当作典型采样间隔；缺失、非正数或异常过大的
    时间差会被替换为该典型值。返回数组长度与 `time_s` 相同，可直接
    作为 occupancy 统计的权重。
    """
    diffs = np.diff(time_s)
    good = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(good) == 0:
        raise ValueError("Cannot estimate behavior sampling interval")

    typical = float(np.median(good))
    dt = np.r_[diffs, typical]
    dt[~np.isfinite(dt) | (dt <= 0) | (dt > 5 * typical)] = typical
    return dt


def bin_head_direction(head_direction_rad, n_bins):
    """
    将弧度制头方向分配到固定数量的方向 bin。

    输入角度会先按 2π 取模，保证负角度或超过 2π 的角度也能落入
    `[0, n_bins - 1]`。返回每个样本的 bin 编号以及 bin 边界。
    """
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    idx = np.searchsorted(edges, head_direction_rad % (2 * np.pi), side="right") - 1
    return np.clip(idx, 0, n_bins - 1), edges


def occupancy_seconds(behavior, n_bins):
    """
    统计动物在每个头方向 bin 中停留的总秒数。

    `behavior` 需要包含 `time_s` 和 `head_direction_rad` 两列。函数会先估计
    每个行为样本的持续时间，再按头方向 bin 汇总为 occupancy，并额外返回
    bin 边界和中心角度，供后续画图或计算 firing rate 使用。
    """
    hd = behavior["head_direction_rad"].to_numpy(float)
    time_s = behavior["time_s"].to_numpy(float)
    dt = behavior_dt_seconds(time_s)
    hd_bin, edges = bin_head_direction(hd, n_bins)
    occ = np.bincount(hd_bin, weights=dt, minlength=n_bins).astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return occ, edges, centers


def spike_counts_by_hd(spike_times, behavior, n_bins):
    """
    将单个神经元的 spike 计数分配到头方向 bin。

    只保留行为记录时间范围内的 spike，然后用线性插值把每个 spike 发生时刻
    映射到对应的头方向，最后按方向 bin 计数。返回长度为 `n_bins` 的计数数组。
    """
    time_s = behavior["time_s"].to_numpy(float)
    hd = behavior["head_direction_rad"].to_numpy(float)
    if len(time_s) < 2:
        raise ValueError("Need at least two behavior samples")

    typical_dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(typical_dt) or typical_dt <= 0:
        raise ValueError("Cannot estimate behavior sampling interval")

    # Alternating cross-validation partitions are non-contiguous. Restrict
    # spikes to sample-to-sample intervals that do not cross a temporal gap, so
    # excluded segments are not accidentally interpolated through.
    next_time = np.r_[time_s[1:], time_s[-1] + typical_dt]
    valid_interval = (next_time > time_s) & ((next_time - time_s) <= 5 * typical_dt)
    interval_idx = np.searchsorted(time_s, spike_times, side="right") - 1
    keep = (
        (interval_idx >= 0)
        & (interval_idx < len(time_s))
        & valid_interval[np.clip(interval_idx, 0, len(time_s) - 1)]
        & (spike_times < next_time[np.clip(interval_idx, 0, len(time_s) - 1)])
    )
    spike_time_kept = np.asarray(spike_times, dtype=float)[keep]
    hd_unwrapped = np.unwrap(hd)
    spike_hd = np.interp(spike_time_kept, time_s, hd_unwrapped) % (2 * np.pi)
    spike_bin, _ = bin_head_direction(spike_hd, n_bins)
    return np.bincount(spike_bin, minlength=n_bins).astype(float)


def rate_from_counts(counts, occ, min_occupancy_s):
    """
    用 spike count 和 occupancy 计算 firing rate。

    occupancy 小于 `min_occupancy_s` 的方向 bin 被视为采样不足，结果填为 NaN；
    其余 bin 使用 `counts / occ` 得到 Hz。`counts` 可以是一维或多维数组，
    但最后一维必须对应方向 bin。
    """
    rate = np.full_like(counts, np.nan, dtype=float)
    valid = occ >= min_occupancy_s
    rate[..., valid] = counts[..., valid] / occ[valid]
    return rate


def mean_normalize(rates):
    """
    对每个 unit 的调谐曲线做均值归一化。

    每一行代表一个 unit。函数用该 unit 在所有有效 bin 上的平均 firing rate
    归一化整条曲线，方便不同放电率量级的神经元放在同一 heatmap 中比较。
    返回归一化矩阵和每个 unit 的原始平均 firing rate。
    """
    normalized = rates.copy()
    unit_means = np.nanmean(normalized, axis=1)
    good = np.isfinite(unit_means) & (unit_means > 0)
    normalized[good] = normalized[good] / unit_means[good, None]
    normalized[~good] = np.nan
    return normalized, unit_means


def split_half_reliability(counts_a, counts_b, occ_a, occ_b, min_occupancy_s):
    """
    用 split-half 方式估计每个 unit 的头方向调谐稳定性。

    两半数据分别计算 firing rate 后，在共同有效的方向 bin 上做 Pearson 相关。
    有效 bin 太少或任一半曲线没有变化时返回 NaN。
    """
    rate_a = rate_from_counts(counts_a, occ_a, min_occupancy_s)
    rate_b = rate_from_counts(counts_b, occ_b, min_occupancy_s)
    out = np.full(rate_a.shape[0], np.nan)

    for i in range(rate_a.shape[0]):
        valid = np.isfinite(rate_a[i]) & np.isfinite(rate_b[i])
        if valid.sum() >= 5 and np.std(rate_a[i, valid]) > 0 and np.std(rate_b[i, valid]) > 0:
            out[i] = np.corrcoef(rate_a[i, valid], rate_b[i, valid])[0, 1]
    return out


def poisson_log_likelihood(test_counts, test_occ, predicted_rate, min_occupancy_s, eps=1e-12):
    """
    Evaluate held-out Poisson log likelihood for an HD rate prediction.

    Constant factorial terms are omitted because they do not affect bandwidth
    selection. Only direction bins with enough held-out occupancy and finite
    predictions are included.
    """
    valid = (test_occ >= min_occupancy_s) & np.isfinite(predicted_rate)
    if valid.sum() < 5:
        return np.nan

    rate = np.maximum(predicted_rate[valid], eps)
    return float(np.sum(test_counts[valid] * np.log(rate) - test_occ[valid] * rate))


def choose_poisson_cv_sigmas(
    counts_a,
    counts_b,
    occ_a,
    occ_b,
    min_occupancy_s,
    sigma_candidates,
    scoring_units=None,
):
    """
    Select recording-level smoothing widths by cross-partition Poisson likelihood.

    Methods B2 writes the validation likelihood with an explicit sum over
    neurons in the same recording. The Gaussian width is therefore selected
    once per partition by summing the held-out Poisson log likelihood over the
    QC-passing HD cells, then the two optimally smoothed partition curves are
    averaged. Returning per-unit arrays filled with the selected recording-level
    widths keeps the saved file format compatible with earlier diagnostics.
    """
    n_units = counts_a.shape[0]
    if scoring_units is None:
        scoring_units = np.ones(n_units, dtype=bool)
    scoring_units = np.asarray(scoring_units, dtype=bool)
    if scoring_units.shape != (n_units,):
        raise ValueError("scoring_units must have one boolean per unit")

    rate_a = rate_from_counts(counts_a, occ_a, min_occupancy_s)
    rate_b = rate_from_counts(counts_b, occ_b, min_occupancy_s)

    scores_a_to_b = np.full(len(sigma_candidates), np.nan)
    scores_b_to_a = np.full(len(sigma_candidates), np.nan)
    for sigma_i, kernel_sigma in enumerate(sigma_candidates):
        pred_a = circular_smooth(rate_a, kernel_sigma)
        pred_b = circular_smooth(rate_b, kernel_sigma)

        total_a_to_b = 0.0
        total_b_to_a = 0.0
        n_valid_a = 0
        n_valid_b = 0
        for unit_i in np.flatnonzero(scoring_units):
            ll_a = poisson_log_likelihood(
                counts_b[unit_i],
                occ_b,
                pred_a[unit_i],
                min_occupancy_s,
            )
            ll_b = poisson_log_likelihood(
                counts_a[unit_i],
                occ_a,
                pred_b[unit_i],
                min_occupancy_s,
            )
            if np.isfinite(ll_a):
                total_a_to_b += ll_a
                n_valid_a += 1
            if np.isfinite(ll_b):
                total_b_to_a += ll_b
                n_valid_b += 1

        if n_valid_a:
            scores_a_to_b[sigma_i] = total_a_to_b
        if n_valid_b:
            scores_b_to_a[sigma_i] = total_b_to_a

    selected_sigma_a = float(sigma_candidates[int(np.nanargmax(scores_a_to_b))])
    selected_sigma_b = float(sigma_candidates[int(np.nanargmax(scores_b_to_a))])
    selected_sigma_a_by_unit = np.full(n_units, selected_sigma_a)
    selected_sigma_b_by_unit = np.full(n_units, selected_sigma_b)

    return (
        selected_sigma_a_by_unit,
        selected_sigma_b_by_unit,
        scores_a_to_b,
        scores_b_to_a,
        rate_a,
        rate_b,
        selected_sigma_a,
        selected_sigma_b,
    )


def segment_rates_for_cv(counts_segments, occ_segments, min_occupancy_s):
    """
    将多个时间段的 spike count 和 occupancy 转成分段 firing rate。

    输入通常来自把一个 session 切成多个连续片段后的统计结果。返回数组形状为
    `(n_segments, n_units, n_bins)`，供交叉验证选择平滑参数使用。
    """
    return np.stack(
        [rate_from_counts(counts_segments[i], occ_segments[i], min_occupancy_s) for i in range(len(occ_segments))]
    )


def choose_cv_sigma(segment_rates, sigma_candidates):
    """
    通过分段交叉验证为每个 unit 选择最佳平滑 sigma。

    每次留出一个时间段作为测试集，其余时间段平均成训练曲线；训练曲线经过
    候选 sigma 平滑后，与测试曲线计算均方误差。每个 unit 选择平均误差最低的
    sigma，并返回完整的 MSE 评分矩阵。
    """
    n_segments, n_units, _ = segment_rates.shape
    chosen = np.full(n_units, np.nan)
    scores = np.full((n_units, len(sigma_candidates)), np.nan)

    for unit_i in range(n_units):
        for sigma_i, sigma in enumerate(sigma_candidates):
            fold_scores = []
            for holdout in range(n_segments):
                train = np.nanmean(np.delete(segment_rates[:, unit_i, :], holdout, axis=0), axis=0)
                test = segment_rates[holdout, unit_i, :]
                pred = circular_smooth(train, sigma)
                valid = np.isfinite(pred) & np.isfinite(test)
                if valid.sum() >= 5:
                    fold_scores.append(np.nanmean((pred[valid] - test[valid]) ** 2))
            if fold_scores:
                scores[unit_i, sigma_i] = np.mean(fold_scores)

        if np.isfinite(scores[unit_i]).any():
            chosen[unit_i] = sigma_candidates[int(np.nanargmin(scores[unit_i]))]

    return chosen, scores


def segment_masks(time_s, n_segments):
    """
    按时间范围把行为样本切成若干连续片段。

    返回布尔 mask 列表，每个 mask 对应一个时间段。最后一段包含右边界，
    这样最大时间点不会因为半开区间而被漏掉。
    """
    edges = np.linspace(time_s[0], time_s[-1], n_segments + 1)
    masks = []
    for i in range(n_segments):
        if i == n_segments - 1:
            masks.append((time_s >= edges[i]) & (time_s <= edges[i + 1]))
        else:
            masks.append((time_s >= edges[i]) & (time_s < edges[i + 1]))
    return masks


def alternating_partition_masks(time_s, n_segments=8):
    """
    Split a recording into alternating temporal partitions.

    Methods B2 divides each session into eight equal-duration temporal
    segments labeled 1-2-1-2-1-2-1-2. Returning the union of alternating
    segments avoids confounding the two cross-validation partitions with slow
    drift across the session.
    """
    segments = segment_masks(time_s, n_segments)
    partition_a = np.zeros_like(segments[0], dtype=bool)
    partition_b = np.zeros_like(segments[0], dtype=bool)
    for segment_i, mask in enumerate(segments):
        if segment_i % 2 == 0:
            partition_a |= mask
        else:
            partition_b |= mask
    return partition_a, partition_b


def process_session(row, n_bins, min_occupancy_s, sigma_candidates):
    """
    处理一个 session，计算并保存所有 unit 的头方向调谐结果。

    `row` 来自 `session_index.csv`，其中包含行为、unit 元数据和 spike 文件路径。
    函数会计算 occupancy、原始 firing rate、交叉验证平滑曲线、均值归一化曲线
    和 split-half reliability，保存为 `.npz`，并返回一行汇总信息供索引表使用。
    """
    subject_id = row["subject_id"]
    behavior = pd.read_parquet(row["behavior_path"]).sort_values("time_s").reset_index(drop=True)
    units = pd.read_parquet(row["units_path"])
    spikes = np.load(row["spikes_path"])

    occ, edges, centers = occupancy_seconds(behavior, n_bins)
    unit_ids = units["unit_id"].astype(int).to_numpy()

    counts = []
    for unit_id in unit_ids:
        counts.append(spike_counts_by_hd(spikes[f"unit_{unit_id}"], behavior, n_bins))
    counts = np.vstack(counts)

    raw_rates = rate_from_counts(counts, occ, min_occupancy_s)

    time_s = behavior["time_s"].to_numpy(float)
    masks2 = alternating_partition_masks(time_s, n_segments=8)
    occ2 = []
    counts2 = []
    for mask in masks2:
        bseg = behavior.loc[mask].reset_index(drop=True)
        occ_seg, _, _ = occupancy_seconds(bseg, n_bins)
        occ2.append(occ_seg)
        counts2.append([spike_counts_by_hd(spikes[f"unit_{unit_id}"], bseg, n_bins) for unit_id in unit_ids])
    occ2 = np.vstack(occ2)
    counts2 = np.stack([np.vstack(c) for c in counts2])

    included_qc = units["included_qc"].astype(bool).to_numpy()
    (
        chosen_sigma_a,
        chosen_sigma_b,
        cv_a_to_b,
        cv_b_to_a,
        rate_a,
        rate_b,
        chosen_sigma_session_a,
        chosen_sigma_session_b,
    ) = choose_poisson_cv_sigmas(
        counts2[0],
        counts2[1],
        occ2[0],
        occ2[1],
        min_occupancy_s,
        sigma_candidates,
        scoring_units=included_qc,
    )
    smoothed_a = np.vstack(
        [
            circular_smooth(rate_a[i], chosen_sigma_a[i] if np.isfinite(chosen_sigma_a[i]) else 2.0)
            for i in range(len(unit_ids))
        ]
    )
    smoothed_b = np.vstack(
        [
            circular_smooth(rate_b[i], chosen_sigma_b[i] if np.isfinite(chosen_sigma_b[i]) else 2.0)
            for i in range(len(unit_ids))
        ]
    )
    smoothed = np.nanmean(np.stack([smoothed_a, smoothed_b]), axis=0)
    normalized, unit_mean_rate_hz = mean_normalize(smoothed)
    chosen_sigma = np.nanmean(np.stack([chosen_sigma_a, chosen_sigma_b]), axis=0)

    split_half_r = split_half_reliability(counts2[0], counts2[1], occ2[0], occ2[1], min_occupancy_s)

    out_path = PROCESSED / f"{subject_id}_hd_tuning_{n_bins}bins.npz"
    np.savez_compressed(
        out_path,
        unit_ids=unit_ids,
        bin_edges_rad=edges,
        bin_centers_rad=centers,
        occupancy_s=occ,
        spike_counts=counts,
        raw_rate_hz=raw_rates,
        smoothed_rate_hz=smoothed,
        smoothed_rate_partition_a_hz=smoothed_a,
        smoothed_rate_partition_b_hz=smoothed_b,
        normalized_rate=normalized,
        unit_mean_rate_hz=unit_mean_rate_hz,
        chosen_sigma_bins=chosen_sigma,
        chosen_sigma_partition_a_bins=chosen_sigma_a,
        chosen_sigma_partition_b_bins=chosen_sigma_b,
        chosen_sigma_session_a_bins=chosen_sigma_session_a,
        chosen_sigma_session_b_bins=chosen_sigma_session_b,
        cv_sigma_candidates=np.asarray(sigma_candidates),
        cv_poisson_a_to_b=cv_a_to_b,
        cv_poisson_b_to_a=cv_b_to_a,
        split_half_r=split_half_r,
        included_qc=included_qc,
    )

    return {
        "subject_id": subject_id,
        "session_id": row["session_id"],
        "n_units": len(unit_ids),
        "n_included_qc": int(units["included_qc"].sum()),
        "total_occupancy_s": float(occ.sum()),
        "median_chosen_sigma_bins": float(np.nanmedian(chosen_sigma)),
        "median_split_half_r": float(np.nanmedian(split_half_r)),
        "tuning_path": str(out_path),
    }


def main():
    """
    命令行入口：批量读取中间数据并生成 HD tuning 结果索引。

    默认使用 100 个方向 bin、最小 occupancy 阈值 0.05 秒，并在多个 sigma 候选值
    中为每个 unit 做交叉验证选择。输出位于 `data/processed`。
    """
    n_bins = 100
    min_occupancy_s = 0.05
    sigma_candidates = np.logspace(-1, 2, 31)

    session_index = pd.read_csv(INTERIM / "session_index.csv")
    if "status" in session_index.columns:
        ok = session_index["status"].fillna("ok").eq("ok")
    else:
        ok = pd.Series(True, index=session_index.index)
    rows = []
    for _, row in tqdm(session_index.loc[ok].iterrows(), total=int(ok.sum()), desc="Computing HD tuning"):
        rows.append(process_session(row, n_bins, min_occupancy_s, sigma_candidates))

    tuning_index = pd.DataFrame(rows)
    tuning_index.to_csv(PROCESSED / "hd_tuning_index.csv", index=False)

    print(tuning_index)
    print("Saved:", PROCESSED / "hd_tuning_index.csv")


if __name__ == "__main__":
    main()
