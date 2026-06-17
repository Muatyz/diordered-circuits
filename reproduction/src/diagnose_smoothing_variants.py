from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from compute_hd_tuning import (
    alternating_partition_masks,
    circular_smooth,
    occupancy_seconds,
    poisson_log_likelihood,
    rate_from_counts,
    spike_counts_by_hd,
)
from utils import sinkhorn_normalize, softplus_inverse


INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")


def percentile_text(values, q=(0, 1, 50, 95, 99, 100)):
    """
    把一组有限数值格式化为分位数文本，便于比较不同 tuning 提取方案。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "no finite values"
    return ", ".join(f"p{qq:g}={vv:.6g}" for qq, vv in zip(q, np.percentile(values, q)))


def finite_corrcoef(x, y):
    """
    只在共同有限的样本上计算 Pearson 相关。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def full_circular_smooth(values, sigma_bins):
    """
    使用完整环形高斯核平滑，同时正确处理 NaN 缺失值。

    与 `circular_smooth` 一致，但显式保留了“不截断核”的语义以便对比。
    缺失 (NaN) 的 bin 在卷积时被忽略，核在每个输出位置上基于可用的
    有限邻域重新归一化。
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

    result = np.full_like(out, np.nan)
    mask = weight_sum > 0
    result[mask] = out[mask] / weight_sum[mask]
    return result


def unitwise_sigma_variant(counts_a, counts_b, occ_a, occ_b, min_occupancy_s, sigma_candidates, smoother):
    """
    按当前代码的读法：每个 neuron、每个 partition 各自用 Poisson CV 选 sigma。
    """
    rate_a = rate_from_counts(counts_a, occ_a, min_occupancy_s)
    rate_b = rate_from_counts(counts_b, occ_b, min_occupancy_s)
    n_units = counts_a.shape[0]
    sigma_a = np.full(n_units, np.nan)
    sigma_b = np.full(n_units, np.nan)
    smoothed_a = np.full_like(rate_a, np.nan, dtype=float)
    smoothed_b = np.full_like(rate_b, np.nan, dtype=float)

    for unit_i in range(n_units):
        scores_a_to_b = []
        scores_b_to_a = []
        preds_a = []
        preds_b = []
        for sigma in sigma_candidates:
            pred_a = smoother(rate_a[unit_i], sigma)
            pred_b = smoother(rate_b[unit_i], sigma)
            preds_a.append(pred_a)
            preds_b.append(pred_b)
            scores_a_to_b.append(
                poisson_log_likelihood(counts_b[unit_i], occ_b, pred_a, min_occupancy_s)
            )
            scores_b_to_a.append(
                poisson_log_likelihood(counts_a[unit_i], occ_a, pred_b, min_occupancy_s)
            )

        scores_a_to_b = np.asarray(scores_a_to_b, dtype=float)
        scores_b_to_a = np.asarray(scores_b_to_a, dtype=float)
        if np.isfinite(scores_a_to_b).any():
            best = int(np.nanargmax(scores_a_to_b))
            sigma_a[unit_i] = sigma_candidates[best]
            smoothed_a[unit_i] = preds_a[best]
        if np.isfinite(scores_b_to_a).any():
            best = int(np.nanargmax(scores_b_to_a))
            sigma_b[unit_i] = sigma_candidates[best]
            smoothed_b[unit_i] = preds_b[best]

    smoothed = np.nanmean(np.stack([smoothed_a, smoothed_b]), axis=0)
    return smoothed, sigma_a, sigma_b


def sessionwise_sigma_variant(counts_a, counts_b, occ_a, occ_b, min_occupancy_s, sigma_candidates, smoother):
    """
    按 B2 公式的另一种自然读法：对同一 recording 内所有 neuron 的 Poisson
    log-likelihood 求和，为每个 partition 选一个共享的 Gaussian width。

    B2 的公式显式对 neuron 指标 i 求和，并报告两个 partition 的 optimal
    smoothing widths 的相关性；这个诊断检验该读法是否更接近论文描述。
    """
    rate_a = rate_from_counts(counts_a, occ_a, min_occupancy_s)
    rate_b = rate_from_counts(counts_b, occ_b, min_occupancy_s)

    total_a_to_b = np.full(len(sigma_candidates), np.nan)
    total_b_to_a = np.full(len(sigma_candidates), np.nan)
    smoothed_a_candidates = []
    smoothed_b_candidates = []

    for sigma_i, sigma in enumerate(sigma_candidates):
        pred_a = smoother(rate_a, sigma)
        pred_b = smoother(rate_b, sigma)
        smoothed_a_candidates.append(pred_a)
        smoothed_b_candidates.append(pred_b)

        score_a = 0.0
        score_b = 0.0
        valid_a = 0
        valid_b = 0
        for unit_i in range(counts_a.shape[0]):
            ll_a = poisson_log_likelihood(counts_b[unit_i], occ_b, pred_a[unit_i], min_occupancy_s)
            ll_b = poisson_log_likelihood(counts_a[unit_i], occ_a, pred_b[unit_i], min_occupancy_s)
            if np.isfinite(ll_a):
                score_a += ll_a
                valid_a += 1
            if np.isfinite(ll_b):
                score_b += ll_b
                valid_b += 1
        if valid_a:
            total_a_to_b[sigma_i] = score_a
        if valid_b:
            total_b_to_a[sigma_i] = score_b

    best_a = int(np.nanargmax(total_a_to_b))
    best_b = int(np.nanargmax(total_b_to_a))
    sigma_a = float(sigma_candidates[best_a])
    sigma_b = float(sigma_candidates[best_b])
    smoothed = np.nanmean(np.stack([smoothed_a_candidates[best_a], smoothed_b_candidates[best_b]]), axis=0)
    return smoothed, sigma_a, sigma_b


def normalize_rows(rates):
    """
    对每个 neuron 的 smoothed tuning curve 做 unit mean normalization。
    """
    out = np.asarray(rates, dtype=float).copy()
    means = np.nanmean(out, axis=1)
    good = np.isfinite(means) & (means > 0)
    out[good] = out[good] / means[good, None]
    out[~good] = np.nan
    return out


def process_session(row, n_bins, min_occupancy_s, sigma_candidates):
    """
    从 interim raw counts 重新构造一个 session 的 partition 数据，并运行诊断变体。
    """
    behavior = pd.read_parquet(row["behavior_path"]).sort_values("time_s").reset_index(drop=True)
    units = pd.read_parquet(row["units_path"])
    spikes = np.load(row["spikes_path"])
    included = units["included_qc"].astype(bool).to_numpy()
    unit_ids = units["unit_id"].astype(int).to_numpy()

    time_s = behavior["time_s"].to_numpy(float)
    masks = alternating_partition_masks(time_s, n_segments=8)
    occ_parts = []
    counts_parts = []
    for mask in masks:
        bseg = behavior.loc[mask].reset_index(drop=True)
        occ_seg, _, _ = occupancy_seconds(bseg, n_bins)
        counts_seg = [
            spike_counts_by_hd(spikes[f"unit_{unit_id}"], bseg, n_bins)
            for unit_id in unit_ids
        ]
        occ_parts.append(occ_seg)
        counts_parts.append(np.vstack(counts_seg))

    counts_a, counts_b = counts_parts
    occ_a, occ_b = occ_parts
    variants = {}
    variants["unitwise_truncated"] = unitwise_sigma_variant(
        counts_a, counts_b, occ_a, occ_b, min_occupancy_s, sigma_candidates, circular_smooth
    )
    variants["unitwise_full"] = unitwise_sigma_variant(
        counts_a, counts_b, occ_a, occ_b, min_occupancy_s, sigma_candidates, full_circular_smooth
    )
    variants["sessionwise_truncated"] = sessionwise_sigma_variant(
        counts_a[included], counts_b[included], occ_a, occ_b, min_occupancy_s, sigma_candidates, circular_smooth
    )
    variants["sessionwise_full"] = sessionwise_sigma_variant(
        counts_a[included], counts_b[included], occ_a, occ_b, min_occupancy_s, sigma_candidates, full_circular_smooth
    )

    output = {}
    for name, (smoothed, sigma_a, sigma_b) in variants.items():
        if np.ndim(sigma_a) == 0:
            normalized = normalize_rows(smoothed)
            output[name] = {
                "normalized": normalized,
                "included": np.ones(normalized.shape[0], dtype=bool),
                "sigma_a": np.asarray([sigma_a], dtype=float),
                "sigma_b": np.asarray([sigma_b], dtype=float),
            }
        else:
            output[name] = {
                "normalized": normalize_rows(smoothed),
                "included": included,
                "sigma_a": np.asarray(sigma_a, dtype=float),
                "sigma_b": np.asarray(sigma_b, dtype=float),
            }
    return output


def summarize_variant(name, per_session):
    """
    汇总一个 tuning 提取变体对目标流形数值性质的影响。
    """
    curves = []
    sigma_a = []
    sigma_b = []
    zeros = []
    for session_result in per_session:
        item = session_result[name]
        included = item["included"]
        normalized = item["normalized"][included]
        curves.append(normalized)
        zeros.extend(np.sum(normalized == 0.0, axis=1).tolist())
        sigma_a.extend(item["sigma_a"][included if len(item["sigma_a"]) == len(included) else slice(None)].tolist())
        sigma_b.extend(item["sigma_b"][included if len(item["sigma_b"]) == len(included) else slice(None)].tolist())

    tuning = np.vstack(curves)
    finite = np.all(np.isfinite(tuning), axis=1)
    tuning = tuning[finite]
    doubly = sinkhorn_normalize(tuning)
    target_current = softplus_inverse(doubly, beta=2.0)

    print(f"\n{name}")
    print("  tuning shape:", tuning.shape)
    print("  sigma A:", percentile_text(sigma_a))
    print("  sigma B:", percentile_text(sigma_b))
    print("  sigma corr:", f"{finite_corrcoef(sigma_a, sigma_b):.6g}")
    print("  zero bins per unit:", percentile_text(zeros, q=(50, 95, 99, 100)))
    print("  units with any zero bin:", int(np.sum(np.asarray(zeros) > 0)))
    print("  tuning values:", percentile_text(tuning))
    print("  doubly normalized values:", percentile_text(doubly))
    print("  target current:", percentile_text(target_current))


def main():
    """
    诊断 raw spike/occupancy 到 tuning curve 的关键歧义点。
    """
    n_bins = 100
    min_occupancy_s = 0.05
    sigma_candidates = np.logspace(-1, 2, 31)

    session_index = pd.read_csv(INTERIM / "session_index.csv")
    ok = session_index["status"].fillna("ok").eq("ok")
    rows = session_index.loc[ok].reset_index(drop=True)

    per_session = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Smoothing variants"):
        per_session.append(process_session(row, n_bins, min_occupancy_s, sigma_candidates))

    for name in ["unitwise_truncated", "unitwise_full", "sessionwise_truncated", "sessionwise_full"]:
        summarize_variant(name, per_session)


if __name__ == "__main__":
    main()
