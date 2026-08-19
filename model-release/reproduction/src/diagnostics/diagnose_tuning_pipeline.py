try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
from pathlib import Path

import numpy as np
import pandas as pd

from plot_figure3_abcd import load_population_tuning
from utils import sinkhorn_normalize, softplus_inverse


PROCESSED = Path("data/processed")
INTERIM = Path("data/interim")


def finite_corrcoef(x, y):
    """
    Compute Pearson correlation on jointly finite bins.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return np.nan
    x0 = x[ok] - np.mean(x[ok])
    y0 = y[ok] - np.mean(y[ok])
    return float(np.dot(x0, y0) / np.sqrt(np.dot(x0, x0) * np.dot(y0, y0)))


def percentile_text(values, q=(0, 1, 50, 95, 99, 100)):
    """
    Format finite percentiles on one line for terminal diagnostics.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "no finite values"
    parts = [f"p{qq:g}={vv:.6g}" for qq, vv in zip(q, np.percentile(values, q))]
    return ", ".join(parts)


def main():
    """
    Audit the head-direction tuning pipeline used to define the target manifold.
    """
    session_index = pd.read_csv(INTERIM / "session_index.csv")
    ok_sessions = session_index["status"].fillna("ok").eq("ok")
    print("Sessions:", int(ok_sessions.sum()), "/", len(session_index))
    print("Included initial:", int(session_index.loc[ok_sessions, "n_included_initial"].sum()))
    print("Included QC:", int(session_index.loc[ok_sessions, "n_included_qc"].sum()))

    tuning_index = pd.read_csv(PROCESSED / "hd_tuning_index.csv")
    split_half = []
    smoothed_partition_corr = []
    chosen_sigma = []
    chosen_sigma_a = []
    chosen_sigma_b = []
    chosen_sigma_session_a = []
    chosen_sigma_session_b = []
    zero_bins_per_unit = []
    mean_rate_hz = []

    for path in tuning_index["tuning_path"]:
        data = np.load(path)
        included = data["included_qc"].astype(bool)
        normalized = data["normalized_rate"][included]
        zero_bins_per_unit.extend(np.sum(normalized == 0.0, axis=1).tolist())
        split_half.extend(data["split_half_r"][included].tolist())
        chosen_sigma.extend(data["chosen_sigma_bins"][included].tolist())
        if "median_chosen_sigma_partition_a_bins" in data.files and "median_chosen_sigma_partition_b_bins" in data.files:
            chosen_sigma_session_a.append(float(data["median_chosen_sigma_partition_a_bins"]))
            chosen_sigma_session_b.append(float(data["median_chosen_sigma_partition_b_bins"]))
        elif "chosen_sigma_session_a_bins" in data.files and "chosen_sigma_session_b_bins" in data.files:
            chosen_sigma_session_a.append(float(data["chosen_sigma_session_a_bins"]))
            chosen_sigma_session_b.append(float(data["chosen_sigma_session_b_bins"]))
        if "chosen_sigma_partition_a_bins" in data.files and "chosen_sigma_partition_b_bins" in data.files:
            chosen_sigma_a.extend(data["chosen_sigma_partition_a_bins"][included].tolist())
            chosen_sigma_b.extend(data["chosen_sigma_partition_b_bins"][included].tolist())
        mean_rate_hz.extend(data["unit_mean_rate_hz"][included].tolist())

        part_a = data["smoothed_rate_partition_a_hz"][included]
        part_b = data["smoothed_rate_partition_b_hz"][included]
        for curve_a, curve_b in zip(part_a, part_b):
            smoothed_partition_corr.append(finite_corrcoef(curve_a, curve_b))

    tuning, angles_rad, metadata = load_population_tuning()
    doubly = sinkhorn_normalize(tuning)
    target_current = softplus_inverse(doubly, beta=2.0)
    column_mean = np.mean(tuning, axis=0)
    nonzero = tuning > 0

    print("Loaded included tuning matrix:", tuning.shape)
    print("Unit mean rate Hz:", percentile_text(mean_rate_hz))
    print("Chosen sigma bins:", percentile_text(chosen_sigma))
    if chosen_sigma_a and chosen_sigma_b:
        sigma_a = np.asarray(chosen_sigma_a, dtype=float)
        sigma_b = np.asarray(chosen_sigma_b, dtype=float)
        sigma_corr = finite_corrcoef(sigma_a, sigma_b)
        log_sigma_corr = finite_corrcoef(np.log10(sigma_a), np.log10(sigma_b))
        print("Partition-specific sigma A:", percentile_text(chosen_sigma_a))
        print("Partition-specific sigma B:", percentile_text(chosen_sigma_b))
        print(f"Partition sigma correlation, unit-weighted saved arrays: {sigma_corr:.6g}")
        print(f"Partition log10-sigma correlation, unit-weighted saved arrays: {log_sigma_corr:.6g}")
    if chosen_sigma_session_a and chosen_sigma_session_b:
        session_sigma_corr = finite_corrcoef(
            np.asarray(chosen_sigma_session_a, dtype=float),
            np.asarray(chosen_sigma_session_b, dtype=float),
        )
        print("Per-session median sigma A:", percentile_text(chosen_sigma_session_a))
        print("Per-session median sigma B:", percentile_text(chosen_sigma_session_b))
        print(f"Partition sigma correlation, per-session medians: {session_sigma_corr:.6g}")
    print("Split-half reliability:", percentile_text(split_half, q=(1, 50, 99)))
    print("Smoothed partition corr:", percentile_text(smoothed_partition_corr, q=(1, 50, 99)))
    print("Zero bins per included unit:", percentile_text(zero_bins_per_unit, q=(50, 95, 99, 100)))
    print("Units with any zero bin:", int(np.sum(np.asarray(zero_bins_per_unit) > 0)))
    print("Raw column mean:", percentile_text(column_mean))
    print("Raw tuning values:", percentile_text(tuning))
    print("Doubly normalized values:", percentile_text(doubly))
    print("Nonzero Sinkhorn relative correction:", percentile_text(np.abs(doubly[nonzero] / tuning[nonzero] - 1.0)))
    print("Target current values:", percentile_text(target_current))


if __name__ == "__main__":
    main()
