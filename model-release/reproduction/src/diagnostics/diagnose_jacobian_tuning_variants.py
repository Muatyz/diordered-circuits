try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from diagnose_discretization import dense_weights_from_factors, max_real_jacobian
from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from diagnose_smoothing_variants import finite_corrcoef, process_session
from plot_figure3_abcd import REPRODUCTION_ROOT
from utils import lowrank_recurrent_drive, sinkhorn_normalize


INTERIM = Path("data/interim")
REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_jacobian_tuning_variants.json"


def finite_percentiles(values, q=(0, 1, 5, 50, 95, 99, 100)):
    """
    Return finite percentiles as JSON-friendly floats.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {f"p{qq:g}": float(vv) for qq, vv in zip(q, np.percentile(values, q))}


def support_counts(values, thresholds=(1e-12, 1e-10, 1e-8, 1e-6, 1e-4)):
    """
    Count target-rate entries below numerical support thresholds.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    total = int(finite.size)
    out = {}
    for threshold in thresholds:
        count = int(np.sum(finite < threshold))
        out[f"lt_{threshold:g}"] = {
            "count": count,
            "fraction": float(count / total) if total else float("nan"),
        }
    return out


def collect_variant(name, per_session):
    """
    Pool one smoothing variant into a population tuning matrix.
    """
    curves = []
    sigma_a = []
    sigma_b = []
    zero_bins = []
    for session_result in per_session:
        item = session_result[name]
        included = item["included"]
        normalized = np.asarray(item["normalized"], dtype=float)[included]
        curves.append(normalized)
        zero_bins.extend(np.sum(normalized == 0.0, axis=1).tolist())

        if len(item["sigma_a"]) == len(included):
            sigma_a.extend(np.asarray(item["sigma_a"], dtype=float)[included].tolist())
            sigma_b.extend(np.asarray(item["sigma_b"], dtype=float)[included].tolist())
        else:
            sigma_a.extend(np.asarray(item["sigma_a"], dtype=float).tolist())
            sigma_b.extend(np.asarray(item["sigma_b"], dtype=float).tolist())

    tuning = np.vstack(curves)
    finite = np.all(np.isfinite(tuning), axis=1)
    return {
        "tuning": tuning[finite],
        "sigma_a": np.asarray(sigma_a, dtype=float),
        "sigma_b": np.asarray(sigma_b, dtype=float),
        "zero_bins_per_unit": np.asarray(zero_bins, dtype=float),
    }


def stability_summary(phi, eps, beta=2.0, regularization=1e-6, sample_bins=(0, 25, 50, 75)):
    """
    Compute A2 fixed-point residuals and sampled B4.2 Jacobian eigenvalues.
    """
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    factor_a, factor_b, diagonal = factors_from_target_current(phi, x_star, regularization)
    drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
    weights = dense_weights_from_factors(factor_a, factor_b, diagonal)
    max_reals = [max_real_jacobian(weights, phi[:, idx], beta) for idx in sample_bins]
    return {
        "eps": float(eps),
        "target_current": finite_percentiles(x_star),
        "factor_a_abs": finite_percentiles(np.abs(factor_a), q=(0, 1, 50, 95, 99, 100)),
        "weight_abs": finite_percentiles(np.abs(weights), q=(0, 50, 95, 99, 100)),
        "flow_residual_l2_div_sqrt_n": finite_percentiles(residual, q=(0, 1, 50, 95, 99, 100)),
        "sample_bins": [int(idx) for idx in sample_bins],
        "jacobian_max_real_c1": [float(value) for value in max_reals],
        "jacobian_max_real_c1_max": float(np.max(max_reals)),
    }


def summarize_variant(name, variant):
    """
    Summarize one B2 smoothing interpretation through A2/B4.2 stability.
    """
    tuning = variant["tuning"]
    phi = sinkhorn_normalize(tuning)
    return {
        "n_neurons": int(tuning.shape[0]),
        "n_angles": int(tuning.shape[1]),
        "sigma_a": finite_percentiles(variant["sigma_a"]),
        "sigma_b": finite_percentiles(variant["sigma_b"]),
        "sigma_corr": float(finite_corrcoef(variant["sigma_a"], variant["sigma_b"])),
        "zero_bins_per_unit": finite_percentiles(variant["zero_bins_per_unit"], q=(50, 95, 99, 100)),
        "tuning": {
            "percentiles": finite_percentiles(tuning),
            "support_counts": support_counts(tuning),
        },
        "doubly_normalized_tuning": {
            "percentiles": finite_percentiles(phi),
            "support_counts": support_counts(phi),
            "neuron_mean": finite_percentiles(np.mean(phi, axis=1), q=(0, 50, 100)),
            "angle_mean": finite_percentiles(np.mean(phi, axis=0), q=(0, 50, 100)),
        },
        "stability": [
            stability_summary(phi, eps=1e-12),
            stability_summary(phi, eps=1e-6),
        ],
    }


def main():
    """
    Test whether B2 smoothing interpretation explains positive Jacobian modes.
    """
    n_bins = 100
    min_occupancy_s = 0.05
    sigma_candidates = np.logspace(-1, 2, 31)

    session_index = pd.read_csv(INTERIM / "session_index.csv")
    ok = session_index["status"].fillna("ok").eq("ok")
    rows = session_index.loc[ok].reset_index(drop=True)

    per_session = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Rebuilding B2 variants"):
        per_session.append(process_session(row, n_bins, min_occupancy_s, sigma_candidates))

    report = {
        "status": "diagnostic_only_no_pipeline_change",
        "question": "Do B2 smoothing variants explain positive real-part Jacobian modes at lambda=1e-6, c=1?",
        "fixed_parameters": {
            "activation_beta": 2.0,
            "regularization": 1e-6,
            "uniform_inhibition_c": 1.0,
            "n_bins": n_bins,
            "min_occupancy_s": min_occupancy_s,
            "sigma_candidates": [float(value) for value in sigma_candidates],
        },
        "variants": {},
        "interpretation": [
            "Only B2 tuning extraction is varied; A2, B4.2, lambda, c, and the Euler dynamics are unchanged.",
            "The unitwise variant follows the literal 'for each neuron' reading and the reported cross-partition sigma correlation over neurons.",
            "The sessionwise variant follows the likelihood expression with an explicit sum over neurons in a mouse/session.",
        ],
    }

    for name in ["unitwise_truncated", "sessionwise_truncated"]:
        variant = collect_variant(name, per_session)
        report["variants"][name] = summarize_variant(name, variant)

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
