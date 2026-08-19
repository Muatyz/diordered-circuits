try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_discretization import dense_weights_from_factors, max_real_jacobian
from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from plot_figure3_abcd import PROCESSED, REPRODUCTION_ROOT, load_population_tuning
from utils import lowrank_recurrent_drive, sinkhorn_normalize


REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_tail_stability_diagnostics.json"


def finite_percentiles(values, q=(0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)):
    """
    Return finite-value percentiles as plain JSON-serializable floats.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    percentiles = np.percentile(values, q)
    return {f"p{qq:g}": float(vv) for qq, vv in zip(q, percentiles)}


def threshold_counts(values, thresholds):
    """
    Count how much of an array falls below several numerical support levels.
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


def load_session_summary():
    """
    Summarize whether the cached dataset selection matches the paper-level count.
    """
    session_index_path = PROCESSED.parent / "interim" / "session_index.csv"
    if not session_index_path.exists():
        return {"available": False}

    session_index = pd.read_csv(session_index_path)
    ok = session_index["status"].fillna("ok").eq("ok") if "status" in session_index.columns else np.ones(len(session_index), dtype=bool)
    return {
        "available": True,
        "sessions_ok": int(np.sum(ok)),
        "sessions_total": int(len(session_index)),
        "included_initial": int(session_index.loc[ok, "n_included_initial"].sum()),
        "included_qc": int(session_index.loc[ok, "n_included_qc"].sum()),
        "extraction_epoch": "wake_square",
    }


def tuning_tail_summary():
    """
    Summarize the support of the reconstructed firing-rate target manifold.
    """
    tuning, angles_rad, metadata = load_population_tuning()
    doubly = sinkhorn_normalize(tuning)
    thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    return {
        "n_neurons": int(tuning.shape[0]),
        "n_angles": int(tuning.shape[1]),
        "metadata_rows": int(len(metadata)),
        "angle_bins_rad_first_last": [float(angles_rad[0]), float(angles_rad[-1])],
        "single_neuron_normalized_tuning": {
            "percentiles": finite_percentiles(tuning),
            "threshold_counts": threshold_counts(tuning, thresholds),
        },
        "doubly_normalized_tuning": {
            "percentiles": finite_percentiles(doubly),
            "threshold_counts": threshold_counts(doubly, thresholds),
            "neuron_mean": finite_percentiles(np.mean(doubly, axis=1), q=(0, 50, 100)),
            "angle_mean": finite_percentiles(np.mean(doubly, axis=0), q=(0, 50, 100)),
        },
    }


def stability_for_inverse_floor(phi, beta, regularization, eps, sample_bins):
    """
    Compute target-current tail, flow residual, and sampled Jacobian stability.
    """
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    a, b, diagonal = factors_from_target_current(phi, x_star, regularization)
    drive = lowrank_recurrent_drive(a, b, diagonal)
    residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
    weights = dense_weights_from_factors(a, b, diagonal)
    max_reals = [max_real_jacobian(weights, phi[:, idx], beta) for idx in sample_bins]
    return {
        "eps": float(eps),
        "target_current_percentiles": finite_percentiles(x_star),
        "factor_a_abs_percentiles": finite_percentiles(np.abs(a), q=(0, 1, 50, 95, 99, 100)),
        "flow_residual_l2_div_sqrt_n": finite_percentiles(residual, q=(0, 1, 50, 95, 99, 100)),
        "sample_bins": [int(idx) for idx in sample_bins],
        "jacobian_max_real_c1": [float(value) for value in max_reals],
        "jacobian_has_positive_sampled_mode": bool(np.max(max_reals) > 0.0),
    }


def current_matrix_stability_summary():
    """
    Recompute sampled local stability under the paper lambda and c settings.
    """
    matrix_path = PROCESSED / "figure3_abcd_weight_matrices.npz"
    data = np.load(matrix_path)
    phi = data["doubly_normalized_tuning"].astype(float)
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])
    sample_bins = [0, 25, 50, 75]

    return {
        "matrix_path": str(matrix_path),
        "activation_beta": beta,
        "regularization": regularization,
        "uniform_inhibition_c": 1.0,
        "weight_formula_version": str(data.get("weight_formula_version", "")),
        "inverse_floor_cases": [
            stability_for_inverse_floor(phi, beta, regularization, eps=1e-12, sample_bins=sample_bins),
            stability_for_inverse_floor(phi, beta, regularization, eps=1e-6, sample_bins=sample_bins),
        ],
    }


def main():
    """
    Diagnose whether the current instability is traceable to target support.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "diagnostic_only_no_pipeline_change",
        "question": "Why does the B2/B3/A2 data-derived target still have positive real-part modes at lambda=1e-6, c=1?",
        "paper_scope": {
            "kept": [
                "B2 wake-square HD tuning with Gaussian smoothing and unit-mean normalization",
                "B4.1 double normalization by Sinkhorn-Knopp",
                "Setting 1 softplus inverse for the input-current target",
                "A2 constrained minimum-norm weights with J_ii=0",
                "B4.2 uniform inhibition c=1",
            ],
            "not_used_as_fix": [
                "target-point feedback",
                "state clipping",
                "relaxing the target manifold before measuring distance",
                "changing lambda away from the paper value",
            ],
        },
        "session_selection": load_session_summary(),
        "tuning_tail": tuning_tail_summary(),
        "local_stability": current_matrix_stability_summary(),
        "interpretation": [
            "The reconstructed target has an extremely small firing-rate tail before the softplus inverse.",
            "Raising only the inverse numerical floor to 1e-6 reduces the negative current tail but does not remove sampled positive Jacobian modes.",
            "The instability is therefore not explained by transpose or low-rank implementation differences; it remains tied to target support or an unreproduced original preprocessing/numerical-support detail.",
            "The paper text reviewed so far states B2/B3/A2/B4 operations but does not explicitly specify a firing-rate floor, current clipping, or target relaxation step.",
        ],
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
