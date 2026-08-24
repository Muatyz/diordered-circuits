try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json

import numpy as np

from diagnose_discretization import dense_weights_from_factors, max_real_jacobian
from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from plot_figure3_abcd import REPRODUCTION_ROOT, load_population_tuning
from utils import lowrank_recurrent_drive, sinkhorn_normalize


REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_rate_floor_stability.json"


def finite_percentiles(values, q=(0, 1, 5, 50, 95, 99, 100)):
    """
    Return finite percentiles as JSON-friendly floats.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {f"p{qq:g}": float(vv) for qq, vv in zip(q, np.percentile(values, q))}


def l2_norm(values):
    """
    Compute an L2 norm without calling BLAS-backed numpy.linalg.
    """
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(values * values)))


def stability_for_floor(raw_tuning, firing_rate_floor, eps=1e-12):
    """
    Apply a diagnostic rate floor before Sinkhorn and summarize local stability.
    """
    beta = 2.0
    regularization = 1e-6
    sample_bins = [0, 25, 50, 75]

    floored = np.maximum(raw_tuning, firing_rate_floor)
    phi = sinkhorn_normalize(floored)
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    factor_a, factor_b, diagonal = factors_from_target_current(phi, x_star, regularization)
    drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
    weights = dense_weights_from_factors(factor_a, factor_b, diagonal)
    max_reals = [max_real_jacobian(weights, phi[:, idx], beta) for idx in sample_bins]
    base_phi = sinkhorn_normalize(raw_tuning)
    relative_change = l2_norm(phi - base_phi) / l2_norm(base_phi)
    return {
        "firing_rate_floor": float(firing_rate_floor),
        "inverse_eps": float(eps),
        "relative_l2_change_from_unfloored_phi": float(relative_change),
        "phi_percentiles": finite_percentiles(phi),
        "target_current_percentiles": finite_percentiles(x_star),
        "factor_a_abs": finite_percentiles(np.abs(factor_a), q=(0, 1, 50, 95, 99, 100)),
        "weight_abs": finite_percentiles(np.abs(weights), q=(0, 50, 95, 99, 100)),
        "flow_residual_l2_div_sqrt_n": finite_percentiles(residual, q=(0, 1, 50, 95, 99, 100)),
        "jacobian_max_real_c1": [float(value) for value in max_reals],
        "jacobian_max_real_c1_max": float(np.max(max_reals)),
    }


def main():
    """
    Diagnose whether missing positive target-rate support explains instability.
    """
    raw_tuning, _, _ = load_population_tuning()
    floors = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
    report = {
        "status": "diagnostic_only_no_pipeline_change",
        "question": "Would a pre-Sinkhorn firing-rate support floor remove positive Jacobian modes?",
        "fixed_parameters": {
            "activation_beta": 2.0,
            "regularization": 1e-6,
            "uniform_inhibition_c": 1.0,
            "inverse_eps": 1e-12,
        },
        "cases": [stability_for_floor(raw_tuning, floor, eps=1e-12) for floor in floors],
        "interpretation_hint": "A floor that stabilizes only after a large relative target change is not evidence for a paper-faithful fix; it only identifies target support as the sensitive axis.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
