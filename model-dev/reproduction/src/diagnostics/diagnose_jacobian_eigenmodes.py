try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json
from pathlib import Path

import numpy as np

from diagnose_discretization import dense_weights_from_factors
from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from plot_figure3_abcd import PROCESSED, REPRODUCTION_ROOT


REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_jacobian_eigenmodes.json"


def build_jacobian(weights, phi_at_angle, beta=2.0, inhibition_c=1.0):
    """
    Build the B4.2 current-space Jacobian at one target-manifold point.
    """
    n_neurons = weights.shape[0]
    derivative = 1.0 - np.exp(-beta * phi_at_angle)
    jacobian = -np.eye(n_neurons)
    jacobian += weights * derivative[None, :]
    jacobian -= (inhibition_c / n_neurons) * derivative[None, :]
    return jacobian


def complex_cosine(vector, reference):
    """
    Compute the absolute cosine between a complex eigenvector and real direction.
    """
    vector = np.asarray(vector)
    reference = np.asarray(reference, dtype=float)
    denom = np.linalg.norm(vector) * np.linalg.norm(reference)
    if denom <= 0:
        return float("nan")
    return float(np.abs(np.vdot(vector, reference)) / denom)


def eigensummary(phi, eps, sample_bins=(0, 25, 50, 75), n_modes=6):
    """
    Summarize leading Jacobian eigenmodes and their tangent alignment.
    """
    beta = 2.0
    regularization = 1e-6
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    factor_a, factor_b, diagonal = factors_from_target_current(phi, x_star, regularization)
    weights = dense_weights_from_factors(factor_a, factor_b, diagonal)
    tangent = 0.5 * (np.roll(x_star, -1, axis=1) - np.roll(x_star, 1, axis=1))
    uniform = np.ones(phi.shape[0])

    output = {}
    for bin_idx in sample_bins:
        jacobian = build_jacobian(weights, phi[:, bin_idx], beta=beta, inhibition_c=1.0)
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        order = np.argsort(eigenvalues.real)[::-1]
        modes = []
        for rank, eig_idx in enumerate(order[:n_modes]):
            eigenvalue = eigenvalues[eig_idx]
            eigenvector = eigenvectors[:, eig_idx]
            modes.append(
                {
                    "rank": rank,
                    "eigenvalue_real": float(eigenvalue.real),
                    "eigenvalue_imag": float(eigenvalue.imag),
                    "cosine_with_current_tangent": complex_cosine(eigenvector, tangent[:, bin_idx]),
                    "cosine_with_uniform_current": complex_cosine(eigenvector, uniform),
                }
            )
        output[str(int(bin_idx))] = modes
    return output


def main():
    """
    Check whether positive real-part modes are tangent or normal modes.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"].astype(float)
    report = {
        "status": "diagnostic_only_no_pipeline_change",
        "question": "Are positive Jacobian modes tangent-like or normal to the target manifold?",
        "parameters": {
            "activation_beta": 2.0,
            "regularization": 1e-6,
            "uniform_inhibition_c": 1.0,
            "sample_bins": [0, 25, 50, 75],
        },
        "cases": {
            "eps_1e-12": eigensummary(phi, eps=1e-12),
            "eps_1e-6": eigensummary(phi, eps=1e-6),
        },
        "interpretation_hint": "A true continuous-attractor tangent mode should have high tangent cosine and near-zero real part; large positive modes with low tangent cosine indicate normal instability.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
