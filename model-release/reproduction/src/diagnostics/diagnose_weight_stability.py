try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import numpy as np

from plot_figure3_abcd import PROCESSED
from utils import (
    invert_spd_cholesky,
    lowrank_recurrent_drive,
    materialize_lowrank_weights,
    nearest_manifold_distance,
    optimized_recurrent_factors,
    simulate_rate_network_with_drive,
    softplus_inverse,
)


def constrained_rows_eq31(phi, beta, regularization, row_indices):
    """
    Construct sampled constrained rows using Appendix A2 Eq. 31.

    Eq. 41 requires the inverse of an N x N matrix. For diagnostics on Windows,
    this row-wise Eq. 31 form instead inverts only Ntheta x Ntheta kernels and
    checks the same constrained solution on a small sample of rows.
    """
    n_neurons, n_angles = phi.shape
    x_star = softplus_inverse(phi, beta=beta)
    base_kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    base_kernel += (regularization * n_angles / n_neurons) * np.eye(n_angles)
    rows = []
    for row_i in row_indices:
        kernel_i = base_kernel - np.outer(phi[row_i], phi[row_i]) / n_neurons
        kernel_inv_i = invert_spd_cholesky(kernel_i)
        row_dual = np.einsum("a,ab->b", x_star[row_i], kernel_inv_i, optimize=False)
        row = np.einsum("a,ja->j", row_dual, phi, optimize=False) / n_neurons
        row[row_i] = 0.0
        rows.append(row)
    return np.asarray(rows)


def summarize_drive(phi, target_current, drive):
    """
    Measure fixed-point residual on the target manifold.
    """
    recurrent = drive(phi.T)
    residual = np.linalg.norm(recurrent - target_current, axis=1) / np.sqrt(phi.shape[0])
    return float(np.median(residual)), float(np.max(residual))


def short_dynamics(phi, target_current, drive, beta, noise_std=0.05):
    """
    Simulate a one-second perturbation test from target-manifold points.
    """
    rng = np.random.default_rng(20260617)
    angle_idx = np.linspace(0, phi.shape[1], 12, endpoint=False, dtype=int)
    initial = target_current[angle_idx] + rng.normal(0.0, noise_std, size=(len(angle_idx), phi.shape[0]))
    _, trajectory = simulate_rate_network_with_drive(
        drive,
        initial,
        duration_s=1.0,
        tau_s=0.05,
        dt_s=0.001,
        record_every_s=1.0,
        inhibition_c=1.0,
        activation_beta=beta,
    )
    distance, _ = nearest_manifold_distance(trajectory[-1], target_current)
    return (
        float(np.median(distance)),
        float(np.max(distance)),
        float(np.min(trajectory[-1])),
        float(np.max(trajectory[-1])),
    )


def main():
    """
    Compare constrained and unconstrained A2 weight constructions.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"]
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])
    target_current = softplus_inverse(phi.T, beta=beta)

    print("Comparing A2 constrained-weight formulas...")
    from utils import optimized_recurrent_weights

    sherman_weights = optimized_recurrent_weights(
        phi,
        regularization=regularization,
        activation_beta=beta,
        dtype=np.float64,
    )
    row_indices = np.array([0, 1, 17, 42, 99, 250, 511, 900, 1200, phi.shape[0] - 1], dtype=int)
    row_indices = np.unique(np.clip(row_indices, 0, phi.shape[0] - 1))
    eq31_rows = constrained_rows_eq31(phi, beta, regularization, row_indices)
    diff = eq31_rows - sherman_weights[row_indices]
    print("  Eq31 sampled rows vs Sherman max abs:", float(np.max(np.abs(diff))))
    print("  Eq31 sampled rows vs Sherman RMS:", float(np.sqrt(np.mean(diff * diff))))
    print("  Sherman abs max/RMS:", float(np.max(np.abs(sherman_weights))), float(np.sqrt(np.mean(sherman_weights**2))))
    print("  Eq31 sampled rows abs max/RMS:", float(np.max(np.abs(eq31_rows))), float(np.sqrt(np.mean(eq31_rows**2))))

    for enforce_zero_diagonal in [False, True]:
        factor_a, factor_b, diagonal = optimized_recurrent_factors(
            phi,
            regularization=regularization,
            activation_beta=beta,
            enforce_zero_diagonal=enforce_zero_diagonal,
        )
        reconstructed = materialize_lowrank_weights(
            factor_a,
            factor_b,
            diagonal if enforce_zero_diagonal else None,
            dtype=np.float64,
        )
        print(
            "  low-rank materialized vs Sherman max abs:",
            float(np.max(np.abs(reconstructed - sherman_weights))),
        )
        drive = lowrank_recurrent_drive(
            factor_a,
            factor_b,
            diagonal if enforce_zero_diagonal else None,
        )
        residual = summarize_drive(phi, target_current, drive)
        dynamics = short_dynamics(phi, target_current, drive, beta)
        print("enforce_zero_diagonal:", enforce_zero_diagonal)
        print("  residual median/max:", residual)
        print("  one-second distance median/max/state-min/state-max:", dynamics)
        print(
            "  factor abs max:",
            float(max(np.max(np.abs(factor_a)), np.max(np.abs(factor_b)), np.max(np.abs(diagonal)))),
        )


if __name__ == "__main__":
    main()
