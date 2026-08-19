try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import numpy as np

from plot_figure3_abcd import PROCESSED
from utils import lowrank_recurrent_drive, nearest_manifold_distance, simulate_rate_network_with_drive

from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from diagnose_discretization import dense_weights_from_factors, max_real_jacobian


def perturbation_test(phi, x_star, drive, beta, duration_s):
    """
    Simulate noisy target perturbations for a given duration.
    """
    rng = np.random.default_rng(20260617)
    angle_idx = np.linspace(0, phi.shape[1], 12, endpoint=False, dtype=int)
    target = x_star.T
    initial = target[angle_idx] + rng.normal(0.0, 0.05, size=(len(angle_idx), phi.shape[0]))
    _, trajectory = simulate_rate_network_with_drive(
        drive,
        initial,
        duration_s=duration_s,
        tau_s=0.05,
        dt_s=0.001,
        record_every_s=duration_s,
        inhibition_c=1.0,
        activation_beta=beta,
    )
    final = trajectory[-1].astype(float)
    if not np.isfinite(final).all():
        return False, np.nan, np.nan, np.inf
    distance, _ = nearest_manifold_distance(final, target)
    return True, float(np.median(distance)), float(np.max(distance)), float(np.max(np.abs(final)))


def main():
    """
    Sweep the paper's regularization parameter for stability diagnostics.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"]
    beta = float(data["activation_beta"])
    eps = 1e-6
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)

    for regularization in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        a, b, diagonal = factors_from_target_current(phi, x_star, regularization)
        drive = lowrank_recurrent_drive(a, b, diagonal)
        residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
        weights = dense_weights_from_factors(a, b, diagonal)
        sample_bins = [0, 25, 50, 75]
        max_reals = [max_real_jacobian(weights, phi[:, idx], beta) for idx in sample_bins]
        finite_1s, dist_med_1s, dist_max_1s, max_abs_1s = perturbation_test(phi, x_star, drive, beta, 1.0)
        finite_30s, dist_med_30s, dist_max_30s, max_abs_30s = perturbation_test(phi, x_star, drive, beta, 30.0)
        print(
            f"lambda={regularization:g} residual_med={np.median(residual):.3g} residual_max={np.max(residual):.3g} "
            f"max_real={max_reals} finite_1s={finite_1s} dist1_med={dist_med_1s:.3g} dist1_max={dist_max_1s:.3g} "
            f"maxabs1={max_abs_1s:.3g} finite_30s={finite_30s} dist30_med={dist_med_30s:.3g} "
            f"dist30_max={dist_max_30s:.3g} maxabs30={max_abs_30s:.3g}"
        )


if __name__ == "__main__":
    main()
