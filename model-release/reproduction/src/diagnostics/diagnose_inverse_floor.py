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
    nearest_manifold_distance,
    simulate_rate_network_with_drive,
    softplus,
)


def softplus_inverse_with_floor(rate, beta, eps):
    """
    Apply the Setting-1 softplus inverse with an explicit numerical rate floor.
    """
    rate = np.maximum(np.asarray(rate, dtype=float), float(eps))
    scaled = beta * rate
    return np.where(scaled > 20.0, rate, np.log(np.expm1(scaled)) / beta)


def factors_from_target_current(phi, x_star, regularization, enforce_zero_diagonal=True):
    """
    Build Appendix-A2 low-rank factors for a supplied input-current target.
    """
    n_neurons, n_angles = phi.shape
    kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    kernel += (regularization * n_angles / n_neurons) * np.eye(n_angles)
    kernel_inv = invert_spd_cholesky(kernel)
    a = np.einsum("ia,ab->ib", x_star, kernel_inv, optimize=False)
    inv_phi = np.einsum("ab,ib->ai", kernel_inv, phi, optimize=False)
    if enforce_zero_diagonal:
        numerator = np.sum(x_star * inv_phi.T, axis=1)
        leverage = np.sum(phi * inv_phi.T, axis=1) / n_neurons
        denom = np.maximum(1.0 - leverage, 1e-12)
        a = a + (numerator / (n_neurons * denom))[:, None] * inv_phi.T
    b = phi / n_neurons
    diagonal = np.sum(a * b, axis=1)
    return a, b, diagonal


def perturbation_test(phi, x_star, drive, beta, duration_s=1.0):
    """
    Simulate from noisy target-manifold initial conditions.
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
    finite = np.isfinite(final).all()
    if finite:
        distance, _ = nearest_manifold_distance(final, target)
        return finite, float(np.median(distance)), float(np.max(distance)), float(np.max(np.abs(final)))
    return finite, np.nan, np.nan, np.inf


def main():
    """
    Sweep numerical softplus-inverse floors without changing the firing-rate target.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"]
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])

    for eps in [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
        a, b, diagonal = factors_from_target_current(phi, x_star, regularization)
        drive = lowrank_recurrent_drive(a, b, diagonal)
        residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
        inverse_mismatch = np.max(np.abs(softplus(x_star, beta=beta) - np.maximum(phi, eps)))
        finite, dist_med, dist_max, max_abs_state = perturbation_test(phi, x_star, drive, beta, duration_s=1.0)
        print(
            f"eps={eps:g} "
            f"x_min={np.min(x_star):.3g} x_p1={np.percentile(x_star, 1):.3g} "
            f"resid_med={np.median(residual):.3g} resid_max={np.max(residual):.3g} "
            f"inverse_mismatch={inverse_mismatch:.3g} "
            f"finite_1s={finite} dist_med={dist_med:.3g} dist_max={dist_max:.3g} "
            f"max_abs_state={max_abs_state:.3g}"
        )

    eps = 1e-6
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    a, b, diagonal = factors_from_target_current(phi, x_star, regularization)
    drive = lowrank_recurrent_drive(a, b, diagonal)
    finite, dist_med, dist_max, max_abs_state = perturbation_test(phi, x_star, drive, beta, duration_s=30.0)
    print(
        f"long_test eps={eps:g} finite_30s={finite} "
        f"dist_med={dist_med:.3g} dist_max={dist_max:.3g} max_abs_state={max_abs_state:.3g}"
    )


if __name__ == "__main__":
    main()
