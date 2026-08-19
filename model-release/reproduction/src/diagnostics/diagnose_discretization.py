try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import numpy as np

from plot_figure3_abcd import load_population_tuning
from utils import (
    invert_spd_cholesky,
    lowrank_recurrent_drive,
    materialize_lowrank_weights,
    nearest_manifold_distance,
    simulate_rate_network_with_drive,
    sinkhorn_normalize,
    softplus,
)


def periodic_linear_resample(values, n_samples):
    """
    Resample circular tuning curves with nonnegative periodic linear interpolation.
    """
    values = np.asarray(values, dtype=float)
    n_old = values.shape[1]
    old_x = np.arange(n_old + 1, dtype=float)
    new_x = np.linspace(0.0, n_old, int(n_samples), endpoint=False)
    out = np.empty((values.shape[0], int(n_samples)), dtype=float)
    for i, row in enumerate(values):
        closed = np.r_[row, row[0]]
        out[i] = np.interp(new_x, old_x, closed)
    return np.maximum(out, 0.0)


def softplus_inverse_with_floor(rate, beta, eps):
    """
    Apply softplus inverse with a numerical rate floor.
    """
    rate = np.maximum(np.asarray(rate, dtype=float), float(eps))
    scaled = beta * rate
    return np.where(scaled > 20.0, rate, np.log(np.expm1(scaled)) / beta)


def factors_from_targets(phi, x_star, regularization):
    """
    Build A2 low-rank factors for supplied target rates and currents.
    """
    n_neurons, n_angles = phi.shape
    kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    kernel += (regularization * n_angles / n_neurons) * np.eye(n_angles)
    kernel_inv = invert_spd_cholesky(kernel)
    a = np.einsum("ia,ab->ib", x_star, kernel_inv, optimize=False)
    inv_phi = np.einsum("ab,ib->ai", kernel_inv, phi, optimize=False)
    numerator = np.sum(x_star * inv_phi.T, axis=1)
    leverage = np.sum(phi * inv_phi.T, axis=1) / n_neurons
    denom = np.maximum(1.0 - leverage, 1e-12)
    a = a + (numerator / (n_neurons * denom))[:, None] * inv_phi.T
    b = phi / n_neurons
    diagonal = np.sum(a * b, axis=1)
    return a, b, diagonal


def dense_weights_from_factors(a, b, diagonal):
    """
    Materialize dense recurrent weights from A2 factors.
    """
    return materialize_lowrank_weights(a, b, diagonal, dtype=np.float64)


def max_real_jacobian(weights, phi_at_angle, beta):
    """
    Estimate the largest-real-part Jacobian eigenvalue at one target point.

    The full 1533 x 1533 LAPACK eigensolver crashes in the local Windows
    environment. A small shifted-Arnoldi problem gives a diagnostic Ritz value
    while using only Jacobian-vector products.
    """
    n_neurons = weights.shape[0]
    derivative = 1.0 - np.exp(-beta * phi_at_angle)
    ones = np.ones(n_neurons, dtype=float)

    def matvec(vector):
        presynaptic = derivative * vector
        recurrent = np.einsum("ij,j->i", weights, presynaptic, optimize=False)
        recurrent -= (1.0 / n_neurons) * float(np.sum(presynaptic)) * ones
        return -vector + recurrent

    n_iter = min(80, n_neurons - 1)
    spectral_shift = 2.0
    rng = np.random.default_rng(20260617)
    starts = [np.ones(n_neurons, dtype=float), rng.normal(size=n_neurons)]
    best = -np.inf
    for start in starts:
        q = np.zeros((n_neurons, n_iter + 1), dtype=float)
        h = np.zeros((n_iter + 1, n_iter), dtype=float)
        q[:, 0] = start / max(np.sqrt(np.sum(start * start)), 1e-12)
        actual_iter = n_iter
        for j in range(n_iter):
            w = matvec(q[:, j]) + spectral_shift * q[:, j]
            for _ in range(2):
                for i in range(j + 1):
                    coeff = float(np.sum(q[:, i] * w))
                    h[i, j] += coeff
                    w -= coeff * q[:, i]
            next_norm = float(np.sqrt(np.sum(w * w)))
            h[j + 1, j] = next_norm
            if next_norm <= 1e-12:
                actual_iter = j + 1
                break
            q[:, j + 1] = w / next_norm
        eigvals = np.linalg.eigvals(h[:actual_iter, :actual_iter]) - spectral_shift
        best = max(best, float(np.max(eigvals.real)))
    return best


def one_second_distance(phi, x_star, drive, beta):
    """
    Simulate a one-second noisy perturbation test.
    """
    rng = np.random.default_rng(20260617)
    angle_idx = np.linspace(0, phi.shape[1], 12, endpoint=False, dtype=int)
    target = x_star.T
    initial = target[angle_idx] + rng.normal(0.0, 0.05, size=(len(angle_idx), phi.shape[0]))
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
    final = trajectory[-1].astype(float)
    if not np.isfinite(final).all():
        return False, np.nan, np.nan, np.inf
    distance, _ = nearest_manifold_distance(final, target)
    return True, float(np.median(distance)), float(np.max(distance)), float(np.max(np.abs(final)))


def main():
    """
    Test whether denser angular discretization improves A2 stability.
    """
    tuning, _, _ = load_population_tuning()
    beta = 2.0
    regularization = 1e-6

    for n_angles in [100, 200, 500, 1000]:
        resampled = tuning if n_angles == tuning.shape[1] else periodic_linear_resample(tuning, n_angles)
        phi = sinkhorn_normalize(resampled)
        for eps in [1e-12, 1e-6]:
            x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
            a, b, diagonal = factors_from_targets(phi, x_star, regularization)
            drive = lowrank_recurrent_drive(a, b, diagonal)
            residual = np.linalg.norm(drive(phi.T) - x_star.T, axis=1) / np.sqrt(phi.shape[0])
            finite, dist_med, dist_max, max_abs = one_second_distance(phi, x_star, drive, beta)
            weights = dense_weights_from_factors(a, b, diagonal)
            sample_bins = np.linspace(0, n_angles, 4, endpoint=False, dtype=int)
            max_reals = [max_real_jacobian(weights, phi[:, idx], beta) for idx in sample_bins]
            print(
                f"n_angles={n_angles} eps={eps:g} "
                f"phi_min={np.min(phi):.3g} x_min={np.min(x_star):.3g} "
                f"resid_med={np.median(residual):.3g} resid_max={np.max(residual):.3g} "
                f"max_real={max_reals} finite_1s={finite} "
                f"dist_med={dist_med:.3g} dist_max={dist_max:.3g} max_abs={max_abs:.3g}"
            )


if __name__ == "__main__":
    main()
