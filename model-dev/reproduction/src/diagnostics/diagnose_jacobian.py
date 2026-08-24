try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import numpy as np

from plot_figure3_abcd import PROCESSED
from utils import optimized_recurrent_weights, softplus_inverse


def softplus_inverse_with_floor(rate, beta, eps):
    """
    Apply the Setting-1 softplus inverse with an explicit numerical rate floor.
    """
    rate = np.maximum(np.asarray(rate, dtype=float), float(eps))
    scaled = beta * rate
    return np.where(scaled > 20.0, rate, np.log(np.expm1(scaled)) / beta)


def weights_for_inverse_floor(phi, beta, regularization, eps):
    """
    Construct dense A2 weights for a supplied inverse floor.
    """
    n_neurons, n_angles = phi.shape
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    kernel = (phi.T @ phi) / n_neurons
    kernel += (regularization * n_angles / n_neurons) * np.eye(n_angles)
    kernel_inv = np.linalg.inv(kernel)
    presynaptic = phi.T / n_neurons

    weights = np.empty((n_neurons, n_neurons), dtype=np.float64)
    inv_phi = kernel_inv @ phi.T
    for i in range(n_neurons):
        u = phi[i] / np.sqrt(n_neurons)
        v = inv_phi[:, i] / np.sqrt(n_neurons)
        denom = max(1.0 - float(u @ v), 1e-12)
        row_dual = x_star[i] @ kernel_inv
        row_dual += (float(x_star[i] @ v) / denom) * v
        weights[i] = row_dual @ presynaptic
    np.fill_diagonal(weights, 0.0)
    return weights


def max_real_jacobian(weights, phi_at_angle, inhibition_c=1.0, beta=2.0):
    """
    Compute the maximum real Jacobian eigenvalue at one target point.
    """
    n_neurons = weights.shape[0]
    derivative = 1.0 - np.exp(-beta * phi_at_angle)
    jacobian = -np.eye(n_neurons)
    jacobian += weights * derivative[None, :]
    jacobian -= (inhibition_c / n_neurons) * derivative[None, :]
    eigvals = np.linalg.eigvals(jacobian)
    return float(np.max(eigvals.real))


def main():
    """
    Diagnose normal stability around the target manifold.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"]
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])
    sample_bins = [0, 25, 50, 75]

    for eps in [1e-12, 1e-8, 1e-6, 1e-4]:
        weights = weights_for_inverse_floor(phi, beta, regularization, eps)
        max_reals = [max_real_jacobian(weights, phi[:, idx], beta=beta) for idx in sample_bins]
        print(
            f"eps={eps:g} weight_abs_max={np.max(np.abs(weights)):.6g} "
            f"weight_rms={np.sqrt(np.mean(weights * weights)):.6g} "
            f"max_real={max_reals}"
        )

    stored_weights = optimized_recurrent_weights(
        phi,
        regularization=regularization,
        activation_beta=beta,
        dtype=np.float64,
    )
    stored_max_reals = [max_real_jacobian(stored_weights, phi[:, idx], beta=beta) for idx in sample_bins]
    print(f"stored_softplus_inverse_default max_real={stored_max_reals}")


if __name__ == "__main__":
    main()
