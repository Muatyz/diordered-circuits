try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import numpy as np

from plot_figure3_abcd import PROCESSED
from utils import (
    lowrank_recurrent_drive,
    nearest_manifold_distance,
    optimized_recurrent_factors,
    optimized_recurrent_weights,
    simulate_rate_network_with_drive,
    softplus,
    softplus_inverse,
)


def percentile_text(values, q=(0, 1, 50, 95, 99, 100)):
    """
    格式化有限值分位数，方便在终端比较数值范围。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "no finite values"
    return ", ".join(f"p{qq:g}={vv:.6g}" for qq, vv in zip(q, np.percentile(values, q)))


def dense_drive(weights):
    """
    根据 dense J 构造 recurrent drive，约定 J_ij 是 j 到 i 的权重。
    """
    weights_t = np.asarray(weights, dtype=float).T

    def drive(rates):
        return np.einsum("tn,nj->tj", np.asarray(rates, dtype=float), weights_t, optimize=False)

    return drive


def flow_residual(drive, x_star, beta, inhibition_c=1.0):
    """
    计算 B4.2 动力学右端在目标流形上的残差范数。
    """
    rates = softplus(x_star, beta=beta)
    recurrent = np.asarray(drive(rates), dtype=float)
    mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
    dx = -x_star + recurrent - inhibition_c * mean_error
    return np.linalg.norm(dx, axis=1) / np.sqrt(x_star.shape[1])


def flow_rhs_from_weights(weights, states, beta=2.0, inhibition_c=1.0):
    """
    Evaluate tau * dx/dt for the finite-N B4.2 dynamics.
    """
    rates = softplus(states, beta=beta)
    recurrent = dense_drive(weights)(rates)
    mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
    return -states + recurrent - inhibition_c * mean_error


def softplus_derivative_from_current(x, beta=2.0):
    """
    Evaluate d softplus(x) / dx in a stable form.
    """
    z = beta * np.asarray(x, dtype=float)
    return np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def jacobian_matrix(weights, phi_at_angle, beta=2.0, inhibition_c=1.0):
    """
    Build the current-space Jacobian for the finite-N B4.2 dynamics.
    """
    n_neurons = weights.shape[0]
    derivative = 1.0 - np.exp(-beta * phi_at_angle)
    jacobian = -np.eye(n_neurons)
    jacobian += weights * derivative[None, :]
    jacobian -= (inhibition_c / n_neurons) * derivative[None, :]
    return jacobian


def max_real_jacobian(weights, phi_at_angle, beta=2.0, inhibition_c=1.0):
    """
    计算一个目标点处线性化 Jacobian 的最大实部。
    """
    jacobian = jacobian_matrix(weights, phi_at_angle, beta=beta, inhibition_c=inhibition_c)
    eigvals = np.linalg.eigvals(jacobian)
    return float(np.max(eigvals.real))


def activation_consistency_check(x_star, phi, beta=2.0):
    """
    Check softplus inverse round-trip and derivative consistency.
    """
    round_trip = softplus(x_star.T, beta=beta) - phi
    h = 1e-5
    analytic = softplus_derivative_from_current(x_star, beta=beta)
    finite_difference = (softplus(x_star + h, beta=beta) - softplus(x_star - h, beta=beta)) / (2.0 * h)
    derivative_error = analytic - finite_difference
    return {
        "round_trip_abs_max": float(np.max(np.abs(round_trip))),
        "round_trip_rms": float(np.sqrt(np.mean(round_trip * round_trip))),
        "derivative_abs_max": float(np.max(np.abs(derivative_error))),
        "derivative_rms": float(np.sqrt(np.mean(derivative_error * derivative_error))),
    }


def jacobian_finite_difference_check(weights, x_star, phi, beta=2.0, inhibition_c=1.0, sample_bins=(0, 25, 50, 75)):
    """
    Compare analytic Jacobian actions against finite differences of the flow.
    """
    rng = np.random.default_rng(20260619)
    rows = []
    eps = 1e-6
    for bin_idx in sample_bins:
        x0 = x_star[bin_idx]
        jacobian = jacobian_matrix(weights, phi[:, bin_idx], beta=beta, inhibition_c=inhibition_c)
        flow_base = flow_rhs_from_weights(weights, x0[None, :], beta=beta, inhibition_c=inhibition_c)[0]
        for direction_idx in range(3):
            direction = rng.normal(size=x0.shape)
            direction /= np.linalg.norm(direction)
            analytic = np.einsum("ij,j->i", jacobian, direction, optimize=False)
            flow_plus = flow_rhs_from_weights(
                weights,
                (x0 + eps * direction)[None, :],
                beta=beta,
                inhibition_c=inhibition_c,
            )[0]
            finite_difference = (flow_plus - flow_base) / eps
            error = analytic - finite_difference
            rows.append(
                {
                    "bin": int(bin_idx),
                    "direction": int(direction_idx),
                    "analytic_norm": float(np.linalg.norm(analytic)),
                    "finite_difference_norm": float(np.linalg.norm(finite_difference)),
                    "abs_error": float(np.linalg.norm(error)),
                    "relative_error": float(np.linalg.norm(error) / max(np.linalg.norm(finite_difference), 1e-30)),
                }
            )
    return rows


def tangent_overlap_spectrum(weights, x_star, phi, beta=2.0, inhibition_c=1.0, sample_bins=(0, 25, 50, 75)):
    """
    Report leading eigenvalues and alignment with the target-manifold tangent.
    """
    tangent = 0.5 * (np.roll(x_star, -1, axis=0) - np.roll(x_star, 1, axis=0))
    rows = []
    for bin_idx in sample_bins:
        jacobian = jacobian_matrix(weights, phi[:, bin_idx], beta=beta, inhibition_c=inhibition_c)
        eigvals, eigvecs = np.linalg.eig(jacobian)
        order = np.argsort(eigvals.real)[::-1]
        reference = tangent[bin_idx]
        reference_norm = np.linalg.norm(reference)
        for rank, eig_idx in enumerate(order[:10]):
            vector = eigvecs[:, eig_idx]
            denom = np.linalg.norm(vector) * reference_norm
            overlap = float(np.abs(np.vdot(vector, reference)) / denom) if denom > 0 else np.nan
            rows.append(
                {
                    "bin": int(bin_idx),
                    "rank": int(rank),
                    "real": float(eigvals[eig_idx].real),
                    "imag": float(eigvals[eig_idx].imag),
                    "tangent_overlap": overlap,
                }
            )
    return rows


def short_dynamics(drive, x_star, beta, tau_s, dt_s, noise_std=0.05, seed=20260618):
    """
    从目标流形附近做短时间扰动仿真，检查是否马上发散。
    """
    rng = np.random.default_rng(seed)
    angle_idx = np.linspace(0, len(x_star), 12, endpoint=False, dtype=int)
    initial = x_star[angle_idx] + rng.normal(0.0, noise_std, size=(len(angle_idx), x_star.shape[1]))
    _, trajectory = simulate_rate_network_with_drive(
        drive,
        initial,
        tau_s=tau_s,
        dt_s=dt_s,
        duration_s=1.0,
        inhibition_c=1.0,
        activation_beta=beta,
        record_every_s=1.0,
    )
    distance, _ = nearest_manifold_distance(trajectory[-1], x_star)
    return {
        "finite": bool(np.isfinite(trajectory).all()),
        "distance": distance,
        "state_min": float(np.nanmin(trajectory[-1])),
        "state_max": float(np.nanmax(trajectory[-1])),
    }


def local_perturbation_scale_test(drive, x_star, beta, noise_stds=(1e-5, 1e-4, 1e-3, 1e-2), duration_s=0.2):
    """
    Simulate small perturbations and measure nearest-manifold distances.
    """
    rows = []
    rng = np.random.default_rng(20260620)
    angle_idx = np.linspace(0, len(x_star), 12, endpoint=False, dtype=int)
    target = x_star[angle_idx]
    for noise_std in noise_stds:
        initial = target + rng.normal(0.0, noise_std, size=target.shape)
        _, trajectory = simulate_rate_network_with_drive(
            drive,
            initial,
            tau_s=0.05,
            dt_s=0.001,
            duration_s=duration_s,
            inhibition_c=1.0,
            activation_beta=beta,
            record_every_s=duration_s,
        )
        initial_distance, _ = nearest_manifold_distance(trajectory[0], x_star)
        final_distance, _ = nearest_manifold_distance(trajectory[-1], x_star)
        rows.append(
            {
                "noise_std": float(noise_std),
                "finite": bool(np.isfinite(trajectory).all()),
                "initial_median": float(np.median(initial_distance)),
                "initial_max": float(np.max(initial_distance)),
                "final_median": float(np.median(final_distance)),
                "final_max": float(np.max(final_distance)),
                "growth_median": float(np.median(final_distance) / max(np.median(initial_distance), 1e-30)),
                "growth_max": float(np.max(final_distance) / max(np.max(initial_distance), 1e-30)),
            }
        )
    return rows


def main():
    """
    检查 A2 权重构造和 B4 动力学实现之间的一致性。
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi_unsorted = data["doubly_normalized_tuning"].astype(float)
    sorted_order = data["sorted_order"]
    phi = phi_unsorted[sorted_order]
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])
    x_star = softplus_inverse(phi.T, beta=beta)

    print("Target rate phi:", phi.shape, percentile_text(phi))
    print("Target current x_star:", x_star.shape, percentile_text(x_star))
    print("Population mean over neurons:", percentile_text(np.mean(phi, axis=0)))
    print("Neuron mean over angles:", percentile_text(np.mean(phi, axis=1)))
    print("Activation round-trip/derivative check:", activation_consistency_check(x_star, phi, beta=beta))

    dense = optimized_recurrent_weights(
        phi,
        regularization=regularization,
        activation_beta=beta,
        dtype=np.float64,
    )
    saved_sorted = data["weights_sorted"].astype(float)
    print("Saved sorted dense vs rebuilt dense max abs:", float(np.max(np.abs(saved_sorted - dense))))
    print("Dense J abs:", percentile_text(np.abs(dense)))
    print("Dense J diag abs max:", float(np.max(np.abs(np.diag(dense)))))

    factor_a, factor_b, diagonal = optimized_recurrent_factors(
        phi,
        regularization=regularization,
        activation_beta=beta,
        enforce_zero_diagonal=True,
    )
    reconstructed = np.einsum("ik,jk->ij", factor_a, factor_b, optimize=False)
    reconstructed_zero_diag = reconstructed.copy()
    np.fill_diagonal(reconstructed_zero_diag, 0.0)
    print("Factor A abs:", percentile_text(np.abs(factor_a)))
    print("Factor diagonal before subtraction:", percentile_text(diagonal))
    print(
        "Low-rank+diag vs dense max/RMS:",
        float(np.max(np.abs(reconstructed_zero_diag - dense))),
        float(np.sqrt(np.mean((reconstructed_zero_diag - dense) ** 2))),
    )

    rng = np.random.default_rng(123)
    test_rates = softplus(x_star[[0, 17, 42]], beta=beta)
    test_rates = np.vstack([test_rates, rng.lognormal(mean=0.0, sigma=0.2, size=(3, phi.shape[0]))])
    lowrank_drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    dense_out = dense_drive(dense)(test_rates)
    lowrank_out = lowrank_drive(test_rates)
    print(
        "Drive dense vs lowrank max/RMS:",
        float(np.max(np.abs(dense_out - lowrank_out))),
        float(np.sqrt(np.mean((dense_out - lowrank_out) ** 2))),
    )

    residual_dense = flow_residual(dense_drive(dense), x_star, beta)
    residual_lowrank = flow_residual(lowrank_drive, x_star, beta)
    print("Flow residual dense:", percentile_text(residual_dense))
    print("Flow residual lowrank:", percentile_text(residual_lowrank))

    sample_bins = [0, 25, 50, 75]
    finite_difference_rows = jacobian_finite_difference_check(
        dense,
        x_star,
        phi,
        beta=beta,
        inhibition_c=1.0,
        sample_bins=sample_bins,
    )
    finite_difference_rel = [row["relative_error"] for row in finite_difference_rows]
    print("Jacobian finite-difference relative error:", percentile_text(finite_difference_rel))
    print("Jacobian dense eig skipped: NumPy/SciPy LAPACK hard-crashes on this Windows stack")
    print("Local perturbation scale test (0.2s):", local_perturbation_scale_test(lowrank_drive, x_star, beta))

    for tau_s, dt_s in [(0.05, 0.001), (1.0, 0.001), (0.05, 0.0002)]:
        result = short_dynamics(lowrank_drive, x_star, beta, tau_s=tau_s, dt_s=dt_s)
        print(
            f"Short dynamics tau={tau_s:g}, dt={dt_s:g}: "
            f"finite={result['finite']}, distance={percentile_text(result['distance'], q=(50, 95, 100))}, "
            f"state_min={result['state_min']:.6g}, state_max={result['state_max']:.6g}"
        )


if __name__ == "__main__":
    main()
