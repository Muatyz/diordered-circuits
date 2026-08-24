try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json
from pathlib import Path

import numpy as np

from plot_figure3_abcd import PROCESSED, REPRODUCTION_ROOT, build_figure3_matrices
from utils import optimized_recurrent_factors, softplus_inverse


REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_jacobian_lowrank_debug.json"


def finite_percentiles(values, q=(0, 1, 5, 50, 95, 99, 100)):
    """
    Return finite percentiles as JSON-friendly floats.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {f"p{qq:g}": float(vv) for qq, vv in zip(q, np.percentile(values, q))}


def softplus_derivative_from_rate(rate, beta):
    """
    Compute d softplus(x) / dx from the target firing rate softplus(x).
    """
    return 1.0 - np.exp(-float(beta) * np.asarray(rate, dtype=float))


def lowrank_recurrent_from_rates(rates, factor_a, factor_b, factor_diagonal=None):
    """
    Apply `A @ B.T - diag(diagonal)` to a batch of firing-rate vectors.
    """
    rates = np.asarray(rates, dtype=float)
    latent = np.einsum("tn,nk->tk", rates, factor_b, optimize=False)
    recurrent = np.einsum("tk,nk->tn", latent, factor_a, optimize=False)
    if factor_diagonal is not None:
        recurrent = recurrent - rates * factor_diagonal
    return recurrent


def jacobian_matvec_factory(factor_a, factor_b, factor_diagonal, rate_at_angle, beta, inhibition_c=1.0):
    """
    Build a Jacobian matvec for the data-derived rate dynamics at one angle.

    The Jacobian is
    `-I + (A B.T - diag(g)) diag(phi') - c/N 1 phi'.T`, where
    `g` is omitted when `factor_diagonal is None`.
    """
    factor_a = np.asarray(factor_a, dtype=float)
    factor_b = np.asarray(factor_b, dtype=float)
    derivative = softplus_derivative_from_rate(rate_at_angle, beta)
    n_neurons = len(derivative)
    ones = np.ones(n_neurons, dtype=float)
    if factor_diagonal is not None:
        factor_diagonal = np.asarray(factor_diagonal, dtype=float)

    def matvec(vector):
        vector = np.asarray(vector, dtype=float)
        presynaptic = derivative * vector
        latent = np.einsum("nk,n->k", factor_b, presynaptic, optimize=False)
        recurrent = np.einsum("nk,k->n", factor_a, latent, optimize=False)
        if factor_diagonal is not None:
            recurrent = recurrent - factor_diagonal * presynaptic
        recurrent = recurrent - (float(inhibition_c) / n_neurons) * np.sum(presynaptic) * ones
        return -vector + recurrent

    return matvec


def exact_unconstrained_lowrank_eigenvalues(factor_a, factor_b, rate_at_angle, beta, inhibition_c=1.0):
    """
    Compute exact non-bulk eigenvalues when the zero-diagonal correction is omitted.

    Without the `J_ii = 0` correction, the Jacobian is `-I + U V.T`; therefore
    all non-bulk eigenvalues are `-1 + eig(V.T U)`.
    """
    factor_a = np.asarray(factor_a, dtype=float)
    factor_b = np.asarray(factor_b, dtype=float)
    derivative = softplus_derivative_from_rate(rate_at_angle, beta)
    n_neurons = len(derivative)

    u = np.concatenate([factor_a, np.ones((n_neurons, 1))], axis=1)
    v = np.concatenate(
        [
            factor_b * derivative[:, None],
            -(float(inhibition_c) / n_neurons) * derivative[:, None],
        ],
        axis=1,
    )
    reduced = np.einsum("nr,ns->rs", v, u, optimize=False)
    return np.linalg.eigvals(reduced) - 1.0


def normalize_vector(vector):
    """
    Normalize a real vector, returning None for numerically zero inputs.
    """
    vector = np.asarray(vector, dtype=float)
    norm = float(np.sqrt(np.sum(vector * vector)))
    if not np.isfinite(norm) or norm <= 1e-14:
        return None
    return vector / norm


def arnoldi_ritz(matvec, n_neurons, initial_vector, n_iter=120, spectral_shift=2.0, reorthogonalize=True):
    """
    Estimate rightmost eigenvalues with shifted Arnoldi and small Ritz spectra.

    Applying Arnoldi to `J + spectral_shift * I` biases the Krylov basis toward
    eigenvalues with large real part, while Ritz values are shifted back before
    reporting. The only eigenvalue call is on the small Hessenberg matrix.
    """
    q0 = normalize_vector(initial_vector)
    if q0 is None:
        raise ValueError("initial_vector has zero norm")

    n_iter = int(min(n_iter, n_neurons - 1))
    q = np.zeros((n_neurons, n_iter + 1), dtype=float)
    h = np.zeros((n_iter + 1, n_iter), dtype=float)
    q[:, 0] = q0
    actual_iter = n_iter

    for j in range(n_iter):
        w = matvec(q[:, j]) + float(spectral_shift) * q[:, j]
        passes = 2 if reorthogonalize else 1
        for _ in range(passes):
            for i in range(j + 1):
                coeff = float(np.sum(q[:, i] * w))
                h[i, j] += coeff
                w = w - coeff * q[:, i]

        next_norm = float(np.sqrt(np.sum(w * w)))
        h[j + 1, j] = next_norm
        if next_norm <= 1e-12:
            actual_iter = j + 1
            break
        q[:, j + 1] = w / next_norm

    h_small = h[:actual_iter, :actual_iter]
    eigvals, eigvecs = np.linalg.eig(h_small)
    eigvals = eigvals - float(spectral_shift)

    residual_scale = h[actual_iter, actual_iter - 1] if actual_iter < h.shape[0] else 0.0
    eigvec_norm = np.sqrt(np.sum(np.abs(eigvecs) ** 2, axis=0))
    residuals = np.abs(residual_scale * eigvecs[-1, :]) / np.maximum(eigvec_norm, 1e-30)
    return eigvals, residuals, q[:, :actual_iter], eigvecs


def complex_cosine_with_real_vector(complex_vector, real_vector):
    """
    Compute absolute cosine between a complex Ritz vector and a real reference.
    """
    ref = normalize_vector(real_vector)
    if ref is None:
        return np.nan
    vec_norm = np.sqrt(np.sum(np.abs(complex_vector) ** 2))
    if not np.isfinite(vec_norm) or vec_norm <= 0:
        return np.nan
    return float(np.abs(np.vdot(complex_vector, ref)) / vec_norm)


def summarize_eigenvalues(eigvals, residuals=None, n_report=8):
    """
    Summarize eigenvalues sorted by real part.
    """
    order = np.argsort(eigvals.real)[::-1]
    rows = []
    for rank, eig_i in enumerate(order[:n_report]):
        item = {
            "rank": int(rank),
            "real": float(eigvals[eig_i].real),
            "imag": float(eigvals[eig_i].imag),
        }
        if residuals is not None:
            item["ritz_residual"] = float(residuals[eig_i])
        rows.append(item)
    return rows


def summarize_constrained_angle(
    factor_a,
    factor_b,
    factor_diagonal,
    phi,
    x_star,
    angle_idx,
    beta,
    inhibition_c,
    n_iter,
    spectral_shift,
    seeds,
):
    """
    Run several shifted-Arnoldi starts for one constrained Jacobian.
    """
    matvec = jacobian_matvec_factory(
        factor_a,
        factor_b,
        factor_diagonal,
        phi[:, angle_idx],
        beta=beta,
        inhibition_c=inhibition_c,
    )
    n_neurons = phi.shape[0]
    tangent = x_star[:, (angle_idx + 1) % phi.shape[1]] - x_star[:, (angle_idx - 1) % phi.shape[1]]
    starts = {
        "tangent": tangent,
        "uniform": np.ones(n_neurons, dtype=float),
    }
    rng = np.random.default_rng(20260617 + int(angle_idx))
    for seed_i in range(seeds):
        starts[f"random_{seed_i}"] = rng.normal(size=n_neurons)

    runs = {}
    best_real = -np.inf
    best_item = None
    for name, initial in starts.items():
        eigvals, residuals, basis, eigvecs = arnoldi_ritz(
            matvec,
            n_neurons,
            initial,
            n_iter=n_iter,
            spectral_shift=spectral_shift,
        )
        order = np.argsort(eigvals.real)[::-1]
        run_rows = []
        for rank, eig_i in enumerate(order[:8]):
            ritz_vector = np.einsum("ij,j->i", basis, eigvecs[:, eig_i], optimize=False)
            item = {
                "rank": int(rank),
                "real": float(eigvals[eig_i].real),
                "imag": float(eigvals[eig_i].imag),
                "ritz_residual": float(residuals[eig_i]),
                "cosine_with_current_tangent": complex_cosine_with_real_vector(ritz_vector, tangent),
                "cosine_with_uniform_current": complex_cosine_with_real_vector(ritz_vector, np.ones(n_neurons)),
            }
            run_rows.append(item)
            if item["real"] > best_real:
                best_real = item["real"]
                best_item = {"start": name, **item}
        runs[name] = run_rows

    return {
        "best_by_real_part": best_item,
        "runs": runs,
    }


def target_flow_residual(factor_a, factor_b, factor_diagonal, phi, x_star, beta, inhibition_c):
    """
    Evaluate autonomous flow residual along the target manifold.
    """
    rates = phi.T
    states = x_star.T
    recurrent = lowrank_recurrent_from_rates(rates, factor_a, factor_b, factor_diagonal)
    mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
    dx = -states + recurrent - float(inhibition_c) * mean_error
    return np.sqrt(np.mean(dx * dx, axis=1))


def main():
    """
    Diagnose tuning, optimized weights, and Jacobian spectra without dense eigensolvers.
    """
    matrix_path = build_figure3_matrices()
    data = np.load(matrix_path)
    sorted_order = data["sorted_order"]
    phi = data["doubly_normalized_tuning"].astype(float)[sorted_order]
    beta = float(data["activation_beta"])
    regularization = float(data["regularization"])
    inhibition_c = 1.0

    factor_a = data["factor_a"].astype(float)[sorted_order]
    factor_b = data["factor_b"].astype(float)[sorted_order]
    factor_diagonal = data["factor_diagonal"].astype(float)[sorted_order]
    x_star = softplus_inverse(phi, beta=beta)

    unconstrained_a, unconstrained_b, _ = optimized_recurrent_factors(
        phi,
        regularization=regularization,
        activation_beta=beta,
        enforce_zero_diagonal=False,
        dtype=np.float64,
    )

    sample_bins = [0, 25, 50, 75]
    flow_residual = target_flow_residual(
        factor_a,
        factor_b,
        factor_diagonal,
        phi,
        x_star,
        beta=beta,
        inhibition_c=inhibition_c,
    )

    report = {
        "status": "debug_no_dense_lapack_eigensolver",
        "matrix_path": str(Path(matrix_path).resolve()),
        "n_neurons": int(phi.shape[0]),
        "n_angles": int(phi.shape[1]),
        "activation_beta": beta,
        "regularization": regularization,
        "inhibition_c": inhibition_c,
        "sample_bins": sample_bins,
        "arnoldi": {
            "n_iter": 100,
            "spectral_shift": 2.0,
            "random_starts_per_angle": 4,
            "note": "Ritz residuals are for the shifted Arnoldi approximation; smaller is more trustworthy.",
        },
        "tuning": {
            "phi_percentiles": finite_percentiles(phi),
            "row_mean_error_abs": finite_percentiles(np.abs(np.mean(phi, axis=1) - 1.0), q=(0, 50, 95, 99, 100)),
            "angle_mean_error_abs": finite_percentiles(np.abs(np.mean(phi, axis=0) - 1.0), q=(0, 50, 95, 99, 100)),
            "target_current_percentiles": finite_percentiles(x_star),
        },
        "weights": {
            "factor_a_abs_percentiles": finite_percentiles(np.abs(factor_a)),
            "factor_b_abs_percentiles": finite_percentiles(np.abs(factor_b)),
            "factor_diagonal_percentiles": finite_percentiles(factor_diagonal),
            "zero_diag_correction_times_gain_percentiles_by_angle": {},
            "target_flow_residual_l2_div_sqrt_n": finite_percentiles(flow_residual, q=(0, 1, 50, 95, 99, 100)),
        },
        "jacobian": {},
    }

    for angle_idx in sample_bins:
        derivative = softplus_derivative_from_rate(phi[:, angle_idx], beta)
        report["weights"]["zero_diag_correction_times_gain_percentiles_by_angle"][str(angle_idx)] = finite_percentiles(
            factor_diagonal * derivative,
            q=(0, 1, 5, 50, 95, 99, 100),
        )

        unconstrained_eigs = exact_unconstrained_lowrank_eigenvalues(
            unconstrained_a,
            unconstrained_b,
            phi[:, angle_idx],
            beta=beta,
            inhibition_c=inhibition_c,
        )
        constrained = summarize_constrained_angle(
            factor_a,
            factor_b,
            factor_diagonal,
            phi,
            x_star,
            angle_idx,
            beta=beta,
            inhibition_c=inhibition_c,
            n_iter=report["arnoldi"]["n_iter"],
            spectral_shift=report["arnoldi"]["spectral_shift"],
            seeds=report["arnoldi"]["random_starts_per_angle"],
        )
        report["jacobian"][str(angle_idx)] = {
            "unconstrained_exact": summarize_eigenvalues(unconstrained_eigs, n_report=8),
            "constrained_arnoldi": constrained,
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")
    for angle_idx in sample_bins:
        best = report["jacobian"][str(angle_idx)]["constrained_arnoldi"]["best_by_real_part"]
        unconstrained = report["jacobian"][str(angle_idx)]["unconstrained_exact"][0]
        print(
            f"bin {angle_idx}: unconstrained max_real={unconstrained['real']:.6g}; "
            f"constrained Ritz best={best['real']:.6g}+{best['imag']:.6g}j "
            f"resid={best['ritz_residual']:.3g} start={best['start']}"
        )


if __name__ == "__main__":
    main()
