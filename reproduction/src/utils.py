import warnings
import time

import numpy as np


def circular_smooth(values, sigma_bins):
    """
    Smooth circular tuning curves with a Gaussian kernel while handling NaNs.

    The last axis is treated as angular bins on a closed circle. Missing bins
    (NaN) are ignored by the kernel and the weights are renormalized locally
    so that finite neighbors contribute correctly. `sigma_bins <= 0` returns a
    float copy without smoothing.
    """
    values = np.asarray(values, dtype=float)
    if sigma_bins <= 0:
        return values.astype(float, copy=True)

    n_bins = values.shape[-1]
    offsets = np.arange(n_bins)
    circular_distance = np.minimum(offsets, n_bins - offsets)
    kernel = np.exp(-0.5 * (circular_distance / float(sigma_bins)) ** 2)
    kernel = kernel / kernel.sum()

    out = np.zeros_like(values, dtype=float)
    weight_sum = np.zeros_like(values, dtype=float)
    for offset, weight in enumerate(kernel):
        rolled = np.roll(values, offset, axis=-1)
        finite_mask = np.isfinite(rolled).astype(float)
        rolled_safe = np.where(np.isfinite(rolled), rolled, 0.0)
        out += weight * rolled_safe
        weight_sum += weight * finite_mask

    # Normalize where we have any finite contribution; otherwise keep NaN.
    result = np.full_like(out, np.nan)
    mask = weight_sum > 0
    result[mask] = out[mask] / weight_sum[mask]
    return result


def gaussian_process_place_fields_1d(
    n_cells,
    n_positions,
    correlation_length_bins,
    threshold=1.0,
    seed=None,
    boundary_mode="reflect",
):
    """
    Sample thresholded 1D Gaussian-process place-cell firing maps.

    Each cell starts from independent unit white noise, is Gaussian-filtered
    along position, and is rescaled to unit RMS before thresholding and
    rectification. This follows the simulation procedure released with
    Mainali et al. (2025). `correlation_length_bins` is the Gaussian-filter
    sigma measured in spatial bins.
    """
    if n_cells <= 0 or n_positions <= 0:
        raise ValueError("n_cells and n_positions must be positive")
    if correlation_length_bins <= 0:
        raise ValueError("correlation_length_bins must be positive")

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:
        raise ImportError("gaussian_process_place_fields_1d requires scipy.ndimage") from exc

    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(int(n_cells), int(n_positions)))
    inputs = gaussian_filter1d(
        inputs,
        sigma=float(correlation_length_bins),
        axis=1,
        mode=boundary_mode,
    )
    rms = np.sqrt(np.mean(inputs * inputs, axis=1, keepdims=True))
    inputs /= np.maximum(rms, 1e-12)
    rates = np.maximum(inputs - float(threshold), 0.0)
    return inputs, rates


def linear_center_of_mass_positions(matrix, positions):
    """
    Estimate each row's center of mass on a non-periodic spatial axis.

    Rows with zero total activity receive the midpoint of `positions`.
    """
    matrix = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    positions = np.asarray(positions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(positions):
        raise ValueError("matrix columns must match the 1D positions")

    mass = np.sum(matrix, axis=1)
    center = np.full(matrix.shape[0], 0.5 * (positions[0] + positions[-1]))
    good = mass > 0
    center[good] = np.einsum("ij,j->i", matrix[good], positions, optimize=False) / mass[good]
    return center


def circular_bin_centers_deg(n_bins, centered=True):
    """
    Return angular bin centers in degrees for a full 360-degree circle.

    When `centered` is true the bins span -180 to 180 degrees, matching
    peak-aligned plots where 0 degrees is the preferred direction.
    """
    if centered:
        return np.linspace(-180.0, 180.0, n_bins, endpoint=False)
    return np.linspace(0.0, 360.0, n_bins, endpoint=False)


def close_circular_trace(x_deg, values):
    """
    Append the first angular bin at the right edge for continuous line plots.

    The returned arrays cover the whole circle exactly once, for example
    -180..180 degrees, without discarding either side of the aligned data.
    """
    x_deg = np.asarray(x_deg, dtype=float)
    values = np.asarray(values, dtype=float)
    if x_deg.ndim != 1:
        raise ValueError("Expected a 1D angular axis")
    if values.shape[-1] != len(x_deg):
        raise ValueError("The last values axis must match x_deg")

    step = float(np.median(np.diff(x_deg)))
    x_closed = np.r_[x_deg, x_deg[-1] + step]
    values_closed = np.concatenate([values, values[..., :1]], axis=-1)
    return x_closed, values_closed


def circular_resample(values, n_samples, axis=0):
    """
    Fourier-resample samples that live on a periodic angular axis.

    The experimental tuning curves are estimated on 100 head-direction bins,
    while the reference simulations use a denser discretization of the same
    circular target manifold. Resampling in Fourier space preserves periodic
    continuity at 0/2pi better than linear interpolation with open endpoints.
    """
    values = np.asarray(values, dtype=float)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if values.shape[axis] == n_samples:
        return values.astype(float, copy=True)

    try:
        from scipy.signal import resample
    except ImportError as exc:
        raise ImportError("circular_resample requires scipy.signal.resample") from exc

    return resample(values, int(n_samples), axis=axis)


def finite_row_mask(matrix, min_fraction=1.0):
    """
    Select rows with at least `min_fraction` finite angular bins.

    This keeps downstream population plots from silently dropping one side of
    the angular axis because of missing bins.
    """
    matrix = np.asarray(matrix)
    return np.mean(np.isfinite(matrix), axis=1) >= min_fraction


def align_rows_to_peak(matrix, target_bin=None):
    """
    Circularly shift each tuning curve so its peak lands on `target_bin`.

    The alignment is performed independently for every row and preserves the
    full number of angular bins, so the returned matrix still represents the
    complete -180 to 180 degree range after plotting.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix shaped (n_units, n_bins)")
    if matrix.shape[1] == 0:
        raise ValueError("Expected at least one angular bin")

    if target_bin is None:
        target_bin = matrix.shape[1] // 2

    peaks = np.nanargmax(matrix, axis=1)
    return np.vstack([np.roll(row, target_bin - peak) for row, peak in zip(matrix, peaks)])


def align_rows_to_circular_com(matrix, angles_rad=None, target_bin=None):
    """
    Circularly shift each row so its circular center of mass lands on target_bin.

    Figure 2E-F in the reference paper uses center-of-mass-aligned tuning
    curves. The COM angle is rounded to the nearest angular bin before rolling,
    preserving the original bin count and avoiding interpolation artifacts.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix shaped (n_units, n_bins)")
    if matrix.shape[1] == 0:
        raise ValueError("Expected at least one angular bin")

    n_bins = matrix.shape[1]
    if target_bin is None:
        target_bin = n_bins // 2
    if angles_rad is None:
        angles_rad = np.linspace(0.0, 2 * np.pi, n_bins, endpoint=False)

    preferred = circular_center_of_mass_angles(matrix, angles_rad)
    preferred_bins = np.mod(np.rint(preferred / (2 * np.pi) * n_bins).astype(int), n_bins)
    return np.vstack([np.roll(row, target_bin - peak) for row, peak in zip(matrix, preferred_bins)])


def mean_normalize_rows(matrix):
    """
    Divide each row by its finite-bin mean.

    Rows with non-positive or non-finite means are filled with NaN because they
    cannot be interpreted as unit-mean-normalized tuning curves.
    """
    matrix = np.asarray(matrix, dtype=float)
    normalized = matrix.copy()
    row_means = np.nanmean(normalized, axis=1)
    good = np.isfinite(row_means) & (row_means > 0)
    normalized[good] = normalized[good] / row_means[good, None]
    normalized[~good] = np.nan
    return normalized, row_means


def population_mean_and_std(aligned_matrix):
    """
    Compute per-angle mean and standard deviation across aligned units.

    The input should be shaped `(n_units, n_bins)` after peak alignment. The
    standard deviation uses the population definition, matching the formula in
    the TODO where the denominator is N.
    """
    aligned_matrix = np.asarray(aligned_matrix, dtype=float)
    return np.nanmean(aligned_matrix, axis=0), np.nanstd(aligned_matrix, axis=0)


def _as_jsonable_number(value):
    """
    Convert NumPy scalars to plain Python numbers for diagnostics dictionaries.
    """
    value = np.asarray(value)
    if value.shape:
        raise ValueError("expected a scalar value")
    return value.item()


def _finite_stats(values):
    """
    Return compact min/max/mean diagnostics for a finite numeric array.
    """
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def prepare_phi_star_for_inverse(
    phi_raw,
    theta_axis=-1,
    neuron_axis=0,
    alpha_floor=1e-4,
    do_double_normalize=True,
    max_sinkhorn_iter=10000,
    sinkhorn_tol=1e-10,
):
    """
    Prepare data-derived firing-rate targets before applying inverse softplus.

    `phi_raw` is expected to contain tuning curves with one neuron axis and one
    circular angle axis. The returned `phi_safe` is shaped `(n_valid_neurons,
    n_theta_bins)`, has strictly positive entries, and has unit theta mean for
    every retained neuron. When `do_double_normalize` is true, every theta bin
    also has unit population mean. Invalid zero-mean neurons are removed and
    reported through `info["valid_neuron_mask"]`.
    """
    phi_raw = np.asarray(phi_raw, dtype=np.float64)
    if phi_raw.ndim != 2:
        raise ValueError("phi_raw must be a 2D matrix")
    if neuron_axis == theta_axis:
        raise ValueError("neuron_axis and theta_axis must be different")
    if not (0.0 < float(alpha_floor) < 1.0):
        raise ValueError("alpha_floor must be between 0 and 1")

    neuron_axis = int(neuron_axis) % phi_raw.ndim
    theta_axis = int(theta_axis) % phi_raw.ndim
    phi = np.moveaxis(phi_raw, (neuron_axis, theta_axis), (0, 1)).astype(np.float64, copy=True)

    info = {
        "phi_raw_shape": list(phi_raw.shape),
        "canonical_shape": list(phi.shape),
        "neuron_axis": int(neuron_axis),
        "theta_axis": int(theta_axis),
        "alpha_floor": float(alpha_floor),
        "do_double_normalize": bool(do_double_normalize),
        "max_sinkhorn_iter": int(max_sinkhorn_iter),
        "sinkhorn_tol": float(sinkhorn_tol),
        "raw_min": float(np.nanmin(phi)),
        "raw_max": float(np.nanmax(phi)),
        "raw_mean": float(np.nanmean(phi)),
        "raw_exact_zero_count": int(np.sum(phi == 0.0)),
        "raw_below_1e-12_count": int(np.sum(phi < 1e-12)),
        "raw_below_1e-8_count": int(np.sum(phi < 1e-8)),
        "raw_below_1e-6_count": int(np.sum(phi < 1e-6)),
        "raw_below_1e-4_count": int(np.sum(phi < 1e-4)),
    }

    if not np.all(np.isfinite(phi)):
        bad_count = int(np.size(phi) - np.sum(np.isfinite(phi)))
        raise ValueError(f"phi_raw contains {bad_count} NaN or infinite values")

    negative_mask = phi < 0.0
    info["negative_count_clipped"] = int(np.sum(negative_mask))
    info["raw_negative_min"] = float(np.min(phi[negative_mask])) if np.any(negative_mask) else None
    if np.any(negative_mask):
        warnings.warn(
            f"Clipping {info['negative_count_clipped']} negative phi_raw values to zero before normalization",
            RuntimeWarning,
            stacklevel=2,
        )
        phi = phi.copy()
        phi[negative_mask] = 0.0

    row_mean = np.mean(phi, axis=1)
    valid_neuron_mask = np.isfinite(row_mean) & (row_mean > 0.0)
    info["n_neurons_raw"] = int(phi.shape[0])
    info["n_theta_bins"] = int(phi.shape[1])
    info["n_valid_neurons"] = int(np.sum(valid_neuron_mask))
    info["n_removed_neurons"] = int(phi.shape[0] - np.sum(valid_neuron_mask))
    info["valid_neuron_mask"] = valid_neuron_mask.tolist()
    if info["n_valid_neurons"] == 0:
        raise ValueError("No neurons have positive finite mean firing rate")

    phi = phi[valid_neuron_mask]
    phi /= np.mean(phi, axis=1, keepdims=True)
    phi = (1.0 - float(alpha_floor)) * phi + float(alpha_floor)

    sinkhorn_iter = 0
    row_err = float(np.max(np.abs(np.mean(phi, axis=1) - 1.0)))
    col_err = float(np.max(np.abs(np.mean(phi, axis=0) - 1.0)))
    if do_double_normalize:
        for sinkhorn_iter in range(1, int(max_sinkhorn_iter) + 1):
            phi /= np.mean(phi, axis=1, keepdims=True)
            phi /= np.mean(phi, axis=0, keepdims=True)
            row_err = float(np.max(np.abs(np.mean(phi, axis=1) - 1.0)))
            col_err = float(np.max(np.abs(np.mean(phi, axis=0) - 1.0)))
            if max(row_err, col_err) < float(sinkhorn_tol):
                break

    info["sinkhorn_iterations"] = int(sinkhorn_iter)
    info["final_row_mean_max_abs_error"] = float(row_err)
    info["final_col_mean_max_abs_error"] = float(col_err)
    info["phi_safe_shape"] = list(phi.shape)
    info["phi_safe_min"] = float(np.min(phi))
    info["phi_safe_max"] = float(np.max(phi))
    info["phi_safe_mean"] = float(np.mean(phi))

    diagnostics = []
    if not np.all(np.isfinite(phi)):
        diagnostics.append("phi_safe contains NaN or infinite values")
    if not np.min(phi) > 0.0:
        diagnostics.append(f"phi_safe minimum is not strictly positive: {np.min(phi):.6g}")
    if row_err >= 1e-8:
        diagnostics.append(f"row mean normalization error {row_err:.6g} exceeds 1e-8")
    if do_double_normalize and col_err >= 1e-8:
        diagnostics.append(f"column mean normalization error {col_err:.6g} exceeds 1e-8")
    if diagnostics:
        raise ValueError("; ".join(diagnostics))

    return phi, info


def softplus_inverse(rate, beta=2.0):
    """
    Convert positive firing rates to input currents for a softplus nonlinearity.

    The inverse is deliberately strict: target-rate floors must be applied by
    `prepare_phi_star_for_inverse`, not hidden inside this function. The
    data-derived network model uses beta=2.
    """
    rate = np.asarray(rate, dtype=np.float64)
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not np.all(np.isfinite(rate)):
        raise ValueError("rate contains NaN or infinite values")
    if np.any(rate <= 0.0):
        raise ValueError("softplus_inverse requires strictly positive rates")

    z = beta * rate
    out = np.empty_like(rate, dtype=np.float64)
    small = z < 20.0
    out[small] = np.log(np.expm1(z[small])) / beta
    out[~small] = rate[~small] + np.log1p(-np.exp(-z[~small])) / beta
    return out


softplus_inv = softplus_inverse


def softplus(x, beta=2.0):
    """
    Evaluate the smooth firing-rate nonlinearity used by the network model.

    The implementation is numerically stable while matching
    `log(1 + exp(beta * x)) / beta`.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.logaddexp(beta * x, 0.0) / beta


def softplus_derivative_from_x(x, beta=2.0):
    """
    Evaluate d softplus(x) / dx as a stable sigmoid(beta * x).
    """
    z = float(beta) * np.asarray(x, dtype=np.float64)
    return np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def softplus_derivative_from_phi(phi, beta=2.0):
    """
    Evaluate the softplus derivative at x_star from phi_star directly.

    Since phi_star = softplus(x_star), the derivative is
    `1 - exp(-beta * phi_star)`, avoiding sigmoid evaluation at large negative
    currents.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi contains NaN or infinite values")
    if np.any(phi <= 0.0):
        raise ValueError("softplus_derivative_from_phi requires strictly positive rates")
    return 1.0 - np.exp(-float(beta) * phi)


def current_space_jacobian(weights, phi_at_angle, beta=2.0, inhibition_c=0.0):
    """
    Build A(theta) = -I + (J - c/N) @ diag(phi_prime(theta)).
    """
    weights = np.asarray(weights, dtype=np.float64)
    phi_at_angle = np.asarray(phi_at_angle, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square")
    n_neurons = weights.shape[0]
    if phi_at_angle.shape != (n_neurons,):
        raise ValueError("phi_at_angle must have one entry per neuron")
    derivative = softplus_derivative_from_phi(phi_at_angle, beta=beta)
    effective_weights = weights - float(inhibition_c) / n_neurons
    jacobian = -np.eye(n_neurons, dtype=np.float64)
    jacobian += effective_weights * derivative[None, :]
    return jacobian


def fixed_point_residual_from_weights(weights, x_star, phi_star, inhibition_c=0.0):
    """
    Compute -x_star + (J - c/N) phi_star + c on target manifold samples.
    """
    weights = np.asarray(weights, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    phi_star = np.asarray(phi_star, dtype=np.float64)
    if phi_star.ndim != 2 or x_star.ndim != 2:
        raise ValueError("x_star and phi_star must be 2D")
    if x_star.shape != phi_star.shape:
        raise ValueError("x_star and phi_star must have the same shape")
    n_neurons = weights.shape[0]
    if phi_star.shape[0] != n_neurons:
        raise ValueError("phi_star must be shaped (n_neurons, n_angles)")
    effective_weights = weights - float(inhibition_c) / n_neurons
    residual = -x_star + np.einsum("ij,jk->ik", effective_weights, phi_star, optimize=False) + float(inhibition_c)
    return residual


def sinkhorn_normalize(matrix, max_iter=10000, tol=1e-10):
    """
    Iteratively normalize rows and columns so both means are one.

    This is the double normalization used before constructing data-derived
    attractor weights: every neuron keeps unit angular mean, while the
    population mean is exactly one at every angle.
    """
    out = np.maximum(np.asarray(matrix, dtype=float), 1e-12).copy()
    for _ in range(max_iter):
        out /= np.nanmean(out, axis=1, keepdims=True)
        out /= np.nanmean(out, axis=0, keepdims=True)
        row_err = np.nanmax(np.abs(np.nanmean(out, axis=1) - 1.0))
        col_err = np.nanmax(np.abs(np.nanmean(out, axis=0) - 1.0))
        if max(row_err, col_err) < tol:
            break
    return out


def invert_spd_cholesky(matrix, jitter=0.0):
    """
    Invert a small symmetric positive-definite matrix without LAPACK calls.

    The local Windows scientific stack can crash inside BLAS/LAPACK routines.
    The A2 dual kernels are only `n_angles x n_angles` (100 x 100 here), so a
    direct Cholesky implementation is fast enough and keeps the computation
    reproducible in this environment.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    work = matrix.copy()
    if jitter:
        work[np.diag_indices(n)] += float(jitter)

    lower = np.zeros_like(work)
    for i in range(n):
        for j in range(i + 1):
            subtotal = float(np.sum(lower[i, :j] * lower[j, :j]))
            if i == j:
                value = work[i, i] - subtotal
                if value <= 0:
                    raise np.linalg.LinAlgError("matrix is not positive definite")
                lower[i, j] = np.sqrt(value)
            else:
                lower[i, j] = (work[i, j] - subtotal) / lower[j, j]

    identity = np.eye(n)
    y = np.zeros_like(work)
    for col in range(n):
        for i in range(n):
            subtotal = float(np.sum(lower[i, :i] * y[:i, col]))
            y[i, col] = (identity[i, col] - subtotal) / lower[i, i]

    inverse = np.zeros_like(work)
    for col in range(n):
        for i in range(n - 1, -1, -1):
            subtotal = float(np.sum(lower[i + 1 :, i] * inverse[i + 1 :, col]))
            inverse[i, col] = (y[i, col] - subtotal) / lower[i, i]

    return inverse


def circular_center_of_mass_angles(matrix, angles_rad):
    """
    Estimate each tuning curve's preferred direction by circular center of mass.

    Rows are neurons and columns are angular bins. The returned angles are in
    `[0, 2*pi)`, suitable for ordering neurons around the latent ring.
    """
    matrix = np.asarray(matrix, dtype=float)
    angles_rad = np.asarray(angles_rad, dtype=float)
    if matrix.shape[1] != len(angles_rad):
        raise ValueError("matrix columns must match angles_rad")

    z = np.einsum("ij,j->i", matrix, np.exp(1j * angles_rad), optimize=False)
    return np.angle(z) % (2 * np.pi)


def optimized_recurrent_weights(phi, regularization=1e-6, activation_beta=2.0, dtype=np.float32):
    """
    Construct minimum-norm recurrent weights from target tuning curves.

    `phi` is the doubly normalized firing-rate manifold shaped
    `(n_neurons, n_angles)`. The implementation follows Appendix A2 of the
    reference paper. The dual kernel is
    `phi.T @ phi / n_neurons + regularization * n_angles / n_neurons * I`,
    matching the discrete-continuous correspondence in Eq. 28. The
    `J_ii = 0` constraint is applied through the row-specific
    leave-one-neuron-out kernel of Eq. 31 using a Sherman-Morrison update, and
    the diagonal is set to zero explicitly.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be shaped (n_neurons, n_angles)")

    n_neurons, n_angles = phi.shape
    x_star = softplus_inverse(phi, beta=activation_beta)
    # Use explicit einsum to avoid Windows BLAS crashes seen with small Gram products.
    kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    kernel += (regularization * n_angles / n_neurons) * np.eye(n_angles)
    kernel_inv = invert_spd_cholesky(kernel)

    weights = np.empty((n_neurons, n_neurons), dtype=dtype)
    inv_phi = np.einsum("ab,ib->ai", kernel_inv, phi, optimize=False)
    for i in range(n_neurons):
        u = phi[i] / np.sqrt(n_neurons)
        v = inv_phi[:, i] / np.sqrt(n_neurons)
        denom = max(1.0 - float(np.sum(u * v)), 1e-12)
        row_dual = np.einsum("a,ab->b", x_star[i], kernel_inv, optimize=False)
        row_dual += (float(np.sum(x_star[i] * v)) / denom) * v
        weights[i] = np.einsum("a,ja->j", row_dual, phi, optimize=False) / n_neurons

    np.fill_diagonal(weights, 0.0)
    return weights


def optimized_recurrent_factors(
    phi,
    regularization=1e-6,
    activation_beta=2.0,
    enforce_zero_diagonal=False,
    dtype=np.float64,
):
    """
    Build low-rank factors for fast multiplication by optimized weights.

    The unconstrained optimized matrix is `A @ B.T`. The returned `diagonal`
    is the diagonal of that product, so simulations can subtract the
    self-connection term without materializing a dense matrix. When
    `enforce_zero_diagonal` is true, `A` includes the same row-specific
    leave-one-neuron-out correction used by `optimized_recurrent_weights`; the
    returned diagonal should still be subtracted to impose `J_ii = 0`.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be shaped (n_neurons, n_angles)")

    n_neurons, n_angles = phi.shape
    x_star = softplus_inverse(phi, beta=activation_beta)
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
    return a.astype(dtype), b.astype(dtype), diagonal.astype(dtype)


def materialize_lowrank_weights(a, b, diagonal=None, dtype=np.float32, chunk_size=128):
    """
    Materialize `A @ B.T - diag(diagonal)` without large BLAS calls.

    The optimized data-derived matrix has low-rank factors with shape
    `(n_neurons, n_angles)`. Building rows in chunks keeps memory predictable
    and avoids the unstable dense matrix multiply path on this Windows setup.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be 2D arrays with matching factor dimension")

    n_rows = a.shape[0]
    n_cols = b.shape[0]
    weights = np.empty((n_rows, n_cols), dtype=dtype)
    for start in range(0, n_rows, int(chunk_size)):
        stop = min(n_rows, start + int(chunk_size))
        weights[start:stop] = np.einsum("ik,jk->ij", a[start:stop], b, optimize=False)

    if diagonal is not None:
        diagonal = np.asarray(diagonal, dtype=float)
        if n_rows != n_cols or diagonal.shape != (n_rows,):
            raise ValueError("diagonal must have one entry per square matrix row")
        idx = np.arange(n_rows)
        weights[idx, idx] -= diagonal.astype(weights.dtype, copy=False)
    return weights


def circulant_from_diagonal_means(matrix):
    """
    Average a square matrix along periodic diagonals to make a circulant matrix.

    The k-th value is the mean of entries `matrix[i, (i-k) mod N]`. Reusing
    those means for every row creates a matrix whose rows are circular shifts.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    idx = np.arange(n)
    diagonal_means = np.empty(n, dtype=float)
    for k in range(n):
        diagonal_means[k] = np.mean(matrix[idx, (idx - k) % n])

    out = np.empty_like(matrix, dtype=float)
    for i in range(n):
        out[i] = diagonal_means[(i - idx) % n]
    return out, diagonal_means


def scramble_residuals(matrix, baseline, rng):
    """
    Add randomly permuted residuals from `matrix - baseline` back to `baseline`.

    This preserves the exact empirical residual distribution and total residual
    energy while destroying its structured alignment with the circulant part.
    """
    residual = np.asarray(matrix) - np.asarray(baseline)
    scrambled = rng.permutation(residual.ravel()).reshape(residual.shape)
    return np.asarray(baseline) + scrambled, scrambled


def pca_basis(samples, n_components=3):
    """
    Compute principal-component axes for row-wise samples by power iteration.

    Returns the sample mean and the first `n_components` right singular
    vectors, allowing high-dimensional network states to be projected for
    visualization. This avoids large SVD/LAPACK calls, which are unstable in
    the local Windows numerical stack.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be shaped (n_samples, n_features)")

    mean = np.mean(samples, axis=0)
    centered = samples - mean
    rng = np.random.default_rng(20260617)
    components = []
    for _ in range(int(n_components)):
        vector = rng.normal(size=centered.shape[1])
        for prev in components:
            vector = vector - float(np.sum(vector * prev)) * prev
        norm = np.sqrt(np.sum(vector * vector))
        vector = vector / max(norm, 1e-12)

        for _ in range(80):
            scores = np.einsum("ij,j->i", centered, vector, optimize=False)
            next_vector = np.einsum("ij,i->j", centered, scores, optimize=False)
            for prev in components:
                next_vector = next_vector - float(np.sum(next_vector * prev)) * prev
            norm = np.sqrt(np.sum(next_vector * next_vector))
            if norm <= 1e-12:
                break
            next_vector = next_vector / norm
            if np.sqrt(np.sum((next_vector - vector) ** 2)) < 1e-7:
                vector = next_vector
                break
            vector = next_vector
        components.append(vector)

    return mean.astype(np.float32), np.asarray(components, dtype=np.float32)


def project_onto_basis(samples, mean, basis):
    """
    Project row-wise samples onto a PCA basis.
    """
    samples = np.asarray(samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    basis = np.asarray(basis, dtype=float)
    return np.einsum("...j,kj->...k", samples - mean, basis, optimize=False)


def nearest_manifold_distance(states, manifold, return_l2=False):
    """
    Compute each state's nearest Euclidean distance to a manifold.

    `states` is shaped `(n_states, n_neurons)` and `manifold` is shaped
    `(n_angles, n_neurons)`. By default this returns the full-state L2 distance
    divided by `sqrt(n_neurons)`, matching the Figure 3F axis. Set
    `return_l2=True` to also return the unnormalized full-state L2 distance.
    """
    states = np.asarray(states, dtype=float)
    manifold = np.asarray(manifold, dtype=float)
    if states.ndim != 2 or manifold.ndim != 2:
        raise ValueError("states and manifold must both be 2D")
    if states.shape[1] != manifold.shape[1]:
        raise ValueError("states and manifold must have the same feature count")

    nearest = np.full(len(states), -1, dtype=int)
    nearest_l2 = np.full(len(states), np.nan, dtype=float)
    finite_states = np.all(np.isfinite(states), axis=1)
    finite_manifold = np.all(np.isfinite(manifold), axis=1)
    manifold_finite = manifold[finite_manifold]
    manifold_indices = np.flatnonzero(finite_manifold)
    if len(manifold_finite) == 0:
        if return_l2:
            return nearest_l2, nearest, nearest_l2
        return nearest_l2, nearest

    manifold_norm = np.sum(manifold_finite * manifold_finite, axis=1)[None, :]
    good_rows = np.flatnonzero(finite_states)
    chunk_size = 512
    for start in range(0, len(good_rows), chunk_size):
        row_idx = good_rows[start : start + chunk_size]
        chunk = states[row_idx]
        state_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        cross = np.einsum("ij,kj->ik", chunk, manifold_finite, optimize=False)
        squared = np.maximum(state_norm + manifold_norm - 2.0 * cross, 0.0)
        local_nearest = np.argmin(squared, axis=1)
        nearest[row_idx] = manifold_indices[local_nearest]
        nearest_l2[row_idx] = np.sqrt(squared[np.arange(len(row_idx)), local_nearest])
    normalized = nearest_l2 / np.sqrt(states.shape[1])
    if return_l2:
        return normalized, nearest, nearest_l2
    return normalized, nearest


def simulate_rate_network(
    weights,
    initial_states,
    tau_s=1.0,
    dt_s=0.05,
    duration_s=30.0,
    inhibition_c=1.0,
    activation_beta=2.0,
    record_every_s=0.5,
    current_clip=None,
    stop_abs=None,
    progress_label=None,
    progress_interval_wall_s=10.0,
):
    """
    Simulate the firing-rate network from several initial conditions at once.

    The state variable is input current `x`; firing rates are `softplus(x)`.
    A global inhibition term stabilizes the population mean firing rate around
    one while preserving doubly normalized target fixed points.
    """
    weights_t = np.asarray(weights, dtype=float).T

    def drive(rates):
        return np.einsum("tn,nj->tj", rates, weights_t, optimize=False)

    return simulate_rate_network_with_drive(
        drive,
        initial_states,
        tau_s=tau_s,
        dt_s=dt_s,
        duration_s=duration_s,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        record_every_s=record_every_s,
        current_clip=current_clip,
        stop_abs=stop_abs,
        progress_label=progress_label,
        progress_interval_wall_s=progress_interval_wall_s,
    )


def simulate_rate_network_with_drive(
    recurrent_drive,
    initial_states,
    tau_s=1.0,
    dt_s=0.05,
    duration_s=30.0,
    inhibition_c=1.0,
    activation_beta=2.0,
    record_every_s=0.5,
    current_clip=None,
    stop_abs=None,
    progress_label=None,
    progress_interval_wall_s=10.0,
):
    """
    Simulate rate dynamics using a custom recurrent-drive function.

    `recurrent_drive(rates)` must return recurrent input for a batch of
    firing-rate states shaped `(n_trials, n_neurons)`. The reference Euler
    dynamics do not clip input currents; `current_clip` is therefore disabled
    by default and exists only for explicit numerical stress tests. `stop_abs`
    does not alter the dynamics; it stops recording once states have already
    exceeded a diagnostic magnitude threshold.
    """
    x = np.asarray(initial_states, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError("initial_states must be shaped (n_trials, n_neurons)")

    n_steps = int(np.round(duration_s / dt_s))
    record_every = max(1, int(np.round(record_every_s / dt_s)))
    drive_scale = dt_s / tau_s

    start_wall = time.perf_counter()
    next_progress_wall = start_wall + progress_interval_wall_s

    times = [0.0]
    trajectory = [x.copy()]
    for step in range(1, n_steps + 1):
        rates = softplus(x, beta=activation_beta)
        recurrent = np.asarray(recurrent_drive(rates), dtype=float)
        mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
        dx = -x + recurrent - inhibition_c * mean_error
        x = x + drive_scale * dx
        if current_clip is not None:
            x = np.clip(x, -float(current_clip), float(current_clip))

        should_stop = False
        if not np.isfinite(x).all():
            should_stop = True
        elif stop_abs is not None and np.max(np.abs(x)) > float(stop_abs):
            should_stop = True

        if step % record_every == 0 or step == n_steps or should_stop:
            times.append(step * dt_s)
            trajectory.append(x.copy())

        if progress_label is not None:
            now = time.perf_counter()
            if now >= next_progress_wall or step == n_steps:
                elapsed = now - start_wall
                simulated_t = step * dt_s
                print(
                    f"[{progress_label}] step {step}/{n_steps}, "
                    f"t={simulated_t:.2f}/{duration_s:.2f}s, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_progress_wall = now + progress_interval_wall_s

        if should_stop:
            if progress_label is not None:
                print(
                    f"[{progress_label}] stopping early at step {step}/{n_steps}; "
                    "state magnitude became non-finite or exceeded stop_abs",
                    flush=True,
                )
            break

    return np.asarray(times), np.asarray(trajectory, dtype=np.float32)


def relax_rate_network_states(
    recurrent_drive,
    initial_states,
    tau_s=0.05,
    dt_s=0.001,
    duration_s=2.0,
    inhibition_c=1.0,
    activation_beta=2.0,
    current_clip=None,
    progress_label=None,
    progress_interval_wall_s=10.0,
):
    """
    Relax input-current states under the rate-network dynamics.

    This is the same Euler update used by `simulate_rate_network_with_drive`,
    but it returns only the final state. It is useful for finding the
    numerically realized fixed-point manifold when regularization makes the
    designed target manifold only approximately invariant.
    """
    x = np.asarray(initial_states, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError("initial_states must be shaped (n_trials, n_neurons)")

    n_steps = int(np.round(duration_s / dt_s))
    drive_scale = dt_s / tau_s

    start_wall = time.perf_counter()
    next_progress_wall = start_wall + progress_interval_wall_s

    for step in range(1, n_steps + 1):
        rates = softplus(x, beta=activation_beta)
        recurrent = np.asarray(recurrent_drive(rates), dtype=float)
        mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
        dx = -x + recurrent - inhibition_c * mean_error
        x = x + drive_scale * dx
        if current_clip is not None:
            x = np.clip(x, -float(current_clip), float(current_clip))

        if progress_label is not None:
            now = time.perf_counter()
            if now >= next_progress_wall or step == n_steps:
                elapsed = now - start_wall
                simulated_t = step * dt_s
                print(
                    f"[{progress_label}] step {step}/{n_steps}, "
                    f"t={simulated_t:.2f}/{duration_s:.2f}s, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_progress_wall = now + progress_interval_wall_s

    return x.astype(np.float32)


def lowrank_recurrent_drive(a, b, diagonal=None):
    """
    Create a recurrent-drive function for low-rank optimized weights.

    When `diagonal` is provided the drive uses `A @ B.T - diag(diagonal)`.
    Passing `None` keeps the unconstrained low-rank weights, which is useful
    when exact target fixed-point matching is more important than removing
    self-connections.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if diagonal is not None:
        diagonal = np.asarray(diagonal, dtype=float)

    def drive(rates):
        rates = np.asarray(rates, dtype=float)
        latent = np.einsum("tn,nk->tk", rates, b, optimize=False)
        recurrent = np.einsum("tk,nk->tn", latent, a, optimize=False)
        if diagonal is not None:
            recurrent = recurrent - rates * diagonal
        return recurrent

    return drive


def circulant_recurrent_drive(first_column):
    """
    Create a fast recurrent-drive function for a circulant weight matrix.

    `first_column[i]` corresponds to the weight from neuron 0 to neuron i, so
    recurrent input is a circular convolution over COM-ordered neurons.
    """
    first_column = np.asarray(first_column, dtype=float)
    fft_kernel = np.fft.rfft(first_column)

    def drive(rates):
        rates = np.asarray(rates, dtype=float)
        return np.fft.irfft(np.fft.rfft(rates, axis=1) * fft_kernel, n=rates.shape[1], axis=1)

    return drive
