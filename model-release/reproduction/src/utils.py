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


def wrapped_gaussian_correlation(delta_theta, sigma, tail_tolerance=1e-14):
    """
    Evaluate the wrapped-Gaussian covariance from paper Eq. 9.

    The paper parameterizes the kernel as
    ``sum_n exp[-sigma**2 * (delta_theta + 2*pi*n)**2 / 2]``.
    Consequently, ``1 / sigma`` is the angular correlation length: increasing
    ``sigma`` narrows the covariance and adds high-frequency structure to
    Gaussian-process samples.
    """
    delta_theta = np.asarray(delta_theta, dtype=float)
    sigma = float(sigma)
    tail_tolerance = float(tail_tolerance)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not 0.0 < tail_tolerance < 1.0:
        raise ValueError("tail_tolerance must lie between zero and one")

    wrapped_delta = (delta_theta + np.pi) % (2.0 * np.pi) - np.pi
    tail_distance = np.sqrt(-2.0 * np.log(tail_tolerance)) / sigma
    image_radius = max(1, int(np.ceil((tail_distance + np.pi) / (2.0 * np.pi))))
    images = np.arange(-image_radius, image_radius + 1, dtype=float)
    shifted = wrapped_delta[..., None] + 2.0 * np.pi * images
    return np.sum(np.exp(-0.5 * sigma * sigma * shifted * shifted), axis=-1)


def wrapped_gaussian_fourier_coefficients(frequencies, sigma):
    """
    Return Fourier-series coefficients of the wrapped covariance.

    This implements paper Eq. 113 under the convention
    ``Gamma(delta) = sum_n Gamma_hat[n] * exp(i*n*delta)``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    normalization = np.sqrt(2.0 * np.pi * sigma * sigma)
    return np.exp(-0.5 * frequencies * frequencies / (sigma * sigma)) / normalization


def sample_circular_gaussian_process(
    n_samples,
    n_angles,
    sigma,
    seed=None,
    white_noise=None,
):
    """
    Sample the circular Gaussian process using its Fourier eigenmodes.

    The covariance matrix on an equally spaced angular grid is circulant, so
    its eigenvectors are discrete Fourier modes. Filtering real white noise by
    the square root of the covariance eigenvalues produces exact samples of
    the discretized process without constructing or factorizing a dense
    covariance matrix. Supplying ``white_noise`` allows several ``sigma``
    values to be compared using identical underlying random coefficients.
    """
    n_samples = int(n_samples)
    n_angles = int(n_angles)
    if n_samples <= 0 or n_angles <= 1:
        raise ValueError("n_samples must be positive and n_angles must exceed one")

    if white_noise is None:
        rng = np.random.default_rng(seed)
        white_noise = rng.normal(size=(n_samples, n_angles))
    else:
        white_noise = np.asarray(white_noise, dtype=float)
        if white_noise.shape != (n_samples, n_angles):
            raise ValueError("white_noise must have shape (n_samples, n_angles)")
        if not np.isfinite(white_noise).all():
            raise ValueError("white_noise must be finite")

    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    covariance_row = wrapped_gaussian_correlation(theta, sigma)
    eigenvalues = np.fft.rfft(covariance_row).real
    roundoff_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * roundoff_scale:
        raise ValueError("discretized wrapped-Gaussian covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    spectrum = np.fft.rfft(white_noise, axis=-1)
    samples = np.fft.irfft(
        spectrum * np.sqrt(eigenvalues)[None, :],
        n=n_angles,
        axis=-1,
    )
    return theta, samples


def normalized_softplus_tuning(input_currents, beta, bias, axis=-1):
    """
    Apply the normalized softplus used by the fitted generative process.

    This is Eq. 10 and Eq. B3 of the paper. The factor ``1 / beta`` sometimes
    included in the softplus definition cancels under mean normalization, so
    the implementation evaluates ``log(1 + exp(beta * (x - b)))`` directly
    and divides each tuning curve by its angular mean.
    """
    input_currents = np.asarray(input_currents, dtype=float)
    beta = float(beta)
    bias = float(bias)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not np.isfinite(input_currents).all():
        raise ValueError("input_currents must be finite")

    unnormalized = np.logaddexp(beta * (input_currents - bias), 0.0)
    normalization = np.mean(unnormalized, axis=axis, keepdims=True)
    if np.any(normalization <= 0.0) or not np.isfinite(normalization).all():
        raise ValueError("normalized softplus received a non-positive normalization")
    return unnormalized / normalization


def tuning_fourier_power_coefficients(tuning_curves, n_modes, axis=-1):
    """
    Compute nonzero Fourier coefficients of an uncentered two-point function.

    For a unit-mean tuning curve ``phi_i(theta)``, the Fourier coefficient of
    the population correlation ``Gamma_phi(delta)`` at mode ``n`` is the
    population mean of ``|phi_hat_i[n]|**2``. This implementation uses that
    identity directly, avoiding construction of the full angle-by-angle
    correlation matrix. Modes 1 through ``n_modes`` are returned; the constant
    mode is omitted exactly as in Figure 5D-E.
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    axis = int(axis)
    n_modes = int(n_modes)
    if tuning_curves.ndim < 2:
        raise ValueError("tuning_curves must contain samples and angular bins")
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")

    n_angles = tuning_curves.shape[axis]
    if n_modes <= 0 or n_modes > n_angles // 2:
        raise ValueError("n_modes must lie between 1 and the angular Nyquist mode")

    spectrum = np.fft.rfft(tuning_curves, axis=axis) / n_angles
    selected = np.take(spectrum, np.arange(1, n_modes + 1), axis=axis)
    sample_axes = tuple(index for index in range(selected.ndim) if index != axis % selected.ndim)
    return np.mean(np.abs(selected) ** 2, axis=sample_axes)


def generated_tuning_fourier_coefficients(input_currents, beta, bias, n_modes):
    """
    Transform Gaussian-process currents and return their rate correlations.

    Each current sample is passed through the paper's normalized softplus
    (Eq. 10/B3), then modes 1 through ``n_modes`` of the generated two-point
    firing-rate correlation are estimated by Monte Carlo averaging.
    """
    rates = normalized_softplus_tuning(
        input_currents,
        beta=beta,
        bias=bias,
        axis=-1,
    )
    return tuning_fourier_power_coefficients(rates, n_modes=n_modes, axis=-1)


def fourier_correlation_error(target_coefficients, generated_coefficients):
    """
    Return the Figure 5D-E squared error across distinct Fourier modes.
    """
    target_coefficients = np.asarray(target_coefficients, dtype=float)
    generated_coefficients = np.asarray(generated_coefficients, dtype=float)
    if target_coefficients.shape != generated_coefficients.shape:
        raise ValueError("target and generated Fourier coefficients must have equal shape")
    if not np.isfinite(target_coefficients).all() or not np.isfinite(generated_coefficients).all():
        raise ValueError("Fourier coefficients must be finite")
    residual = target_coefficients - generated_coefficients
    return float(np.sum(residual * residual))


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


def circular_peak_z_scores(tuning_curves):
    """
    计算每条环形调谐曲线所有局部极大值的曲线内 z-score。

    局部峰严格定义为某个角度 bin 的值同时大于左右两个环形邻居，
    因此第一个和最后一个 bin 也会彼此比较。z-score 使用每条曲线自身
    沿角度的均值和总体标准差，匹配 Figure 6C 的定义；不额外引入
    prominence、峰间距或后处理平滑。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的有限调谐曲线，
            或形状为 ``(n_angles,)`` 的单条曲线。

    Returns:
        ``(peak_mask, peak_z_scores)``。两者形状均为
        ``(n_curves, n_angles)``；非峰位置的 z-score 为 ``NaN``。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim == 1:
        tuning_curves = tuning_curves[None, :]
    if tuning_curves.ndim != 2 or tuning_curves.shape[1] < 3:
        raise ValueError(
            "tuning_curves must be shaped (n_curves, n_angles) with at least 3 angles"
        )
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")

    means = np.mean(tuning_curves, axis=1)
    standard_deviations = np.std(tuning_curves, axis=1)
    peak_mask = (
        (tuning_curves > np.roll(tuning_curves, 1, axis=1))
        & (tuning_curves > np.roll(tuning_curves, -1, axis=1))
    )
    peak_z = np.full(tuning_curves.shape, np.nan, dtype=float)
    valid_rows = standard_deviations > 0.0
    z_scores = np.zeros_like(tuning_curves)
    z_scores[valid_rows] = (
        tuning_curves[valid_rows] - means[valid_rows, None]
    ) / standard_deviations[valid_rows, None]
    peak_z[peak_mask & valid_rows[:, None]] = z_scores[
        peak_mask & valid_rows[:, None]
    ]
    return peak_mask, peak_z


def circular_peak_counts(tuning_curves, z_thresholds):
    """
    按多个 z-score 阈值统计每条环形调谐曲线的峰数量。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的调谐曲线。
        z_thresholds: 一个或多个检测阈值；只有 ``z > z_threshold`` 的
            局部峰才会计数，使用原文规定的严格不等式。

    Returns:
        形状为 ``(n_thresholds, n_curves)`` 的整数峰数矩阵。
    """
    thresholds = np.atleast_1d(np.asarray(z_thresholds, dtype=float))
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError("z_thresholds must be a finite one-dimensional array")

    _, peak_z = circular_peak_z_scores(tuning_curves)
    return np.stack(
        [np.sum(peak_z > threshold, axis=1) for threshold in thresholds],
        axis=0,
    )


def ranked_circular_peak_heights(tuning_curves, z_threshold, n_ranks=3):
    """
    提取每条环形调谐曲线中 z-score 合格峰的降序峰高。

    Figure 6D 使用 ``z_threshold=1``，并分别汇总每条曲线最高、第二高和
    第三高的局部峰。若某条曲线没有足够多的合格峰，对应位置保持
    ``NaN``，使后续密度估计只使用真正存在该 rank 峰的神经元。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的调谐曲线。
        z_threshold: 峰自身必须严格超过的曲线内 z-score 阈值。
        n_ranks: 每条曲线最多返回多少个最高峰。

    Returns:
        形状为 ``(n_curves, n_ranks)`` 的峰高矩阵，按行降序排列。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    n_ranks = int(n_ranks)
    if n_ranks <= 0:
        raise ValueError("n_ranks must be positive")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold):
        raise ValueError("z_threshold must be finite")

    _, peak_z = circular_peak_z_scores(tuning_curves)
    qualifying = peak_z > z_threshold
    candidate_heights = np.where(qualifying, tuning_curves, -np.inf)
    sorted_heights = np.sort(candidate_heights, axis=1)[:, ::-1]
    ranked = np.full((tuning_curves.shape[0], n_ranks), np.nan, dtype=float)
    available = min(n_ranks, sorted_heights.shape[1])
    ranked[:, :available] = sorted_heights[:, :available]
    ranked[~np.isfinite(ranked)] = np.nan
    return ranked


def reflect_circular_curves_about_com(tuning_curves, angles_rad=None):
    """
    将每条环形调谐曲线围绕其 circular center of mass 做镜像反射。

    对离散角度网格，目标值 ``v(2 * theta_COM - theta)`` 通常落在两个
    bin 之间，因此使用周期线性插值，而不把 COM 舍入到最近 bin。
    这直接离散化 Figure 6E 的连续反射定义，并正确处理 0/2π 边界。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的有限曲线，
            或形状为 ``(n_angles,)`` 的单条曲线。
        angles_rad: 可选均匀环形角度网格；默认是 ``[0, 2π)``。

    Returns:
        与输入二维形式相同的反射曲线，以及每条曲线的 COM 角度。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim == 1:
        tuning_curves = tuning_curves[None, :]
    if tuning_curves.ndim != 2 or tuning_curves.shape[1] < 3:
        raise ValueError(
            "tuning_curves must be shaped (n_curves, n_angles) with at least 3 angles"
        )
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")

    n_angles = tuning_curves.shape[1]
    if angles_rad is None:
        angles_rad = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    angles_rad = np.asarray(angles_rad, dtype=float)
    if angles_rad.shape != (n_angles,):
        raise ValueError("angles_rad must contain one angle per curve bin")
    expected_step = 2.0 * np.pi / n_angles
    wrapped_steps = np.mod(np.diff(np.append(angles_rad, angles_rad[0] + 2.0 * np.pi)), 2.0 * np.pi)
    if not np.allclose(wrapped_steps, expected_step, rtol=1e-7, atol=1e-10):
        raise ValueError("angles_rad must be a uniform circular grid")

    com_angles = circular_center_of_mass_angles(tuning_curves, angles_rad)
    origin = float(angles_rad[0])
    source_angles = (
        2.0 * com_angles[:, None] - angles_rad[None, :]
    ) % (2.0 * np.pi)
    source_positions = ((source_angles - origin) % (2.0 * np.pi)) / expected_step
    lower = np.floor(source_positions).astype(np.int64) % n_angles
    fraction = source_positions - np.floor(source_positions)
    upper = (lower + 1) % n_angles
    row_indices = np.arange(tuning_curves.shape[0])[:, None]
    reflected = (
        (1.0 - fraction) * tuning_curves[row_indices, lower]
        + fraction * tuning_curves[row_indices, upper]
    )
    return reflected, com_angles


def circular_flip_correlations(tuning_curves, angles_rad=None):
    """
    计算调谐曲线与其 COM 镜像之间的逐曲线 Pearson 相关。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的调谐曲线。
        angles_rad: 可选均匀环形角度网格。

    Returns:
        形状为 ``(n_curves,)`` 的 ``rho_flip``。任一方零方差时返回 NaN。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim == 1:
        tuning_curves = tuning_curves[None, :]
    reflected, _ = reflect_circular_curves_about_com(
        tuning_curves,
        angles_rad=angles_rad,
    )
    centered = tuning_curves - np.mean(tuning_curves, axis=1, keepdims=True)
    reflected_centered = reflected - np.mean(
        reflected,
        axis=1,
        keepdims=True,
    )
    numerator = np.sum(centered * reflected_centered, axis=1)
    denominator = np.sqrt(
        np.sum(centered * centered, axis=1)
        * np.sum(reflected_centered * reflected_centered, axis=1)
    )
    correlations = np.full(tuning_curves.shape[0], np.nan, dtype=float)
    valid = denominator > 0.0
    correlations[valid] = numerator[valid] / denominator[valid]
    return np.clip(correlations, -1.0, 1.0)


def head_direction_information_content(tuning_curves):
    """
    计算每条调谐曲线的头朝向信息量（bits/spike）。

    对均匀角度先验，离散 Skaggs information content 为
    ``mean_theta[(v / mean(v)) * log2(v / mean(v))]``。零放电率 bin
    按数学极限 ``0 * log2(0) = 0`` 处理，不引入 epsilon。该量对整条
    曲线乘以正数保持不变，因此适用于原始 firing rate 或 unit-mean
    normalized tuning curves。

    Args:
        tuning_curves: 形状为 ``(n_curves, n_angles)`` 的非负有限曲线，
            或形状为 ``(n_angles,)`` 的单条曲线。

    Returns:
        形状为 ``(n_curves,)`` 的信息量。均值不为正的曲线返回 NaN。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim == 1:
        tuning_curves = tuning_curves[None, :]
    if tuning_curves.ndim != 2 or tuning_curves.shape[1] == 0:
        raise ValueError("tuning_curves must be shaped (n_curves, n_angles)")
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")
    if np.any(tuning_curves < 0.0):
        raise ValueError("tuning_curves must be non-negative")

    means = np.mean(tuning_curves, axis=1)
    information = np.full(tuning_curves.shape[0], np.nan, dtype=float)
    valid_rows = means > 0.0
    ratios = np.zeros_like(tuning_curves)
    ratios[valid_rows] = tuning_curves[valid_rows] / means[valid_rows, None]
    terms = np.zeros_like(ratios)
    positive = ratios > 0.0
    terms[positive] = ratios[positive] * np.log2(ratios[positive])
    information[valid_rows] = np.mean(terms[valid_rows], axis=1)
    return information


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


def empirical_two_point_correlation(tuning_curves):
    """
    Compute the uncentered empirical two-point function from Eq. 4.

    Rows of `tuning_curves` are neurons and columns are angular bins. The
    returned matrix is `tuning_curves.T @ tuning_curves / n_neurons`; no
    subtraction of either neuron-wise or angle-wise means is performed.
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim != 2 or tuning_curves.shape[0] == 0:
        raise ValueError("tuning_curves must contain at least one neuron")
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")

    return np.einsum(
        "ia,ib->ab",
        tuning_curves,
        tuning_curves,
        optimize=False,
    ) / tuning_curves.shape[0]


def relative_circulant_error(matrix):
    """
    Measure a square matrix's relative distance from its circulant projection.

    Periodic diagonals are averaged with `circulant_from_diagonal_means`, and
    the Frobenius norm of the residual is divided by the matrix norm. A value
    of zero indicates exact circular translation symmetry.
    """
    matrix = np.asarray(matrix, dtype=float)
    circulant, _ = circulant_from_diagonal_means(matrix)
    denominator = max(float(np.linalg.norm(matrix)), 1e-12)
    return float(np.linalg.norm(matrix - circulant) / denominator)


def kuiper_uniformity_test_asymptotic(angles_rad, max_terms=100):
    """
    Apply the one-sample Kuiper test for circular uniformity.

    Angles are wrapped onto `[0, 2*pi)`. The statistic is `V = D+ + D-`.
    The p-value uses the standard finite-sample-corrected asymptotic Kuiper
    series, which is appropriate for the session sizes in Figure 4A.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_rad = angles_rad[np.isfinite(angles_rad)]
    if angles_rad.size < 2:
        raise ValueError("at least two finite angles are required")

    uniform = np.sort(np.mod(angles_rad, 2.0 * np.pi) / (2.0 * np.pi))
    n = uniform.size
    ranks = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(ranks / n - uniform))
    d_minus = float(np.max(uniform - (ranks - 1.0) / n))
    statistic = d_plus + d_minus

    root_n = np.sqrt(float(n))
    scaled = (root_n + 0.155 + 0.24 / root_n) * statistic
    terms = np.arange(1, int(max_terms) + 1, dtype=float)
    squared = terms * terms
    p_value = 2.0 * np.sum(
        (4.0 * squared * scaled * scaled - 1.0)
        * np.exp(-2.0 * squared * scaled * scaled)
    )
    return statistic, float(np.clip(p_value, 0.0, 1.0))


def benjamini_hochberg(p_values):
    """
    Return Benjamini-Hochberg adjusted p-values in the original order.

    The monotonic correction is applied from largest to smallest rank, and
    adjusted values are clipped to the probability interval.
    """
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p_values must lie between zero and one")

    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


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


def circular_fourier_derivative(values, period=2.0 * np.pi, axis=-1):
    """
    Differentiate periodic samples using their trigonometric interpolant.

    The head-direction target manifold is defined on a circle. A Fourier
    derivative therefore treats the first and last angular bins as neighbors
    and avoids the artificial boundary introduced by an ordinary gradient.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[axis] < 2:
        raise ValueError("the periodic axis must contain at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or infinite entries")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")

    n_samples = values.shape[axis]
    spacing = float(period) / n_samples
    frequencies = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=spacing)
    reshape = [1] * values.ndim
    reshape[axis] = n_samples
    spectrum = np.fft.fft(values, axis=axis)
    derivative = np.fft.ifft(
        spectrum * (1j * frequencies.reshape(reshape)),
        axis=axis,
    )
    return derivative.real.astype(np.float64)


def optimized_recurrent_velocity_factors(
    phi,
    tau_s=0.05,
    regularization=1e-6,
    activation_beta=2.0,
    enforce_zero_diagonal=True,
    dtype=np.float64,
):
    """
    Build the static and angular-velocity factors from paper Eq. 5.

    The modulated connectivity is represented as
    `J(omega) = J_static + omega * J_velocity`. The velocity target is
    `tau * d x_star / d theta`, so the network flow on the target manifold is
    `d x_star / dt = omega * d x_star / d theta`. Both terms use the same
    regularized inverse tuning kernel and the same `J_ii = 0` constraint.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("phi must be shaped (n_neurons, n_angles)")
    if float(tau_s) <= 0.0:
        raise ValueError("tau_s must be positive")

    n_neurons, n_angles = phi.shape
    x_star = softplus_inverse(phi, beta=activation_beta)
    velocity_target = float(tau_s) * circular_fourier_derivative(
        x_star,
        period=2.0 * np.pi,
        axis=1,
    )
    kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    kernel += (float(regularization) * n_angles / n_neurons) * np.eye(n_angles)
    kernel_inv = invert_spd_cholesky(kernel)
    inv_phi = np.einsum("ab,ib->ai", kernel_inv, phi, optimize=False)

    def factors_for_target(target):
        """
        Solve the shared constrained ridge problem for one target current.
        """
        factor_a = np.einsum("ia,ab->ib", target, kernel_inv, optimize=False)
        if enforce_zero_diagonal:
            numerator = np.sum(target * inv_phi.T, axis=1)
            leverage = np.sum(phi * inv_phi.T, axis=1) / n_neurons
            denom = np.maximum(1.0 - leverage, 1e-12)
            factor_a += (numerator / (n_neurons * denom))[:, None] * inv_phi.T
        factor_b = phi / n_neurons
        diagonal = np.sum(factor_a * factor_b, axis=1)
        return factor_a.astype(dtype), diagonal.astype(dtype)

    static_a, static_diagonal = factors_for_target(x_star)
    velocity_a, velocity_diagonal = factors_for_target(velocity_target)
    factor_b = (phi / n_neurons).astype(dtype)
    return static_a, velocity_a, factor_b, static_diagonal, velocity_diagonal


def overlap_order_parameter(target_rate, population_activity):
    """
    Compute m(theta, t) = N^-1 sum_i phi_i*(theta) phi_i(t).

    `target_rate` is shaped `(n_neurons, n_angles)`. Activity can be one state
    `(n_neurons,)` or a time series `(n_times, n_neurons)`. The returned array
    is always shaped `(n_angles, n_times)`.
    """
    target_rate = np.asarray(target_rate, dtype=np.float64)
    activity = np.asarray(population_activity, dtype=np.float64)
    if target_rate.ndim != 2:
        raise ValueError("target_rate must be shaped (n_neurons, n_angles)")
    if activity.ndim == 1:
        activity = activity[None, :]
    if activity.ndim != 2 or activity.shape[1] != target_rate.shape[0]:
        raise ValueError("population_activity must contain one value per neuron")
    return np.einsum(
        "ia,ti->at",
        target_rate,
        activity,
        optimize=False,
    ) / target_rate.shape[0]


def simulate_velocity_modulated_rate_network(
    static_a,
    velocity_a,
    factor_b,
    static_diagonal,
    velocity_diagonal,
    initial_states,
    angular_velocity,
    tau_s=0.05,
    dt_s=0.001,
    inhibition_c=1.0,
    activation_beta=2.0,
    record_every_s=0.1,
    progress_label=None,
    progress_interval_wall_s=10.0,
):
    """
    Integrate data-derived dynamics with the Eq. 5 velocity-modulated weights.

    `angular_velocity` is sampled once per Euler step and may be shaped
    `(n_steps,)` for one trajectory or `(n_steps, n_trials)` for a batch. The
    function records input-current states at time zero and every requested
    interval, including the final state.
    """
    static_a = np.asarray(static_a, dtype=np.float64)
    velocity_a = np.asarray(velocity_a, dtype=np.float64)
    factor_b = np.asarray(factor_b, dtype=np.float64)
    static_diagonal = np.asarray(static_diagonal, dtype=np.float64)
    velocity_diagonal = np.asarray(velocity_diagonal, dtype=np.float64)
    x = np.asarray(initial_states, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("initial_states must be shaped (n_trials, n_neurons)")
    if static_a.shape != velocity_a.shape or static_a.shape != factor_b.shape:
        raise ValueError("all low-rank factors must have matching shapes")
    if x.shape[1] != static_a.shape[0]:
        raise ValueError("initial_states and factors have different neuron counts")
    if static_diagonal.shape != (x.shape[1],) or velocity_diagonal.shape != (x.shape[1],):
        raise ValueError("diagonal factors must contain one value per neuron")
    if float(tau_s) <= 0.0 or float(dt_s) <= 0.0:
        raise ValueError("tau_s and dt_s must be positive")

    omega = np.asarray(angular_velocity, dtype=np.float64)
    if omega.ndim == 1:
        omega = omega[:, None]
    if omega.ndim != 2:
        raise ValueError("angular_velocity must be one- or two-dimensional")
    if omega.shape[1] == 1 and x.shape[0] > 1:
        omega = np.repeat(omega, x.shape[0], axis=1)
    if omega.shape[1] != x.shape[0]:
        raise ValueError("angular_velocity must have one column per trial")

    n_steps = omega.shape[0]
    record_every = max(1, int(np.round(float(record_every_s) / float(dt_s))))
    drive_scale = float(dt_s) / float(tau_s)
    start_wall = time.perf_counter()
    next_progress_wall = start_wall + float(progress_interval_wall_s)
    times = [0.0]
    trajectory = [x.copy()]

    for step in range(n_steps):
        rates = softplus(x, beta=activation_beta)
        latent = rates @ factor_b
        static_drive = latent @ static_a.T - rates * static_diagonal
        velocity_drive = latent @ velocity_a.T - rates * velocity_diagonal
        recurrent = static_drive + omega[step, :, None] * velocity_drive
        mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
        x += drive_scale * (-x + recurrent - float(inhibition_c) * mean_error)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(
                f"velocity-modulated dynamics became non-finite at step {step + 1}"
            )

        completed = step + 1
        if completed % record_every == 0 or completed == n_steps:
            times.append(completed * float(dt_s))
            trajectory.append(x.copy())

        if progress_label is not None:
            now = time.perf_counter()
            if now >= next_progress_wall or completed == n_steps:
                print(
                    f"[{progress_label}] step {completed}/{n_steps}, "
                    f"t={completed * float(dt_s):.2f}s, "
                    f"elapsed={now - start_wall:.1f}s",
                    flush=True,
                )
                next_progress_wall = now + float(progress_interval_wall_s)

    return np.asarray(times), np.asarray(trajectory, dtype=np.float64)


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


def nearest_circular_manifold_distance(states, manifold, return_l2=False):
    """
    Compute distance to the nearest point on a closed piecewise-linear manifold.

    `manifold[k]` and `manifold[(k + 1) % n_angles]` define one segment of the
    circular target manifold. Minimizing over every segment allows the nearest
    coordinate to lie between sampled angular bins, so tangential drift along
    the ring is not counted as off-manifold distance.

    The returned coordinate is a fractional manifold index: `k + alpha`, where
    `alpha` lies in `[0, 1]` along segment `k`. Distances use the Figure 3F
    normalization, full-state L2 divided by `sqrt(n_neurons)`.
    """
    states = np.asarray(states, dtype=float)
    manifold = np.asarray(manifold, dtype=float)
    if states.ndim != 2 or manifold.ndim != 2:
        raise ValueError("states and manifold must both be 2D")
    if states.shape[1] != manifold.shape[1]:
        raise ValueError("states and manifold must have the same feature count")
    if len(manifold) < 2:
        raise ValueError("a circular manifold requires at least two sampled points")

    nearest_coordinate = np.full(len(states), np.nan, dtype=float)
    nearest_l2 = np.full(len(states), np.nan, dtype=float)
    finite_states = np.all(np.isfinite(states), axis=1)
    finite_segments = np.all(np.isfinite(manifold), axis=1) & np.all(
        np.isfinite(np.roll(manifold, -1, axis=0)),
        axis=1,
    )
    segment_indices = np.flatnonzero(finite_segments)
    if len(segment_indices) == 0:
        if return_l2:
            return nearest_l2, nearest_coordinate, nearest_l2
        return nearest_l2, nearest_coordinate

    starts = manifold[segment_indices]
    vectors = np.roll(manifold, -1, axis=0)[segment_indices] - starts
    start_norm = np.sum(starts * starts, axis=1)
    vector_norm = np.sum(vectors * vectors, axis=1)
    good_rows = np.flatnonzero(finite_states)

    state_chunk_size = 256
    segment_chunk_size = 256
    for state_start in range(0, len(good_rows), state_chunk_size):
        row_idx = good_rows[state_start : state_start + state_chunk_size]
        chunk = states[row_idx]
        state_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        best_squared = np.full(len(row_idx), np.inf, dtype=float)
        best_segment = np.full(len(row_idx), -1, dtype=int)
        best_alpha = np.zeros(len(row_idx), dtype=float)

        for segment_start in range(0, len(segment_indices), segment_chunk_size):
            segment_stop = min(segment_start + segment_chunk_size, len(segment_indices))
            starts_part = starts[segment_start:segment_stop]
            vectors_part = vectors[segment_start:segment_stop]
            vector_norm_part = vector_norm[segment_start:segment_stop]

            state_dot_start = np.einsum("ij,kj->ik", chunk, starts_part, optimize=False)
            state_dot_vector = np.einsum("ij,kj->ik", chunk, vectors_part, optimize=False)
            start_dot_vector = np.sum(starts_part * vectors_part, axis=1)[None, :]
            projection_numerator = state_dot_vector - start_dot_vector
            alpha = np.divide(
                projection_numerator,
                vector_norm_part[None, :],
                out=np.zeros_like(projection_numerator),
                where=vector_norm_part[None, :] > 0,
            )
            alpha = np.clip(alpha, 0.0, 1.0)

            squared = (
                state_norm
                + start_norm[None, segment_start:segment_stop]
                - 2.0 * state_dot_start
                - 2.0 * alpha * projection_numerator
                + alpha * alpha * vector_norm_part[None, :]
            )
            squared = np.maximum(squared, 0.0)
            local_segment = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(len(row_idx)), local_segment]
            improved = local_squared < best_squared
            if np.any(improved):
                best_squared[improved] = local_squared[improved]
                best_segment[improved] = segment_start + local_segment[improved]
                best_alpha[improved] = alpha[np.arange(len(row_idx)), local_segment][improved]

        valid_best = best_segment >= 0
        nearest_l2[row_idx[valid_best]] = np.sqrt(best_squared[valid_best])
        nearest_coordinate[row_idx[valid_best]] = (
            segment_indices[best_segment[valid_best]] + best_alpha[valid_best]
        ) % len(manifold)

    normalized = nearest_l2 / np.sqrt(states.shape[1])
    if return_l2:
        return normalized, nearest_coordinate, nearest_l2
    return normalized, nearest_coordinate


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


def dense_recurrent_drive(weights):
    """
    Create a batched recurrent-drive function for a dense weight matrix.

    Network states are stored row-wise as `(n_trials, n_neurons)`, whereas
    `weights[i, j]` maps presynaptic neuron `j` to postsynaptic neuron `i`.
    The returned function therefore computes `rates @ weights.T`.
    """
    weights_t = np.asarray(weights, dtype=np.float32).T.copy()

    def drive(rates):
        rates = np.asarray(rates, dtype=np.float32)
        return rates @ weights_t

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
