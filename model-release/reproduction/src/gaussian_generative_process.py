from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import (
    align_rows_to_circular_com,
    circular_flip_correlations,
    circular_peak_counts,
    head_direction_information_content,
    normalized_softplus_tuning,
    ranked_circular_peak_heights,
    sample_circular_gaussian_process,
)


@dataclass(frozen=True)
class HeterogeneousTuningDataset:
    """
    Clark Gaussian generative process 生成的一组异质性环形调谐曲线。

    Attributes:
        theta_rad: 形状为 ``(n_angles,)`` 的环形角度网格，范围为 ``[0, 2π)``。
        input_currents: 形状为 ``(n_neurons, n_angles)`` 的 Gaussian process
            输入电流 ``x_star(theta)``。
        firing_rates: 形状为 ``(n_neurons, n_angles)`` 的归一化 firing-rate
            调谐曲线 ``phi_star(theta)``，每行沿角度的均值为 1。
        sigma: wrapped-Gaussian covariance 的 Fourier decay scale。
        beta: normalized softplus 的陡峭度。
        bias: normalized softplus 的软阈值 ``b``。
        seed: 用于生成该数据集的随机种子。
    """

    theta_rad: np.ndarray
    input_currents: np.ndarray
    firing_rates: np.ndarray
    sigma: float
    beta: float
    bias: float
    seed: int | None

    @property
    def n_neurons(self):
        """
        返回数据集中的神经元数量。
        """
        return int(self.firing_rates.shape[0])

    @property
    def n_angles(self):
        """
        返回 teacher manifold 的角度离散点数量。
        """
        return int(self.firing_rates.shape[1])

    @property
    def x_star(self):
        """
        返回与 Clark 记号一致的目标输入电流 ``x_star(theta)``。
        """
        return self.input_currents

    @property
    def phi_star(self):
        """
        返回与 Clark 记号一致的目标 firing rates ``phi_star(theta)``。
        """
        return self.firing_rates

    def save_npz(self, path, compressed=True):
        """
        将假设性调谐曲线数据集保存为可供 learning 子项目读取的 NPZ。

        Args:
            path: 输出文件路径。
            compressed: 为真时使用 ``np.savez_compressed``。

        Returns:
            实际写入的 ``Path``。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if compressed else np.savez
        save(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            theta_rad=self.theta_rad,
            input_currents=self.input_currents,
            firing_rates=self.firing_rates,
            x_star=self.input_currents,
            phi_star=self.firing_rates,
            sigma=np.asarray(self.sigma),
            beta=np.asarray(self.beta),
            bias=np.asarray(self.bias),
            seed=np.asarray(-1 if self.seed is None else self.seed, dtype=np.int64),
            normalization=np.asarray("unit_angular_mean"),
            source=np.asarray("Clark Gaussian generative process, Eq. 9-10/B3"),
        )
        return path


def generate_heterogeneous_tuning_dataset(
    n_neurons,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
    dtype=np.float32,
):
    """
    批量生成可作为 teacher manifold 的异质性头朝向调谐曲线。

    该函数严格采用 Clark 原文的两步生成过程：首先按 Eq. 9 从环形
    Gaussian process 采样 ``x_star(theta)``，随后按 Eq. 10/B3 应用
    normalized softplus 得到 ``phi_star(theta)``。输出数组约定为
    ``(n_neurons, n_angles)``，可直接作为后续 batch solver、local learning
    或 RNN target manifold 的输入。

    Args:
        n_neurons: 要生成的神经元数量。
        n_angles: ``[0, 2π)`` 上的角度离散点数量。
        sigma: Gaussian process covariance 的 Fourier decay scale。
        beta: normalized softplus 的陡峭度。
        bias: normalized softplus 的软阈值。
        seed: 可选随机种子；相同参数和种子产生完全相同的数据集。
        dtype: 返回电流和 firing-rate 数组的数据类型。

    Returns:
        ``HeterogeneousTuningDataset``，同时包含 latent currents 和 rates。
    """
    n_neurons = int(n_neurons)
    n_angles = int(n_angles)
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if n_angles <= 1:
        raise ValueError("n_angles must exceed one")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")

    theta, currents = sample_circular_gaussian_process(
        n_samples=n_neurons,
        n_angles=n_angles,
        sigma=float(sigma),
        seed=seed,
    )
    rates = normalized_softplus_tuning(
        currents,
        beta=float(beta),
        bias=float(bias),
        axis=-1,
    )
    return HeterogeneousTuningDataset(
        theta_rad=theta.astype(dtype, copy=False),
        input_currents=currents.astype(dtype, copy=False),
        firing_rates=rates.astype(dtype, copy=False),
        sigma=float(sigma),
        beta=float(beta),
        bias=float(bias),
        seed=None if seed is None else int(seed),
    )


def iter_heterogeneous_tuning_batches(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
    dtype=np.float32,
):
    """
    分批生成大规模异质性调谐曲线，避免一次持有完整数据集。

    随机数生成器在各 batch 之间连续推进，因此每个神经元都是独立样本。
    该接口适合在线 local learning、Monte Carlo 统计以及 Figure 5F 中
    ``N_samples=100,000`` 的生成相关函数估计。

    Args:
        total_neurons: 所有 batch 合计的神经元数量。
        batch_size: 单个 batch 的最大神经元数量。
        n_angles: 环形角度离散点数量。
        sigma: Gaussian process covariance 参数。
        beta: normalized softplus 的陡峭度。
        bias: normalized softplus 的软阈值。
        seed: 可选随机种子。
        dtype: 每个 batch 中电流和 firing-rate 数组的数据类型。

    Yields:
        连续的 ``HeterogeneousTuningDataset`` batch。
    """
    total_neurons = int(total_neurons)
    batch_size = int(batch_size)
    n_angles = int(n_angles)
    if total_neurons <= 0 or batch_size <= 0:
        raise ValueError("total_neurons and batch_size must be positive")

    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    generated = 0
    while generated < total_neurons:
        current_size = min(batch_size, total_neurons - generated)
        white_noise = rng.normal(size=(current_size, n_angles))
        _, currents = sample_circular_gaussian_process(
            n_samples=current_size,
            n_angles=n_angles,
            sigma=float(sigma),
            white_noise=white_noise,
        )
        rates = normalized_softplus_tuning(
            currents,
            beta=float(beta),
            bias=float(bias),
            axis=-1,
        )
        yield HeterogeneousTuningDataset(
            theta_rad=theta.astype(dtype, copy=False),
            input_currents=currents.astype(dtype, copy=False),
            firing_rates=rates.astype(dtype, copy=False),
            sigma=float(sigma),
            beta=float(beta),
            bias=float(bias),
            seed=None if seed is None else int(seed),
        )
        generated += current_size


def circular_two_point_correlation(tuning_curves):
    """
    计算随角度差变化的未中心化二点相关函数 ``Gamma_phi(delta_theta)``。

    Args:
        tuning_curves: 形状为 ``(n_neurons, n_angles)`` 的调谐曲线。

    Returns:
        形状为 ``(n_angles,)`` 的相关函数。第 ``k`` 项对应角度差
        ``2πk / n_angles``，且不进行均值扣除。
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim != 2 or tuning_curves.shape[0] == 0:
        raise ValueError("tuning_curves must be shaped (n_neurons, n_angles)")
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")

    spectrum = np.fft.fft(tuning_curves, axis=-1)
    autocorrelation = np.fft.ifft(
        np.abs(spectrum) ** 2,
        axis=-1,
    ).real / tuning_curves.shape[1]
    return np.mean(autocorrelation, axis=0)


def generated_two_point_correlation(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
):
    """
    使用流式生成器估计大样本极限下的 ``Gamma_phi(delta_theta)``。

    Args:
        total_neurons: Monte Carlo 神经元样本总数。
        batch_size: 每批生成的神经元数。
        n_angles: 环形角度离散点数量。
        sigma: Gaussian process covariance 参数。
        beta: normalized softplus 的陡峭度。
        bias: normalized softplus 的软阈值。
        seed: 可选随机种子。

    Returns:
        ``(theta_lag_rad, correlation)``；两者形状均为 ``(n_angles,)``。
    """
    correlation_sum = np.zeros(int(n_angles), dtype=np.float64)
    count = 0
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        batch_count = batch.n_neurons
        correlation_sum += batch_count * circular_two_point_correlation(
            batch.firing_rates
        )
        count += batch_count
    theta_lag = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    return theta_lag, correlation_sum / count


def generated_aligned_population_profile(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
):
    """
    流式估计 COM 对齐后生成调谐曲线的均值和总体标准差。

    Figure 6A-B 对生成曲线执行与实验数据 Figure 2E-F 相同的操作：
    每条 unit-mean 调谐曲线先按 circular center of mass 对齐，再跨样本
    计算逐角度均值和总体标准差。函数累计一阶矩与二阶矩，因此无需在
    内存中同时保存 ``N_samples=100,000`` 条曲线。

    Args:
        total_neurons: Gaussian generative process 的样本总数。
        batch_size: 每批生成和对齐的神经元数量。
        n_angles: ``[0, 2π)`` 上的角度 bin 数。
        sigma: Gaussian process covariance 的 Fourier decay scale。
        beta: normalized softplus 的陡峭度。
        bias: normalized softplus 的软阈值。
        seed: 可选随机种子。

    Returns:
        ``(delta_theta_rad, mean_profile, std_profile)``。角度范围为
        ``[-π, π)``，零点位于数组中央；标准差采用总体定义（分母为 N）。
    """
    total_neurons = int(total_neurons)
    batch_size = int(batch_size)
    n_angles = int(n_angles)
    if total_neurons <= 0 or batch_size <= 0:
        raise ValueError("total_neurons and batch_size must be positive")
    if n_angles <= 1:
        raise ValueError("n_angles must exceed one")

    value_sum = np.zeros(n_angles, dtype=np.float64)
    squared_sum = np.zeros(n_angles, dtype=np.float64)
    count = 0
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        aligned = align_rows_to_circular_com(
            batch.firing_rates,
            angles_rad=batch.theta_rad,
        )
        value_sum += np.sum(aligned, axis=0, dtype=np.float64)
        squared_sum += np.sum(aligned * aligned, axis=0, dtype=np.float64)
        count += aligned.shape[0]

    mean_profile = value_sum / count
    variance_profile = squared_sum / count - mean_profile * mean_profile
    std_profile = np.sqrt(np.maximum(variance_profile, 0.0))
    delta_theta = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return delta_theta, mean_profile, std_profile


def generated_peak_count_distribution(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    z_thresholds,
    max_peak_count,
    seed=None,
):
    """
    流式估计 Figure 6C 的生成调谐曲线峰数分布。

    每批生成 normalized-softplus 调谐曲线后，直接按环形局部极大值及
    曲线内 z-score 计数。输出包含从 0 到 ``max_peak_count`` 的概率；
    更大的峰数会单独报告为 overflow，而不会静默截断到末尾类别。

    Args:
        total_neurons: 生成曲线总数。
        batch_size: 单批曲线数量。
        n_angles: 环形角度 bin 数。
        sigma: Gaussian process Fourier decay scale。
        beta: normalized softplus 陡峭度。
        bias: normalized softplus 软阈值。
        z_thresholds: Figure 6C 使用的 z-score 阈值。
        max_peak_count: 显式统计的最大峰数。
        seed: 可选随机种子。

    Returns:
        ``(probabilities, overflow_counts)``。概率形状为
        ``(n_thresholds, max_peak_count + 1)``。
    """
    thresholds = np.atleast_1d(np.asarray(z_thresholds, dtype=float))
    max_peak_count = int(max_peak_count)
    if max_peak_count < 0:
        raise ValueError("max_peak_count must be non-negative")

    histograms = np.zeros(
        (len(thresholds), max_peak_count + 1),
        dtype=np.int64,
    )
    overflow = np.zeros(len(thresholds), dtype=np.int64)
    count = 0
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        batch_counts = circular_peak_counts(batch.firing_rates, thresholds)
        for threshold_index, counts in enumerate(batch_counts):
            in_range = counts <= max_peak_count
            histograms[threshold_index] += np.bincount(
                counts[in_range],
                minlength=max_peak_count + 1,
            )
            overflow[threshold_index] += int(np.sum(~in_range))
        count += batch.n_neurons

    return histograms / count, overflow


def generated_ranked_peak_heights(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    z_threshold,
    n_ranks=3,
    seed=None,
):
    """
    流式生成 Figure 6D 所需的最高若干个合格峰高度。

    每个 batch 按 Figure 6C 的环形峰定义筛选 ``z > z_threshold`` 的峰，
    再在每条曲线内部按峰高降序排列。最终仅保存 ``n_ranks`` 列，因而
    即使生成 100,000 条曲线也不需要保留完整调谐曲线矩阵。

    Args:
        total_neurons: 生成曲线总数。
        batch_size: 每批曲线数量。
        n_angles: 环形角度 bin 数。
        sigma: Gaussian process Fourier decay scale。
        beta: normalized softplus 陡峭度。
        bias: normalized softplus 软阈值。
        z_threshold: 峰检测的曲线内 z-score 阈值。
        n_ranks: 每条曲线保留的最高峰数量。
        seed: 可选随机种子。

    Returns:
        形状为 ``(total_neurons, n_ranks)`` 的峰高矩阵；缺失 rank 为 NaN。
    """
    ranked_batches = []
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        ranked_batches.append(
            ranked_circular_peak_heights(
                batch.firing_rates,
                z_threshold=z_threshold,
                n_ranks=n_ranks,
            )
        )
    return np.vstack(ranked_batches)


def generated_flip_correlations(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
):
    """
    流式生成 Figure 6E 的 COM 镜像相关系数。

    Args:
        total_neurons: 生成曲线总数。
        batch_size: 每批曲线数量。
        n_angles: 环形角度 bin 数。
        sigma: Gaussian process Fourier decay scale。
        beta: normalized softplus 陡峭度。
        bias: normalized softplus 软阈值。
        seed: 可选随机种子。

    Returns:
        形状为 ``(total_neurons,)`` 的 ``rho_flip`` 数组。
    """
    correlations = []
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        correlations.append(
            circular_flip_correlations(
                batch.firing_rates,
                angles_rad=batch.theta_rad,
            )
        )
    return np.concatenate(correlations)


def generated_head_direction_information(
    total_neurons,
    batch_size,
    n_angles,
    sigma,
    beta,
    bias,
    seed=None,
):
    """
    流式生成 Figure 6F 的头朝向信息量（bits/spike）。

    Args:
        total_neurons: 生成曲线总数。
        batch_size: 每批曲线数量。
        n_angles: 环形角度 bin 数。
        sigma: Gaussian process Fourier decay scale。
        beta: normalized softplus 陡峭度。
        bias: normalized softplus 软阈值。
        seed: 可选随机种子。

    Returns:
        形状为 ``(total_neurons,)`` 的 ``I_HD`` 数组。
    """
    information_batches = []
    for batch in iter_heterogeneous_tuning_batches(
        total_neurons=total_neurons,
        batch_size=batch_size,
        n_angles=n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=seed,
        dtype=np.float64,
    ):
        information_batches.append(
            head_direction_information_content(batch.firing_rates)
        )
    return np.concatenate(information_batches)
