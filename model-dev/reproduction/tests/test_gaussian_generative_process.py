import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import (  # noqa: E402
    circular_flip_correlations,
    circular_peak_counts,
    circular_peak_z_scores,
    fourier_correlation_error,
    generated_tuning_fourier_coefficients,
    head_direction_information_content,
    normalized_softplus_tuning,
    ranked_circular_peak_heights,
    reflect_circular_curves_about_com,
    sample_circular_gaussian_process,
    tuning_fourier_power_coefficients,
    wrapped_gaussian_correlation,
    wrapped_gaussian_fourier_coefficients,
)
from gaussian_generative_process import (  # noqa: E402
    circular_two_point_correlation,
    generate_heterogeneous_tuning_dataset,
    generated_aligned_population_profile,
    generated_peak_count_distribution,
    generated_flip_correlations,
    generated_head_direction_information,
    generated_ranked_peak_heights,
    iter_heterogeneous_tuning_batches,
)
from utils import align_rows_to_circular_com, population_mean_and_std  # noqa: E402


class GaussianGenerativeProcessTests(unittest.TestCase):
    def test_wrapped_gaussian_matches_fourier_series(self):
        """
        Eq. 9 and Eq. 113 should describe the same periodic covariance.
        """
        sigma = 1.42
        delta = np.linspace(-np.pi, np.pi, 101)
        frequencies = np.arange(-40, 41)
        coefficients = wrapped_gaussian_fourier_coefficients(frequencies, sigma)
        fourier = np.sum(
            coefficients[:, None] * np.exp(1j * frequencies[:, None] * delta[None, :]),
            axis=0,
        ).real
        direct = wrapped_gaussian_correlation(delta, sigma)
        np.testing.assert_allclose(fourier, direct, rtol=1e-11, atol=1e-12)

    def test_larger_sigma_narrows_covariance(self):
        """
        Increasing sigma must reduce off-center covariance, as in Figure 5A.
        """
        delta = np.asarray([0.0, 1.0])
        broad = wrapped_gaussian_correlation(delta, sigma=0.8)
        narrow = wrapped_gaussian_correlation(delta, sigma=2.25)
        self.assertTrue(np.isclose(broad[0], 1.0, atol=1e-5))
        self.assertTrue(np.isclose(narrow[0], 1.0, atol=1e-12))
        self.assertLess(narrow[1], broad[1])

    def test_fourier_sampler_recovers_target_covariance(self):
        """
        Many Fourier samples should reproduce the specified circular covariance.
        """
        n_samples = 12000
        n_angles = 64
        sigma = 1.6
        theta, samples = sample_circular_gaussian_process(
            n_samples=n_samples,
            n_angles=n_angles,
            sigma=sigma,
            seed=1234,
        )
        empirical = np.mean(samples[:, :1] * samples, axis=0)
        target = wrapped_gaussian_correlation(theta, sigma)
        np.testing.assert_allclose(empirical, target, atol=0.035, rtol=0.06)

    def test_normalized_softplus_has_unit_angular_mean(self):
        """
        Eq. B3 normalization should set every sampled tuning curve's mean to one.
        """
        rng = np.random.default_rng(10)
        currents = rng.normal(size=(8, 256))
        rates = normalized_softplus_tuning(currents, beta=2.76, bias=1.73)
        np.testing.assert_allclose(np.mean(rates, axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(rates > 0.0))

    def test_fourier_power_matches_circulant_correlation_transform(self):
        """
        Direct power averaging must equal Fourier transforming Gamma(delta).
        """
        rng = np.random.default_rng(31)
        tuning = rng.lognormal(size=(17, 64))
        tuning /= np.mean(tuning, axis=1, keepdims=True)
        direct = tuning_fourier_power_coefficients(tuning, n_modes=12)

        correlation_by_lag = np.asarray(
            [
                np.mean(tuning * np.roll(tuning, -lag, axis=1))
                for lag in range(tuning.shape[1])
            ]
        )
        transformed = np.fft.rfft(correlation_by_lag) / tuning.shape[1]
        np.testing.assert_allclose(direct, transformed[1:13].real, atol=1e-12)

    def test_generated_coefficients_and_error_are_self_consistent(self):
        """
        Equal generated and target coefficients must produce exactly zero loss.
        """
        _, currents = sample_circular_gaussian_process(
            n_samples=128,
            n_angles=64,
            sigma=1.42,
            seed=8,
        )
        coefficients = generated_tuning_fourier_coefficients(
            currents,
            beta=2.76,
            bias=1.73,
            n_modes=10,
        )
        self.assertEqual(fourier_correlation_error(coefficients, coefficients), 0.0)

    def test_dataset_api_is_reproducible_and_unit_normalized(self):
        """
        学习阶段使用的数据集接口应当可复现，并保持每条曲线角度均值为 1。
        """
        first = generate_heterogeneous_tuning_dataset(
            n_neurons=11,
            n_angles=64,
            sigma=1.4,
            beta=2.6,
            bias=2.08,
            seed=44,
        )
        second = generate_heterogeneous_tuning_dataset(
            n_neurons=11,
            n_angles=64,
            sigma=1.4,
            beta=2.6,
            bias=2.08,
            seed=44,
        )
        np.testing.assert_array_equal(first.input_currents, second.input_currents)
        np.testing.assert_array_equal(first.firing_rates, second.firing_rates)
        self.assertIs(first.x_star, first.input_currents)
        self.assertIs(first.phi_star, first.firing_rates)
        np.testing.assert_allclose(
            np.mean(first.firing_rates, axis=1),
            1.0,
            atol=2e-7,
        )
        self.assertEqual(first.firing_rates.shape, (11, 64))

    def test_streaming_batches_preserve_requested_total(self):
        """
        流式生成器的最后一个 batch 应自动截断到指定总神经元数。
        """
        batches = list(
            iter_heterogeneous_tuning_batches(
                total_neurons=10,
                batch_size=4,
                n_angles=32,
                sigma=1.4,
                beta=2.6,
                bias=2.08,
                seed=9,
            )
        )
        self.assertEqual([batch.n_neurons for batch in batches], [4, 4, 2])
        self.assertEqual(sum(batch.n_neurons for batch in batches), 10)

    def test_circular_two_point_correlation_matches_direct_lag_average(self):
        """
        FFT 实现应与逐 lag 计算的未中心化相关函数完全一致。
        """
        rng = np.random.default_rng(7)
        tuning = rng.uniform(size=(13, 32))
        fft_result = circular_two_point_correlation(tuning)
        direct = np.asarray(
            [
                np.mean(tuning * np.roll(tuning, -lag, axis=1))
                for lag in range(tuning.shape[1])
            ]
        )
        np.testing.assert_allclose(fft_result, direct, atol=1e-12)

    def test_streamed_aligned_profile_matches_full_population_statistics(self):
        """
        Figure 6A-B 的流式矩统计应等于一次性生成后的直接均值和标准差。
        """
        parameters = {
            "total_neurons": 23,
            "n_angles": 32,
            "sigma": 1.4,
            "beta": 2.6,
            "bias": 2.08,
            "seed": 71,
        }
        _, streamed_mean, streamed_std = generated_aligned_population_profile(
            batch_size=7,
            **parameters,
        )
        dataset = generate_heterogeneous_tuning_dataset(
            n_neurons=parameters["total_neurons"],
            n_angles=parameters["n_angles"],
            sigma=parameters["sigma"],
            beta=parameters["beta"],
            bias=parameters["bias"],
            seed=parameters["seed"],
            dtype=np.float64,
        )
        aligned = align_rows_to_circular_com(
            dataset.firing_rates,
            angles_rad=dataset.theta_rad,
        )
        expected_mean, expected_std = population_mean_and_std(aligned)
        np.testing.assert_allclose(streamed_mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(streamed_std, expected_std, atol=1e-12)

    def test_circular_peak_detection_wraps_across_boundary(self):
        """
        Figure 6C 峰检测必须把最后和第一个角度 bin 视为相邻位置。
        """
        tuning = np.array(
            [
                [4.0, 1.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 3.0],
            ]
        )
        peak_mask, peak_z = circular_peak_z_scores(tuning)
        np.testing.assert_array_equal(
            peak_mask,
            np.array(
                [
                    [True, False, False, False],
                    [False, True, False, True],
                ]
            ),
        )
        self.assertTrue(np.isfinite(peak_z[0, 0]))
        self.assertTrue(np.isnan(peak_z[0, 1]))

    def test_peak_count_uses_strict_curvewise_z_threshold(self):
        """
        只有局部峰自身 z-score 严格大于阈值时才应计入峰数。
        """
        tuning = np.array([[0.0, 2.0, 0.0, 1.0]])
        _, peak_z = circular_peak_z_scores(tuning)
        lower_peak_z = float(peak_z[0, 3])
        counts = circular_peak_counts(
            tuning,
            [lower_peak_z - 1e-12, lower_peak_z],
        )
        np.testing.assert_array_equal(counts[:, 0], np.array([2, 1]))

    def test_streamed_peak_distribution_matches_full_generation(self):
        """
        Figure 6C 的流式峰数分布应等于一次性生成后的直接统计。
        """
        parameters = {
            "total_neurons": 29,
            "n_angles": 32,
            "sigma": 1.4,
            "beta": 2.6,
            "bias": 2.08,
            "seed": 81,
        }
        thresholds = np.array([1.0, 0.5, 0.0])
        streamed, overflow = generated_peak_count_distribution(
            batch_size=7,
            z_thresholds=thresholds,
            max_peak_count=5,
            **parameters,
        )
        dataset = generate_heterogeneous_tuning_dataset(
            n_neurons=parameters["total_neurons"],
            n_angles=parameters["n_angles"],
            sigma=parameters["sigma"],
            beta=parameters["beta"],
            bias=parameters["bias"],
            seed=parameters["seed"],
            dtype=np.float64,
        )
        counts = circular_peak_counts(dataset.firing_rates, thresholds)
        expected = np.stack(
            [
                np.bincount(row, minlength=6)[:6] / row.size
                for row in counts
            ]
        )
        np.testing.assert_allclose(streamed, expected, atol=1e-12)
        np.testing.assert_array_equal(overflow, np.zeros(3, dtype=int))

    def test_ranked_peak_heights_are_descending_and_thresholded(self):
        """
        Figure 6D 应按峰高而非角度顺序排列，并保留缺失 rank 为 NaN。
        """
        tuning = np.array(
            [
                [0.0, 4.0, 0.0, 2.0, 0.0, 3.0],
                [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        ranked = ranked_circular_peak_heights(
            tuning,
            z_threshold=0.0,
            n_ranks=3,
        )
        np.testing.assert_allclose(ranked[0], np.array([4.0, 3.0, 2.0]))
        self.assertEqual(float(ranked[1, 0]), 5.0)
        self.assertTrue(np.isnan(ranked[1, 1:]).all())

    def test_streamed_ranked_heights_match_full_generation(self):
        """
        Figure 6D 的流式峰高提取应等于一次性生成后的直接结果。
        """
        parameters = {
            "total_neurons": 31,
            "n_angles": 32,
            "sigma": 1.4,
            "beta": 2.6,
            "bias": 2.08,
            "seed": 91,
        }
        streamed = generated_ranked_peak_heights(
            batch_size=8,
            z_threshold=1.0,
            n_ranks=3,
            **parameters,
        )
        dataset = generate_heterogeneous_tuning_dataset(
            n_neurons=parameters["total_neurons"],
            n_angles=parameters["n_angles"],
            sigma=parameters["sigma"],
            beta=parameters["beta"],
            bias=parameters["bias"],
            seed=parameters["seed"],
            dtype=np.float64,
        )
        expected = ranked_circular_peak_heights(
            dataset.firing_rates,
            z_threshold=1.0,
            n_ranks=3,
        )
        np.testing.assert_allclose(streamed, expected, equal_nan=True)

    def test_flip_correlation_is_one_for_com_symmetric_curve(self):
        """
        围绕自身 COM 对称的周期曲线应具有接近 1 的 flip correlation。
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        center = 0.37
        distance = np.angle(np.exp(1j * (theta - center)))
        tuning = np.exp(-0.5 * (distance / 0.45) ** 2)[None, :]
        reflected, com = reflect_circular_curves_about_com(tuning, theta)
        correlation = circular_flip_correlations(tuning, theta)
        self.assertLess(abs(np.angle(np.exp(1j * (com[0] - center)))), 1e-3)
        np.testing.assert_allclose(reflected, tuning, atol=2e-3)
        self.assertGreater(float(correlation[0]), 0.99999)

    def test_flip_correlation_detects_asymmetric_curve(self):
        """
        加入偏心次峰应显著降低围绕 COM 的镜像相关。
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        main = np.exp(-0.5 * (np.angle(np.exp(1j * theta)) / 0.5) ** 2)
        side = 0.7 * np.exp(
            -0.5 * (np.angle(np.exp(1j * (theta - 1.4))) / 0.25) ** 2
        )
        correlation = circular_flip_correlations((main + side)[None, :])
        self.assertLess(float(correlation[0]), 0.9)

    def test_streamed_flip_correlations_match_full_generation(self):
        """
        Figure 6E 的流式相关应等于一次性生成后的直接计算。
        """
        parameters = {
            "total_neurons": 27,
            "n_angles": 32,
            "sigma": 1.4,
            "beta": 2.6,
            "bias": 2.08,
            "seed": 101,
        }
        streamed = generated_flip_correlations(
            batch_size=7,
            **parameters,
        )
        dataset = generate_heterogeneous_tuning_dataset(
            n_neurons=parameters["total_neurons"],
            n_angles=parameters["n_angles"],
            sigma=parameters["sigma"],
            beta=parameters["beta"],
            bias=parameters["bias"],
            seed=parameters["seed"],
            dtype=np.float64,
        )
        expected = circular_flip_correlations(
            dataset.firing_rates,
            angles_rad=dataset.theta_rad,
        )
        np.testing.assert_allclose(streamed, expected, atol=1e-12)

    def test_hd_information_is_zero_for_uniform_curve(self):
        """
        完全均匀的调谐曲线不携带头朝向信息。
        """
        information = head_direction_information_content(np.ones((2, 16)))
        np.testing.assert_allclose(information, np.zeros(2), atol=1e-15)

    def test_hd_information_one_bin_code_equals_log2_bin_count(self):
        """
        所有放电集中在一个等概率角度 bin 时信息量应为 log2(N_theta)。
        """
        tuning = np.zeros((1, 8))
        tuning[0, 3] = 8.0
        information = head_direction_information_content(tuning)
        self.assertAlmostEqual(float(information[0]), 3.0, places=12)

    def test_hd_information_is_scale_invariant_and_handles_zeros(self):
        """
        bits/spike 信息量应对 firing-rate 整体缩放不变，且允许零 bin。
        """
        tuning = np.array([[0.0, 1.0, 3.0, 0.0]])
        first = head_direction_information_content(tuning)
        second = head_direction_information_content(7.5 * tuning)
        np.testing.assert_allclose(first, second, atol=1e-15)
        self.assertTrue(np.isfinite(first).all())

    def test_streamed_hd_information_matches_full_generation(self):
        """
        Figure 6F 的流式信息量应等于一次性生成后的直接计算。
        """
        parameters = {
            "total_neurons": 33,
            "n_angles": 32,
            "sigma": 1.4,
            "beta": 2.6,
            "bias": 2.08,
            "seed": 111,
        }
        streamed = generated_head_direction_information(
            batch_size=8,
            **parameters,
        )
        dataset = generate_heterogeneous_tuning_dataset(
            n_neurons=parameters["total_neurons"],
            n_angles=parameters["n_angles"],
            sigma=parameters["sigma"],
            beta=parameters["beta"],
            bias=parameters["bias"],
            seed=parameters["seed"],
            dtype=np.float64,
        )
        expected = head_direction_information_content(dataset.firing_rates)
        np.testing.assert_allclose(streamed, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
