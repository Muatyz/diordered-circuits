import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from compute_hd_tuning import (  # noqa: E402
    behavior_dt_seconds,
    bin_head_direction,
    choose_poisson_cv_sigmas,
    occupancy_seconds,
    spike_counts_by_hd,
)
from reproduction_config import load_figure3_config  # noqa: E402
from utils import (  # noqa: E402
    benjamini_hochberg,
    circular_fourier_derivative,
    circulant_recurrent_drive,
    current_space_jacobian,
    dense_recurrent_drive,
    empirical_two_point_correlation,
    fixed_point_residual_from_weights,
    kuiper_uniformity_test_asymptotic,
    materialize_lowrank_weights,
    nearest_circular_manifold_distance,
    nearest_manifold_distance,
    overlap_order_parameter,
    optimized_recurrent_factors,
    optimized_recurrent_velocity_factors,
    prepare_phi_star_for_inverse,
    relative_circulant_error,
    simulate_velocity_modulated_rate_network,
    softplus,
    softplus_derivative_from_phi,
    softplus_inverse,
)


class HeadDirectionTuningTests(unittest.TestCase):
    def test_figure3_shared_config_has_consistent_network_parameters(self):
        """
        Figure 3A-L must share one explicit network parameter source.
        """
        config = load_figure3_config()
        network = config["network"]
        self.assertEqual(network["regularization"], 1e-4)
        self.assertEqual(network["inhibition_c"], 1.0)
        self.assertEqual(network["tau_s"], 0.05)
        self.assertEqual(network["dt_s"], 0.001)
        self.assertEqual(config["panels_jkl"]["velocity_bin_s"], 1.0)
        self.assertEqual(config["panels_jkl"]["neural_spike_bin_s"], 0.1)

    def test_bin_head_direction_wraps_to_valid_bins(self):
        """
        Head-direction binning should wrap negative and 2pi angles onto the ring.
        """
        angles = np.array([-0.01, 0.0, 0.49 * np.pi, 2 * np.pi, 2 * np.pi + 0.01])
        bins, _ = bin_head_direction(angles, n_bins=4)
        np.testing.assert_array_equal(bins, np.array([3, 0, 0, 0, 0]))

    def test_behavior_dt_replaces_partition_gaps_with_typical_interval(self):
        """
        Non-contiguous CV partitions should not assign gap duration to occupancy.
        """
        time_s = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        np.testing.assert_allclose(behavior_dt_seconds(time_s), np.ones(5))

    def test_spike_counts_do_not_interpolate_across_temporal_gap(self):
        """
        Spikes that fall between alternating CV segments must be discarded.
        """
        behavior = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 10.0, 11.0],
                "head_direction_rad": [0.1, 0.1, 0.1, np.pi, np.pi],
            }
        )
        counts = spike_counts_by_hd(np.array([0.5, 6.0, 10.5]), behavior, n_bins=4)
        np.testing.assert_array_equal(counts, np.array([1.0, 0.0, 1.0, 0.0]))

        occ, _, _ = occupancy_seconds(behavior, n_bins=4)
        self.assertAlmostEqual(float(occ.sum()), 5.0)

    def test_softplus_inverse_round_trip(self):
        """
        The target-current conversion should invert the saved positive rates.
        """
        rates = np.array([1e-12, 1e-6, 0.1, 1.0, 10.0])
        recovered = softplus(softplus_inverse(rates, beta=2.0), beta=2.0)
        np.testing.assert_allclose(recovered, rates, rtol=1e-10, atol=1e-12)

    def test_softplus_inverse_rejects_nonpositive_rates(self):
        """
        Floors must be explicit preprocessing, not hidden inverse clipping.
        """
        with self.assertRaises(ValueError):
            softplus_inverse(np.array([0.0, 1.0]), beta=2.0)

    def test_prepare_phi_positive_and_normalized(self):
        """
        The preprocessing floor should keep unit means and strict positivity.
        """
        phi_raw = np.array(
            [
                [0.0, 2.0, 0.0, 2.0],
                [1.0, 0.5, 2.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        phi_safe, info = prepare_phi_star_for_inverse(phi_raw, alpha_floor=1e-4)

        self.assertEqual(phi_safe.shape, (2, 4))
        self.assertEqual(info["n_removed_neurons"], 1)
        self.assertGreater(float(np.min(phi_safe)), 0.0)
        np.testing.assert_allclose(np.mean(phi_safe, axis=1), np.ones(2), atol=1e-10)
        np.testing.assert_allclose(np.mean(phi_safe, axis=0), np.ones(4), atol=1e-10)

    def test_no_inf_in_x_star_after_preprocessing(self):
        """
        Exact zeros in raw tuning should no longer produce infinite currents.
        """
        phi_raw = np.array([[0.0, 2.0, 0.0, 2.0], [1.5, 0.0, 1.5, 1.0]])
        phi_safe, _ = prepare_phi_star_for_inverse(phi_raw, alpha_floor=1e-4)
        x_star = softplus_inverse(phi_safe, beta=2.0)

        self.assertTrue(np.isfinite(x_star).all())
        self.assertGreater(float(np.min(phi_safe)), 0.0)

    def test_jacobian_matches_finite_difference(self):
        """
        The current-space Jacobian must be -I + J @ diag(phi_prime).
        """
        weights = np.array([[0.2, -0.4], [0.5, 0.1]], dtype=float)
        x0 = np.array([-0.7, 0.4], dtype=float)
        phi0 = softplus(x0, beta=2.0)
        jacobian = current_space_jacobian(weights, phi0, beta=2.0, inhibition_c=0.3)
        rng = np.random.default_rng(123)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)

        def rhs(x):
            rates = softplus(x, beta=2.0)
            return -x + (weights - 0.3 / 2.0) @ rates + 0.3

        eps = 1e-6
        finite_difference = (rhs(x0 + eps * direction) - rhs(x0)) / eps
        analytic = jacobian @ direction
        rel_error = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(finite_difference)
        self.assertLess(rel_error, 1e-5)

    def test_fixed_point_residual_is_small(self):
        """
        Unconstrained A2 low-rank weights should match the target manifold.
        """
        rng = np.random.default_rng(20260617)
        phi_raw = rng.gamma(shape=2.0, scale=1.0, size=(8, 3))
        phi_safe, _ = prepare_phi_star_for_inverse(phi_raw, alpha_floor=1e-4)
        x_star = softplus_inverse(phi_safe, beta=2.0)
        factor_a, factor_b, diagonal = optimized_recurrent_factors(
            phi_safe,
            regularization=0.0,
            activation_beta=2.0,
            enforce_zero_diagonal=False,
        )
        weights = materialize_lowrank_weights(factor_a, factor_b, diagonal=None, dtype=np.float64)
        residual = fixed_point_residual_from_weights(weights, x_star, phi_safe)
        self.assertLess(float(np.sqrt(np.mean(residual * residual))), 1e-6)

    def test_softplus_derivative_from_phi_identity(self):
        """
        The phi-based derivative should equal sigmoid(beta * x_star).
        """
        rates = np.array([1e-4, 0.2, 1.0, 4.0])
        x_star = softplus_inverse(rates, beta=2.0)
        from_x = (softplus(x_star + 1e-6, beta=2.0) - softplus(x_star - 1e-6, beta=2.0)) / 2e-6
        np.testing.assert_allclose(softplus_derivative_from_phi(rates, beta=2.0), from_x, rtol=1e-6, atol=1e-8)

    def test_circular_manifold_distance_allows_between_bin_drift(self):
        """
        Motion between angular samples must remain on the interpolated manifold.
        """
        manifold = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        state = np.array([[0.75, 0.25]])

        sampled_distance, _ = nearest_manifold_distance(state, manifold)
        continuous_distance, coordinate = nearest_circular_manifold_distance(state, manifold)

        self.assertGreater(float(sampled_distance[0]), 0.0)
        self.assertAlmostEqual(float(continuous_distance[0]), 0.0, places=12)
        self.assertAlmostEqual(float(coordinate[0]), 0.25, places=12)

    def test_circular_manifold_distance_wraps_last_segment(self):
        """
        The final and first angular samples must form a valid closing segment.
        """
        manifold = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        state = np.array([[0.5, -0.5]])
        distance, coordinate = nearest_circular_manifold_distance(state, manifold)

        self.assertAlmostEqual(float(distance[0]), 0.0, places=12)
        self.assertAlmostEqual(float(coordinate[0]), 3.5, places=12)

    def test_dense_and_circulant_drives_match_explicit_matrix(self):
        """
        Optimized drive helpers must preserve the Jij orientation convention.
        """
        first_column = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.column_stack([np.roll(first_column, shift) for shift in range(4)])
        rates = np.array([[0.5, 1.0, 1.5, 2.0], [2.0, 0.0, 1.0, 0.5]])
        expected = rates @ weights.T

        np.testing.assert_allclose(dense_recurrent_drive(weights)(rates), expected, atol=1e-6)
        np.testing.assert_allclose(circulant_recurrent_drive(first_column)(rates), expected, atol=1e-12)

    def test_circular_fourier_derivative_is_periodic_and_exact_for_modes(self):
        """
        Fourier differentiation should recover low circular harmonics.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        values = np.vstack([np.sin(theta), np.cos(2.0 * theta)])
        expected = np.vstack([np.cos(theta), -2.0 * np.sin(2.0 * theta)])
        np.testing.assert_allclose(
            circular_fourier_derivative(values, axis=1),
            expected,
            atol=1e-11,
        )

    def test_overlap_order_parameter_matches_definition(self):
        """
        The reusable overlap helper must implement Eq. 6 without centering.
        """
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        activity = np.array([[2.0, 1.0], [0.5, 1.5]])
        expected = target.T @ activity.T / 2.0
        np.testing.assert_allclose(overlap_order_parameter(target, activity), expected)

    def test_figure4_two_point_function_is_uncentered(self):
        """
        Figure 4C must implement Eq. 4 without subtracting a mean curve.
        """
        tuning = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = tuning.T @ tuning / 2.0
        np.testing.assert_allclose(empirical_two_point_correlation(tuning), expected)

    def test_relative_circulant_error_detects_translation_symmetry(self):
        """
        A circulant matrix should have zero projection error.
        """
        first_column = np.array([1.0, 0.5, -0.2, 0.5])
        circulant = np.column_stack(
            [np.roll(first_column, shift) for shift in range(len(first_column))]
        )
        self.assertLess(relative_circulant_error(circulant), 1e-12)

        perturbed = circulant.copy()
        perturbed[0, 1] += 0.7
        self.assertGreater(relative_circulant_error(perturbed), 0.01)

    def test_kuiper_and_bh_helpers_return_valid_probabilities(self):
        """
        Figure 4A uniformity diagnostics should be finite and correctly ordered.
        """
        uniform_angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
        statistic, p_value = kuiper_uniformity_test_asymptotic(uniform_angles)
        self.assertGreater(statistic, 0.0)
        self.assertGreater(p_value, 0.5)

        adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.5]))
        np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.5]))

    def test_velocity_factors_generate_target_tangent_flow(self):
        """
        Eq. 5 factors should map target rates to tau times the manifold tangent.
        """
        rng = np.random.default_rng(20260618)
        phi = 0.2 + rng.gamma(shape=2.0, scale=0.5, size=(12, 8))
        static_a, velocity_a, factor_b, static_d, velocity_d = (
            optimized_recurrent_velocity_factors(
                phi,
                tau_s=0.05,
                regularization=0.0,
                enforce_zero_diagonal=True,
            )
        )
        static_weights = materialize_lowrank_weights(
            static_a,
            factor_b,
            diagonal=static_d,
            dtype=np.float64,
        )
        velocity_weights = materialize_lowrank_weights(
            velocity_a,
            factor_b,
            diagonal=velocity_d,
            dtype=np.float64,
        )
        x_star = softplus_inverse(phi, beta=2.0)
        tangent_target = 0.05 * circular_fourier_derivative(x_star, axis=1)
        np.testing.assert_allclose(static_d, np.sum(static_a * factor_b, axis=1))
        np.testing.assert_allclose(velocity_d, np.sum(velocity_a * factor_b, axis=1))
        np.testing.assert_allclose(static_weights @ phi, x_star, atol=1e-7)
        np.testing.assert_allclose(velocity_weights @ phi, tangent_target, atol=1e-7)

    def test_zero_velocity_modulated_simulation_preserves_exact_target(self):
        """
        An exactly fitted target state should remain fixed when omega is zero.
        """
        rng = np.random.default_rng(20260619)
        phi = 0.2 + rng.gamma(shape=2.0, scale=0.5, size=(10, 6))
        factors = optimized_recurrent_velocity_factors(
            phi,
            regularization=0.0,
            enforce_zero_diagonal=True,
        )
        x0 = softplus_inverse(phi[:, 0], beta=2.0)
        times, trajectory = simulate_velocity_modulated_rate_network(
            *factors,
            initial_states=x0,
            angular_velocity=np.zeros(10),
            dt_s=0.001,
            record_every_s=0.01,
            inhibition_c=0.0,
        )
        self.assertAlmostEqual(float(times[-1]), 0.01)
        np.testing.assert_allclose(trajectory[-1, 0], x0, atol=1e-7)

    def test_poisson_cv_selects_smoothing_width_per_unit(self):
        """
        Unitwise CV must not overwrite all neurons with a session-pooled sigma.
        """
        n_bins = 60
        theta = np.arange(n_bins)
        occupancy = np.full(n_bins, 100.0)

        narrow_distance = np.minimum(np.mod(theta - 10, n_bins), np.mod(10 - theta, n_bins))
        narrow = 0.1 + 5.0 * np.exp(-0.5 * (narrow_distance / 2.0) ** 2)

        broad_distance = np.minimum(np.mod(theta - 30, n_bins), np.mod(30 - theta, n_bins))
        broad_test = 0.1 + 5.0 * np.exp(-0.5 * (broad_distance / 10.0) ** 2)
        broad_train = np.maximum(
            broad_test * (1.0 + 1.5 * np.sin(theta * 2.0 * np.pi / 3.0)),
            0.01,
        )

        counts_a = np.vstack([narrow, broad_train]) * occupancy
        counts_b = np.vstack([narrow, broad_test]) * occupancy
        sigma_candidates = np.array([0.1, 1.0, 3.0, 8.0])

        sigma_a, sigma_b, *_ = choose_poisson_cv_sigmas(
            counts_a,
            counts_b,
            occupancy,
            occupancy,
            min_occupancy_s=0.0,
            sigma_candidates=sigma_candidates,
            scoring_units=np.array([True, True]),
        )

        np.testing.assert_array_equal(sigma_a, np.array([0.1, 3.0]))
        np.testing.assert_array_equal(sigma_b, np.array([0.1, 0.1]))


if __name__ == "__main__":
    unittest.main()
