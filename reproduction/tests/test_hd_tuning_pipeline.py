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
from utils import (  # noqa: E402
    current_space_jacobian,
    fixed_point_residual_from_weights,
    materialize_lowrank_weights,
    optimized_recurrent_factors,
    prepare_phi_star_for_inverse,
    softplus,
    softplus_derivative_from_phi,
    softplus_inverse,
)


class HeadDirectionTuningTests(unittest.TestCase):
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
