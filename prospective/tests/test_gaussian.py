import numpy as np

from prospective.common.geometry import normalized_gaussian, signed_circular_difference, uniform_positions


def test_normalized_gaussian_integrates_to_one_away_from_boundaries():
    x = uniform_positions(4000, 100.0)
    profile = normalized_gaussian(x, 50.0, 5.0)
    assert np.isclose(profile.sum() * (100.0 / len(x)), 1.0, atol=1e-8)


def test_periodic_gaussian_is_continuous_across_boundary():
    x = np.array([99.0, 1.0])
    profile = normalized_gaussian(x, 0.0, 5.0, length=100.0, periodic=True)
    assert np.allclose(profile[0], profile[1])
    assert np.allclose(signed_circular_difference(np.array([99.0, 1.0]), 0.0, 100.0), [-1.0, 1.0])

