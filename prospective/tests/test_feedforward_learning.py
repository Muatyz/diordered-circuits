import numpy as np

from prospective.plasticity.feedforward import feedforward_update, feedforward_update_rate


def test_local_rule_has_correct_pre_post_orientation_and_signs():
    weights = np.array([[1.0, 4.0], [1.0, 4.0]])
    pre = np.array([2.0, 0.0])
    post = np.array([1.0, 0.5])
    derivative = feedforward_update_rate(weights, pre, post, eta=1.0, alpha=1.0, beta=0.5)
    assert derivative.shape == (2, 2)
    assert derivative[0, 0] > 0
    assert derivative[0, 1] < 0
    assert np.allclose(derivative[1], 0.5 * derivative[0])


def test_eta_zero_leaves_weights_unchanged():
    weights = np.ones((2, 3))
    updated, delta, clipped = feedforward_update(
        weights, np.ones(3), np.ones(2), dt=0.1, eta=0.0,
        alpha=1.0, beta=0.5, nonnegative_clip=True,
    )
    assert np.array_equal(updated, weights)
    assert not np.any(delta)
    assert not np.any(clipped)


def test_preclip_depression_remains_observable():
    weights = np.array([[1.0]])
    updated, delta, clipped = feedforward_update(
        weights, np.array([0.0]), np.array([1.0]), dt=2.0, eta=1.0,
        alpha=1.0, beta=1.0, nonnegative_clip=True,
    )
    assert delta[0, 0] == -2.0
    assert clipped[0, 0]
    assert updated[0, 0] == 0.0

