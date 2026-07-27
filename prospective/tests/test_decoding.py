import numpy as np

from prospective.analysis.decoding import circular_center, learned_competitive_positions, linear_center


def test_linear_center():
    assert linear_center(np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 4.0])) == 2.0


def test_circular_center_does_not_jump_at_boundary():
    center = circular_center(np.array([1.0, 1.0]), np.array([99.0, 1.0]), 100.0)
    assert center < 2.0 or center > 98.0


def test_learned_positions_come_from_row_peaks():
    weights = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 0.0]])
    positions = np.array([0.0, 10.0, 20.0])
    assert np.array_equal(learned_competitive_positions(weights, positions), [10.0, 0.0])

