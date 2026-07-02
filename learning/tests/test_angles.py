from __future__ import annotations

import numpy as np

from learning.common.angles import (
    collapse_activity_by_theta,
    circular_difference,
    make_theta_hd_pref,
    make_vafidis_paired_theta_hd_pref,
    peak_decode,
    pva_decode,
    pva_vector_strength,
    wrap_angle,
)


def test_wrap_angle_range() -> None:
    wrapped_angle = wrap_angle(np.array([-np.pi - 0.1, 0.0, np.pi + 0.1]))
    assert np.all(wrapped_angle >= -np.pi)
    assert np.all(wrapped_angle < np.pi)
    assert np.allclose(wrapped_angle, np.array([np.pi - 0.1, 0.0, -np.pi + 0.1]))


def test_circular_difference_uses_short_arc() -> None:
    difference = circular_difference(0.05, -0.05)
    assert np.isclose(difference, 0.10)


def test_pva_decode_recovers_single_bump() -> None:
    theta_hd_pref = make_theta_hd_pref(16)
    r_hd = np.zeros(16)
    r_hd[4] = 1.0
    decoded_angle = pva_decode(theta_hd_pref, r_hd)
    assert np.isclose(decoded_angle, theta_hd_pref[4])


def test_peak_decode_collapses_vafidis_partner_cells() -> None:
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(8)
    r_hd = np.zeros(8)
    r_hd[4] = 0.9
    r_hd[5] = 1.0
    decoded_angle = peak_decode(theta_hd_pref, r_hd)
    assert np.isclose(decoded_angle, 0.0)


def test_peak_decode_returns_center_of_highest_plateau() -> None:
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(8)
    r_hd = np.zeros(8)
    r_hd[2:6] = 1.0
    decoded_angle = peak_decode(theta_hd_pref, r_hd)
    assert np.isclose(decoded_angle, -np.pi / 4.0)


def test_collapse_activity_by_theta_averages_partner_cells() -> None:
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(8)
    r_hd = np.arange(8, dtype=float)
    unique_theta, collapsed_rates = collapse_activity_by_theta(theta_hd_pref, r_hd)
    assert np.allclose(unique_theta, np.array([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]))
    assert np.allclose(collapsed_rates, np.array([0.5, 2.5, 4.5, 6.5]))


def test_pva_vector_strength_rejects_uniform_activity() -> None:
    theta_hd_pref = make_theta_hd_pref(16)
    uniform_activity = np.ones(16)
    bump_activity = np.zeros(16)
    bump_activity[4] = 1.0
    assert pva_vector_strength(theta_hd_pref, uniform_activity) < 1e-12
    assert np.isclose(pva_vector_strength(theta_hd_pref, bump_activity), 1.0)


def test_vafidis_hd_preferences_are_odd_even_pairs() -> None:
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(8)
    expected_theta_hd_pref = np.array(
        [
            -np.pi,
            -np.pi,
            -np.pi / 2.0,
            -np.pi / 2.0,
            0.0,
            0.0,
            np.pi / 2.0,
            np.pi / 2.0,
        ]
    )
    assert np.allclose(theta_hd_pref, expected_theta_hd_pref)
