from __future__ import annotations

import numpy as np

from learning.analysis.weights import (
    compute_weight_eigenvalues,
    mean_negative_source_offset,
    mean_source_offset,
    sort_weight_matrices_by_hd_preference,
    summarize_eigenvalue_pair_degeneracy,
    summarize_weight_structure,
)
from learning.common.angles import make_theta_hd_pref


def test_weight_sorting_reorders_both_axes_without_changing_sparsity() -> None:
    w_hd_to_hd = np.arange(16, dtype=float).reshape(4, 4)
    w_hr_to_hd = np.arange(16, 32, dtype=float).reshape(4, 4)
    preference = np.asarray([1.0, -2.0, -1.0, 2.0])
    result = sort_weight_matrices_by_hd_preference(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        theta_hd_preference=preference,
    )
    np.testing.assert_array_equal(result["hd_order"], [1, 2, 0, 3])
    np.testing.assert_array_equal(
        result["w_hd_to_hd"],
        w_hd_to_hd[np.ix_([1, 2, 0, 3], [1, 2, 0, 3])],
    )
    assert np.count_nonzero(result["w_hd_to_hd"] == 0.0) == np.count_nonzero(
        w_hd_to_hd == 0.0
    )


def test_weight_summary_reports_relative_near_zero_fraction() -> None:
    w_hd_to_hd = np.eye(4)
    w_hr_to_hd = np.eye(4)
    summary = summarize_weight_structure(w_hd_to_hd, w_hr_to_hd)
    assert np.isclose(summary["hd_to_hd_near_zero_1pct_max_fraction"], 0.75)
    assert np.isclose(summary["hr_to_hd_near_zero_1pct_max_fraction"], 0.75)


def test_positive_and_negative_source_offsets_are_reported_separately() -> None:
    theta_hd_pref = make_theta_hd_pref(8)
    weight_target_by_source = np.zeros((8, 8))
    weight_target_by_source[0, 1] = 2.0
    weight_target_by_source[0, 7] = -3.0

    excitatory_offset = mean_source_offset(
        weight_target_by_source=weight_target_by_source,
        theta_target_pref=theta_hd_pref,
        theta_source_pref=theta_hd_pref,
        use_absolute_weight=False,
    )
    inhibitory_offset = mean_negative_source_offset(
        weight_target_by_source=weight_target_by_source,
        theta_target_pref=theta_hd_pref,
        theta_source_pref=theta_hd_pref,
    )

    assert np.isclose(excitatory_offset, np.pi / 4.0)
    assert np.isclose(inhibitory_offset, -np.pi / 4.0)


def test_spectral_degeneracy_detects_nonconstant_ring_pair() -> None:
    theta_hd_pref = make_theta_hd_pref(12)
    weight_matrix = np.cos(theta_hd_pref[:, None] - theta_hd_pref[None, :])

    eigenvalues = compute_weight_eigenvalues(weight_matrix)
    summary = summarize_eigenvalue_pair_degeneracy(weight_matrix=weight_matrix)

    assert eigenvalues.shape == (12,)
    assert summary["nonconstant_pair_count"] == 5.0
    assert summary["first_nonconstant_pair_gap_norm"] < 1e-12
