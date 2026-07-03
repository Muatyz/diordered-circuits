from __future__ import annotations

import numpy as np

from learning.analysis.weights import (
    compute_weight_eigenvalues,
    mean_negative_source_offset,
    mean_source_offset,
    summarize_eigenvalue_pair_degeneracy,
)
from learning.common.angles import make_theta_hd_pref


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
