from __future__ import annotations

import numpy as np

from learning.connectivity.constraints import constrain_w_hd_to_hd, constrain_w_hr_to_hd
from learning.plasticity.predictive_local import update_predictive_local_weights


def test_predictive_local_update_respects_hd_diagonal_and_bounds() -> None:
    n_theta = 6
    w_hd_to_hd = np.zeros((n_theta, n_theta))
    w_hr_to_hd = np.zeros((n_theta, n_theta))
    delta_w_hd_to_hd = np.zeros_like(w_hd_to_hd)
    delta_w_hr_to_hd = np.zeros_like(w_hr_to_hd)
    e_hd = np.ones(n_theta)
    p_hd = np.ones(n_theta)
    p_hr = np.ones(n_theta)
    (
        next_w_hd_to_hd,
        next_w_hr_to_hd,
        next_delta_w_hd_to_hd,
        next_delta_w_hr_to_hd,
    ) = update_predictive_local_weights(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        delta_w_hd_to_hd=delta_w_hd_to_hd,
        delta_w_hr_to_hd=delta_w_hr_to_hd,
        e_hd=e_hd,
        p_hd=p_hd,
        p_hr=p_hr,
        dt=0.1,
        tau_delta=0.1,
        eta_hd_to_hd=1.0,
        eta_hr_to_hd=1.0,
        w_hd_to_hd_min=0.0,
        w_hd_to_hd_max=0.2,
        w_hr_to_hd_min=-0.2,
        w_hr_to_hd_max=0.2,
    )
    assert np.allclose(np.diag(next_w_hd_to_hd), 0.0)
    assert np.max(next_w_hd_to_hd) <= 0.2
    assert np.min(next_w_hr_to_hd) >= -0.2
    assert np.all(next_w_hr_to_hd >= 0.0)
    assert np.allclose(next_delta_w_hd_to_hd, 1.0)
    assert np.allclose(next_delta_w_hr_to_hd, 1.0)


def test_hr_to_hd_antisymmetric_balance_removes_common_mode() -> None:
    w_hr_to_hd = np.array(
        [
            [1.0, 2.0, 0.5, -0.5],
            [-0.5, 0.25, 1.5, -1.0],
        ]
    )
    balanced_w_hr_to_hd = constrain_w_hr_to_hd(
        w_hr_to_hd,
        lower_bound=-2.0,
        upper_bound=2.0,
        balance_mode="antisymmetric_wings",
    )
    n_hr_per_wing = w_hr_to_hd.shape[1] // 2
    w_lhr_to_hd = balanced_w_hr_to_hd[:, :n_hr_per_wing]
    w_rhr_to_hd = balanced_w_hr_to_hd[:, n_hr_per_wing:]

    assert np.allclose(w_lhr_to_hd + w_rhr_to_hd, 0.0)
    assert np.allclose(
        w_rhr_to_hd - w_lhr_to_hd,
        w_hr_to_hd[:, n_hr_per_wing:] - w_hr_to_hd[:, :n_hr_per_wing],
    )


def test_hd_to_hd_symmetric_constraint_removes_antisymmetric_component() -> None:
    w_hd_to_hd = np.array(
        [
            [0.0, 1.0, -0.5],
            [0.25, 0.0, 2.0],
            [0.75, -1.0, 0.0],
        ]
    )
    constrained_w_hd_to_hd = constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=-2.0,
        upper_bound=2.0,
        symmetry_mode="symmetric",
    )

    assert np.allclose(constrained_w_hd_to_hd, constrained_w_hd_to_hd.T)
    assert np.allclose(np.diag(constrained_w_hd_to_hd), 0.0)


def test_hd_to_hd_zero_sum_constraint_removes_common_mode() -> None:
    w_hd_to_hd = np.ones((5, 5))
    np.fill_diagonal(w_hd_to_hd, 0.0)
    w_hd_to_hd[0, 1] += 2.0
    w_hd_to_hd[1, 0] += 2.0

    constrained_w_hd_to_hd = constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=-5.0,
        upper_bound=5.0,
        symmetry_mode="symmetric",
        balance_mode="zero_sum",
    )

    assert np.allclose(constrained_w_hd_to_hd, constrained_w_hd_to_hd.T)
    assert np.allclose(np.diag(constrained_w_hd_to_hd), 0.0)
    assert abs(np.mean(constrained_w_hd_to_hd)) < 1e-6
    assert np.max(constrained_w_hd_to_hd) > 0.0
    assert np.min(constrained_w_hd_to_hd) < 0.0
