"""Unit tests for the non-invasive diagnostics around the released LearnPI code."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "original" / "basin_test.py"
MODULE_SPEC = importlib.util.spec_from_file_location("original_basin_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
basin_test = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(basin_test)


def test_pva_and_peak_decode_preserve_original_paired_hd_geometry() -> None:
    preferred_direction_deg = np.repeat([0.0, 120.0, 240.0], 2)
    firing_rate = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    theta_pva_deg, pva_strength = basin_test.decode_pva_history(
        firing_rate,
        preferred_direction_deg,
    )
    theta_peak_deg, contrast = basin_test.decode_peak_history(firing_rate)

    assert np.allclose(theta_pva_deg, [0.0, 120.0, 240.0])
    assert np.allclose(theta_peak_deg, [0.0, 120.0, 240.0])
    assert np.allclose(pva_strength, 1.0)
    assert np.all(contrast > 0.0)


def test_endpoint_clustering_merges_a_basin_across_zero_degrees() -> None:
    clustered = basin_test.cluster_circular_endpoints(
        np.asarray([359.0, 1.0, 88.5, 91.0, 180.0]),
        diameter_tolerance_deg=5.0,
    )

    assert np.allclose(clustered["center_deg"], [0.0, 89.75, 180.0])
    assert clustered["occupancy"].tolist() == [2, 2, 1]


def test_complete_link_clustering_does_not_chain_a_continuous_ring() -> None:
    endpoint_deg = np.linspace(0.0, 360.0, 36, endpoint=False)
    clustered = basin_test.cluster_circular_endpoints(
        endpoint_deg,
        diameter_tolerance_deg=5.0,
    )

    assert clustered["center_deg"].size == endpoint_deg.size
    assert np.all(clustered["occupancy"] == 1)


def test_endpoint_clustering_ignores_undefined_decodes() -> None:
    clustered = basin_test.cluster_circular_endpoints(
        np.asarray([10.0, np.nan, 12.0, np.inf]),
        diameter_tolerance_deg=3.0,
    )

    assert clustered["center_deg"].size == 1
    assert clustered["occupancy"].tolist() == [2]


def test_public_archive_params_are_completed_with_release_defaults() -> None:
    params = basin_test.normalize_release_params(
        {
            "dt": 5e-4,
            "n_neu": 60,
            "v0": 2,
            "v_max": 720,
            "A": 4,
            "sigma": 0.15,
            "inh": -1,
            "inh_rot": -1.5,
            "every_perc": 1,
            "avg_err": 10,
            "n_sigma": 0,
            "exc": 4,
            "tau_s": 65,
            "gain": 1,
        }
    )

    assert params["M"] == 4
    assert params["tau_d"] == 100
    assert params["fmax"] == 0.15
    assert params["vary_w_rot"] is False


def test_late_drift_unwraps_a_zero_degree_crossing() -> None:
    drift_deg_s = basin_test.estimate_late_drift_deg_s(
        np.asarray([[358.0, 359.0, 0.0, 1.0, 2.0]]),
        np.arange(5.0),
        late_fraction=1.0,
    )

    assert np.allclose(drift_deg_s, [1.0])


def test_sampling_skips_the_release_codes_pre_bump_placeholder() -> None:
    sampled = basin_test._sample_indices(101, dt=0.001, sample_interval=0.025)

    assert sampled.tolist() == [1, 26, 51, 76, 100]


def test_runner_freezes_inputs_weights_and_reuses_reconstructed_w_rot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {
        "dt": 0.001,
        "n_neu": 4,
        "v0": 2,
        "v_max": 720,
        "M": 4,
        "sigma": 0.15,
        "inh": -1,
        "inh_rot": -1.5,
        "every_perc": 1,
        "avg_err": 1,
        "n_sigma": 0,
        "exc": 4,
        "tau_s": 65,
        "gain": 0.5,
    }
    network_path = tmp_path / "network.npz"
    np.savez(
        network_path,
        w=np.zeros((4, 8), dtype=float),
        params=np.asarray(params, dtype=object),
    )
    calls: list[dict[str, object]] = []

    def fake_simulate(t_run, theta0, simulation_params, **kwargs):
        calls.append(
            {
                "gain": simulation_params["gain"],
                "train": kwargs["train"],
                "day": kwargs["day"],
                "stab": kwargs["stab"],
                "w_rot": kwargs["w_rot"],
                "theta0": theta0.copy(),
            }
        )
        firing_rate = np.zeros((4, theta0.size), dtype=float)
        firing_rate[0:2] = 1.0
        reconstructed_w_rot = (
            np.eye(4) if kwargs["w_rot"] is None else kwargs["w_rot"]
        )
        return kwargs["w"], reconstructed_w_rot, firing_rate, firing_rate, firing_rate

    monkeypatch.setitem(sys.modules, "fly_rec", SimpleNamespace(simulate=fake_simulate))
    basin_test.run_uniform_bump_basin_test(
        network_path,
        initial_condition_count=2,
        duration=0.002,
        sample_interval=0.001,
    )

    assert len(calls) == 2
    assert all(call["gain"] == 1.0 for call in calls)
    assert all(call["train"] is False for call in calls)
    assert all(call["day"] is False for call in calls)
    assert all(call["stab"] is True for call in calls)
    assert calls[0]["w_rot"] is None
    assert np.array_equal(calls[1]["w_rot"], np.eye(4))
    assert np.allclose(calls[0]["theta0"], 270.0)
    assert np.allclose(calls[1]["theta0"], 90.0)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 180.0])
def test_endpoint_clustering_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="diameter_tolerance_deg"):
        basin_test.cluster_circular_endpoints([0.0], tolerance)
