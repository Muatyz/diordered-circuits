import numpy as np
import matplotlib.pyplot as plt

from prospective.animation.feedforward_scene import FeedforwardScene, aggregate_profile_by_position, baseline_subtracted_center
from prospective.animation.storyboard import global_training_frame_indices, top_update_connections
from prospective.config.schema import ExperimentConfig


def test_top_update_connections_matches_exact_flat_ranking():
    delta = np.array([[0.1, -4.0], [3.0, 0.2]])
    post, pre = top_update_connections(delta, 2)
    assert list(zip(post, pre)) == [(0, 1), (1, 0)]


def test_top_k_is_capped_by_matrix_size():
    post, pre = top_update_connections(np.ones((2, 2)), 99)
    assert len(post) == len(pre) == 4


def test_scene_constructs_and_updates_from_exact_saved_arrays():
    config = ExperimentConfig()
    frames = 2
    n_in = config.geometry.n_input
    n_comp = config.geometry.n_competitive
    history = {
        "time": np.array([1.0, 1.1]),
        "tutor_position": np.array([20.0, 22.0]),
        "tutor_rate": np.ones((frames, n_in)),
        "membrane": np.ones((frames, n_comp)),
        "adaptation": 0.2 * np.ones((frames, n_comp)),
        "rate": 0.1 * np.ones((frames, n_comp)),
        "weights": np.ones((frames, n_comp, n_in)),
        "delta_weights": 0.01 * np.ones((frames, n_comp, n_in)),
        "clipped": np.zeros((frames, n_comp, n_in), dtype=bool),
    }
    scene = FeedforwardScene(history, config, title="test")
    artists = scene.update(0)
    assert artists
    assert np.allclose(scene.artists.u_profile.get_ydata(), 1.0)
    assert np.allclose(scene.artists.v_profile.get_ydata(), 0.2)
    scene.fig.canvas.draw()
    network_box = scene.ax_network.get_position()
    matrix_box = scene.ax_matrix.get_position()
    assert network_box.x1 < matrix_box.x0
    assert matrix_box.x1 < scene.colorbar_axes[0].get_position().x0
    plt.close(scene.fig)


def test_scene_does_not_highlight_arbitrary_connections_when_update_is_zero():
    config = ExperimentConfig()
    n_in = config.geometry.n_input
    n_comp = config.geometry.n_competitive
    history = {
        "time": np.array([0.0, 0.1]),
        "tutor_position": np.array([0.0, 2.0]),
        "tutor_rate": np.ones((2, n_in)),
        "membrane": np.zeros((2, n_comp)),
        "adaptation": np.zeros((2, n_comp)),
        "rate": np.zeros((2, n_comp)),
        "weights": np.ones((2, n_comp, n_in)),
        "delta_weights": np.zeros((2, n_comp, n_in)),
        "clipped": np.zeros((2, n_comp, n_in), dtype=bool),
    }
    scene = FeedforwardScene(history, config, title="test")
    scene.update(0)
    assert len(scene.artists.update_connections.get_segments()) == 0
    assert "delta J = 0" in scene.artists.equation_text.get_text()
    plt.close(scene.fig)


def test_duplicate_learned_positions_are_averaged_without_smoothing():
    positions, values = aggregate_profile_by_position(
        np.array([0.0, 0.0, 2.0]), np.array([1.0, 3.0, 5.0])
    )
    assert np.array_equal(positions, [0.0, 2.0])
    assert np.array_equal(values, [2.0, 5.0])


def test_profile_center_ignores_uniform_baseline():
    positions = np.array([0.0, 5.0, 10.0])
    assert baseline_subtracted_center(positions, np.array([2.0, 5.0, 2.0])) == 5.0


def test_global_training_frames_cover_both_endpoints_uniformly():
    config = ExperimentConfig()
    times = np.linspace(0.0, 300.0, 6001)
    indices = global_training_frame_indices(times, config)
    assert indices.size == config.animation.global_frame_count
    assert indices[0] == 0
    assert indices[-1] == times.size - 1
    assert np.all(np.diff(indices) > 0)
