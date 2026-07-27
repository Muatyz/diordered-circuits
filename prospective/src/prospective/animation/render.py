"""Render neural-, learning-, and full-training-time clips from an existing run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from prospective.animation.feedforward_scene import FeedforwardScene
from prospective.animation.storyboard import (
    global_training_frame_indices,
    learning_frame_indices,
    neural_frame_indices,
)
from prospective.config import load_config
from prospective.io.run_dir import write_json
from prospective.io.save_load import load_npz


def _writer_and_suffix(config):
    requested = config.animation.output_format
    if requested == "mp4" and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=config.animation.fps, bitrate=2400), "mp4", "ffmpeg"
    if animation.writers.is_available("pillow"):
        return animation.PillowWriter(fps=config.animation.fps), "gif", "pillow"
    raise RuntimeError("neither ffmpeg nor Pillow animation writer is available")


def _render_clip(
    history,
    config,
    indices,
    output_stem: Path,
    title: str,
    *,
    mode: str,
) -> tuple[Path, str]:
    physical_span = float(history["time"][indices[-1]] - history["time"][indices[0]])
    playback_duration = len(indices) / config.animation.fps
    playback_acceleration = physical_span / playback_duration if playback_duration > 0 else None
    scene = FeedforwardScene(
        history,
        config,
        title=title,
        mode=mode,
        playback_acceleration=playback_acceleration,
    )
    movie = animation.FuncAnimation(scene.fig, lambda frame: scene.update(int(frame)), frames=indices, interval=1000 / config.animation.fps, blit=False)
    writer, suffix, backend = _writer_and_suffix(config)
    output = output_stem.with_suffix(f".{suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(indices)
    progress = tqdm(
        total=frame_count,
        desc=f"render {output.name}",
        unit="frame",
        dynamic_ncols=True,
        disable=not config.animation.render_progress,
    )

    def update_progress(frame_number: int, total_frames: int) -> None:
        """Advance to Matplotlib's actual encoded-frame count."""

        completed = min(frame_number + 1, frame_count)
        progress.update(max(0, completed - progress.n))

    try:
        movie.save(output, writer=writer, dpi=130, progress_callback=update_progress)
        progress.update(max(0, frame_count - progress.n))
    finally:
        progress.close()
        plt.close(scene.fig)
    return output, backend


def _existing_movie(output_stem: Path) -> Path | None:
    """Find a previously rendered nonempty movie independent of backend."""

    for suffix in (".mp4", ".gif"):
        candidate = output_stem.with_suffix(suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def render_run(
    run_dir: str | Path,
    *,
    animation_config_path: str | Path | None = None,
    clip: str = "both",
) -> Path:
    """Render selected timescales and a manifest without rerunning simulation."""

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    if animation_config_path is not None:
        render_config = load_config(animation_config_path)
        config = replace(config, animation=render_config.animation)
    history = load_npz(run_dir / "training_history.npz")
    if "weights" not in history:
        raise ValueError("source run lacks animation arrays; train with animation.enabled=true")
    output_dir = run_dir / "animations" / "feedforward_mechanism"
    neural_indices = neural_frame_indices(history["time"], config)
    learning_indices = learning_frame_indices(history["time"], config)
    global_indices = global_training_frame_indices(history["time"], config)
    if clip not in {"both", "all", "neural", "learning", "global"}:
        raise ValueError("clip must be both, all, neural, learning, or global")
    neural_path = _existing_movie(output_dir / "neural_dynamics")
    learning_path = _existing_movie(output_dir / "learning_evolution")
    global_path = _existing_movie(output_dir / "global_training_dynamics")
    neural_backend = "existing" if neural_path is not None else "not_rendered"
    learning_backend = "existing" if learning_path is not None else "not_rendered"
    global_backend = "existing" if global_path is not None else "not_rendered"
    if clip in {"both", "neural"}:
        neural_path, neural_backend = _render_clip(
            history,
            config,
            neural_indices,
            output_dir / "neural_dynamics",
            "Neural dynamics: tutor sampling, adaptation lag, and local update",
            mode="neural",
        )
    elif clip == "all":
        neural_path, neural_backend = _render_clip(
            history,
            config,
            neural_indices,
            output_dir / "neural_dynamics",
            "Neural dynamics: tutor sampling, adaptation lag, and local update",
            mode="neural",
        )
    if clip in {"both", "all", "learning"}:
        learning_path, learning_backend = _render_clip(
            history,
            config,
            learning_indices,
            output_dir / "learning_evolution",
            "Learning-time montage: random J toward local Gaussian structure",
            mode="learning",
        )
    if clip in {"all", "global"}:
        global_path, global_backend = _render_clip(
            history,
            config,
            global_indices,
            output_dir / "global_training_dynamics",
            "Global training dynamics: complete evolution of neural state and J(t)",
            mode="global",
        )
    write_json(output_dir / "manifest.json", {
        "source_run": str(run_dir.resolve()),
        "scientific_frame_source": "training_history.npz exact samples",
        "neural_output": neural_path.name if neural_path is not None else None,
        "learning_output": learning_path.name if learning_path is not None else None,
        "global_output": global_path.name if global_path is not None else None,
        "neural_backend": neural_backend,
        "learning_backend": learning_backend,
        "global_backend": global_backend,
        "fps": config.animation.fps,
        "render_progress": config.animation.render_progress,
        "neural_window_seconds": config.animation.neural_window_seconds,
        "learning_hold_seconds": config.animation.learning_hold_seconds,
        "render_config": str(Path(animation_config_path).resolve()) if animation_config_path is not None else None,
        "connection_rule": config.animation.update_highlight_mode,
        "display_top_k_connections": config.animation.display_top_k_connections,
        "neural_sample_indices": neural_indices.tolist(),
        "learning_sample_indices": learning_indices.tolist(),
        "global_sample_indices": global_indices.tolist(),
        "global_time_span_seconds": [
            float(history["time"][global_indices[0]]),
            float(history["time"][global_indices[-1]]),
        ],
        "global_average_playback_acceleration": (
            float(history["time"][global_indices[-1]] - history["time"][global_indices[0]])
            / (len(global_indices) / config.animation.fps)
        ),
        "global_matrix_row_order": "final learned competitive-position order",
        "variable_interpolation": "none",
        "boundary_mode": config.geometry.boundary_mode,
        "warning": "externally driven feedforward representation; not an autonomous attractor",
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", help="optional render-only config; its animation section overrides the source run")
    parser.add_argument("--clip", choices=["both", "all", "neural", "learning", "global"], default="both")
    args = parser.parse_args()
    print(render_run(args.run_dir, animation_config_path=args.config, clip=args.clip))


if __name__ == "__main__":
    main()
