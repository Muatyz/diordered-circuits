"""Typed configuration schema for prospective toy experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentSection:
    """Names and locates a reproducible experiment run."""

    name: str = "feedforward_toy"
    seed: int = 11
    output_root: str = "runs"


@dataclass(frozen=True)
class GeometrySection:
    """Defines the one-dimensional tutor and competitive populations."""

    length: float = 100.0
    n_input: int = 32
    n_competitive: int = 32
    boundary_mode: str = "paper_reset"


@dataclass(frozen=True)
class TutorSection:
    """Parameters of the unidirectionally moving Gaussian tutor."""

    speed: float = 20.0
    sigma: float = 5.0
    integrated_drive: float = 30.0
    initial_position: float = 0.0


@dataclass(frozen=True)
class NeuralSection:
    """Fast membrane and slow firing-rate-adaptation parameters."""

    dt: float = 0.005
    tau_u: float = 0.015
    tau_v: float = 0.6
    adaptation_strength: float = 0.2
    inhibition_strength: float = 0.8


@dataclass(frozen=True)
class FeedforwardLearningSection:
    """Local Hebbian and weight-dependent decay parameters."""

    enabled: bool = True
    eta: float = 0.01
    alpha: float = 1.0
    beta: float = 0.5
    initial_weight_scale: float = 0.25
    nonnegative_clip: bool = True


@dataclass(frozen=True)
class SimulationSection:
    """Numerical integration, sampling, and safety controls."""

    duration: float = 120.0
    state_sample_interval_steps: int = 20
    weight_snapshot_interval_steps: int = 1000
    divergence_threshold: float = 1.0e6
    progress: bool = True


@dataclass(frozen=True)
class AnalysisSection:
    """Post-processing choices that never affect model dynamics."""

    transient_duration: float = 10.0
    boundary_margin_sigmas: float = 2.5
    bump_strength_min: float = 0.05
    fit_gaussian_for_diagnostics: bool = True


@dataclass(frozen=True)
class AnimationSection:
    """Offline rendering choices for mechanism animations."""

    enabled: bool = False
    fps: int = 24
    render_progress: bool = True
    display_top_k_connections: int = 48
    update_highlight_mode: str = "top_k_absolute_delta"
    render_matrix_inset: bool = True
    neural_window_seconds: float = 20.0
    learning_frame_interval_cycles: int = 1
    learning_hold_seconds: float = 0.5
    global_frame_count: int = 720
    output_format: str = "mp4"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete validated configuration for a feedforward experiment."""

    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    geometry: GeometrySection = field(default_factory=GeometrySection)
    tutor: TutorSection = field(default_factory=TutorSection)
    neural: NeuralSection = field(default_factory=NeuralSection)
    feedforward_learning: FeedforwardLearningSection = field(default_factory=FeedforwardLearningSection)
    simulation: SimulationSection = field(default_factory=SimulationSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    animation: AnimationSection = field(default_factory=AnimationSection)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        """Build and validate a nested configuration from parsed YAML."""

        config = cls(
            experiment=ExperimentSection(**values.get("experiment", {})),
            geometry=GeometrySection(**values.get("geometry", {})),
            tutor=TutorSection(**values.get("tutor", {})),
            neural=NeuralSection(**values.get("neural", {})),
            feedforward_learning=FeedforwardLearningSection(**values.get("feedforward_learning", {})),
            simulation=SimulationSection(**values.get("simulation", {})),
            analysis=AnalysisSection(**values.get("analysis", {})),
            animation=AnimationSection(**values.get("animation", {})),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready nested dictionary."""

        return asdict(self)

    def validate(self) -> None:
        """Reject invalid physics and unsafe numerical settings early."""

        g, t, n, l, s, a = (
            self.geometry,
            self.tutor,
            self.neural,
            self.feedforward_learning,
            self.simulation,
            self.animation,
        )
        if g.length <= 0 or g.n_input < 2 or g.n_competitive < 2:
            raise ValueError("geometry length must be positive and each population needs at least two neurons")
        if g.boundary_mode not in {"paper_reset", "periodic_ring"}:
            raise ValueError("boundary_mode must be 'paper_reset' or 'periodic_ring'")
        if t.sigma <= 0 or t.integrated_drive <= 0:
            raise ValueError("tutor sigma and integrated_drive must be positive")
        if n.dt <= 0 or n.tau_u <= 0 or n.tau_v <= 0:
            raise ValueError("dt and neural time constants must be positive")
        if n.adaptation_strength < 0 or n.inhibition_strength < 0:
            raise ValueError("adaptation and inhibition strengths must be nonnegative")
        if n.dt > min(n.tau_u, n.tau_v):
            raise ValueError("Euler dt must not exceed the fastest neural time constant")
        if l.eta < 0 or l.alpha <= 0 or l.initial_weight_scale < 0:
            raise ValueError("learning eta/initial scale must be nonnegative and alpha positive")
        if not 0 < l.beta < 2:
            raise ValueError("the Gaussian equilibrium theory requires 0 < beta < 2")
        if s.duration <= 0 or s.state_sample_interval_steps < 1 or s.weight_snapshot_interval_steps < 1:
            raise ValueError("duration and sample intervals must be positive")
        if s.divergence_threshold <= 0:
            raise ValueError("divergence_threshold must be positive")
        if self.analysis.transient_duration < 0:
            raise ValueError("transient_duration must be nonnegative")
        if (
            a.fps < 1
            or a.display_top_k_connections < 1
            or a.learning_frame_interval_cycles < 1
            or a.global_frame_count < 2
        ):
            raise ValueError("animation fps, top-k, cycle interval, and global frame count must be positive")
        if a.neural_window_seconds <= 0 or a.learning_hold_seconds <= 0:
            raise ValueError("animation window and learning hold duration must be positive")
        if a.update_highlight_mode != "top_k_absolute_delta":
            raise ValueError("only top_k_absolute_delta is currently supported")
        if a.output_format not in {"mp4", "gif", "html"}:
            raise ValueError("animation output_format must be mp4, gif, or html")
