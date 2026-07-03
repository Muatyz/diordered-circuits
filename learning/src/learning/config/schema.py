"""Dataclass schema for Vafidis toy-model configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import pi
from typing import Any


VAFIDIS_TAU_S = 0.065
VAFIDIS_HR_INHIBITION = 1.5
VAFIDIS_VELOCITY_GAIN = 1.0 / (2.0 * pi)
VAFIDIS_OU_STD_RAD = 225.0 * pi / 180.0
VAFIDIS_TEST_VELOCITY_RAD = 500.0 * pi / 180.0
VAFIDIS_MODEL_CELLS = 60
VAFIDIS_VISUAL_SIGMA = 0.15
VAFIDIS_VISUAL_KAPPA = 1.0 / (4.0 * VAFIDIS_VISUAL_SIGMA * VAFIDIS_VISUAL_SIGMA)

# Vafidis uses wHD = Aactive / fmax = 2 / 0.15 = 13.33 because rates are in
# kHz with fmax = 0.15. The toy activation is normalized to max 1, so the
# equivalent fixed HD-to-HR voltage scale is Aactive itself.
VAFIDIS_ACTIVE_INPUT_RANGE = 2.0


@dataclass
class ActivationConfig:
    name: str = "sigmoid"
    gain: float = 2.5
    bias: float = 1.0


@dataclass
class ModelInitConfig:
    w_hd_to_hd_mode: str = "zeros"
    w_hr_to_hd_mode: str = "zeros"
    w_hd_to_hd_scale: float = 0.03
    w_hr_to_hd_scale: float = 0.02
    local_sigma: float = 0.45
    random_jitter: float = 0.001


@dataclass
class ModelConfig:
    n_theta: int = VAFIDIS_MODEL_CELLS
    n_hr: int = VAFIDIS_MODEL_CELLS
    tau_s: float = VAFIDIS_TAU_S
    tau_hd_to_hr: float | None = None
    tau_l_hd: float = 0.01
    p_distal_to_proximal: float = 2.0 / 3.0
    b_hd: float = 0.6
    b_hr: float = VAFIDIS_HR_INHIBITION
    w_hd_to_hr_strength: float = VAFIDIS_ACTIVE_INPUT_RANGE
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    init: ModelInitConfig = field(default_factory=ModelInitConfig)


@dataclass
class SimulationConfig:
    seed: int = 11
    dt: float = 0.01
    recurrent_warmup_duration: float = 0.0
    freeze_hd_to_hd_after_warmup: bool = False
    train_duration: float = 160.0
    bump_test_duration: float = 4.0
    darkness_test_duration: float = 6.0
    cue_duration: float = 0.35
    pi_cue_duration: float | None = None
    recue_duration: float = 1.0
    save_interval_steps: int = 40
    theta0: float = 0.0
    progress: bool = True


@dataclass
class LearningRuleConfig:
    eta_hd_to_hd: float = 3.0
    eta_hr_to_hd: float = 20.0
    hd_to_hd_learning_velocity_threshold: float | None = None
    tau_delta: float = 0.10
    w_hd_to_hd_min: float = -1.5
    w_hd_to_hd_max: float = 2.0
    w_hr_to_hd_min: float = -1.5
    w_hr_to_hd_max: float = 1.5
    hd_to_hd_symmetry_mode: str = "none"
    hd_to_hd_balance_mode: str = "none"
    hr_to_hd_balance_mode: str = "none"


@dataclass
class VisualConfig:
    amplitude: float = 2.0
    kappa: float = VAFIDIS_VISUAL_KAPPA
    baseline: float = 1.5
    normalize_peak: bool = True
    light_excitation: float = 0.0
    proximal_scale: float = 1.0


@dataclass
class VelocityConfig:
    process: str = "ou"
    mean: float = 0.0
    std: float = VAFIDIS_OU_STD_RAD
    tau: float = 0.50
    clip: float | None = 4.0 * pi
    k_vel: float = VAFIDIS_VELOCITY_GAIN


@dataclass
class TestsConfig:
    darkness_angular_velocity: float = VAFIDIS_TEST_VELOCITY_RAD
    gain_velocities: list[float] = field(
        default_factory=lambda: [
            -VAFIDIS_TEST_VELOCITY_RAD,
            -0.8 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.6 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.4 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.2 * VAFIDIS_TEST_VELOCITY_RAD,
            0.0,
            0.2 * VAFIDIS_TEST_VELOCITY_RAD,
            0.4 * VAFIDIS_TEST_VELOCITY_RAD,
            0.6 * VAFIDIS_TEST_VELOCITY_RAD,
            0.8 * VAFIDIS_TEST_VELOCITY_RAD,
            VAFIDIS_TEST_VELOCITY_RAD,
        ]
    )


@dataclass
class PathsConfig:
    runs_root: str = "runs/vafidis_toy"
    reports_root: str = "reports"


@dataclass
class ExperimentConfig:
    experiment_name: str = "vafidis_toy"
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    learning_rule: LearningRuleConfig = field(default_factory=LearningRuleConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    tests: TestsConfig = field(default_factory=TestsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for field_name, field_value in values.items():
        if not hasattr(instance, field_name):
            raise ValueError(f"Unknown config field: {field_name}")
        current_value = getattr(instance, field_name)
        if hasattr(current_value, "__dataclass_fields__") and isinstance(field_value, dict):
            setattr(instance, field_name, _merge_dataclass(current_value, field_value))
        else:
            setattr(instance, field_name, field_value)
    return instance


def experiment_config_from_dict(config_dict: dict[str, Any]) -> ExperimentConfig:
    model_config = config_dict.get("model")
    if isinstance(model_config, dict) and "n_theta" in model_config and "n_hr" not in model_config:
        model_config["n_hr"] = model_config["n_theta"]
    config = ExperimentConfig()
    return _merge_dataclass(config, config_dict)
