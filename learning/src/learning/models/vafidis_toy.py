"""Vafidis-style predictive local plasticity toy model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from learning.common.angles import make_vafidis_paired_theta_hd_pref, peak_decode, pva_decode, wrap_angle
from learning.common.arrays import assert_finite, check_model_arrays
from learning.config.schema import ExperimentConfig
from learning.connectivity.initialize import (
    initialize_w_hd_to_hd,
    initialize_w_hd_to_hr,
    initialize_w_hr_to_hd,
)
from learning.dynamics.activation import apply_activation
from learning.dynamics.hd_dynamics import (
    HD_DISTAL_NORMALIZATION_MODES,
    PROXIMAL_INTEGRATION_FORWARD_EULER,
    PROXIMAL_INTEGRATION_METHODS,
    compute_hd_distal_pathway_drives,
    compute_hd_compartments,
    compute_v_hd_proximal_steady_state,
    euler_update_i_hd_distal_from_pathway_drives,
    euler_update_v_hd_distal,
    update_v_hd_proximal,
)
from learning.dynamics.hr_dynamics import compute_i_hr, euler_update_r_hd_to_hr_lp
from learning.plasticity.predictive_local import compute_e_hd, update_predictive_local_weights
from learning.plasticity.traces import euler_update_psp_trace
from learning.stimuli.velocity import make_i_vel_to_hr
from learning.stimuli.visual import (
    generate_heterogeneous_visual_profiles,
    make_heterogeneous_i_vis_to_hd,
    make_i_vis_to_hd,
    make_zero_i_vis_to_hd,
)


@dataclass(frozen=True)
class VafidisToyParams:
    n_theta: int
    n_hr: int
    dt: float
    proximal_integration_method: str
    tau_s: float
    tau_hd_to_hr: float
    tau_l_hd: float
    p_distal_to_proximal: float
    c_hd_proximal: float
    g_l_hd_proximal: float
    g_d_hd_to_proximal: float
    b_hd: float
    b_hr: float
    hd_distal_normalization: str
    activation_name: str
    activation_gain: float
    activation_bias: float
    activation_max_rate: float
    k_vel: float
    visual_profile: str
    visual_amplitude: float
    visual_kappa: float
    visual_baseline: float
    visual_normalize_peak: bool
    visual_light_excitation: float
    visual_proximal_scale: float
    visual_noise_std: float
    visual_heterogeneous_sigma: float
    visual_heterogeneous_beta: float
    visual_heterogeneous_bias: float
    visual_heterogeneous_n_angles: int
    visual_heterogeneous_seed: int
    visual_heterogeneous_alignment: str
    visual_heterogeneous_normalization: str
    visual_heterogeneous_population_sampling: str
    visual_heterogeneous_master_n_hd_cells: int | None
    tau_delta: float
    eta_hd_to_hd: float
    eta_hr_to_hd: float
    w_hd_to_hd_min: float | None
    w_hd_to_hd_max: float | None
    w_hr_to_hd_min: float | None
    w_hr_to_hd_max: float | None
    hd_to_hd_symmetry_mode: str
    hd_to_hd_balance_mode: str
    hr_to_hd_balance_mode: str
    zero_hd_to_hd_diagonal: bool

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "VafidisToyParams":
        n_hr = config.model.n_hr
        if n_hr != config.model.n_theta:
            raise ValueError("The Vafidis-style toy model expects model.n_hr == model.n_theta")
        if n_hr % 2 != 0:
            raise ValueError("model.n_hr must be even so left/right HR wings have equal size")
        hd_distal_normalization = config.model.hd_distal_normalization.lower()
        if hd_distal_normalization not in HD_DISTAL_NORMALIZATION_MODES:
            raise ValueError(
                "model.hd_distal_normalization must be one of "
                f"{sorted(HD_DISTAL_NORMALIZATION_MODES)}"
            )
        c_hd_proximal = float(config.model.c_hd_proximal)
        g_l_hd_proximal = float(config.model.g_l_hd_proximal)
        g_d_hd_to_proximal = float(config.model.g_d_hd_to_proximal)
        if c_hd_proximal <= 0.0:
            raise ValueError("model.c_hd_proximal must be positive")
        if g_l_hd_proximal < 0.0 or g_d_hd_to_proximal < 0.0:
            raise ValueError("HD proximal conductances must be non-negative")
        total_proximal_conductance = g_l_hd_proximal + g_d_hd_to_proximal
        if total_proximal_conductance <= 0.0:
            raise ValueError("HD proximal conductances must have a positive sum")
        proximal_integration_method = str(
            config.simulation.proximal_integration_method
        ).lower()
        if proximal_integration_method not in PROXIMAL_INTEGRATION_METHODS:
            raise ValueError(
                "simulation.proximal_integration_method must be one of "
                f"{sorted(PROXIMAL_INTEGRATION_METHODS)}"
            )
        expected_distal_attenuation = (
            g_d_hd_to_proximal / total_proximal_conductance
        )
        if not np.isclose(
            config.model.p_distal_to_proximal,
            expected_distal_attenuation,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "model.p_distal_to_proximal must equal "
                "g_d_hd_to_proximal / (g_d_hd_to_proximal + g_l_hd_proximal)"
            )
        proximal_euler_factor = (
            config.simulation.dt * total_proximal_conductance / c_hd_proximal
        )
        if (
            proximal_integration_method == PROXIMAL_INTEGRATION_FORWARD_EULER
            and proximal_euler_factor >= 2.0
        ):
            raise ValueError(
                "simulation.dt is unstable for the Vafidis Eq. 4 forward-Euler "
                "update; require dt * (gL + gD) / C < 2"
            )
        activation_max_rate = float(config.model.activation.max_rate)
        if activation_max_rate <= 0.0:
            raise ValueError("model.activation.max_rate must be positive")
        return cls(
            n_theta=config.model.n_theta,
            n_hr=n_hr,
            dt=config.simulation.dt,
            proximal_integration_method=proximal_integration_method,
            tau_s=config.model.tau_s,
            tau_hd_to_hr=(
                config.model.tau_s
                if config.model.tau_hd_to_hr is None
                else config.model.tau_hd_to_hr
            ),
            tau_l_hd=config.model.tau_l_hd,
            p_distal_to_proximal=config.model.p_distal_to_proximal,
            c_hd_proximal=c_hd_proximal,
            g_l_hd_proximal=g_l_hd_proximal,
            g_d_hd_to_proximal=g_d_hd_to_proximal,
            b_hd=config.model.b_hd,
            b_hr=config.model.b_hr,
            hd_distal_normalization=hd_distal_normalization,
            activation_name=config.model.activation.name,
            activation_gain=config.model.activation.gain,
            activation_bias=config.model.activation.bias,
            activation_max_rate=activation_max_rate,
            k_vel=config.velocity.k_vel,
            visual_profile=config.visual.profile,
            visual_amplitude=config.visual.amplitude,
            visual_kappa=config.visual.kappa,
            visual_baseline=config.visual.baseline,
            visual_normalize_peak=config.visual.normalize_peak,
            visual_light_excitation=config.visual.light_excitation,
            visual_proximal_scale=config.visual.proximal_scale,
            visual_noise_std=config.visual.noise_std,
            visual_heterogeneous_sigma=config.visual.heterogeneous_sigma,
            visual_heterogeneous_beta=config.visual.heterogeneous_beta,
            visual_heterogeneous_bias=config.visual.heterogeneous_bias,
            visual_heterogeneous_n_angles=config.visual.heterogeneous_n_angles,
            visual_heterogeneous_seed=(
                config.simulation.seed + config.visual.heterogeneous_seed_offset
            ),
            visual_heterogeneous_alignment=config.visual.heterogeneous_alignment,
            visual_heterogeneous_normalization=config.visual.heterogeneous_normalization,
            visual_heterogeneous_population_sampling=(
                config.visual.heterogeneous_population_sampling
            ),
            visual_heterogeneous_master_n_hd_cells=(
                config.visual.heterogeneous_master_n_hd_cells
            ),
            tau_delta=config.learning_rule.tau_delta,
            eta_hd_to_hd=config.learning_rule.eta_hd_to_hd,
            eta_hr_to_hd=config.learning_rule.eta_hr_to_hd,
            w_hd_to_hd_min=config.learning_rule.w_hd_to_hd_min,
            w_hd_to_hd_max=config.learning_rule.w_hd_to_hd_max,
            w_hr_to_hd_min=config.learning_rule.w_hr_to_hd_min,
            w_hr_to_hd_max=config.learning_rule.w_hr_to_hd_max,
            hd_to_hd_symmetry_mode=config.learning_rule.hd_to_hd_symmetry_mode,
            hd_to_hd_balance_mode=config.learning_rule.hd_to_hd_balance_mode,
            hr_to_hd_balance_mode=config.learning_rule.hr_to_hd_balance_mode,
            zero_hd_to_hd_diagonal=config.learning_rule.zero_hd_to_hd_diagonal,
        )


@dataclass
class VafidisToyState:
    time: float
    theta_true: float
    angular_velocity: float
    theta_hd_pref: np.ndarray
    visual_tuning_profiles: np.ndarray | None
    r_hd_to_hr_lp: np.ndarray
    i_hr: np.ndarray
    r_hr: np.ndarray
    i_hd_from_hd: np.ndarray
    i_hd_from_lhr: np.ndarray
    i_hd_from_rhr: np.ndarray
    i_hd_distal: np.ndarray
    v_hd_distal: np.ndarray
    v_hd_ss: np.ndarray
    v_hd_proximal: np.ndarray
    i_vis_to_hd: np.ndarray
    r_hd: np.ndarray
    e_hd: np.ndarray
    p_hd_synaptic: np.ndarray
    p_hd: np.ndarray
    p_hr_synaptic: np.ndarray
    p_hr: np.ndarray
    delta_w_hd_to_hd: np.ndarray
    delta_w_hr_to_hd: np.ndarray
    w_hd_to_hd: np.ndarray
    w_hr_to_hd: np.ndarray
    w_hd_to_hr: np.ndarray

    @property
    def theta_hd_decoded(self) -> float:
        return pva_decode(self.theta_hd_pref, self.r_hd)

    @property
    def theta_hd_decoded_peak(self) -> float:
        return peak_decode(self.theta_hd_pref, self.r_hd)

    @property
    def w_lhr_to_hd(self) -> np.ndarray:
        n_hr_per_wing = self.w_hr_to_hd.shape[1] // 2
        return self.w_hr_to_hd[:, :n_hr_per_wing]

    @property
    def w_rhr_to_hd(self) -> np.ndarray:
        n_hr_per_wing = self.w_hr_to_hd.shape[1] // 2
        return self.w_hr_to_hd[:, n_hr_per_wing:]

    def copy(self) -> "VafidisToyState":
        return VafidisToyState(
            time=float(self.time),
            theta_true=float(self.theta_true),
            angular_velocity=float(self.angular_velocity),
            theta_hd_pref=self.theta_hd_pref.copy(),
            visual_tuning_profiles=(
                None
                if self.visual_tuning_profiles is None
                else self.visual_tuning_profiles.copy()
            ),
            r_hd_to_hr_lp=self.r_hd_to_hr_lp.copy(),
            i_hr=self.i_hr.copy(),
            r_hr=self.r_hr.copy(),
            i_hd_from_hd=self.i_hd_from_hd.copy(),
            i_hd_from_lhr=self.i_hd_from_lhr.copy(),
            i_hd_from_rhr=self.i_hd_from_rhr.copy(),
            i_hd_distal=self.i_hd_distal.copy(),
            v_hd_distal=self.v_hd_distal.copy(),
            v_hd_ss=self.v_hd_ss.copy(),
            v_hd_proximal=self.v_hd_proximal.copy(),
            i_vis_to_hd=self.i_vis_to_hd.copy(),
            r_hd=self.r_hd.copy(),
            e_hd=self.e_hd.copy(),
            p_hd_synaptic=self.p_hd_synaptic.copy(),
            p_hd=self.p_hd.copy(),
            p_hr_synaptic=self.p_hr_synaptic.copy(),
            p_hr=self.p_hr.copy(),
            delta_w_hd_to_hd=self.delta_w_hd_to_hd.copy(),
            delta_w_hr_to_hd=self.delta_w_hr_to_hd.copy(),
            w_hd_to_hd=self.w_hd_to_hd.copy(),
            w_hr_to_hd=self.w_hr_to_hd.copy(),
            w_hd_to_hr=self.w_hd_to_hr.copy(),
        )


def _activation(params: VafidisToyParams, voltage: np.ndarray) -> np.ndarray:
    return apply_activation(
        voltage,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
        max_rate=params.activation_max_rate,
    )


def _make_i_vis_to_hd_proximal(
    *,
    theta_hd_pref: np.ndarray,
    theta_true: float,
    params: VafidisToyParams,
    visual_tuning_profiles: np.ndarray | None = None,
    visual_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complete light-dependent current entering paper Eq. 4.

    Vafidis et al. Eq. 4 injects visual input and light-only excitation into
    the axon-proximal compartment. ``visual.proximal_scale`` is an optional
    experimental current gain; the paper-aligned baseline sets it to one.
    """
    visual_profile = params.visual_profile.lower()
    if visual_profile == "von_mises":
        i_vis_to_hd_raw = make_i_vis_to_hd(
            theta_hd_pref=theta_hd_pref,
            theta_true=theta_true,
            amplitude=params.visual_amplitude,
            kappa=params.visual_kappa,
            baseline=params.visual_baseline,
            normalize_peak=params.visual_normalize_peak,
        )
    elif visual_profile == "heterogeneous_gaussian_process":
        if visual_tuning_profiles is None:
            raise ValueError("heterogeneous visual profile requires generated tuning profiles")
        i_vis_to_hd_raw = make_heterogeneous_i_vis_to_hd(
            tuning_profiles=visual_tuning_profiles,
            theta_true=theta_true,
            amplitude=params.visual_amplitude,
            baseline=params.visual_baseline,
        )
    else:
        raise ValueError(f"Unknown visual.profile: {params.visual_profile}")
    i_vis_to_hd = params.visual_proximal_scale * (
        i_vis_to_hd_raw + params.visual_light_excitation
    )
    if visual_noise is not None:
        if visual_noise.shape != i_vis_to_hd.shape:
            raise ValueError("visual_noise must match the HD visual input shape")
        i_vis_to_hd = i_vis_to_hd + visual_noise
    return i_vis_to_hd


def validate_vafidis_toy_state(state: VafidisToyState, params: VafidisToyParams) -> None:
    check_model_arrays(
        r_hd=state.r_hd,
        r_hr=state.r_hr,
        w_hd_to_hd=state.w_hd_to_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        w_hd_to_hr=state.w_hd_to_hr,
        n_theta=params.n_theta,
        n_hr=params.n_hr,
    )
    for array_name in [
        "r_hd_to_hr_lp",
        "i_hr",
        "i_hd_from_hd",
        "i_hd_from_lhr",
        "i_hd_from_rhr",
        "i_hd_distal",
        "v_hd_distal",
        "v_hd_ss",
        "v_hd_proximal",
        "i_vis_to_hd",
        "e_hd",
        "p_hd_synaptic",
        "p_hd",
        "p_hr_synaptic",
        "p_hr",
        "delta_w_hd_to_hd",
        "delta_w_hr_to_hd",
    ]:
        assert_finite(getattr(state, array_name), array_name)
    for pathway_name in ["i_hd_from_hd", "i_hd_from_lhr", "i_hd_from_rhr"]:
        if getattr(state, pathway_name).shape != (params.n_theta,):
            raise ValueError(f"{pathway_name} must have shape (n_theta,)")
    if state.visual_tuning_profiles is not None:
        expected_rows = params.n_theta
        if state.visual_tuning_profiles.ndim != 2 or state.visual_tuning_profiles.shape[0] != expected_rows:
            raise ValueError("visual_tuning_profiles must have shape (n_theta, n_angles)")
        assert_finite(state.visual_tuning_profiles, "visual_tuning_profiles")
    if params.zero_hd_to_hd_diagonal and not np.allclose(
        np.diag(state.w_hd_to_hd), 0.0
    ):
        raise ValueError("w_hd_to_hd diagonal must be zero")


def initialize_vafidis_toy_state(
    *,
    config: ExperimentConfig,
    rng: np.random.Generator,
    theta_true: float | None = None,
) -> VafidisToyState:
    params = VafidisToyParams.from_config(config)
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(params.n_theta)
    visual_tuning_profiles = None
    if params.visual_profile.lower() == "heterogeneous_gaussian_process":
        visual_tuning_profiles = generate_heterogeneous_visual_profiles(
            theta_hd_pref=theta_hd_pref,
            n_angles=params.visual_heterogeneous_n_angles,
            sigma=params.visual_heterogeneous_sigma,
            beta=params.visual_heterogeneous_beta,
            bias=params.visual_heterogeneous_bias,
            seed=params.visual_heterogeneous_seed,
            alignment=params.visual_heterogeneous_alignment,
            normalization=params.visual_heterogeneous_normalization,
            population_sampling=params.visual_heterogeneous_population_sampling,
            master_n_hd_cells=params.visual_heterogeneous_master_n_hd_cells,
        )
    initial_theta_true = wrap_angle(config.simulation.theta0 if theta_true is None else theta_true)
    w_hd_to_hd = initialize_w_hd_to_hd(
        n_theta=params.n_theta,
        mode=config.model.init.w_hd_to_hd_mode,
        scale=config.model.init.w_hd_to_hd_scale,
        local_sigma=config.model.init.local_sigma,
        random_jitter=config.model.init.random_jitter,
        rng=rng,
        lower_bound=params.w_hd_to_hd_min,
        upper_bound=params.w_hd_to_hd_max,
        symmetry_mode=params.hd_to_hd_symmetry_mode,
        balance_mode=params.hd_to_hd_balance_mode,
        zero_diagonal=params.zero_hd_to_hd_diagonal,
    )
    w_hr_to_hd = initialize_w_hr_to_hd(
        n_theta=params.n_theta,
        n_hr=params.n_hr,
        mode=config.model.init.w_hr_to_hd_mode,
        scale=config.model.init.w_hr_to_hd_scale,
        local_sigma=config.model.init.local_sigma,
        random_jitter=config.model.init.random_jitter,
        rng=rng,
        lower_bound=params.w_hr_to_hd_min,
        upper_bound=params.w_hr_to_hd_max,
        balance_mode=params.hr_to_hd_balance_mode,
    )
    w_hd_to_hr = initialize_w_hd_to_hr(
        n_theta=params.n_theta,
        n_hr=params.n_hr,
        strength=config.model.w_hd_to_hr_strength,
    )
    i_hd_distal = np.zeros(params.n_theta, dtype=float)
    v_hd_distal = np.zeros(params.n_theta, dtype=float)
    i_vis_to_hd = _make_i_vis_to_hd_proximal(
        theta_hd_pref=theta_hd_pref,
        theta_true=float(initial_theta_true),
        params=params,
        visual_tuning_profiles=visual_tuning_profiles,
    )
    v_hd_proximal = compute_v_hd_proximal_steady_state(
        v_hd_distal=v_hd_distal,
        i_vis_to_hd=i_vis_to_hd,
        g_l_hd_proximal=params.g_l_hd_proximal,
        g_d_hd_to_proximal=params.g_d_hd_to_proximal,
    )
    v_hd_distal, v_hd_ss, v_hd_proximal = compute_hd_compartments(
        v_hd_distal=v_hd_distal,
        v_hd_proximal=v_hd_proximal,
        p_distal_to_proximal=params.p_distal_to_proximal,
    )
    r_hd = _activation(params, v_hd_proximal)
    r_hd_to_hr_lp = r_hd.copy()
    i_hr = compute_i_hr(
        w_hd_to_hr=w_hd_to_hr,
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_vel_to_hr=np.zeros(params.n_hr, dtype=float),
        b_hr=params.b_hr,
    )
    r_hr = _activation(params, i_hr)
    i_hd_from_hd, i_hd_from_lhr, i_hd_from_rhr = compute_hd_distal_pathway_drives(
        w_hd_to_hd=w_hd_to_hd,
        r_hd=r_hd,
        w_hr_to_hd=w_hr_to_hd,
        r_hr=r_hr,
        normalization=params.hd_distal_normalization,
    )
    r_hd_distal_prediction = _activation(params, v_hd_ss)
    e_hd = compute_e_hd(r_hd=r_hd, r_hd_distal_prediction=r_hd_distal_prediction)
    p_hd_synaptic = np.zeros(params.n_theta, dtype=float)
    p_hd = np.zeros(params.n_theta, dtype=float)
    p_hr_synaptic = np.zeros(params.n_hr, dtype=float)
    p_hr = np.zeros(params.n_hr, dtype=float)
    delta_w_hd_to_hd = np.zeros((params.n_theta, params.n_theta), dtype=float)
    delta_w_hr_to_hd = np.zeros((params.n_theta, params.n_hr), dtype=float)
    state = VafidisToyState(
        time=0.0,
        theta_true=float(initial_theta_true),
        angular_velocity=0.0,
        theta_hd_pref=theta_hd_pref,
        visual_tuning_profiles=visual_tuning_profiles,
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_hr=i_hr,
        r_hr=r_hr,
        i_hd_from_hd=i_hd_from_hd,
        i_hd_from_lhr=i_hd_from_lhr,
        i_hd_from_rhr=i_hd_from_rhr,
        i_hd_distal=i_hd_distal,
        v_hd_distal=v_hd_distal,
        v_hd_ss=v_hd_ss,
        v_hd_proximal=v_hd_proximal,
        i_vis_to_hd=i_vis_to_hd,
        r_hd=r_hd,
        e_hd=e_hd,
        p_hd_synaptic=p_hd_synaptic,
        p_hd=p_hd,
        p_hr_synaptic=p_hr_synaptic,
        p_hr=p_hr,
        delta_w_hd_to_hd=delta_w_hd_to_hd,
        delta_w_hr_to_hd=delta_w_hr_to_hd,
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        w_hd_to_hr=w_hd_to_hr,
    )
    validate_vafidis_toy_state(state, params)
    return state


def step_vafidis_toy(
    *,
    state: VafidisToyState,
    params: VafidisToyParams,
    angular_velocity: float,
    visual_teacher: bool,
    training: bool,
    visual_noise: np.ndarray | None = None,
    i_hd_distal_noise: np.ndarray | None = None,
    i_hd_proximal_noise: np.ndarray | None = None,
    i_hr_noise: np.ndarray | None = None,
) -> VafidisToyState:
    """Advance one step, with optional IID noise at all synaptic inputs."""
    for noise_name, noise_value, expected_shape in (
        ("i_hd_distal_noise", i_hd_distal_noise, (params.n_theta,)),
        ("i_hd_proximal_noise", i_hd_proximal_noise, (params.n_theta,)),
        ("i_hr_noise", i_hr_noise, (params.n_hr,)),
    ):
        if noise_value is not None:
            if np.asarray(noise_value).shape != expected_shape:
                raise ValueError(f"{noise_name} must have shape {expected_shape}")
            assert_finite(np.asarray(noise_value, dtype=float), noise_name)
    theta_true = float(wrap_angle(state.theta_true + angular_velocity * params.dt))
    i_vis_to_hd = (
        _make_i_vis_to_hd_proximal(
            theta_hd_pref=state.theta_hd_pref,
            theta_true=theta_true,
            params=params,
            visual_tuning_profiles=state.visual_tuning_profiles,
            visual_noise=visual_noise,
        )
        if visual_teacher
        else make_zero_i_vis_to_hd(params.n_theta)
    )
    i_vel_to_hr = make_i_vel_to_hr(
        n_hr=params.n_hr,
        angular_velocity=angular_velocity,
        k_vel=params.k_vel,
    )

    # Match the ordered fixed-step data stream in the released LearnPI network:
    # distal current -> distal voltage -> PSP -> HD-to-HR delay/HR -> proximal
    # voltage -> prediction error -> plasticity. The proximal substep uses the
    # configured Euler or exact-linear Eq. (4) update. All population drives
    # below deliberately use the old rates and weights from ``state``.
    i_hd_from_hd, i_hd_from_lhr, i_hd_from_rhr = compute_hd_distal_pathway_drives(
        w_hd_to_hd=state.w_hd_to_hd,
        r_hd=state.r_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        r_hr=state.r_hr,
        normalization=params.hd_distal_normalization,
    )
    i_hd_distal = euler_update_i_hd_distal_from_pathway_drives(
        i_hd_distal=state.i_hd_distal,
        i_hd_from_hd=i_hd_from_hd,
        i_hd_from_lhr=i_hd_from_lhr,
        i_hd_from_rhr=i_hd_from_rhr,
        b_hd=params.b_hd,
        dt=params.dt,
        tau_s=params.tau_s,
    )
    if i_hd_distal_noise is not None:
        i_hd_distal = i_hd_distal + (params.dt / params.tau_s) * np.asarray(
            i_hd_distal_noise,
            dtype=float,
        )
    v_hd_distal = euler_update_v_hd_distal(
        v_hd_distal=state.v_hd_distal,
        i_hd_distal=i_hd_distal,
        dt=params.dt,
        tau_l_hd=params.tau_l_hd,
    )

    p_hd_synaptic, p_hd = euler_update_psp_trace(
        p_synaptic=state.p_hd_synaptic,
        p_trace=state.p_hd,
        r_pre=state.r_hd,
        dt=params.dt,
        tau_s=params.tau_s,
        tau_l=params.tau_l_hd,
    )
    p_hr_synaptic, p_hr = euler_update_psp_trace(
        p_synaptic=state.p_hr_synaptic,
        p_trace=state.p_hr,
        r_pre=state.r_hr,
        dt=params.dt,
        tau_s=params.tau_s,
        tau_l=params.tau_l_hd,
    )

    r_hd_to_hr_lp = euler_update_r_hd_to_hr_lp(
        r_hd_to_hr_lp=state.r_hd_to_hr_lp,
        r_hd=state.r_hd,
        dt=params.dt,
        tau_s=params.tau_hd_to_hr,
    )
    i_hr = compute_i_hr(
        w_hd_to_hr=state.w_hd_to_hr,
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_vel_to_hr=i_vel_to_hr,
        b_hr=params.b_hr,
    )
    if i_hr_noise is not None:
        i_hr = i_hr + np.asarray(i_hr_noise, dtype=float)
    r_hr = _activation(params, i_hr)

    i_hd_proximal = i_vis_to_hd
    if i_hd_proximal_noise is not None:
        i_hd_proximal = i_hd_proximal + np.asarray(
            i_hd_proximal_noise,
            dtype=float,
        )
    v_hd_proximal = update_v_hd_proximal(
        v_hd_proximal=state.v_hd_proximal,
        v_hd_distal=v_hd_distal,
        i_vis_to_hd=i_hd_proximal,
        dt=params.dt,
        c_hd_proximal=params.c_hd_proximal,
        g_l_hd_proximal=params.g_l_hd_proximal,
        g_d_hd_to_proximal=params.g_d_hd_to_proximal,
        integration_method=params.proximal_integration_method,
    )
    v_hd_distal, v_hd_ss, v_hd_proximal = compute_hd_compartments(
        v_hd_distal=v_hd_distal,
        v_hd_proximal=v_hd_proximal,
        p_distal_to_proximal=params.p_distal_to_proximal,
    )
    r_hd = _activation(params, v_hd_proximal)
    r_hd_distal_prediction = _activation(params, v_hd_ss)
    e_hd = compute_e_hd(r_hd=r_hd, r_hd_distal_prediction=r_hd_distal_prediction)

    if training:
        (
            w_hd_to_hd,
            w_hr_to_hd,
            delta_w_hd_to_hd,
            delta_w_hr_to_hd,
        ) = update_predictive_local_weights(
            w_hd_to_hd=state.w_hd_to_hd,
            w_hr_to_hd=state.w_hr_to_hd,
            delta_w_hd_to_hd=state.delta_w_hd_to_hd,
            delta_w_hr_to_hd=state.delta_w_hr_to_hd,
            e_hd=e_hd,
            p_hd=p_hd,
            p_hr=p_hr,
            dt=params.dt,
            tau_delta=params.tau_delta,
            eta_hd_to_hd=params.eta_hd_to_hd,
            eta_hr_to_hd=params.eta_hr_to_hd,
            w_hd_to_hd_min=params.w_hd_to_hd_min,
            w_hd_to_hd_max=params.w_hd_to_hd_max,
            w_hr_to_hd_min=params.w_hr_to_hd_min,
            w_hr_to_hd_max=params.w_hr_to_hd_max,
            hd_to_hd_symmetry_mode=params.hd_to_hd_symmetry_mode,
            hd_to_hd_balance_mode=params.hd_to_hd_balance_mode,
            hr_to_hd_balance_mode=params.hr_to_hd_balance_mode,
            zero_hd_to_hd_diagonal=params.zero_hd_to_hd_diagonal,
        )
    else:
        w_hd_to_hd = state.w_hd_to_hd.copy()
        w_hr_to_hd = state.w_hr_to_hd.copy()
        delta_w_hd_to_hd = state.delta_w_hd_to_hd.copy()
        delta_w_hr_to_hd = state.delta_w_hr_to_hd.copy()

    next_state = VafidisToyState(
        time=state.time + params.dt,
        theta_true=theta_true,
        angular_velocity=float(angular_velocity),
        theta_hd_pref=state.theta_hd_pref.copy(),
        visual_tuning_profiles=state.visual_tuning_profiles,
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_hr=i_hr,
        r_hr=r_hr,
        i_hd_from_hd=i_hd_from_hd,
        i_hd_from_lhr=i_hd_from_lhr,
        i_hd_from_rhr=i_hd_from_rhr,
        i_hd_distal=i_hd_distal,
        v_hd_distal=v_hd_distal,
        v_hd_ss=v_hd_ss,
        v_hd_proximal=v_hd_proximal,
        i_vis_to_hd=i_vis_to_hd,
        r_hd=r_hd,
        e_hd=e_hd,
        p_hd_synaptic=p_hd_synaptic,
        p_hd=p_hd,
        p_hr_synaptic=p_hr_synaptic,
        p_hr=p_hr,
        delta_w_hd_to_hd=delta_w_hd_to_hd,
        delta_w_hr_to_hd=delta_w_hr_to_hd,
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        w_hd_to_hr=state.w_hd_to_hr.copy(),
    )
    validate_vafidis_toy_state(next_state, params)
    return next_state
