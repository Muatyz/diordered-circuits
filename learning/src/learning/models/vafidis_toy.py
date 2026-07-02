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
    compute_hd_compartments,
    euler_update_i_hd_distal,
    euler_update_v_hd_distal,
)
from learning.dynamics.hr_dynamics import compute_i_hr, euler_update_r_hd_to_hr_lp
from learning.plasticity.predictive_local import compute_e_hd, update_predictive_local_weights
from learning.plasticity.traces import euler_update_psp_trace
from learning.stimuli.velocity import make_i_vel_to_hr
from learning.stimuli.visual import make_i_vis_to_hd, make_zero_i_vis_to_hd


@dataclass(frozen=True)
class VafidisToyParams:
    n_theta: int
    n_hr: int
    dt: float
    tau_s: float
    tau_hd_to_hr: float
    tau_l_hd: float
    p_distal_to_proximal: float
    b_hd: float
    b_hr: float
    activation_name: str
    activation_gain: float
    activation_bias: float
    k_vel: float
    visual_amplitude: float
    visual_kappa: float
    visual_baseline: float
    visual_normalize_peak: bool
    visual_light_excitation: float
    visual_proximal_scale: float
    tau_delta: float
    eta_hd_to_hd: float
    eta_hr_to_hd: float
    w_hd_to_hd_min: float
    w_hd_to_hd_max: float
    w_hr_to_hd_min: float
    w_hr_to_hd_max: float
    hd_to_hd_symmetry_mode: str
    hd_to_hd_balance_mode: str
    hr_to_hd_balance_mode: str

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "VafidisToyParams":
        n_hr = config.model.n_hr
        if n_hr != config.model.n_theta:
            raise ValueError("The Vafidis-style toy model expects model.n_hr == model.n_theta")
        if n_hr % 2 != 0:
            raise ValueError("model.n_hr must be even so left/right HR wings have equal size")
        return cls(
            n_theta=config.model.n_theta,
            n_hr=n_hr,
            dt=config.simulation.dt,
            tau_s=config.model.tau_s,
            tau_hd_to_hr=(
                config.model.tau_s
                if config.model.tau_hd_to_hr is None
                else config.model.tau_hd_to_hr
            ),
            tau_l_hd=config.model.tau_l_hd,
            p_distal_to_proximal=config.model.p_distal_to_proximal,
            b_hd=config.model.b_hd,
            b_hr=config.model.b_hr,
            activation_name=config.model.activation.name,
            activation_gain=config.model.activation.gain,
            activation_bias=config.model.activation.bias,
            k_vel=config.velocity.k_vel,
            visual_amplitude=config.visual.amplitude,
            visual_kappa=config.visual.kappa,
            visual_baseline=config.visual.baseline,
            visual_normalize_peak=config.visual.normalize_peak,
            visual_light_excitation=config.visual.light_excitation,
            visual_proximal_scale=config.visual.proximal_scale,
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
        )


@dataclass
class VafidisToyState:
    time: float
    theta_true: float
    angular_velocity: float
    theta_hd_pref: np.ndarray
    r_hd_to_hr_lp: np.ndarray
    i_hr: np.ndarray
    r_hr: np.ndarray
    i_hd_distal: np.ndarray
    v_hd_distal: np.ndarray
    v_hd_ss: np.ndarray
    v_hd_proximal: np.ndarray
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
            r_hd_to_hr_lp=self.r_hd_to_hr_lp.copy(),
            i_hr=self.i_hr.copy(),
            r_hr=self.r_hr.copy(),
            i_hd_distal=self.i_hd_distal.copy(),
            v_hd_distal=self.v_hd_distal.copy(),
            v_hd_ss=self.v_hd_ss.copy(),
            v_hd_proximal=self.v_hd_proximal.copy(),
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
    )


def _make_i_vis_to_hd_proximal(
    *,
    theta_hd_pref: np.ndarray,
    theta_true: float,
    params: VafidisToyParams,
) -> np.ndarray:
    """Return the effective axon-proximal visual drive.

    Vafidis et al. Eq. 4 injects visual input and light-only excitation into
    the axon-proximal compartment.  In the steady-state reduction used by the
    toy model, Eq. 31 scales that current by 1 / (gD + gL).
    """
    i_vis_to_hd_raw = make_i_vis_to_hd(
        theta_hd_pref=theta_hd_pref,
        theta_true=theta_true,
        amplitude=params.visual_amplitude,
        kappa=params.visual_kappa,
        baseline=params.visual_baseline,
        normalize_peak=params.visual_normalize_peak,
    )
    return params.visual_proximal_scale * (
        i_vis_to_hd_raw + params.visual_light_excitation
    )


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
        "i_hd_distal",
        "v_hd_distal",
        "v_hd_ss",
        "v_hd_proximal",
        "e_hd",
        "p_hd_synaptic",
        "p_hd",
        "p_hr_synaptic",
        "p_hr",
        "delta_w_hd_to_hd",
        "delta_w_hr_to_hd",
    ]:
        assert_finite(getattr(state, array_name), array_name)
    if not np.allclose(np.diag(state.w_hd_to_hd), 0.0):
        raise ValueError("w_hd_to_hd diagonal must be zero")


def initialize_vafidis_toy_state(
    *,
    config: ExperimentConfig,
    rng: np.random.Generator,
    theta_true: float | None = None,
) -> VafidisToyState:
    params = VafidisToyParams.from_config(config)
    theta_hd_pref = make_vafidis_paired_theta_hd_pref(params.n_theta)
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
    )
    v_hd_distal, v_hd_ss, v_hd_proximal = compute_hd_compartments(
        v_hd_distal=v_hd_distal,
        i_vis_to_hd=i_vis_to_hd,
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
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_hr=i_hr,
        r_hr=r_hr,
        i_hd_distal=i_hd_distal,
        v_hd_distal=v_hd_distal,
        v_hd_ss=v_hd_ss,
        v_hd_proximal=v_hd_proximal,
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


def make_visual_input_for_state(
    *,
    state: VafidisToyState,
    params: VafidisToyParams,
    visual_teacher: bool,
) -> np.ndarray:
    if not visual_teacher:
        return make_zero_i_vis_to_hd(params.n_theta)
    return _make_i_vis_to_hd_proximal(
        theta_hd_pref=state.theta_hd_pref,
        theta_true=state.theta_true,
        params=params,
    )


def step_vafidis_toy(
    *,
    state: VafidisToyState,
    params: VafidisToyParams,
    angular_velocity: float,
    visual_teacher: bool,
    training: bool,
) -> VafidisToyState:
    """Advance the toy model by one Euler step."""
    theta_true = float(wrap_angle(state.theta_true + angular_velocity * params.dt))
    i_vis_to_hd = (
        _make_i_vis_to_hd_proximal(
            theta_hd_pref=state.theta_hd_pref,
            theta_true=theta_true,
            params=params,
        )
        if visual_teacher
        else make_zero_i_vis_to_hd(params.n_theta)
    )
    i_vel_to_hr = make_i_vel_to_hr(
        n_hr=params.n_hr,
        angular_velocity=angular_velocity,
        k_vel=params.k_vel,
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
    r_hr = _activation(params, i_hr)

    i_hd_distal = euler_update_i_hd_distal(
        i_hd_distal=state.i_hd_distal,
        w_hd_to_hd=state.w_hd_to_hd,
        r_hd=state.r_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        r_hr=state.r_hr,
        b_hd=params.b_hd,
        dt=params.dt,
        tau_s=params.tau_s,
    )
    v_hd_distal = euler_update_v_hd_distal(
        v_hd_distal=state.v_hd_distal,
        i_hd_distal=i_hd_distal,
        dt=params.dt,
        tau_l_hd=params.tau_l_hd,
    )
    v_hd_distal, v_hd_ss, v_hd_proximal = compute_hd_compartments(
        v_hd_distal=v_hd_distal,
        i_vis_to_hd=i_vis_to_hd,
        p_distal_to_proximal=params.p_distal_to_proximal,
    )
    r_hd = _activation(params, v_hd_proximal)
    r_hd_distal_prediction = _activation(params, v_hd_ss)
    e_hd = compute_e_hd(r_hd=r_hd, r_hd_distal_prediction=r_hd_distal_prediction)

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
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        i_hr=i_hr,
        r_hr=r_hr,
        i_hd_distal=i_hd_distal,
        v_hd_distal=v_hd_distal,
        v_hd_ss=v_hd_ss,
        v_hd_proximal=v_hd_proximal,
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
