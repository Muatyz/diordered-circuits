"""Canonical frozen, zero-input dynamics for attractor diagnostics.

The Vafidis toy step stores several derived arrays.  Five population blocks
affect a later autonomous step: the low-pass HD-to-HR rate, the one-step-
lagged HR rate, the HD distal current, and the distal and proximal HD
voltages.  Keeping ``r_hr`` explicit is important because the released
implementation uses the previous HR rate when updating the HD distal current;
keeping ``v_hd_proximal`` explicit is required by paper Eq. (4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from learning.common.angles import pva_decode
from learning.dynamics.activation import apply_activation
from learning.dynamics.hd_dynamics import (
    effective_hd_distal_weight_matrices,
    proximal_voltage_update_coefficients,
)

if TYPE_CHECKING:
    from learning.models.vafidis_toy import VafidisToyParams, VafidisToyState


AUTONOMOUS_STATE_COMPONENTS = (
    "r_hd_to_hr_lp",
    "r_hr",
    "i_hd_distal",
    "v_hd_distal",
    "v_hd_proximal",
)


@dataclass(frozen=True)
class FrozenAutonomousDynamics:
    """Frozen weights and exact discrete map used by darkness tests."""

    params: VafidisToyParams
    theta_hd_pref: np.ndarray
    w_hd_to_hd: np.ndarray
    w_hr_to_hd: np.ndarray
    w_hd_to_hr: np.ndarray

    @classmethod
    def from_state(
        cls,
        *,
        params: VafidisToyParams,
        state: VafidisToyState,
    ) -> "FrozenAutonomousDynamics":
        return cls(
            params=params,
            theta_hd_pref=np.asarray(state.theta_hd_pref, dtype=float).copy(),
            w_hd_to_hd=np.asarray(state.w_hd_to_hd, dtype=float).copy(),
            w_hr_to_hd=np.asarray(state.w_hr_to_hd, dtype=float).copy(),
            w_hd_to_hr=np.asarray(state.w_hd_to_hr, dtype=float).copy(),
        )

    @property
    def state_dimension(self) -> int:
        return 4 * self.params.n_theta + self.params.n_hr

    @property
    def component_slices(self) -> dict[str, slice]:
        n_theta = self.params.n_theta
        n_hr = self.params.n_hr
        return {
            "r_hd_to_hr_lp": slice(0, n_theta),
            "r_hr": slice(n_theta, n_theta + n_hr),
            "i_hd_distal": slice(n_theta + n_hr, 2 * n_theta + n_hr),
            "v_hd_distal": slice(2 * n_theta + n_hr, 3 * n_theta + n_hr),
            "v_hd_proximal": slice(3 * n_theta + n_hr, 4 * n_theta + n_hr),
        }

    def pack_state(self, state: VafidisToyState) -> np.ndarray:
        """Pack the minimal Markov state in a documented block order."""
        vector = np.concatenate(
            [np.asarray(getattr(state, name), dtype=float) for name in AUTONOMOUS_STATE_COMPONENTS]
        )
        self._validate_vector(vector)
        return vector

    def unpack_state(self, state_vector: np.ndarray) -> dict[str, np.ndarray]:
        """Return views of a packed state keyed by physical variable name."""
        state_vector = np.asarray(state_vector, dtype=float)
        self._validate_vector(state_vector)
        return {
            name: state_vector[current_slice]
            for name, current_slice in self.component_slices.items()
        }

    def _validate_vector(self, state_vector: np.ndarray) -> None:
        if state_vector.shape != (self.state_dimension,):
            raise ValueError(
                "autonomous state must have shape "
                f"({self.state_dimension},), got {state_vector.shape}"
            )
        if not np.all(np.isfinite(state_vector)):
            raise FloatingPointError("autonomous state contains NaN or Inf")

    def _activation(self, voltage: np.ndarray) -> np.ndarray:
        return apply_activation(
            voltage,
            activation_name=self.params.activation_name,
            gain=self.params.activation_gain,
            bias=self.params.activation_bias,
            max_rate=self.params.activation_max_rate,
        )

    def _activation_derivative(self, voltage: np.ndarray) -> np.ndarray:
        if self.params.activation_name != "sigmoid":
            raise ValueError(
                "analytic autonomous Jacobian currently requires sigmoid activation"
            )
        firing_rate = self._activation(voltage)
        return self.params.activation_gain * firing_rate * (
            1.0 - firing_rate / self.params.activation_max_rate
        )

    def hd_rate(self, state_vector: np.ndarray) -> np.ndarray:
        blocks = self.unpack_state(state_vector)
        return self._activation(blocks["v_hd_proximal"])

    def firing_rate_state(self, state_vector: np.ndarray) -> np.ndarray:
        """Return the HD+HR firing-rate observable used only for PCA plots.

        The autonomous flow and Jacobian remain defined on the full canonical
        Markov state.  Concatenating the current HD and HR rates gives a
        neuron-level observable analogous to the hidden-rate state plotted by
        Ramesan et al., without treating the three-PC projection as dynamics.
        """
        blocks = self.unpack_state(state_vector)
        return np.concatenate([self.hd_rate(state_vector), blocks["r_hr"]])

    def decoded_heading(self, state_vector: np.ndarray) -> float:
        return pva_decode(self.theta_hd_pref, self.hd_rate(state_vector))

    def step(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply the exact frozen, visual-off, zero-velocity discrete map."""
        blocks = self.unpack_state(state_vector)
        params = self.params
        r_hd = self.hd_rate(state_vector)

        effective_w_hd_to_hd, effective_w_hr_to_hd = (
            effective_hd_distal_weight_matrices(
                w_hd_to_hd=self.w_hd_to_hd,
                w_hr_to_hd=self.w_hr_to_hd,
                normalization=params.hd_distal_normalization,
            )
        )
        distal_drive = (
            effective_w_hd_to_hd @ r_hd
            + effective_w_hr_to_hd @ blocks["r_hr"]
            - params.b_hd
        )
        synaptic_fraction = params.dt / params.tau_s
        i_hd_distal_next = blocks["i_hd_distal"] + synaptic_fraction * (
            -blocks["i_hd_distal"] + distal_drive
        )
        distal_voltage_fraction = params.dt / params.tau_l_hd
        v_hd_distal_next = blocks["v_hd_distal"] + distal_voltage_fraction * (
            -blocks["v_hd_distal"] + i_hd_distal_next
        )
        hd_to_hr_fraction = params.dt / params.tau_hd_to_hr
        r_hd_to_hr_lp_next = blocks["r_hd_to_hr_lp"] + hd_to_hr_fraction * (
            -blocks["r_hd_to_hr_lp"] + r_hd
        )
        i_hr_next = self.w_hd_to_hr @ r_hd_to_hr_lp_next - params.b_hr
        r_hr_next = self._activation(i_hr_next)
        proximal_retention, proximal_coupling, _ = (
            proximal_voltage_update_coefficients(
                dt=params.dt,
                c_hd_proximal=params.c_hd_proximal,
                g_l_hd_proximal=params.g_l_hd_proximal,
                g_d_hd_to_proximal=params.g_d_hd_to_proximal,
                integration_method=params.proximal_integration_method,
            )
        )
        v_hd_proximal_next = (
            proximal_retention * blocks["v_hd_proximal"]
            + proximal_coupling * v_hd_distal_next
        )
        next_vector = np.concatenate(
            [
                r_hd_to_hr_lp_next,
                r_hr_next,
                i_hd_distal_next,
                v_hd_distal_next,
                v_hd_proximal_next,
            ]
        )
        self._validate_vector(next_vector)
        return next_vector

    def flow(self, state_vector: np.ndarray) -> np.ndarray:
        """Return the Euler-equivalent vector field ``(G_dt(x)-x)/dt``."""
        return (self.step(state_vector) - np.asarray(state_vector, dtype=float)) / self.params.dt

    def flow_jacobian(self, state_vector: np.ndarray) -> np.ndarray:
        """Return the analytic Jacobian of :meth:`flow`.

        This is the Jacobian of the exact discrete implementation converted to
        an Euler-equivalent flow.  The algebraic HR update therefore produces
        fast rates of order ``-1 / dt``; those modes are part of the current
        implementation and are intentionally not hidden by a reduced model.
        """
        blocks = self.unpack_state(state_vector)
        params = self.params
        n_theta = params.n_theta
        n_hr = params.n_hr
        slices = self.component_slices
        map_jacobian = np.zeros(
            (self.state_dimension, self.state_dimension), dtype=float
        )

        hd_voltage = blocks["v_hd_proximal"]
        d_r_hd_d_va = self._activation_derivative(hd_voltage)
        hd_to_hr_fraction = params.dt / params.tau_hd_to_hr
        lp_retention = 1.0 - hd_to_hr_fraction
        d_lp_d_lp = lp_retention * np.eye(n_theta)
        d_lp_d_va = hd_to_hr_fraction * np.diag(d_r_hd_d_va)
        map_jacobian[slices["r_hd_to_hr_lp"], slices["r_hd_to_hr_lp"]] = d_lp_d_lp
        map_jacobian[slices["r_hd_to_hr_lp"], slices["v_hd_proximal"]] = d_lp_d_va

        r_hd_to_hr_lp_next = blocks["r_hd_to_hr_lp"] + hd_to_hr_fraction * (
            -blocks["r_hd_to_hr_lp"] + self._activation(hd_voltage)
        )
        i_hr_next = self.w_hd_to_hr @ r_hd_to_hr_lp_next - params.b_hr
        d_r_hr_d_i_hr = self._activation_derivative(i_hr_next)
        hr_gain_matrix = d_r_hr_d_i_hr[:, None] * self.w_hd_to_hr
        map_jacobian[slices["r_hr"], slices["r_hd_to_hr_lp"]] = (
            lp_retention * hr_gain_matrix
        )
        map_jacobian[slices["r_hr"], slices["v_hd_proximal"]] = (
            hr_gain_matrix * (hd_to_hr_fraction * d_r_hd_d_va)[None, :]
        )

        effective_w_hd_to_hd, effective_w_hr_to_hd = (
            effective_hd_distal_weight_matrices(
                w_hd_to_hd=self.w_hd_to_hd,
                w_hr_to_hd=self.w_hr_to_hd,
                normalization=params.hd_distal_normalization,
            )
        )
        synaptic_fraction = params.dt / params.tau_s
        distal_current_retention = 1.0 - synaptic_fraction
        d_i_d_r_hr = synaptic_fraction * effective_w_hr_to_hd
        d_i_d_i = distal_current_retention * np.eye(n_theta)
        d_i_d_va = synaptic_fraction * (
            effective_w_hd_to_hd * d_r_hd_d_va[None, :]
        )
        map_jacobian[slices["i_hd_distal"], slices["r_hr"]] = d_i_d_r_hr
        map_jacobian[slices["i_hd_distal"], slices["i_hd_distal"]] = d_i_d_i
        map_jacobian[slices["i_hd_distal"], slices["v_hd_proximal"]] = d_i_d_va

        distal_voltage_fraction = params.dt / params.tau_l_hd
        voltage_retention = 1.0 - distal_voltage_fraction
        map_jacobian[slices["v_hd_distal"], slices["r_hr"]] = (
            distal_voltage_fraction * d_i_d_r_hr
        )
        map_jacobian[slices["v_hd_distal"], slices["i_hd_distal"]] = (
            distal_voltage_fraction * d_i_d_i
        )
        map_jacobian[slices["v_hd_distal"], slices["v_hd_distal"]] = (
            voltage_retention * np.eye(n_theta)
        )
        map_jacobian[slices["v_hd_distal"], slices["v_hd_proximal"]] = (
            distal_voltage_fraction * d_i_d_va
        )

        proximal_retention, proximal_coupling, _ = (
            proximal_voltage_update_coefficients(
                dt=params.dt,
                c_hd_proximal=params.c_hd_proximal,
                g_l_hd_proximal=params.g_l_hd_proximal,
                g_d_hd_to_proximal=params.g_d_hd_to_proximal,
                integration_method=params.proximal_integration_method,
            )
        )
        map_jacobian[slices["v_hd_proximal"], slices["r_hr"]] = (
            proximal_coupling * distal_voltage_fraction * d_i_d_r_hr
        )
        map_jacobian[slices["v_hd_proximal"], slices["i_hd_distal"]] = (
            proximal_coupling * distal_voltage_fraction * d_i_d_i
        )
        map_jacobian[slices["v_hd_proximal"], slices["v_hd_distal"]] = (
            proximal_coupling * voltage_retention * np.eye(n_theta)
        )
        map_jacobian[slices["v_hd_proximal"], slices["v_hd_proximal"]] = (
            proximal_retention * np.eye(n_theta)
            + proximal_coupling * distal_voltage_fraction * d_i_d_va
        )

        return (map_jacobian - np.eye(self.state_dimension)) / params.dt
