"""PSP trace dynamics."""

from __future__ import annotations

import numpy as np


def euler_update_psp_trace(
    *,
    p_synaptic: np.ndarray,
    p_trace: np.ndarray,
    r_pre: np.ndarray,
    dt: float,
    tau_s: float,
    tau_l: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler update for the PSP filter in Vafidis et al. Eq. 13-14.

    The two cascaded first-order filters implement the double-exponential
    kernel H(t) from the paper.  The second stage uses the just-updated
    synaptic stage, matching the update order in the released LearnPI code.
    """
    next_p_synaptic = p_synaptic + (dt / tau_s) * (-p_synaptic + r_pre)
    next_p_trace = p_trace + (dt / tau_l) * (-p_trace + next_p_synaptic)
    return next_p_synaptic, next_p_trace
