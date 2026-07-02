"""Weight matrix plots."""

from __future__ import annotations

from pathlib import Path

from learning.plotting.backend import use_headless_backend

use_headless_backend()

import matplotlib.pyplot as plt
import numpy as np


def plot_weight_matrix(
    *,
    weight_matrix: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "coolwarm",
    x_label: str = "source neuron index [unitless]",
    y_label: str = "target HD neuron index [unitless]",
    colorbar_label: str = "synaptic weight [a.u.]",
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.14, top=0.88)
    max_abs_weight = float(np.nanmax(np.abs(weight_matrix))) if weight_matrix.size else 1.0
    if max_abs_weight <= 0.0:
        max_abs_weight = 1.0
    mesh = axis.imshow(
        weight_matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-max_abs_weight if np.min(weight_matrix) < 0.0 else 0.0,
        vmax=max_abs_weight,
        extent=extent,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    fig.colorbar(mesh, ax=axis, label=colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)
