"""Fast competitive-layer dynamics."""

from prospective.dynamics.activation import divisive_quadratic_rate
from prospective.dynamics.competitive import competitive_euler_step

__all__ = ["competitive_euler_step", "divisive_quadratic_rate"]

