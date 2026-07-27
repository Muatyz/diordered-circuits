import numpy as np
import pytest

from prospective.theory.equilibrium import equilibrium_widths
from prospective.theory.hermite import prospective_gamma


def test_beta_half_matches_paper_width_identity():
    sigma_j, sigma_u = equilibrium_widths(0.5, 5.0)
    assert np.isclose(sigma_j, 5.0)
    assert np.isclose(sigma_u / np.sqrt(2.0), 5.0)


def test_widths_diverge_toward_beta_two():
    assert equilibrium_widths(1.9, 5.0)[0] > equilibrium_widths(1.0, 5.0)[0]
    with pytest.raises(ValueError):
        equilibrium_widths(2.0, 5.0)


def test_zero_speed_has_zero_hermite_shift():
    gamma, lag = prospective_gamma(speed=0.0, adaptation_strength=0.2, tau_u=0.015, tau_v=0.6, sigma_u=7.0)
    assert gamma == 0.0
    assert lag == 0.0

