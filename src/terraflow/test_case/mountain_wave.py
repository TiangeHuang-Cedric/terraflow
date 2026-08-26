from __future__ import annotations
from typing import TYPE_CHECKING
from jax import numpy as jnp

from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase

if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet

class MountainWaveTestCase(AbstractBaroclinicModelTestCase):
    """Baroclinic wave test case with a mountain perturbation."""
    T0  :float  # K
    h0  :float  # m
    d   :float  # m
    u0  :float  # m s-1
    lonc:float  # rad
    latc:float  # rad
    kap :float
    psp :float  # Pa
    N   :float  # s-1

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "mz"
        self.name = "Mountain Wave"
        self.description = "Baroclinic wave test case with a mountain perturbation."
        self.references = (
            "Jablonowski, C., Lauritzen, P., Nair, R. & Taylor, M. "
            "Idealized test cases for the dynamical cores of Atmospheric General Circulation Models: "
            "A proposal for the NCAR ASP 2008 summer colloquium.",
        )
        self.keywords = ("baroclinic", "mountain wave", "atmosphere")
        self.T0 = 288.0     # K
        self.h0 = 2000.0    # m
        self.d = 1.5e6
        self.u0 = 20.0      # m s-1
        self.lonc = jnp.pi / 2.0
        self.latc = jnp.pi / 6.0
        self.kap = 2.0 / 7.0
        self.psp = 93000.0  # Pa
        self.N = 0.0182     # s-1
        super().__init__(para)