from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase

if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet


class SteadyStateTestCase(AbstractBaroclinicModelTestCase):
    alpha: float 
    u0   : float   # m s-1
    t0   : float   # K
    gamma: float 
    dt   : float 
    eta0 : float 
    etat : float   # Tropopause level

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "ss"
        self.name = "Steady State"
        self.description = "Steady state test case for baroclinic dynamics."
        self.references = (
            "Jablonowski, C. & Williamson, D. L. "
            "A baroclinic instability test case for atmospheric model dynamical cores. "
            "Quarterly Journal of the Royal Meteorological Society 132, 2943–2975 (2006).",

            "Jablonowski, C., Lauritzen, P., Nair, R. & Taylor, M. "
            "Idealized test cases for the dynamical cores of Atmospheric General Circulation Models: "
            "A proposal for the NCAR ASP 2008 summer colloquium."
        )
        self.keywords = ("baroclinic", "steady state", "atmosphere")
        self.alpha = 0.0
        self.u0    = 35.0   # m s-1
        self.t0    = 288.0  # K
        self.gamma = 0.005  # K m-1
        self.dt    = 4.8e5  # K
        self.eta0  = 0.252
        self.etat  = 0.2    # Tropopause level
        super().__init__(para)
