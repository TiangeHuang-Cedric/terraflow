from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class RossbyHaurwitzWaveTestCase(AbstractBaroclinicModelTestCase):
    """3D Rossby-Haurwitz wave test case."""
    n    : int
    u0   : float  # m s-1
    t0   : float  # K
    gamma: float  # K m-1
    pref : float  # Pa
    M    : float 

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "rh"
        self.name = "3D Rossby-Haurwitz Wave"
        self.description = "3D Rossby-Haurwitz wave test case for baroclinic dynamics."
        self.references = (
            "Jablonowski, C., Lauritzen, P., Nair, R. & Taylor, M. "
            "Idealized test cases for the dynamical cores of Atmospheric General Circulation Models: "
            "A proposal for the NCAR ASP 2008 summer colloquium.",
        )
        self.keywords = ("baroclinic", "Rossby-Haurwitz wave", "atmosphere")
        self.n     = 4
        self.u0    = 50
        self.t0    = 288    # K
        self.gamma = 0.0065 # K m-1
        self.pref  = 955e2  # Pa
        self.M     = self.u0 / (self.n * para.radius.unwrap())
        super().__init__(para)
