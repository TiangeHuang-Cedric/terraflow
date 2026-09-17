import math


from terraflow.util.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractShallowWaterModelTestCase


class JetZonalFlowTestCase(AbstractShallowWaterModelTestCase):
    """Jet zonal flow test case (Galewsky et al. 2004).

    This test case simulates a zonal jet flow in a shallow water atmosphere model.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    u_max: float
    lat0 : float 
    lat1 : float
    lat2 : float
    en   : float
    alpha: float
    beta : float
    gh0  : float
    ghd  : float

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "swm_jz"
        self.name = "Jet Zonal Flow"
        self.description = (
            "A test case simulating a zonal jet flow in a shallow water "
            "atmosphere model."
        )
        self.references = (
            "Galewsky, J., Scott, R. K., & Polvani, L. M. (2004). "
            "An initial-value problem for testing numerical models of the global shallow-water equations. "
            "Tellus A: Dynamic Meteorology and Oceanography, 56(5), 429-440.",
        )
        self.keywords = ("shallow water", "jet zonal flow", "atmosphere")
        self.u_max = 80.0
        self.lat0  = math.pi / 7.0
        self.lat1  = math.pi / 2.0
        self.lat1  = self.lat1 - self.lat0
        self.en    = float(math.exp(-4.0 / ((self.lat1 - self.lat0) ** 2)))
        self.lat2  = math.pi / 4.0
        self.alpha = 1.0 / 3.0
        self.beta  = 1.0 / 15.0
        self.gh0   = 1.0e4 * para.gravity.unwrap()
        self.ghd   = 120.0 * para.gravity.unwrap()
        super().__init__(para)
