from __future__ import annotations
from typing import TYPE_CHECKING
import math


from terraflow.test_case.abstract import AbstractShallowWaterModelTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class MountainZonalFlowTestCase(AbstractShallowWaterModelTestCase):
    """Mountain zonal flow test case (Williamson et al. 1992, Test Case 5).

    This test case simulates a zonal flow over an isolated mountain.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    alpha: float 
    gz0  : float 
    gzs0 : float 
    u0   : float 
    lon0 : float 
    lat0 : float 
    R    : float 

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "swm_mz"
        self.name = "Mountain Zonal Flow"
        self.description = (
            "A test case simulating zonal flow over a mountain in a shallow water "
            "atmosphere model."
        )
        self.references = (
            "Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N. (1992). "
            "A standard test set for numerical approximations to the shallow water equations in spherical geometry. "
            "Journal of Computational Physics, 102(1), 211-224.",
        )
        self.keywords = ("shallow water", "mountain zonal flow", "atmosphere")
        self.alpha = 0.0
        self.gz0   = 5960.0 * para.gravity.unwrap()
        self.gzs0  = 2000.0 * para.gravity.unwrap()
        self.u0    = 20.0
        self.lon0  = math.pi * 1.5
        self.lat0  = math.pi / 6.0
        self.R     = math.pi / 9.0
        super().__init__(para)
    