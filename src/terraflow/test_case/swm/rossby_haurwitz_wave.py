from __future__ import annotations
from typing import TYPE_CHECKING
from terraflow.test_case.abstract import AbstractShallowWaterModelTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class RossbyHaurwitzWaveTestCase(AbstractShallowWaterModelTestCase):
    """Rossby-Haurwitz wave test case (Williamson et al. 1992, Test Case 6).

    This test case simulates a Rossby-Haurwitz wave in a shallow water
    atmosphere model.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    omg: float
    K  : float 
    gz0: float
    R  : float 

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "swm_rh"
        self.name = "Rossby-Haurwitz Wave"
        self.description = (
            "A test case simulating a Rossby-Haurwitz wave in a shallow water "
            "atmosphere model."
        )
        self.references = (
            "Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N. (1992). "
            "A standard test set for numerical approximations to the shallow water equations in spherical geometry. "
            "Journal of Computational Physics, 102(1), 211-224.",
        )
        self.keywords = ("shallow water", "Rossby-Haurwitz wave", "atmosphere")
        self.omg = 7.848e-6
        self.K   = 7.848e-6
        self.gz0 = 8000.0 * para.gravity.unwrap()
        self.R   = 4
        super().__init__(para)