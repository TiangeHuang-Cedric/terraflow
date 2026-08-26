from jax import numpy as jnp

from terraflow.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractAdvectionModel2DTestCase


class SolidRotationTestCase(AbstractAdvectionModel2DTestCase):
    """Solid body rotation test case for advection dynamics."""
    period: float 
    h0    : float 
    lon0  : float 
    lat0  : float 
    alpha : float 
    u0    : float 

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "adv_sr"
        self.name = "Solid Rotation"
        self.description = "Solid body rotation test case for advection dynamics."
        self.references = (
            "Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N. (1992). "
            "A standard test set for numerical approximations to the shallow water equations in spherical geometry. "
            "Journal of Computational Physics, 102(1), 211-224.",
        )
        self.keywords = ("advection", "solid rotation", "atmosphere")
        self.period = 12 * 86400.0
        self.h0     = 1000.0
        self.lon0   = 3 * jnp.pi / 2
        self.lat0   = 0.0
        self.alpha  = jnp.pi / 2
        self.u0     = 2 * jnp.pi * para.radius.unwrap() / self.period
        super().__init__(para)