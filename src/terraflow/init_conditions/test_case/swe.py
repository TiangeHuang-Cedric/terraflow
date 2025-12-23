from typing import Dict, Union
from jax import numpy as jnp
import equinox as eqx

from terraflow.util import typing, math
from terraflow.planet import PlanetParameterSet

Float = typing.FloatType
Array = typing.Array


class ShallowWaterEquationTestCase(eqx.Module):
    """Base class for shallow water equation test cases.

    This class uses Equinox Module which makes it compatible with JAX transformations.
    Static fields (para, name, description, etc.) are marked to exclude them from pytree,
    as they don't change during computation and don't need to be traced by JAX.
    """
    _para: PlanetParameterSet = eqx.field()
    name: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    references: tuple[str, ...] = eqx.field(static=True)
    keywords: tuple[str, ...] = eqx.field(static=True)

    @property
    def para(self) -> PlanetParameterSet:
        return self._para

    def get_info(self) -> Dict[str, Union[str, tuple[str, ...]]]:
        return {
            "name": self.name,
            "description": self.description,
            "references": self.references,
            "keywords": self.keywords
        }

    def init_normal_velocity(
        self,
        lon: Array,
        lat: Array,
        cartesian: Array,
        edge_normal: Array,
    ) -> Array:
        """Initialize normal velocity at edges.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        u = self.init_u(lon, lat)
        v = self.init_v(lon, lat)
        velocity_3d_cart = math.convert_vector_ltln2cart(u, v, cartesian)
        return jnp.sum(velocity_3d_cart * edge_normal, axis=-1)

    def init_thickness(
        self,
        lon: Array,
        lat: Array,
    ) -> Array:
        """Initialize thickness field (height - topography).

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        h = self.init_height(lon, lat)
        hs = self.init_topography(lon, lat)
        return h - hs

    def init_topography(self, lon: Array, lat: Array) -> Array:
        """Initialize topography field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Topography initialization is not implemented."
        )

    def init_u(self, lon: Array, lat: Array) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "U velocity initialization is not implemented."
        )

    def init_v(self, lon: Array, lat: Array,) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "V velocity initialization is not implemented."
        )

    def init_height(self, lon: Array, lat: Array,) -> Array:
        """Initialize height field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Height initialization is not implemented."
        )


class MountainZonalFlowTestCase(ShallowWaterEquationTestCase):
    """Mountain zonal flow test case (Williamson et al. 1992, Test Case 5).

    This test case simulates a zonal flow over an isolated mountain.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    alpha: float = eqx.field(default=0.0)
    h0: float = eqx.field(default=5960.0)
    hs0: float = eqx.field(default=2000.0)
    u0: float = eqx.field(default=20.0)
    clon: float = eqx.field(default=jnp.pi * 1.5)
    clat: float = eqx.field(default=jnp.pi / 6.0)
    mountain_radius: float = eqx.field(default=jnp.pi / 9.0)

    def __init__(self, para: PlanetParameterSet):
        self._para = para
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
        self.h0 = 5960.0
        self.hs0 = 2000.0
        self.u0 = 20.0
        self.clon = jnp.pi * 1.5
        self.clat = jnp.pi / 6.0
        self.mountain_radius = jnp.pi / 9.0

    def init_topography(self, lon: Array, lat: Array,) -> Array:
        """Initialize mountain topography."""
        mountain_radius = self.mountain_radius
        clon = self.clon
        clat = self.clat
        hs0 = self.hs0
        r = jnp.sqrt(
            jnp.minimum(
                jnp.ones_like(lat) * mountain_radius ** 2,
                (lat - clat) ** 2 + (lon - clon) ** 2)
        )
        return hs0 * (1.0 - r / mountain_radius)

    def init_height(self, lon: Array, lat: Array, ) -> Array:
        """Initialize height field for mountain zonal flow."""
        gravity = self.para.gravity.unwrap()
        r_earth = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        alpha = self.alpha
        h0 = self.h0
        u0 = self.u0

        h = r_earth * omega * u0 + (u0 ** 2) / 2.0
        h *= (-jnp.cos(lon) * jnp.cos(lat) * jnp.sin(alpha)
              + jnp.sin(lat) * jnp.cos(alpha)) ** 2
        return (gravity * h0 - h) / gravity

    def init_u(self, lon: Array, lat: Array) -> Array:
        """Initialize zonal velocity component."""
        u0 = self.u0
        alpha = self.alpha
        return u0 * (jnp.cos(lat) * jnp.cos(alpha)
                     + jnp.cos(lon) * jnp.sin(lat) * jnp.sin(alpha))

    def init_v(self, lon: Array, lat: Array,) -> Array:
        """Initialize meridional velocity component."""
        u0 = self.u0
        alpha = self.alpha
        return -u0 * jnp.sin(lon) * jnp.sin(alpha)


def select_testcase(
    case: str,
    para: PlanetParameterSet
) -> ShallowWaterEquationTestCase:
    """Select and create a shallow water test case.

    Args:
        case: Name of the test case (e.g., "mountain_zonal_flow")
        para: Earth parameter set. If None, creates default atmosphere parameters.

    Returns:
        An instance of the requested test case.
    """
    if case == "mountain_zonal_flow":
        test = MountainZonalFlowTestCase(para)
    else:
        raise NotImplementedError(f"Testcase '{case}' is not implemented.")
    return test
