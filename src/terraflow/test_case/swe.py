from jax import numpy as jnp
import jax
import quadax
from loguru import logger

from terraflow.util import typing, math
from terraflow.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractTestCase

FloatType = typing.FloatType
Float = typing.Float
Array = typing.Array


class ShallowWaterEquationTestCase(AbstractTestCase):
    """Base class for shallow water dynamics test cases."""

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

    def init_geopotential_height(
        self,
        lon: Array,
        lat: Array,
    ) -> Array:
        """Initialize geopotential height field (height - topography).

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return self.init_height(lon, lat)

    def init_topography(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize topography field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Topography initialization is not implemented."
        )

    def init_u(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "U velocity initialization is not implemented."
        )

    def init_v(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "V velocity initialization is not implemented."
        )

    def init_height(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
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
    alpha: Array
    h0: Array
    hs0: Array
    u0: Array
    clon: Array
    clat: Array
    mountain_radius: Array

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
        dev = para.dev()
        float_type = para.dev.float_type
        self.alpha = jnp.array(
            0.0, device=dev, dtype=float_type
        )
        self.h0 = jnp.array(
            5960.0, device=dev, dtype=float_type
        )
        self.hs0 = jnp.array(
            2000.0, device=dev, dtype=float_type
        )
        self.u0 = jnp.array(
            20.0, device=dev, dtype=float_type
        )
        self.clon = jnp.array(
            jnp.pi * 1.5, device=dev, dtype=float_type
        )
        self.clat = jnp.array(
            jnp.pi / 6.0, device=dev, dtype=float_type
        )
        self.mountain_radius = jnp.array(
            jnp.pi / 9.0, device=dev, dtype=float_type
        )
        logger.success("Initialized mountain zonal flow test case.")

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


class RossbyHaurwitzWaveTestCase(ShallowWaterEquationTestCase):
    """Rossby-Haurwitz wave test case (Williamson et al. 1992, Test Case 6).

    This test case simulates a Rossby-Haurwitz wave in a shallow water
    atmosphere model.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    omg: Array
    K: Array
    h0: Array
    R: Array

    def __init__(self, para: PlanetParameterSet):
        self._para = para
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
        dev = para.dev()
        float_type = para.dev.float_type
        self.omg = jnp.array(
            7.848e-6, device=dev, dtype=float_type
        )
        self.K = jnp.array(
            7.848e-6, device=dev, dtype=float_type
        )
        self.h0 = jnp.array(
            8000.0, device=dev, dtype=float_type
        )
        self.R = jnp.array(
            4, device=dev, dtype=float_type
        )
        logger.success("Initialized Rossby-Haurwitz wave test case.")

    def init_topography(self, lon: Array, lat: Array) -> Array:
        """Initialize topography (flat for Rossby-Haurwitz wave)."""
        return jnp.zeros_like(lon)

    def init_height(self, lon: Array, lat: Array) -> Array:
        """
        Initialize height field for Rossby-Haurwitz wave.

        gz = gz0 + a² A(φ) + a² B(φ) cosRλ + a² C(φ) cos2Rλ

        A(φ) = 1/2 ω (2 Ω + ω) cos²φ + 1/4 ω² cos²ᴿφ ((R + 1) cos²φ + (2 R² - R - 2) - 2 R² cos⁻²φ)
        B(φ) = 2 (Ω + ω) ω cosᴿφ ((R² + 2 R + 2) - (R + 1)² cos²φ) / (R + 1) / (R + 2)
        C(φ) = 1/4 ω² cos²ᴿφ ((R + 1) cos²φ - (R + 2))
        """
        gravity = self.para.gravity.unwrap()
        h0 = self.h0
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        omg = self.omg
        R = self.R
        K = self.K

        gz0 = gravity * h0
        cos_lat = jnp.cos(lat)
        a = (
            0.5 * omg * (2 * omega + omg) * cos_lat ** 2
            + 0.25 * K**2 * (
                (R + 1) * cos_lat**(2 * R + 2)
                + (2 * R**2 - R - 2) * cos_lat**(2 * R)
                - 2 * R**2 * cos_lat**(2 * R - 2)
            )
        )
        b = 2 * (omega + omg) * omg * cos_lat**R * (
            R**2 + 2 * R + 2 - (R + 1)**2 * cos_lat**2
        ) / ((R + 1) * (R + 2))
        c = 0.25 * omg**2 * cos_lat**(2 * R) * ((R + 1) * cos_lat**2 - R - 2)
        gz = gz0 + radius**2 * \
            (a + b * jnp.cos(R * lon) + c * jnp.cos(2 * R * lon))
        return gz / gravity

    def init_u(self, lon: Array, lat: Array) -> Array:
        """
        Initialize zonal velocity component.

        u = a ω (cosφ + R cosᴿ⁻¹φ sin²φ cosRλ - cosᴿ⁺¹φ cosRλ)
        """
        radius = self.para.radius.unwrap()
        omg = self.omg
        R = self.R
        K = self.K
        cos_lat = jnp.cos(lat)
        sin_lat = jnp.sin(lat)
        a = omg * cos_lat
        b = K * R * cos_lat ** (R - 1) * sin_lat ** 2 * jnp.cos(R * lon)
        c = K * cos_lat ** (R + 1) * jnp.cos(R * lon)
        u = radius * (a + b - c)
        return u

    def init_v(self, lon: Array, lat: Array,) -> Array:
        """
        Initialize meridional velocity component.

        v = - a ω R cosᴿ⁻¹φ sinφ sinRλ
        """
        radius = self.para.radius.unwrap()
        omg = self.omg
        R = self.R
        cos_lat = jnp.cos(lat)
        sin_lat = jnp.sin(lat)
        a = R * cos_lat ** (R - 1) * sin_lat * jnp.sin(R * lon)
        v = -radius * omg * a
        return v


class JetZonalFlowTestCase(ShallowWaterEquationTestCase):
    """Jet zonal flow test case (Galewsky et al. 2004).

    This test case simulates a zonal jet flow in a shallow water atmosphere model.
    All numerical parameters are stored as JAX-traceable attributes.
    """
    u_max: Array
    lat0: Array
    lat1: Array
    lat2: Array
    en: Array
    alpha: Array
    beta: Array
    h0: Array
    hs0: Array
    hd: Array

    def __init__(self, para: PlanetParameterSet):
        self._para = para
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
        dev = para.dev()
        float_type = para.dev.float_type
        self.u_max = jnp.array(
            80.0, device=dev, dtype=float_type
        )
        self.lat0 = jnp.array(
            jnp.pi / 7.0, device=dev, dtype=float_type
        )
        self.lat1 = jnp.array(
            jnp.pi / 2.0, device=dev, dtype=float_type
        )
        self.lat1 = self.lat1 - self.lat0
        self.en = jnp.exp(-4.0 / ((self.lat1 - self.lat0) ** 2))
        self.lat2 = jnp.array(
            jnp.pi / 4.0, device=dev, dtype=float_type
        )
        self.alpha = jnp.array(
            1.0 / 3.0, device=dev, dtype=float_type)
        self.beta = jnp.array(
            1.0 / 15.0, device=dev, dtype=float_type
        )
        self.h0 = jnp.array(
            1.0e4, device=dev, dtype=float_type
        )
        self.hs0 = jnp.array(
            0.0, device=dev, dtype=float_type
        )
        self.hd = jnp.array(
            120.0, device=dev, dtype=float_type
        )
        logger.success("Initialized jet zonal flow test case.")

    def init_topography(self, lon: Array, lat: Array,) -> Array:
        """Initialize topography (flat for jet zonal flow)."""
        return jnp.ones_like(lon) * self.hs0

    def init_height(self, lon: Array, lat: Array,) -> Array:
        """Initialize height field for jet zonal flow."""
        lon_flat = jnp.asarray(lon.flatten())
        lat_flat = jnp.asarray(lat.flatten())
        n_cells = lon_flat.shape[0]

        def loop_fn(_, idx) -> tuple[None, Array]:
            return None, self._init_height(lon_flat[idx], lat_flat[idx])

        _, height_flatten = jax.lax.scan(
            loop_fn,
            init=None,
            xs=jnp.arange(n_cells),
        )
        return height_flatten.reshape(lon.shape)

    def init_u(self, lon: Array, lat: Array) -> Array:
        lat0 = self.lat0
        lat1 = self.lat1
        en = self.en
        umax = self.u_max
        return jnp.where(
            (lat > lat0) & (lat < lat1),
            umax / en * jnp.exp(
                1.0 / ((lat - lat0) * (lat - lat1))
            ),
            0.0
        )

    def init_v(self, lon: Array, lat: Array,) -> Array:
        return jnp.zeros_like(lon)

    def _init_height(
        self,
        lon: Float[Array, ""],
        lat: Float[Array, ""],
        epsabs: float = 1e-6,
        epsrel: float = 1e-12
    ) -> Float[Array, ""]:
        gravity = self.para.gravity.unwrap()
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        h0 = self.h0
        hd = self.hd
        lat2 = self.lat2
        beta = self.beta
        alpha = self.alpha
        gh0 = gravity * h0

        def u_fn(phi: Array) -> Array: return self.init_u(lon, phi)

        def coriolis_fn(phi: Array) -> Array: return 2 * omega * jnp.sin(phi)

        def integrand(phi: Array) -> Array:
            u = u_fn(phi)
            f = coriolis_fn(phi)
            return radius * u * (f + u / radius * jnp.tan(phi))

        # Avoid issues with quadax when lat is exactly -pi/2
        lat = jnp.where(lat <= -jnp.pi / 2, -jnp.pi / 2 + 1e-10, lat)

        gh_int, _ = quadax.quadgk(
            integrand,  # type: ignore
            jnp.array([-jnp.pi / 2, lat]),
            epsabs=epsabs,
            epsrel=epsrel
        )

        h = (gh0 - gh_int) / gravity
        lon_converted = jnp.where(lon > jnp.pi, lon - 2 * jnp.pi, lon)
        h_extra = hd * jnp.cos(lat)
        h_extra *= jnp.exp(-(lon_converted / alpha) ** 2)
        h_extra *= jnp.exp(-((lat2 - lat) / beta) ** 2)
        return h + h_extra
