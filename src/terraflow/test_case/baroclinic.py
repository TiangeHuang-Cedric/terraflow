from jax import numpy as jnp
import jax

from terraflow.util import typing, math, formula, log
from terraflow.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractTestCase

Float = typing.Float
Int = typing.Int
Array = typing.Array


class AbstractBaroclinicEquationTestCase(AbstractTestCase):
    """Base class for baroclinic dynamics test cases."""

    def init_normal_velocity(
        self,
        lon: Array,
        lat: Array,
        eta: Array,
        cartesian: Array,
        edge_normal: Array,
    ) -> Array:
        """Initialize normal velocity at edges.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        u = self.init_u(lon, lat, eta)
        v = self.init_v(lon, lat, eta)
        velocity_3d_cart = math.convert_vector_ltln2cart(u, v, cartesian)
        return jnp.sum(velocity_3d_cart * edge_normal, axis=-1)

    def init_mgs(self, lon: Array, lat: Array, gzs: Array, *args, **kwargs) -> Array:
        """Initialize dry mass field at surface.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Dry mass surface initialization is not implemented."
        )

    def init_gzs(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize geopotential surface field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Geopotential surface initialization is not implemented."
        )

    def init_gz(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize geopotential field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Geopotential initialization is not implemented."
        )

    def init_u(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "U velocity initialization is not implemented."
        )

    def init_v(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "V velocity initialization is not implemented."
        )

    def init_t(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize temperature field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Temperature initialization is not implemented."
        )

    def init_pt(self, lon: Array, lat: Array, eta: Array, mg: Array,  *args, **kwargs) -> Array:
        """Initialize potential temperature field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Potential temperature initialization is not implemented."
        )


class SteadyStateTestCase(AbstractBaroclinicEquationTestCase):
    alpha: Array
    u0: Array  # m s-1
    t0: Array  # K
    gamma: Array
    dt: Array
    eta0: Array
    etat: Array  # Tropopause level

    def __init__(self, para: PlanetParameterSet):
        self._para = para
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
        dev = para.dev()
        float_type = para.dev.float_type
        self.alpha = jnp.array(
            0.0, device=dev, dtype=float_type
        )
        self.u0 = jnp.array(
            35.0, device=dev, dtype=float_type
        )  # m s-1
        self.t0 = jnp.array(
            288.0, device=dev, dtype=float_type
        )  # K
        self.gamma = jnp.array(
            0.005, device=dev, dtype=float_type
        )  # K m-1
        self.dt = jnp.array(
            4.8e5, device=dev, dtype=float_type
        )  # K
        self.eta0 = jnp.array(
            0.252, device=dev, dtype=float_type
        )
        self.etat = jnp.array(
            0.2, device=dev, dtype=float_type
        )  # Tropopause level
        log.logger.success("Initialized baroclinic steady state test case.")

    def init_mgs(self, lon: Array, lat: Array, gzs: Array) -> Array:
        """Initialize dry mass field for steady state test case."""
        return jnp.ones_like(lon) * 1.0e5

    def init_u(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize zonal (u) velocity component for steady state test case."""
        etav = (eta - self.eta0) * jnp.pi / 2

        def _wrap(etav: Array):
            return self.u0 * jnp.cos(etav) ** 1.5 * jnp.sin(2 * lat) ** 2

        return jax.vmap(_wrap, in_axes=0, out_axes=0)(etav)

    def init_v(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize meridional (v) velocity component for steady state test case."""
        dtype = lon.dtype
        return jnp.zeros((eta.shape[0], lat.shape[0]), dtype=dtype)

    def init_t(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize temperature field for steady state test case."""
        rd = self.para.rd.unwrap()
        g = self.para.gravity.unwrap()
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)

        def _wrap(eta: Array) -> Array:
            mask = (self.etat <= eta) & (eta <= 1.0)
            etav = (eta - self.eta0) * jnp.pi / 2
            t_bar = jnp.where(
                mask,
                self.t0 * eta ** (rd * self.gamma / g),
                (self.t0 * eta ** (rd * self.gamma / g)
                 + self.dt * (self.etat - eta) ** 5)
            )
            return t_bar + 3.0 / 4.0 * eta * jnp.pi * self.u0 / rd * jnp.sin(etav) * jnp.sqrt(jnp.cos(etav)) * (
                (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
                * 2 * self.u0 * jnp.cos(etav) ** 1.5
                + (8 / 5 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - jnp.pi / 4)
                * radius * omega
            )

        return jax.vmap(_wrap, in_axes=0, out_axes=0)(eta)

    def init_gz(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize geopotential field for steady state test case."""
        rd = self.para.rd.unwrap()
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        g = self.para.gravity.unwrap()
        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)

        def _wrap(eta: Array):
            etav = (eta - self.eta0) * jnp.pi / 2
            mask = (self.etat <= eta) & (eta <= 1.0)
            gz_bar = jnp.where(
                mask,
                self.t0 * g / self.gamma * (1 - eta ** (rd * self.gamma / g)),
                self.t0 * g / self.gamma * (1 - eta ** (rd * self.gamma / g)) - rd * self.dt * (
                    (jnp.log(eta / self.etat) + 137 / 60) * self.etat ** 5 - 5 * self.etat ** 4 * eta +
                    5 * self.etat ** 3 * eta ** 2 - 10 / 3 * self.etat ** 2 * eta ** 3 +
                    5 / 4 * self.etat * eta ** 4 - 1 / 5 * eta ** 5
                )
            )
            return gz_bar + self.u0 * jnp.cos(etav) ** 1.5 * (
                (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
                * self.u0 * jnp.cos(etav) ** 1.5
                + (8 / 15 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - jnp.pi / 4)
                * radius * omega
            )

        return jax.vmap(_wrap, in_axes=0, out_axes=0)(eta)

    def init_gzs(self, lon: Array, lat: Array) -> Array:
        """Initialize geopotential surface field for steady state test case."""
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        etav = (1 - self.eta0) * jnp.pi / 2
        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)
        return self.u0 * jnp.cos(etav) ** 1.5 * (
            (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
            * self.u0 * jnp.cos(etav) ** 1.5
            + (8 / 5 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - jnp.pi / 4)
            * radius * omega
        )


class MountainWaveTestCase(AbstractBaroclinicEquationTestCase):
    """Baroclinic wave test case with a mountain perturbation."""
    T0: Array  # K
    h0: Array  # m
    d: Array  # m
    u0: Array  # m s-1
    lonc: Array  # rad
    latc: Array  # rad
    kap: Array
    psp: Array  # Pa
    N: Array  # s-1

    def __init__(self, para: PlanetParameterSet):
        self._para = para
        self.alias_name = "mz"
        self.name = "Mountain Wave"
        self.description = "Baroclinic wave test case with a mountain perturbation."
        self.references = (
            "Jablonowski, C., Lauritzen, P., Nair, R. & Taylor, M. "
            "Idealized test cases for the dynamical cores of Atmospheric General Circulation Models: "
            "A proposal for the NCAR ASP 2008 summer colloquium.",
        )
        self.keywords = ("baroclinic", "mountain wave", "atmosphere")
        dev = para.dev()
        float_type = para.dev.float_type
        self.T0 = jnp.array(
            288.0, device=dev, dtype=float_type
        )  # K
        self.h0 = jnp.array(
            2000.0, device=dev, dtype=float_type
        )  # m
        self.d = jnp.array(
            1.5e6, device=dev, dtype=float_type
        )
        self.u0 = jnp.array(
            20.0, device=dev, dtype=float_type
        )  # m s-1
        self.lonc = jnp.array(
            jnp.pi / 2.0, device=dev, dtype=float_type
        )
        self.latc = jnp.array(
            jnp.pi / 6.0, device=dev, dtype=float_type
        )
        self.kap = jnp.array(
            2.0 / 7.0, device=dev, dtype=float_type
        )
        self.psp = jnp.array(
            93000.0, device=dev, dtype=float_type
        )  # Pa
        self.N = jnp.array(
            0.0182, device=dev, dtype=float_type
        )  # s-1
        log.logger.success("Initialized baroclinic mountain wave test case.")

    def init_u(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize zonal (u) velocity component for mountain wave test case."""
        cos_lat = jnp.cos(lat)
        res = jnp.expand_dims(self.u0 * cos_lat, axis=0)
        return jnp.tile(res, (eta.shape[0], 1))

    def init_v(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize meridional (v) velocity component for mountain wave test case."""
        dtype = lon.dtype
        return jnp.zeros((eta.shape[0], lat.shape[0]), dtype=dtype)

    def init_gzs(self, lon: Array, lat: Array) -> Array:
        """Initialize geopotential field for mountain wave test case."""
        d = self.d
        h0 = self.h0
        latc = self.latc
        lonc = self.lonc
        radius = self.para.radius.unwrap()
        g = self.para.gravity.unwrap()
        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)
        r = radius * jnp.acos(
            jnp.sin(latc) * sin_lat
            + jnp.cos(latc) * cos_lat * jnp.cos(lon - lonc)
        )
        return g * h0 * jnp.exp(-(r / d)**2)

    def init_mgs(self, lon: Array, lat: Array, gzs: Array) -> Array:
        """Initialize dry mass field for mountain wave test case."""
        sin_lat = jnp.sin(lat)
        radius = self.para.radius.unwrap()
        omega = self.para.omega.unwrap()
        g = self.para.gravity.unwrap()
        u0 = self.u0
        psp = self.psp
        N = self.N
        kap = self.kap
        return psp * jnp.exp(
            (
                -0.5 * radius * N**2 * u0 / (g**2 * kap)
                * (u0 / radius + 2.0 * omega) * (sin_lat**2 - 1.0)
            )
            - N**2 / (g**2 * kap) * gzs
        )

    def init_pt(self, lon: Array, lat: Array, eta: Array, mg: Array) -> Array:
        """Initialize potential temperature field for mountain wave test case."""
        return formula.dry_potential_temperature(
            t=jnp.array(288.0, dtype=lon.dtype),
            p=mg,
            para=self.para,
        )


class RossbyHaurwitzWaveTestCase(AbstractBaroclinicEquationTestCase):
    """3D Rossby-Haurwitz wave test case."""
    n: Int[Array, ""]
    u0: Float[Array, ""]  # m s-1
    t0: Float[Array, ""]  # K
    gamma: Float[Array, ""]  # K m-1
    pref: Float[Array, ""]  # Pa
    M: Float[Array, ""]

    def __init__(self, para: PlanetParameterSet):
        self._para = para
        self.alias_name = "rh"
        self.name = "3D Rossby-Haurwitz Wave"
        self.description = "3D Rossby-Haurwitz wave test case for baroclinic dynamics."
        self.references = (
            "Jablonowski, C., Lauritzen, P., Nair, R. & Taylor, M. "
            "Idealized test cases for the dynamical cores of Atmospheric General Circulation Models: "
            "A proposal for the NCAR ASP 2008 summer colloquium.",
        )
        self.keywords = ("baroclinic", "Rossby-Haurwitz wave", "atmosphere")
        dev = para.dev()
        int_type = para.dev.int_type
        float_type = para.dev.float_type
        self.n = jnp.array(
            4, device=dev, dtype=int_type
        )
        self.u0 = jnp.array(
            50, device=dev, dtype=float_type
        )
        self.t0 = jnp.array(
            288, device=dev, dtype=float_type
        )  # K
        self.gamma = jnp.array(
            0.0065, device=dev, dtype=float_type
        )  # K m-1
        self.pref = jnp.array(
            955e2, device=dev, dtype=float_type
        )  # Pa
        self.M = jnp.array(
            self.u0 / (self.n * self.para.radius.unwrap()), device=dev, dtype=float_type
        )
        log.logger.success("Initialized 3D Rossby-Haurwitz wave test case.")

    def init_u(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize zonal (u) velocity component for Rossby-Haurwitz wave test case."""
        n = self.n
        M = self.M
        a = self.para.radius.unwrap()
        cos_lat = jnp.cos(lat)
        sin_lat = jnp.sin(lat)
        u = (
            a * M * cos_lat
            + a * M * cos_lat**(n - 1) * jnp.cos(n * lon) *
            (n * sin_lat**2 - cos_lat**2)
        )
        u = jnp.expand_dims(u, axis=0)
        return jnp.tile(u, (eta.shape[0], 1))

    def init_v(self, lon: Array, lat: Array, eta: Array) -> Array:
        """Initialize meridional (v) velocity component for Rossby-Haurwitz wave test case."""
        n = self.n
        M = self.M
        a = self.para.radius.unwrap()
        cos_lat = jnp.cos(lat)
        sin_lat = jnp.sin(lat)
        v = - a * M * n * cos_lat**(n - 1) * sin_lat * jnp.sin(n * lon)
        v = jnp.expand_dims(v, axis=0)
        return jnp.tile(v, (eta.shape[0], 1))

    def init_gzs(self, lon: Array, lat: Array) -> Array:
        """Initialize geopotential surface field for Rossby-Haurwitz wave test case."""
        return jnp.zeros_like(lon)

    def init_mgs(self, lon: Array, lat: Array, gzs: Array) -> Array:
        """Initialize dry mass field for Rossby-Haurwitz wave test case."""
        g = self.para.gravity.unwrap()
        omega = self.para.omega.unwrap()
        a = self.para.radius.unwrap()
        rd = self.para.rd.unwrap()
        n = self.n
        t0 = self.t0
        pref = self.pref
        gamma = self.gamma
        M = self.M
        cos_lat = jnp.cos(lat)
        a_lat = (
            M * (2 * omega + M) / 2 * cos_lat**2
            + M**2 / 4 * cos_lat**(2 * n) * (
                (n + 1) * cos_lat**2 + (2 * n**2 - n - 2)
            )
            - n**2 * M**2 / 2 * cos_lat**(2 * (n - 1))
        )
        b_lat = (
            2 * (omega + M) * M / (n + 1) / (n + 2) * cos_lat**n *
            ((n**2 + 2 * n + 2) - (n + 1)**2 * cos_lat**2)
        )
        c_lat = M**2 / 4 * cos_lat**(2 * n) * ((n + 1) * cos_lat**2 - (n + 2))
        phi_p = a**2 * (
            a_lat + b_lat * jnp.cos(n * lon)
            + c_lat * jnp.cos(2 * n * lon)
        )
        return pref * (1 + gamma / g / t0 * phi_p)**(g / gamma / rd)

    def init_pt(self, lon: Array, lat: Array, eta: Array, mg: Array) -> Array:
        """Initialize potential temperature field for Rossby-Haurwitz wave test case."""
        g = self.para.gravity.unwrap()
        rd = self.para.rd.unwrap()
        t0 = self.t0
        gamma = self.gamma
        pref = self.pref
        t = t0 * (mg / pref)**(gamma * rd / g)
        return formula.dry_potential_temperature(
            t=t,
            p=mg,
            para=self.para,
        )

    def init_gz(self, lon: Array, lat: Array, eta: Array, mg: Array) -> Array:
        """Initialize geopotential field for Rossby-Haurwitz wave test case."""
        g = self.para.gravity.unwrap()
        rd = self.para.rd.unwrap()
        t0 = self.t0
        gamma = self.gamma
        pref = self.pref
        return g * t0 / gamma * (1 - (mg / pref) ** (gamma * rd / g))
