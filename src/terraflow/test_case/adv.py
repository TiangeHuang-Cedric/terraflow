from jax import numpy as jnp
import datetime

from terraflow.util import typing, math, log
from terraflow.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractTestCase
from terraflow.discretization.abstract import TracerMetaData

FloatType = typing.FloatType
Float = typing.Float
Array = typing.Array


class AbstractAdvectionEquationTestCase(AbstractTestCase):
    pass


class AbstractAdvectionEquation2DTestCase(AbstractAdvectionEquationTestCase):
    """Base class for 2D advection dynamics test cases."""

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

    def init_tracers_meta(self, dt_adv: datetime.timedelta, *args, **kwargs) -> dict[str, TracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_tracers(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize tracer fields.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_background(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize background field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Background initialization is not implemented."
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


class AbstractAdvectionEquation3DTestCase(AbstractAdvectionEquationTestCase):
    """Base class for 3D advection dynamics test cases."""

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

    def init_tracers_meta(self, dt_adv: datetime.timedelta, *args, **kwargs) -> dict[str, TracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_tracers(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize tracer fields.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_background(self, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize background field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Background initialization is not implemented."
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


class SolidRotationTestCase(AbstractAdvectionEquation2DTestCase):
    """Solid body rotation test case for advection dynamics."""
    period: Array
    h0: Array
    lon0: Array
    lat0: Array
    alpha: Array
    u0: Array

    def __init__(self, para: PlanetParameterSet):
        self._para = para
        self.name = "Solid Rotation"
        self.description = "Solid body rotation test case for advection dynamics."
        self.references = (
            "Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N. (1992). "
            "A standard test set for numerical approximations to the shallow water equations in spherical geometry. "
            "Journal of Computational Physics, 102(1), 211-224.",
        )
        self.keywords = ("advection", "solid rotation", "atmosphere")
        dev = para.dev()
        float_type = para.dev.float_type
        self.period = jnp.array(12 * 86400.0, device=dev, dtype=float_type)
        self.h0 = jnp.array(1000.0, device=dev, dtype=float_type)
        self.lon0 = jnp.array(3 * jnp.pi / 2, device=dev, dtype=float_type)
        self.lat0 = jnp.array(0.0, device=dev, dtype=float_type)
        self.alpha = jnp.array(jnp.pi / 2, device=dev, dtype=float_type)
        self.u0 = 2 * jnp.pi * self.para.radius.unwrap() / self.period
        log.logger.success(f"Initialized {self.name} test case. ")

    def init_tracers_meta(self, dt_adv: datetime.timedelta) -> dict[str, TracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return {
            "q1": TracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Cosine bell tracer",
            ),
            "q2": TracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q2",
                long_name="Slotted cylinder tracer",
            ),
            "q3": TracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q3",
                long_name="Gaussian hill tracer",
            ),
            "q4": TracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q4",
                long_name="Cosine one tracer",
            ),
        }

    def init_tracers(self, lon: Array, lat: Array) -> Array:
        """Initialize tracer fields.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        radius = self.para.radius.unwrap()
        lon0 = self.lon0
        lat0 = self.lat0
        h0 = self.h0
        r0 = radius / 3.0
        r = math.great_circle(radius, lon0, lat0, lon, lat)
        # Cosine bell
        q1 = jnp.where(
            r < r0,
            0.5 * h0 * (1 + jnp.cos(jnp.pi * r / r0)),
            0.0
        )
        # Slotted cylinders
        qmax = 1.0
        qmin = 0.1
        r = radius * 0.5
        r0 = math.great_circle(radius, lon0, lat0, lon, lat)
        cond1 = jnp.logical_and(r0 <= r, jnp.abs(lon - lon0) >= r / radius / 6)
        cond2 = jnp.logical_and(
            jnp.logical_and(r0 <= r, jnp.abs(lon - lon0) < r / radius / 6),
            (lat - lat0) < -5/12 * r / radius
        )
        q2 = jnp.where(
            cond1,
            qmax,
            jnp.where(
                cond2,
                qmax,
                qmin
            )
        )
        # Gaussian hill
        r = math.great_circle(radius, lon0, lat0, lon, lat)
        q3 = jnp.exp(-5 * r**2 / radius**2)
        # Cosine one
        q4 = jnp.ones_like(lon)
        return jnp.stack([q1, q2, q3, q4], axis=0)

    def init_background(self, lon: Array, lat: Array) -> Array:
        """Initialize background field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return jnp.ones_like(lon)

    def init_u(self, lon: Array, lat: Array) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        alpha = self.alpha
        u0 = self.u0
        return u0 * (jnp.cos(lat) * jnp.cos(alpha) + jnp.sin(lat) * jnp.cos(lon) * jnp.sin(alpha))

    def init_v(self, lon: Array, lat: Array) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        alpha = self.alpha
        u0 = self.u0
        return -u0 * jnp.sin(lon) * jnp.sin(alpha)
