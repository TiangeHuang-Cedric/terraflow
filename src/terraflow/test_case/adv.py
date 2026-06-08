from jax import numpy as jnp
import jax
import datetime
import equinox as eqx

from terraflow.util import typing, math, log
from terraflow.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractTestCase
from terraflow.discretization.abstract import AbstractTracerMetaData

FloatType = typing.FloatType
Float = typing.Float
Array = typing.Array


class AbstractAdvectionEquationTestCase(AbstractTestCase):
    pass


class AbstractAdvectionEquation2DTestCase(AbstractAdvectionEquationTestCase):
    """Base class for 2D advection dynamics test cases."""

    def init_normal_velocity(
        self,
        time_in_seconds: Array,
        lon: Array,
        lat: Array,
        cartesian: Array,
        edge_normal: Array,
    ) -> Array:
        """Initialize normal velocity at edges.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        u = self.init_u(time_in_seconds, lon, lat)
        v = self.init_v(time_in_seconds, lon, lat)
        velocity_3d_cart = math.convert_vector_ltln2cart(u, v, cartesian)
        return jnp.sum(velocity_3d_cart * edge_normal, axis=-1)

    def init_tracers_meta(self, dt_adv: datetime.timedelta, *args, **kwargs) -> dict[str, AbstractTracerMetaData]:
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

    def init_u(self, time_in_seconds: Array, lon: Array, lat: Array, *args, **kwargs) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "U velocity initialization is not implemented."
        )

    def init_v(self, time_in_seconds: Array, lon: Array, lat: Array, *args, **kwargs) -> Array:
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
        time_in_seconds: Array,
        lon: Array,
        lat: Array,
        eta: Array,
        cartesian: Array,
        edge_normal: Array,
        *args,
        **kwargs
    ) -> Array:
        """Initialize normal velocity at edges.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        u = self.init_u(time_in_seconds, lon, lat, eta, *args, **kwargs)
        v = self.init_v(time_in_seconds, lon, lat, eta, *args, **kwargs)
        velocity_3d_cart = math.convert_vector_ltln2cart(u, v, cartesian)
        return jnp.sum(velocity_3d_cart * edge_normal, axis=-1)

    def init_tracers_meta(self, dt_adv: datetime.timedelta, *args, **kwargs) -> dict[str, AbstractTracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_tracers(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize tracer fields.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError

    def init_background(self, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize background field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Background initialization is not implemented."
        )

    def init_u(self, time_in_seconds: Array, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "U velocity initialization is not implemented."
        )

    def init_v(self, time_in_seconds: Array, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "V velocity initialization is not implemented."
        )

    def init_we_lev(self, time_in_seconds: Array, lon: Array, lat: Array, eta: Array, *args, **kwargs) -> Array:
        """Initialize vertical (we) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Vertical velocity initialization is not implemented."
        )

    def init_mfz_lev(self, lon: Array, lat: Array, eta: Array, deta: Array, *args, **kwargs) -> Array:
        """Initialize mass flux in vertical direction at levels.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        raise NotImplementedError(
            "Vertical Mass flux initialization is not implemented."
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
        self.alias_name = "sr"
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

    def init_tracers_meta(self, dt_adv: datetime.timedelta) -> dict[str, AbstractTracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return {
            "q1": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Cosine bell tracer",
            ),
            "q2": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q2",
                long_name="Slotted cylinder tracer",
            ),
            "q3": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q3",
                long_name="Gaussian hill tracer",
            ),
            "q4": AbstractTracerMetaData(
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

    def init_u(self, time_in_seconds: Array, lon: Array, lat: Array) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        alpha = self.alpha
        u0 = self.u0
        return u0 * (jnp.cos(lat) * jnp.cos(alpha) + jnp.sin(lat) * jnp.cos(lon) * jnp.sin(alpha))

    def init_v(self, time_in_seconds: Array, lon: Array, lat: Array) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        alpha = self.alpha
        u0 = self.u0
        return -u0 * jnp.sin(lon) * jnp.sin(alpha)


class DeformationalFlowTestCase(AbstractAdvectionEquation2DTestCase):
    """Deformational flow test case for advection dynamics."""
    case_n: int = eqx.field(static=True)
    period: Array
    lon1: Array
    lat1: Array
    lon2: Array
    lat2: Array

    def __init__(self, para: PlanetParameterSet, case_n: int):
        self._para = para
        self.case_n = case_n
        self.alias_name = f"dc{case_n}"
        self.name = f"Deformational Flow Case {case_n}"
        self.description = f"Deformational flow test case {case_n} for advection dynamics."
        self.references = (
            "Blossey, P. N. & Durran, D. R. "
            "Selective monotonicity preservation in scalar advection. "
            "J. Comput. Phys. 227, 5160–5183 (2008).",
        )
        self.keywords = ("advection", "deformational flow", "atmosphere")
        dev = para.dev()
        float_type = para.dev.float_type
        self.period = jnp.array(12 * 86400.0, device=dev, dtype=float_type)
        if case_n == 1:
            self.lon1 = jnp.array(
                jnp.pi,
                device=dev,
                dtype=float_type
            )
            self.lat1 = jnp.array(
                jnp.pi / 3.0,
                device=dev,
                dtype=float_type
            )
            self.lon2 = jnp.array(
                jnp.pi,
                device=dev,
                dtype=float_type
            )
            self.lat2 = jnp.array(
                -jnp.pi / 3.0,
                device=dev,
                dtype=float_type
            )
        elif case_n == 3:
            self.lon1 = jnp.array(
                jnp.pi * 3.0 / 4.0,
                device=dev,
                dtype=float_type
            )
            self.lat1 = jnp.array(
                0.0,
                device=dev,
                dtype=float_type
            )
            self.lon2 = jnp.array(
                jnp.pi * 5.0 / 4.0,
                device=dev,
                dtype=float_type
            )
            self.lat2 = jnp.array(
                0.0,
                device=dev,
                dtype=float_type
            )
        elif case_n == 2 or case_n == 4:
            self.lon1 = jnp.array(
                jnp.pi * 5.0 / 6.0,
                device=dev,
                dtype=float_type
            )
            self.lat1 = jnp.array(
                0.0,
                device=dev,
                dtype=float_type
            )
            self.lon2 = jnp.array(
                jnp.pi * 7.0 / 6.0,
                device=dev,
                dtype=float_type
            )
            self.lat2 = jnp.array(
                0.0,
                device=dev,
                dtype=float_type
            )
        else:
            raise ValueError(
                f"Invalid case number: {case_n}. Supported cases are from 1 to 4.")
        log.logger.success(f"Initialized {self.name} test case. ")

    def init_tracers_meta(self, dt_adv: datetime.timedelta) -> dict[str, AbstractTracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return {
            "q1": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Cosine hills tracer",
            ),
            "q2": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q2",
                long_name="Slotted cylinder tracer",
            ),
            "q3": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q3",
                long_name="Gaussian hills tracer",
            ),
            "q4": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q4",
                long_name="Uniform one tracer",
            ),
        }

    def init_tracers(self, lon: Array, lat: Array) -> Array:
        radius = self.para.radius.unwrap()
        lon1 = self.lon1
        lat1 = self.lat1
        lon2 = self.lon2
        lat2 = self.lat2
        x1 = math.lonlat2xyz(lon1, lat1, radius) / radius
        x2 = math.lonlat2xyz(lon2, lat2, radius) / radius
        # Cosine hills
        qmax = 1.0
        qmin = 0.1
        c = 0.9
        r = radius * 0.5
        r1 = math.great_circle(radius, lon1, lat1, lon, lat)
        r2 = math.great_circle(radius, lon2, lat2, lon, lat)
        q1 = jnp.where(
            r1 < r,
            qmin + c * qmax * 0.5 * (1 + jnp.cos(jnp.pi * r1 / r)),
            jnp.where(
                r2 < r,
                qmin + c * qmax * 0.5 * (1 + jnp.cos(jnp.pi * r2 / r)),
                qmin
            )
        )
        # Slotted cylinder
        qmax = 1.0
        qmin = 0.1
        r = radius * 0.5
        r1 = math.great_circle(radius, lon1, lat1, lon, lat)
        r2 = math.great_circle(radius, lon2, lat2, lon, lat)
        cond1 = jnp.logical_or(
            jnp.logical_and(r1 <= r, jnp.abs(lon - lon1) >= r / radius / 6),
            jnp.logical_and(r2 <= r, jnp.abs(lon - lon2) >= r / radius / 6)
        )
        cond2 = jnp.logical_and(
            jnp.logical_and(
                r1 <= r, jnp.abs(lon - lon1) < r / radius / 6
            ),
            lat - lat1 < -5.0 / 12.0 * (r / radius)
        )
        cond3 = jnp.logical_and(
            jnp.logical_and(
                r2 <= r, jnp.abs(lon - lon2) < r / radius / 6
            ),
            lat - lat2 > 5.0 / 12.0 * (r / radius)
        )
        q2 = jnp.where(
            cond1,
            qmax,
            jnp.where(
                cond2,
                qmax,
                jnp.where(
                    cond3,
                    qmax,
                    qmin
                )
            )
        )
        # Gaussian hills
        qmax = 0.95
        c = 5.0
        x = math.lonlat2xyz(lon, lat, radius) / radius
        q3 = qmax * (
            jnp.exp(-c * jnp.sum((x - x1)**2, axis=-1))
            + jnp.exp(-c * jnp.sum((x - x2)**2, axis=-1))
        )
        # Uniform one
        q4 = jnp.ones_like(lon)
        return jnp.stack([q1, q2, q3, q4], axis=0)

    def init_background(self, lon: Array, lat: Array) -> Array:
        """Initialize background field.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return jnp.ones_like(lon)

    def init_u(self, time_in_seconds: Array, lon: Array, lat: Array) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        if self.case_n == 1:
            # case 1
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            u = k * jnp.sin(lon / 2.0)**2 * jnp.sin(2 * lat) * cos_t
        elif self.case_n == 2:
            # case 2
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            u = k * jnp.sin(lon)**2 * jnp.sin(2 * lat) * cos_t
        elif self.case_n == 3:
            # case 3
            radius = self.para.radius.unwrap()
            period = self.period
            k = 5.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            u = (
                -k * jnp.sin(lon / 2.0)**2 * jnp.sin(2 * lat)
                * jnp.cos(lat)**2 * cos_t
            )
        elif self.case_n == 4:
            # case 4
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            c1 = jnp.pi * 2 * time_in_seconds / period
            c2 = jnp.pi * 2 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            lon = lon - c1
            u = (
                k * jnp.sin(lon)**2 * jnp.sin(2 * lat) * cos_t
                + c2 * jnp.cos(lat)
            )
        else:
            raise ValueError(
                f"Invalid case number: {self.case_n}. Supported cases are from 1 to 4."
            )
        return u

    def init_v(self, time_in_seconds: Array, lon: Array, lat: Array) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        if self.case_n == 1:
            # case 1
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            v = k / 2.0 * jnp.sin(lon) * jnp.cos(lat) * cos_t
        elif self.case_n == 2:
            # case 2
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            v = k * jnp.sin(2 * lon) * jnp.cos(lat) * cos_t
        elif self.case_n == 3:
            # case 3
            radius = self.para.radius.unwrap()
            period = self.period
            k = 5.0 * radius / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            v = k / 2.0 * jnp.sin(lon) * jnp.cos(lat)**3 * cos_t
        elif self.case_n == 4:
            # case 4
            radius = self.para.radius.unwrap()
            period = self.period
            k = 10.0 * radius / period
            c1 = jnp.pi * 2 * time_in_seconds / period
            cos_t = jnp.cos(jnp.pi * time_in_seconds / period)
            lon = lon - c1
            v = k * jnp.sin(2 * lon) * jnp.cos(lat) * cos_t
        else:
            raise ValueError(
                f"Invalid case number: {self.case_n}. Supported cases are from 1 to 4."
            )
        return v


class DeformationalFlowCase1(DeformationalFlowTestCase):
    """Deformational flow test case 1 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=1)


class DeformationalFlowCase2(DeformationalFlowTestCase):
    """Deformational flow test case 2 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=2)


class DeformationalFlowCase3(DeformationalFlowTestCase):
    """Deformational flow test case 3 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=3)


class DeformationalFlowCase4(DeformationalFlowTestCase):
    """Deformational flow test case 4 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=4)


class DCMIP12TestCase(AbstractAdvectionEquation3DTestCase):
    """DCMIP 1-2 3D advection test case."""
    T0: Array
    p0: Array
    K0: Array
    u0: Array
    w0: Array
    z1: Array
    z2: Array
    z0: Array
    ztop: Array
    tau: Array

    def __init__(self, para: PlanetParameterSet):
        self._para = para
        self.alias_name = "dcmip12"
        self.name = "DCMIP 1-2 3D Advection"
        self.description = "DCMIP 1-2 3D advection test case."
        self.references = (
            "Ullrich, P. A. et al. Dynamical Core Model Intercomparison Project (DCMIP) Test Case Document.",
        )
        self.keywords = ("advection", "3D", "DCMIP", "atmosphere")
        dev = para.dev()
        float_type = para.dev.float_type
        self.T0 = jnp.array(300.0, device=dev, dtype=float_type)
        self.p0 = jnp.array(1.0e5, device=dev, dtype=float_type)
        self.K0 = jnp.array(5, device=dev, dtype=float_type)
        self.u0 = jnp.array(40.0, device=dev, dtype=float_type)
        self.w0 = jnp.array(0.15, device=dev, dtype=float_type)
        self.z1 = jnp.array(2000.0, device=dev, dtype=float_type)
        self.z2 = jnp.array(5000.0, device=dev, dtype=float_type)
        z0 = 0.5 * (self.z1 + self.z2)
        self.z0 = jnp.array(z0, device=dev, dtype=float_type)
        self.ztop = jnp.array(12000.0, device=dev, dtype=float_type)
        self.tau = jnp.array(86400.0, device=dev, dtype=float_type)
        rd = para.rd.unwrap()
        self.rho0 = self.p0 / (rd * self.T0)
        log.logger.success(f"Initialized {self.name} test case. ")

    def init_tracers_meta(self, dt_adv: datetime.timedelta) -> dict[str, AbstractTracerMetaData]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return {
            "q1": AbstractTracerMetaData(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Test tracer",
            ),
        }

    def init_tracers(self, lon: Array, lat: Array, eta: Array, gz: Array) -> Array:
        """Initialize tracer fields.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        g = self.para.gravity.unwrap()
        z = gz / g
        z1 = self.z1
        z2 = self.z2
        z0 = self.z0
        q = jnp.where(
            jnp.logical_and(z1 < z, z < z2),
            0.5 * (1 + jnp.cos(2 * jnp.pi * (z - z0) / (z2 - z1))),
            0.0
        )
        return jnp.stack([q], axis=0)

    def init_u(
        self,
        time_in_seconds: Array,
        lon: Array,
        lat: Array,
        eta: Array,
        *args,
        **kwargs
    ) -> Array:
        """Initialize zonal (u) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        u0 = self.u0

        def _wrap(eta: Array):
            return u0 * jnp.cos(lat)

        return jax.vmap(_wrap, in_axes=0, out_axes=0)(eta)

    def init_v(
        self,
        time_in_seconds: Array,
        lon: Array,
        lat: Array,
        eta: Array,
        mg_e: Array,
        gz_e: Array,
        *args,
        **kwargs
    ) -> Array:
        """Initialize meridional (v) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        cos_t = jnp.cos(jnp.pi * time_in_seconds / self.tau)
        rd = self.para.rd.unwrap()
        T0 = self.T0
        w0 = self.w0
        rho0 = self.rho0
        K0 = self.K0
        ztop = self.ztop
        g = self.para.gravity.unwrap()
        radius = self.para.radius.unwrap()
        rho_e = mg_e / (rd * T0)

        def _wrap(gz_e: Array, rho_e: Array):
            return -radius * w0 * jnp.pi * rho0 / (K0 * ztop * rho_e) * (
                jnp.cos(lat) * jnp.sin(K0 * lat) *
                jnp.cos(jnp.pi * gz_e / (g * ztop)) * cos_t
            )

        return jax.vmap(_wrap, in_axes=(0, 0), out_axes=0)(gz_e, rho_e)

    def init_we_lev(self, time_in_seconds: Array, lon: Array, lat: Array, eta: Array, gz_lev: Array) -> Array:
        """Initialize vertical (w) velocity component.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        cos_t = jnp.cos(jnp.pi * time_in_seconds / self.tau)
        g = self.para.gravity.unwrap()
        rho0 = self.rho0
        K0 = self.K0
        ztop = self.ztop
        p0 = self.p0
        w0 = self.w0
        res = -g * w0 * rho0 / K0 * (
            -2 * jnp.sin(K0 * lat) * jnp.sin(lat)
            + K0 * jnp.cos(lat) * jnp.cos(K0 * lat)
        ) * jnp.sin(jnp.pi * gz_lev[1:-1, ...] / (g * ztop)) * cos_t / p0
        return jnp.pad(res, ((1, 1), (0, 0)), mode='constant')

    def init_mfz_lev(self, lon: Array, lat: Array, eta: Array, deta: Array, we_lev: Array, dmg: Array) -> Array:
        """Initialize mass flux in vertical direction.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        res = we_lev[1:-1, ...] * dmg[1:, ...] / deta[1:-1, None]
        return jnp.pad(res, ((1, 1), (0, 0)), mode='constant')
