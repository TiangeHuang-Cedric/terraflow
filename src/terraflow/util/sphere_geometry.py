from typing import Sequence, Dict
from jax import numpy as jnp
import jax

from terraflow.util import typing
from terraflow.util.math import normalize

Float = typing.Float
FloatType = typing.FloatType
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


def convert_longitude_range_in_degrees(
    lon: Array,
    from_range: Sequence[int],
    to_range: Sequence[int]
) -> Array:
    """
    Convert longitude values between different ranges.
    Args:
        lon (jnp.ndarray): Longitude values (in degrees) to be converted.
        from_range (Sequence[Union[str, int]]): Original range of longitude values.
            Supported ranges are [-180, 180], [0, 360].
        to_range (Sequence[Union[str, int]]): Target range for longitude values.
            Supported ranges are [-180, 180], [0, 360].
    Returns:
        jnp.ndarray: Converted longitude values.
    Raises:
        ValueError: If the specified ranges are not supported.
    """
    range_180Wto180E = [[-180, 180]]
    range_0to360 = [[0, 360]]
    supported_ranges = range_180Wto180E + range_0to360
    if from_range not in supported_ranges:
        raise ValueError(f"From range: {from_range} is not supported.")
    if to_range not in supported_ranges:
        raise ValueError(f"To range: {to_range} is not supported.")

    if from_range in range_180Wto180E and to_range in range_0to360:
        return jnp.where(lon < 0, lon + 360.0, lon)
    elif from_range in range_0to360 and to_range in range_180Wto180E:
        return jnp.where(lon > 180.0, lon - 360.0, lon)
    elif from_range == to_range:
        return lon
    else:
        raise ValueError(
            f"Convert from range: {from_range} to range: {to_range} "
            f"is not supported."
        )


def convert_longitude_range_in_radians(
    lon: Array,
    from_range: Sequence[str],
    to_range: Sequence[str]
) -> Array:
    """
    Convert longitude values between different ranges.
    Args:
        lon (jnp.ndarray): Longitude values (in degrees) to be converted.
        from_range (Sequence[Union[str, int]]): Original range of longitude values.
            Supported ranges are ["-pi", "pi"], ["0", "2*pi"].
        to_range (Sequence[Union[str, int]]): Target range for longitude values.
            Supported ranges are ["-pi", "pi"], ["0", "2*pi"].
    Returns:
        jnp.ndarray: Converted longitude values.
    Raises:
        ValueError: If the specified ranges are not supported.
    """
    range_180Wto180E = [["-pi", "pi"]]
    range_0to360 = [["0", "2*pi"]]
    supported_ranges = range_180Wto180E + range_0to360
    if from_range not in supported_ranges:
        raise ValueError(f"From range: {from_range} is not supported.")
    if to_range not in supported_ranges:
        raise ValueError(f"To range: {to_range} is not supported.")

    if from_range in range_180Wto180E and to_range in range_0to360:
        return jnp.where(lon < 0, lon + 2 * jnp.pi, lon)
    elif from_range in range_0to360 and to_range in range_180Wto180E:
        return jnp.where(lon > jnp.pi, lon - 2 * jnp.pi, lon)
    elif from_range == to_range:
        return lon
    else:
        raise ValueError(
            f"Convert from range: {from_range} to range: {to_range} "
            f"is not supported."
        )


def arclength(R: FloatType, u1: Array, u2: Array, axis=-1) -> Array:
    """
    Calculate the arclength between two points on a unit sphere.
    Args:
        R (FloatType): Radius of the sphere.
        u1(Array): Cartesian coordinates of the first point.
        u2(Array): Cartesian coordinates of the second point.
    Returns:
        Arclength between the two points.
    """
    dot_product = jnp.sum(
        (u1 + u2) * (u1 + u2),
        axis=axis,
        keepdims=False
    )
    return jnp.where(
        dot_product == 0.0, 
        jnp.pi * R,
        jnp.where(
            dot_product > 4.0,
            0.0,
            R * 2 * jnp.arctan(jnp.sqrt((4 - dot_product) / dot_product))
        )
    )


def spherical_excess(R: FloatType, a: Array, b: Array, c: Array) -> Array:
    """
    Calculate the spherical excess of a spherical triangle given side lengths.
    Args:
        a(jnp.ndarray): Length of side opposite to vertex A.
        b(jnp.ndarray): Length of side opposite to vertex B.
        c(jnp.ndarray): Length of side opposite to vertex C.
    Returns:
        jnp.ndarray: Spherical excess of the triangle.
    """
    s = (a + b + c) / 2
    # Calculate spherical excess according to L'Huilier's Formula
    tmp = (jnp.tan(s / (2 * R)) * jnp.tan((s - a) / (2 * R))
           * jnp.tan((s - b) / (2 * R)) * jnp.tan((s - c) / (2 * R)))
    return jnp.where(
        tmp < 0,
        0.0,
        4.0 * R**2 * jnp.arctan(jnp.sqrt(tmp))
    )


def spherical_triangle_area(
    R : FloatType,
    p1: Array,
    p2: Array,
    p3: Array,
    axis: int = -1
) -> Array:
    """
    Calculate the area of a spherical triangle given its vertices.
    Args:
        R (FloatType): Radius of the sphere.
        p1(Array): Cartesian coordinates of the first vertex.
        p2(Array): Cartesian coordinates of the second vertex.
        p3(Array): Cartesian coordinates of the third vertex.
    Returns:
        Float: Area of the spherical triangle.
    """
    u1 = normalize(p1, axis=axis)
    u2 = normalize(p2, axis=axis)
    u3 = normalize(p3, axis=axis)

    a = arclength(R, u1, u2, axis=axis)
    b = arclength(R, u2, u3, axis=axis)
    c = arclength(R, u3, u1, axis=axis)

    # According to Girard's Therem, sphtri_area = spherical_excess
    area = spherical_excess(R, a, b, c)
    return area


def vector_convert_lonlat2xyz(
    R: FloatType,
    u: Float[Array, "npoint"],
    v: Float[Array, "npoint"],
    xyz: Float[Array, "npoint 3"]
) -> Float[Array, "npoint"]:
    """
    Convert vector components from latitude-longitude to Cartesian coordinates.
    Args:
        u (jnp.ndarray): Vector component in the latitude direction.
        v (jnp.ndarray): Vector component in the longitude direction.
        point (jnp.ndarray): Cartesian coordinates of the point.
    Returns:
        jnp.ndarray: Vector in Cartesian coordinates.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = jnp.linalg.norm(xyz[:, :2] / R, axis=-1)
    mask = rho == 0.0
    ux = jnp.where(mask, v  , -u * (y / (R * rho)) - (v * x * z) / (R**2 * rho))
    uy = jnp.where(mask, u  ,  u * (x / (R * rho)) - (v * y * z) / (R**2 * rho))
    uz = jnp.where(mask, 0.0,  v * (rho))
    return jnp.stack(jnp.asarray([ux, uy, uz]), axis=-1)


def vector_convert_xyz2lonlat(
    R: FloatType,
    vec: Float[Array, "n_point 3"],
    xyz: Float[Array, "n_point 3"]
) -> Dict[str, jnp.ndarray]:
    """
    Convert vector components from Cartesian to latitude-longitude coordinates.

    Args:
        vec (jnp.ndarray): Vector in Cartesian coordinates.
        point (jnp.ndarray): Cartesian coordinates of the point.
    Returns:
        jnp.ndarray: Vector components in latitude-longitude coordinates.
    """
    # See mathworks "spherical basis representation of vectors",
    # assume a unit sphere        
    rho = jnp.linalg.norm(xyz[:, :2] / R, axis=-1)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]  # (n_point,)

    eaz = jnp.stack(
        [-y / (R * rho)       , x / (R * rho)        , jnp.zeros_like(z)],
        axis=-1
    )  # (n_point, 3)
    eel = jnp.stack(
        [-x * z / (R**2 * rho), -y * z / (R**2 * rho), rho              ],
        axis=-1
    )  # (n_point, 3)
    u = jnp.sum(vec / R * eaz, axis=-1)  # (n_point,)
    v = jnp.sum(vec / R * eel, axis=-1)  # (n_point,)
    mask = rho < 10 * jnp.finfo(vec.dtype).eps  # (n_point,)
    u = jnp.where(mask, vec[:, 1] / R, u)
    v = jnp.where(mask, vec[:, 0] / R, v)
    return {"u": u, "v": v}


def _project_sphere2plane(
    R  : FloatType,
    op0: Float[Array, "3"],
    op1: Float[Array, "3"]
) -> Float[Array, "3"]:
    """Project the vector from OP0 to OP1 onto the tangent plane at OP0.

    Args:
        R  : FloatType, radius of the sphere
        op0: Float[Array, "3"], cartesian coordinates of reference (tangent) point.
        op1: Float[Array, "3"], cartesian coordinates of the point to project.

    Returns:
        P0P1: Float[Array, "3"] representing the projected vector in the tangent plane.
    """
    # norms and cosine of angular separation
    n0  = jnp.linalg.norm(op0, axis=-1)
    n1  = jnp.linalg.norm(op1, axis=-1)
    eps = jnp.finfo(op0.dtype).eps
    # guard against zero norms
    n0 = jnp.maximum(n0, eps * 100)
    n1 = jnp.maximum(n1, eps * 100)
    cos_lambda = jnp.dot(op0, op1) / (n0 * n1)
    # clamp cos to valid domain to avoid NaNs from acos
    cos_lambda = jnp.clip(cos_lambda, -1.0 + 1e-15, 1.0 - 1e-15)
    lam = jnp.arccos(cos_lambda)
    # compute OP2, following julia: OP2 = OP1 / cos_lam
    op2  = op1 / cos_lambda
    p0p2 = op2 - op0
    # avoid division by zero in tan(lam)
    tan_lam = jnp.tan(lam)
    tan_lam = jnp.where(
        jnp.abs(tan_lam) < eps,
        jnp.sign(tan_lam) * eps,
        tan_lam
    )
    p0p1 = p0p2 * (lam / jnp.asarray(tan_lam))
    return p0p1


# vectorized/projected version for many OP1s given a single OP0
project_sphere2plane = jax.vmap(
    _project_sphere2plane, in_axes=(None, None, 0), out_axes=0
)


def great_circle(
    R: Float[Array, "..."] | float,
    lon1: Float[Array, "..."] | float,
    lat1: Float[Array, "..."] | float,
    lon2: Float[Array, "..."] | float,
    lat2: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    return R * jnp.arccos(
        jnp.minimum(
            1.0,
            jnp.maximum(
                -1.0,
                jnp.sin(lat1) * jnp.sin(lat2)
                + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon1 - lon2)
            )
        )
    )


def lonlat2xyz(
    lon: Float[Array, "..."] | float,
    lat: Float[Array, "..."] | float,
    radius: Float[Array, "..."] | float
) -> Float[Array, "... 3"]:
    """Convert longitude and latitude to Cartesian coordinates.

    Args:
        lon: Float[Array, "..."], longitude in radians.
        lat: Float[Array, "..."], latitude in radians.
        radius: Float[Array, "..."], radius of the sphere.

    Returns:
        Float[Array, "... 3"], Cartesian coordinates corresponding to the input longitude and latitude.
    """
    cos_lat = jnp.cos(lat)
    x = radius * cos_lat * jnp.cos(lon)
    y = radius * cos_lat * jnp.sin(lon)
    z = radius * jnp.sin(lat)
    return jnp.stack([x, y, z], axis=-1)