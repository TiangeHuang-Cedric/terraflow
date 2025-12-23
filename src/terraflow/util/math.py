from typing import Sequence, Dict
from functools import partial
from jax import numpy as jnp
from jax import lax, jit
import jax

import terraflow.util.typing as typing

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


def roll_nonnegative_vector(x: Array, shift: int) -> Array:
    """
    Roll the elements of the input vector by a specified shift,
    while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x (jnp.ndarray): Input vector.
        shift (int): Number of positions to roll the elements.
    Returns:
        jnp.ndarray: Rolled array of the same shape as input.
    """
    n = jnp.sum(x >= 0, axis=-1, keepdims=False)
    length = x.shape[0]

    def _rotate(valid_len):
        idx = jnp.arange(length)
        mask = idx < valid_len
        rotate_idx = jnp.where(mask, (idx + shift) % valid_len, idx)
        return jnp.take(x, jnp.asarray(rotate_idx))

    return lax.cond(n > 0, _rotate, lambda _: x, n)


@partial(jit, static_argnames=['shift'])
def roll_nonnegative(x: Array, shift: int) -> Array:
    """
    Roll the elements of the input array by a specified shift along
    the last axis, while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x:
        shift:
    Returns:
        Rolled array of the same shape as input.
    """
    return jax.vmap(roll_nonnegative_vector, in_axes=(0, None))(x, shift)


def rotate_nonnegative_vector(x: Array) -> Array:
    """
    Rotate the elements of the input vector to the left by one position,
    while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x (jnp.ndarray): Input vector.
    Returns:
        jnp.ndarray: Rotated array of the same shape as input.
    """
    n = jnp.sum(x >= 0, axis=-1, keepdims=False)
    length = x.shape[0]

    def _rotate(valid_len):
        idx = jnp.arange(length)
        mask = idx < valid_len
        rotate_idx = jnp.where(mask, (idx-1) % valid_len, idx)
        return jnp.take(x, jnp.asarray(rotate_idx))

    return lax.cond(n > 0, _rotate, lambda _: x, n)


@jit
def rotate_nonnegative(x: Array) -> Array:
    """
    Rotate the elements of the input array to the left by one position along
    the last axis, while keeping the relative order of non-negative elements.
    Negative elements are treated as invalid and remain in their positions.
    Args:
        x: Input array.
    Returns:
        Rotated array of the same shape as input.
    """
    return jax.vmap(rotate_nonnegative_vector)(x)


def normalize(p: Array, axis: int = -1) -> Array:
    """
    Normalize a vector to have unit length.
    Args:
        p (jnp.ndarray): Input vector of shape (..., 3).
    Returns:
        Normalized vector of the same shape as input.
    """
    norm: Array = jnp.linalg.norm(p, axis=axis, keepdims=True)
    denom: Array = jnp.where(norm == 0, 1.0, norm)
    return p / denom


def arclength(u1: Array, u2: Array, axis=-1) -> Array:
    """
    Calculate the arclength between two points on a unit sphere.
    Args:
        u1(Array): Cartesian coordinates of the first point.
        u2(Array): Cartesian coordinates of the second point.
    Returns:
        Arclength between the two points.
    """
    dot_product = jnp.sum((u1 + u2) * (u1 + u2), axis=axis, keepdims=False)
    return jnp.where(
        dot_product == 0.0, jnp.pi,
        jnp.where(
            dot_product > 4.0,
            0.0,
            2 * jnp.arctan(jnp.sqrt((4 - dot_product) / dot_product))
        )
    )


def spherical_excess(a: Array, b: Array, c: Array) -> Array:
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
    tmp = (jnp.tan(s / 2) * jnp.tan((s - a) / 2)
           * jnp.tan((s - b) / 2) * jnp.tan((s - c) / 2))
    return jnp.where(
        tmp < 0,
        0.0,
        4.0 * jnp.arctan(jnp.sqrt(tmp))
    )


def spherical_triangle_area(
    p1: Array,
    p2: Array,
    p3: Array,
    axis: int = -1
) -> Array:
    """
    Calculate the area of a spherical triangle given its vertices.
    Args:
        p1(Array): Cartesian coordinates of the first vertex.
        p2(Array): Cartesian coordinates of the second vertex.
        p3(Array): Cartesian coordinates of the third vertex.
    Returns:
        Float: Area of the spherical triangle.
    """
    u1 = normalize(p1, axis=axis)
    u2 = normalize(p2, axis=axis)
    u3 = normalize(p3, axis=axis)

    a = arclength(u1, u2, axis=axis)
    b = arclength(u2, u3, axis=axis)
    c = arclength(u3, u1, axis=axis)

    # According to Girard's Therem, sphtri_area = spherical_excess
    area = spherical_excess(a, b, c)
    det = jax.vmap(jax.scipy.linalg.det)(jnp.stack([p1, p2, p3], axis=-1))
    area = jnp.where(det < 0, -area, area)

    return area


def convert_vector_ltln2cart(
    u: Array,
    v: Array,
    point: Array
) -> Array:
    """
    Convert vector components from latitude-longitude to Cartesian coordinates.
    Args:
        u (jnp.ndarray): Vector component in the latitude direction.
        v (jnp.ndarray): Vector component in the longitude direction.
        point (jnp.ndarray): Cartesian coordinates of the point.
    Returns:
        jnp.ndarray: Vector in Cartesian coordinates.
    """
    x, y, z = point[:, 0], point[:, 1], point[:, 2]
    rho = jnp.linalg.norm(jnp.stack([x, y], axis=-1), axis=-1)
    mask = rho == 0.0
    ux = jnp.where(mask, v, -u * (y / rho) - v * (x * z) / rho)
    uy = jnp.where(mask, u, u * (x / rho) - v * (y * z) / rho)
    uz = jnp.where(mask, 0.0, v * (rho))
    return jnp.stack(jnp.asarray([ux, uy, uz]), axis=-1)


def convert_vector_cart2ltln(
    uxyz: Float[Array, "n_point 3"],
    point: Float[Array, "n_point 3"]
) -> Dict[str, jnp.ndarray]:
    """
    Convert vector components from Cartesian to latitude-longitude coordinates.

    Args:
        uxyz (jnp.ndarray): Vector in Cartesian coordinates.
        point (jnp.ndarray): Cartesian coordinates of the point.
    Returns:
        jnp.ndarray: Vector components in latitude-longitude coordinates.
    """
    radius = jnp.linalg.norm(point, axis=-1)  # (n_point,)
    rvec = point / radius[:, None]  # (n_point, 3)
    x, y, z = rvec[:, 0], rvec[:, 1], rvec[:, 2]  # (n_point,)

    rho = jnp.linalg.norm(jnp.stack([x, y], axis=-1), axis=-1)  # (n_point,)

    # see mathworks "spherical basis representation of vectors",
    # assume a unit sphere
    eaz = jnp.stack(
        [-y / rho, x / rho, jnp.zeros_like(z)],
        axis=-1
    )  # (n_point, 3)
    eel = jnp.stack(
        [-x * z / rho, -y * z / rho, rho],
        axis=-1
    )  # (n_point, 3)
    u = jnp.sum(uxyz * eaz, axis=-1)  # (n_point,)
    v = jnp.sum(uxyz * eel, axis=-1)  # (n_point,)
    mask = rho < 10 * jnp.finfo(uxyz.dtype).eps  # (n_point,)
    u = jnp.where(mask, uxyz[:, 1], u)
    v = jnp.where(mask, uxyz[:, 0], v)
    return {"u": u, "v": v}


def _project_sphere2plane(
    op0: Float[Array, "3"],
    op1: Float[Array, "3"]
) -> Float[Array, "3"]:
    """Project the vector from OP0 to OP1 onto the tangent plane at OP0.

    Args:
        op0: Float[Array, "3"], cartesian coordinates of reference (tangent) point.
        op1: Float[Array, "3"], cartesian coordinates of the point to project.

    Returns:
        P0P1: Float[Array, "3"] representing the projected vector in the tangent plane.
    """
    # norms and cosine of angular separation
    n0 = jnp.linalg.norm(op0, axis=-1)
    n1 = jnp.linalg.norm(op1, axis=-1)
    eps = jnp.finfo(op0.dtype).eps
    # guard against zero norms
    n0 = jnp.maximum(n0, eps * 100)
    n1 = jnp.maximum(n1, eps * 100)
    cos_lambda = jnp.dot(op0, op1) / (n0 * n1)
    # clamp cos to valid domain to avoid NaNs from acos
    cos_lambda = jnp.clip(cos_lambda, -1.0 + 1e-15, 1.0 - 1e-15)
    lam = jnp.arccos(cos_lambda)
    # compute OP2, following julia: OP2 = OP1 / cos_lam
    op2 = op1 / cos_lambda
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
    _project_sphere2plane, in_axes=(None, 0), out_axes=0
)
