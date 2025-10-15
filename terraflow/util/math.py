import numpy as np
import jax
from jax import numpy as jnp
from jax import lax, jit
from functools import partial

import terraflow.util.typing as typing

Array = typing.Array
Int = typing.Int
Float = typing.Float

jnp.roll

def roll_nonnegative_vector(x: jnp.ndarray, shift:int) -> jnp.ndarray:
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
    n = jnp.sum(x>=0, axis=-1, keepdims=False)
    length = x.shape[0]
    
    def _rotate(valid_len):
        idx = jnp.arange(length)
        mask = idx < valid_len
        rotate_idx = jnp.where(mask, (idx+shift) % valid_len, idx)
        return jnp.take(x, rotate_idx)
    
    return lax.cond(n > 0, _rotate, lambda _: x, n)
    

@partial(jit, static_argnames=['shift'])
def roll_nonnegative(x: jnp.ndarray, shift:int) -> jnp.ndarray:
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


def rotate_nonnegative_vector(x: jnp.ndarray) -> jnp.ndarray:
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
        return jnp.take(x, rotate_idx)

    return lax.cond(n > 0, _rotate, lambda _: x, n)


@jit
def rotate_nonnegative(x: jnp.ndarray) -> jnp.ndarray:
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


def normalize(p: jnp.ndarray, axis=-1) -> jnp.ndarray:
    """
    Normalize a vector to have unit length.
    Args:
        p (jnp.ndarray): Input vector of shape (..., 3).
    Returns:
        Normalized vector of the same shape as input.
    """
    norm = jnp.linalg.norm(p, axis=axis, keepdims=True)
    return p / jnp.where(norm == 0, 1.0, norm)


def arclength(u1: jnp.ndarray, u2: Array, axis=-1) -> Float:
    """
    Calculate the arclength between two points on a unit sphere.
    Args:
        u1(Array): Cartesian coordinates of the first point.
        u2(Array): Cartesian coordinates of the second point.
    Returns:
        Arclength between the two points.
    """
    dot_product = jnp.sum((u1 + u2) * (u1 + u2), axis=axis, keepdims=False)
    return jnp.where(dot_product == 0.0, jnp.pi,
                     jnp.where(
                         dot_product > 4.0,
                         0.0,
                         2 * jnp.arctan(
                             jnp.sqrt((4 - dot_product) / dot_product))))


def spherical_excess(
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray) -> jnp.ndarray:
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
        tmp < 0, 0.0, 4.0 * jnp.arctan(jnp.sqrt(tmp)))


def spherical_triangle_area(
        p1: jnp.ndarray,
        p2: jnp.ndarray,
        p3: jnp.ndarray,
        axis=-1) -> jnp.ndarray:
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
