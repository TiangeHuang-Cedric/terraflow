from functools import partial
from jax import numpy as jnp
from jax import lax, jit
import jax

import terraflow.util.typing as typing

Float = typing.Float
FloatType = typing.FloatType
Array = typing.Array
ArrayWithUnits = typing.ArrayWithUnits


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


def pinv_svd(A: Float[Array, "m n"], rtol=1e-15):
    """Compute the Moore-Penrose pseudoinverse of matrix A using SVD.

    Args:
        A: Float[Array, "m n"], input matrix to be pseudoinverted.
        rtol: float, relative tolerance for small singular values.

    Returns:
        A_pinv: Float[Array, "n m"], pseudoinverse of A.
    """
    U, S, VT = jnp.linalg.svd(A, full_matrices=False)
    tol = rtol * jnp.max(S)
    S_inv = jnp.linalg.pinv(jnp.diag(S), rcond=tol)
    A_pinv = VT.T @ S_inv @ U.T
    return A_pinv


def exp_two_values(
    val1: Float[Array, "..."],
    val2: Float[Array, "..."],
    x0: Float[Array, "..."],
    x1: Float[Array, "..."],
    x: Float[Array, "..."]
) -> Float[Array, "..."]:
    eps = jnp.finfo(val1.dtype).eps
    w = jnp.exp((x - x0)**2 * jnp.log(eps) / (x1 - x0)**2)
    return w * val1 + (1 - w) * val2