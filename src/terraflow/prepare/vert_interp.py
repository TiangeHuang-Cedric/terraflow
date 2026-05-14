from typing import Optional
import jax.numpy as jnp
import jax

from terraflow.util import typing


Float = typing.Float
Array = typing.Array


def vert_interp_linear(
    x1: Float[Array, "nlev_x1 ..."],
    data1: Float[Array, "nlev_x1 ..."],
    x2: Float[Array, "nlev_x2"],
    left: Optional[str] = None,
    right: Optional[str] = None
) -> Float[Array, "nlev_x2"]:
    """
    Interpolate a variable using linear interpolation.

    Args:
        x1 (Array): Source coordinates (shape: nlev_x1, ...).
        data1 (Array): Variable defined on source coordinates (shape: nlev_x1, ...).
        x2 (Array): Target coordinates to interpolate to (shape: nlev_x2, ...).
        left (str, optional): Value to return for x2 < x1[0]. Valid options are 'extrapolate' or None.
            Defaults to None.
        right (str, optional): Value to return for x2 > x1[-1]. Valid options are 'extrapolate' or None.
            Defaults to None.
    Returns:
        Array: Variable interpolated to target coordinates (shape: nlev_x2, ...).
    """
    hori_shape = data1.shape[1:]
    nlev_x1 = data1.shape[0]
    ncol = int(jnp.prod(jnp.array(hori_shape)))

    data1_cols = data1.reshape(nlev_x1, ncol)

    if x1.ndim == 1:
        x1_cols = jnp.broadcast_to(x1[:, None], (nlev_x1, ncol))
    else:
        x1_cols = x1.reshape(nlev_x1, ncol)

    # Perform interpolation along the horizontal axis for each point
    if x2.ndim == 1 and x2.shape[0] != ncol:
        # If x2 is 1D but does not match the horizontal shape of data1,
        # will evaluate interpolation for each column of data1 separately.
        interp_func = jax.vmap(
            lambda x1, data1: jnp.interp(
                x2,
                x1,
                data1,
                left=left,
                right=right
            ),
            in_axes=(1, 1),
            out_axes=1
        )
        res_cols = interp_func(x1_cols, data1_cols)
        res_cols = res_cols.reshape((x2.shape[0],) + hori_shape)
    elif x2.ndim == 1 and x2.shape[0] == ncol:
        # If x2 is 1D and matches the horizontal shape of data1,
        # will interpolate each column of data1 to the same x2 coordinates.
        x2_ = x2.reshape(1, -1)
        interp_func = jax.vmap(
            lambda x1, data1, x2: jnp.interp(
                x2,
                x1,
                data1,
                left=left,
                right=right
            ),
            in_axes=(1, 1, 1),
            out_axes=1
        )
        res_cols = interp_func(x1_cols, data1_cols, x2_)
        res_cols = res_cols.reshape(hori_shape)
    elif x2.ndim >= 2 and int(jnp.prod(jnp.array(x2.shape[1:]))) == ncol:
        # If x2 is N-Dimensional and matches the horizontal shape of data1,
        # will interpolate each column of data1 to the corresponding x2 coordinates.
        interp_func = jax.vmap(
            lambda x1, data1, x2: jnp.interp(
                x2,
                x1,
                data1,
                left=left,
                right=right
            ),
            in_axes=(1, 1, 1),
            out_axes=1
        )
        res_cols = interp_func(x1_cols, data1_cols, x2)
        res_cols = res_cols.reshape((x2.shape[0],) + hori_shape)
    else:
        raise ValueError(
            "x2 must be either 1D or N-Dimensional array which matches the horizontal shape of data1."
        )

    return res_cols


def vert_interp_log_linear(
    x1: jnp.ndarray,
    data1: jnp.ndarray,
    x2: jnp.ndarray,
    left: str | None = "extrapolate",
    right: str | None = "extrapolate"
) -> jnp.ndarray:
    return vert_interp_linear(
        x1=jnp.log(x1),
        data1=data1,
        x2=jnp.log(x2),
        left=left,
        right=right
    )
