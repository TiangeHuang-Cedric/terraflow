from typing import Optional
import jax.numpy as jnp
import jax

from terraflow.util import typing


Float = typing.Float
Array = typing.Array


def interp2plev(
    x: Float[Array, "nlev_m ..."],
    mlev: Float[Array, "nlev_m ..."],
    plev: Float[Array, "nlev_p"],
    left: Optional[str] = None,
    right: Optional[str] = None
) -> Float[Array, "nlev_p ..."]:
    """
    Interpolate a variable from model levels to pressure levels.

    Args:
        x (Array): Variable defined on model levels (shape: nlev_m, ...).
        mlev (Array): Model level pressures (shape: nlev_m).
        plev (Array): Pressure levels to interpolate to (shape: nlev_p).
        left (str, optional): Value to return for plev < mlev[0]. Valid options are 'extrapolate' or None.
            Defaults to None.
        right (str, optional): Value to return for plev > mlev[-1]. Valid options are 'extrapolate' or None.
            Defaults to None.
    Returns:
        Array: Variable interpolated to pressure levels (shape: nlev_p, ...).
    """
    hori_shape = x.shape[1:]
    nlev_m = x.shape[0]
    ncol = int(jnp.prod(jnp.array(hori_shape)))

    x_cols = x.reshape(nlev_m, ncol)

    if mlev.ndim == 1:
        mlev_cols = jnp.broadcast_to(mlev[:, None], (nlev_m, ncol))
    else:
        mlev_cols = mlev.reshape(nlev_m, ncol)

    # Perform interpolation along the horizontal axis for each point
    interp_func = jax.vmap(
        lambda mlev, x: jnp.interp(
            plev,
            mlev,
            x,
            left=left,
            right=right
        ),
        in_axes=(1, 1),
        out_axes=1
    )
    res_cols = interp_func(mlev_cols, x_cols)

    return res_cols.reshape((plev.shape[0],) + hori_shape)
