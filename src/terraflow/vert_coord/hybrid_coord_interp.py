from terraflow.util import typing
from terraflow.prepare.vert_interp import VerticalLinearInterpolator


Float = typing.Float
Array = typing.Array


def interp2plev(
    x: Float[Array, "nlev_m ..."],
    mlev: Float[Array, "nlev_m ..."],
    plev: Float[Array, "nlev_p"],
    backend: str = "jax",
    allow_extrap: bool = False
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
    interpolator = VerticalLinearInterpolator(
        backend=backend,
        allow_extrap=allow_extrap
    )
    return interpolator(
        x1=mlev,
        data1=x,
        x2=plev
    )
