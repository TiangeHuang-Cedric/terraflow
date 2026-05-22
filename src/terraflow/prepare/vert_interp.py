from functools import partial
import jax.numpy as jnp
import jax

from terraflow.util import typing


Float = typing.Float
Array = typing.Array


class VerticalLinearInterpolator:

    def __init__(
        self,
        backend: str = "native",
        allow_extrap: bool = False,
    ):
        if backend == "native":
            self.linear = NativeVerticalLinearInterpolator()
        elif backend == "jax":
            self.linear = JaxVerticalLinearInterpolator()
        else:
            raise ValueError(f"Unsupported vertical interpolation backend: {backend}")
        self.allow_extrap = allow_extrap

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        return self.linear(
            x1=x1,
            data1=data1,
            x2=x2,
            allow_extrap=self.allow_extrap
        )


class VerticalLogLinearInterpolator:

    def __init__(
        self,
        backend: str = "native",
        allow_extrap: bool = False,
    ):
        if backend == "native":
            self.log_linear = NativeVerticalLogLinearInterpolator()
        elif backend == "jax":
            self.log_linear = JaxVerticalLogLinearInterpolator()
        else:
            raise ValueError(f"Unsupported vertical interpolation backend: {backend}")
        self.allow_extrap = allow_extrap

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        return self.log_linear(
            x1=x1,
            data1=data1,
            x2=x2,
            allow_extrap=self.allow_extrap
        )


# -----------------------------------------------------------------------------
# Native interpolation implementation.
# -----------------------------------------------------------------------------
class NativeVerticalLinearInterpolator:

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
        allow_extrap: bool = False,
    ) -> Float[Array, "..."]:
        return _native_vert_interp_linear(
            x1=x1,
            data1=data1,
            x2=x2,
            allow_extrap=allow_extrap
        )


class NativeVerticalLogLinearInterpolator:

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
        allow_extrap: bool = False,
    ) -> Float[Array, "..."]:
        return _native_vert_interp_log_linear(
            x1=x1,
            data1=data1,
            x2=x2,
            allow_extrap=allow_extrap
        )


@partial(jax.jit, static_argnames=("allow_extrap",))
def _linear_interp_columns(
    x1_cols: jnp.ndarray,
    data1_cols: jnp.ndarray,
    x2_cols: jnp.ndarray,
    allow_extrap: bool,
) -> jnp.ndarray:
    """Interpolate all columns in parallel on device."""

    nlev_x1 = x1_cols.shape[0]

    def _interp_one_column(
        x1_col: jnp.ndarray,
        data1_col: jnp.ndarray,
        x2_col: jnp.ndarray,
    ) -> jnp.ndarray:
        # `searchsorted(..., side="right")` matches F90 upper-bound behavior:
        # x == x1[-1] falls into the upper-edge branch.
        i2 = jnp.searchsorted(x1_col, x2_col, side="right")

        in_range = (i2 > 0) & (i2 < nlev_x1)
        below = i2 == 0

        if allow_extrap:
            i1_idx = jnp.where(in_range, i2 - 1, jnp.where(below, 0, nlev_x1 - 2))
            i2_idx = jnp.where(in_range, i2, jnp.where(below, 1, nlev_x1 - 1))
        else:
            i1_idx = jnp.where(in_range, i2 - 1, jnp.where(below, 0, nlev_x1 - 1))
            i2_idx = jnp.where(in_range, i2, jnp.where(below, 0, nlev_x1 - 1))

        x_lo = x1_col[i1_idx]
        x_hi = x1_col[i2_idx]
        same = i1_idx == i2_idx

        wgt1 = jnp.where(same, 0.5, (x_hi - x2_col) / (x_hi - x_lo))
        wgt2 = jnp.where(same, 0.5, (x2_col - x_lo) / (x_hi - x_lo))

        return data1_col[i1_idx] * wgt1 + data1_col[i2_idx] * wgt2

    return jax.vmap(_interp_one_column, in_axes=(1, 1, 1), out_axes=1)(
        x1_cols,
        data1_cols,
        x2_cols,
    )


@partial(jax.jit, static_argnames=("allow_extrap",))
def _log_linear_interp_columns(
    x1_cols: jnp.ndarray,
    data1_cols: jnp.ndarray,
    x2_cols: jnp.ndarray,
    allow_extrap: bool,
) -> jnp.ndarray:
    """Interpolate all columns in parallel on device."""

    nlev_x1 = x1_cols.shape[0]

    def _interp_one_column(
        x1_col: jnp.ndarray,
        data1_col: jnp.ndarray,
        x2_col: jnp.ndarray,
    ) -> jnp.ndarray:
        # `searchsorted(..., side="right")` matches F90 upper-bound behavior:
        # x == x1[-1] falls into the upper-edge branch.
        i2 = jnp.searchsorted(x1_col, x2_col, side="right")

        in_range = (i2 > 0) & (i2 < nlev_x1)
        below = i2 == 0

        if allow_extrap:
            i1_idx = jnp.where(in_range, i2 - 1, jnp.where(below, 0, nlev_x1 - 2))
            i2_idx = jnp.where(in_range, i2, jnp.where(below, 1, nlev_x1 - 1))
        else:
            i1_idx = jnp.where(in_range, i2 - 1, jnp.where(below, 0, nlev_x1 - 1))
            i2_idx = jnp.where(in_range, i2, jnp.where(below, 0, nlev_x1 - 1))

        x_lo = x1_col[i1_idx]
        x_hi = x1_col[i2_idx]
        same = i1_idx == i2_idx

        wgt1 = jnp.where(same, 0.5, jnp.log(x_hi / x2_col) / jnp.log(x_hi / x_lo))
        wgt2 = jnp.where(same, 0.5, jnp.log(x2_col / x_lo) / jnp.log(x_hi / x_lo))

        return data1_col[i1_idx] * wgt1 + data1_col[i2_idx] * wgt2

    return jax.vmap(_interp_one_column, in_axes=(1, 1, 1), out_axes=1)(
        x1_cols,
        data1_cols,
        x2_cols,
    )


def _native_vert_interp_linear(
    x1: Float[Array, "nlev_x1 ..."],
    data1: Float[Array, "nlev_x1 ..."],
    x2: Float[Array, "..."],
    allow_extrap: bool = False,
) -> Float[Array, "..."]:
    """
    Native implementation of linear interpolation.

    Semantics:
    - `allow_extrap=True`: linearly extrapolate outside the source range using
      the first/last two levels.
    - `allow_extrap=False`: clamp to boundary levels outside the source range.

    Args:
        x1: Source coordinates, shape `(nlev_x1,)` or `(nlev_x1, ...)`.
        data1: Source values, shape `(nlev_x1, ...)`.
        x2: Target coordinates, supports four forms:
            1) scalar `()`: returns `(...)`;
            2) shared 1D levels `(nlev_x2,)`: returns `(nlev_x2, ...)`;
            3) one scalar per column, shape `data1.shape[1:]`: returns `(...)`;
            4) one profile per column, shape `(nlev_x2, ...)`: returns same shape.
    """
    x1 = jnp.asarray(x1)
    data1 = jnp.asarray(data1)
    x2 = jnp.asarray(x2)

    if data1.ndim < 1:
        raise ValueError("data1 must have at least 1 dimension")

    nlev_x1 = data1.shape[0]
    if nlev_x1 < 2:
        raise ValueError("nlev_x1 must be >= 2")

    hori_shape = data1.shape[1:]
    ncol = int(jnp.prod(jnp.array(hori_shape))) if hori_shape else 1

    data1_cols = data1.reshape(nlev_x1, ncol)

    if x1.ndim == 1:
        if x1.shape[0] != nlev_x1:
            raise ValueError("x1 length must match data1.shape[0]")
        x1_cols = jnp.broadcast_to(x1[:, None], (nlev_x1, ncol))
    else:
        if x1.shape[0] != nlev_x1 or x1.shape[1:] != hori_shape:
            raise ValueError("x1 shape must be (nlev_x1,) or match data1 shape")
        x1_cols = x1.reshape(nlev_x1, ncol)

    if x2.ndim == 0:
        x2_cols = jnp.broadcast_to(x2.reshape(1, 1), (1, ncol))
        out_shape = hori_shape
    elif x2.shape == hori_shape:
        # One target value per horizontal column.
        x2_cols = x2.reshape(1, ncol)
        out_shape = hori_shape
    elif x2.ndim == 1:
        # Shared target levels for all horizontal columns.
        x2_cols = jnp.broadcast_to(x2[:, None], (x2.shape[0], ncol))
        out_shape = (x2.shape[0],) + hori_shape
    elif x2.shape[1:] == hori_shape:
        x2_cols = x2.reshape(x2.shape[0], ncol)
        out_shape = x2.shape
    else:
        raise ValueError("x2 shape is incompatible with data1 horizontal shape")

    res_cols = _linear_interp_columns(x1_cols, data1_cols, x2_cols, allow_extrap=allow_extrap)
    return res_cols.reshape(out_shape)


def _native_vert_interp_log_linear(
    x1: Float[Array, "nlev_x1 ..."],
    data1: Float[Array, "nlev_x1 ..."],
    x2: Float[Array, "..."],
    allow_extrap: bool = False,
) -> Float[Array, "..."]:
    """
    Native implementation of log-linear interpolation.

    Semantics:
    - `allow_extrap=True`: linearly extrapolate outside the source range using
      the first/last two levels.
    - `allow_extrap=False`: clamp to boundary levels outside the source range.

    Args:
        x1: Source coordinates, shape `(nlev_x1,)` or `(nlev_x1, ...)`.
        data1: Source values, shape `(nlev_x1, ...)`.
        x2: Target coordinates, supports four forms:
            1) scalar `()`: returns `(...)`;
            2) shared 1D levels `(nlev_x2,)`: returns `(nlev_x2, ...)`;
            3) one scalar per column, shape `data1.shape[1:]`: returns `(...)`;
            4) one profile per column, shape `(nlev_x2, ...)`: returns same shape.
    """
    x1 = jnp.asarray(x1)
    data1 = jnp.asarray(data1)
    x2 = jnp.asarray(x2)

    if data1.ndim < 1:
        raise ValueError("data1 must have at least 1 dimension")

    nlev_x1 = data1.shape[0]
    if nlev_x1 < 2:
        raise ValueError("nlev_x1 must be >= 2")

    hori_shape = data1.shape[1:]
    ncol = int(jnp.prod(jnp.array(hori_shape))) if hori_shape else 1

    data1_cols = data1.reshape(nlev_x1, ncol)

    if x1.ndim == 1:
        if x1.shape[0] != nlev_x1:
            raise ValueError("x1 length must match data1.shape[0]")
        x1_cols = jnp.broadcast_to(x1[:, None], (nlev_x1, ncol))
    else:
        if x1.shape[0] != nlev_x1 or x1.shape[1:] != hori_shape:
            raise ValueError("x1 shape must be (nlev_x1,) or match data1 shape")
        x1_cols = x1.reshape(nlev_x1, ncol)

    if x2.ndim == 0:
        x2_cols = jnp.broadcast_to(x2.reshape(1, 1), (1, ncol))
        out_shape = hori_shape
    elif x2.shape == hori_shape:
        # One target value per horizontal column.
        x2_cols = x2.reshape(1, ncol)
        out_shape = hori_shape
    elif x2.ndim == 1:
        # Shared target levels for all horizontal columns.
        x2_cols = jnp.broadcast_to(x2[:, None], (x2.shape[0], ncol))
        out_shape = (x2.shape[0],) + hori_shape
    elif x2.shape[1:] == hori_shape:
        x2_cols = x2.reshape(x2.shape[0], ncol)
        out_shape = x2.shape
    else:
        raise ValueError("x2 shape is incompatible with data1 horizontal shape")

    res_cols = _log_linear_interp_columns(x1_cols, data1_cols, x2_cols, allow_extrap=allow_extrap)
    return res_cols.reshape(out_shape)


# -----------------------------------------------------------------------------
# Jax built-in interpolation implementation.
# -----------------------------------------------------------------------------
class JaxVerticalLinearInterpolator:

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
        allow_extrap: bool = False,
    ) -> Float[Array, "..."]:
        if allow_extrap:
            left = "extrapolate"
            right = "extrapolate"
        else:
            left = None
            right = None
        return _jax_vert_interp_linear(
            x1=x1,
            data1=data1,
            x2=x2,
            left=left,
            right=right
        )


class JaxVerticalLogLinearInterpolator:

    def __call__(
        self,
        x1: Float[Array, "nlev_x1 ..."],
        data1: Float[Array, "nlev_x1 ..."],
        x2: Float[Array, "..."],
        allow_extrap: bool = False,
    ) -> Float[Array, "..."]:
        if allow_extrap:
            left = "extrapolate"
            right = "extrapolate"
        else:
            left = None
            right = None
        return _jax_vert_interp_log_linear(
            x1=x1,
            data1=data1,
            x2=x2,
            left=left,
            right=right
        )


def _jax_vert_interp_linear(
    x1: Float[Array, "nlev_x1 ..."],
    data1: Float[Array, "nlev_x1 ..."],
    x2: Float[Array, "nlev_x2"],
    left: str | None = None,
    right: str | None = None
) -> Float[Array, "nlev_x2"]:
    """
    Interpolate a variable using linear interpolation.

    Args:
        x1 (Array): Source coordinates (shape: nlev_x1, ...).
        data1 (Array): Variable defined on source coordinates (shape: nlev_x1, ...).
        x2 (Array): Target coordinates to interpolate to (shape: nlev_x2, ...).
        left (str, optional): Value to return for x2 < x1[0].
            Valid options are 'extrapolate' or None.
        right (str, optional): Value to return for x2 > x1[-1].
            Valid options are 'extrapolate' or None.
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
        # evaluate interpolation for each column of data1 separately.
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
        # interpolate each column of data1 to the same x2 coordinates.
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
        # interpolate each column to its corresponding x2 coordinates.
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
            "x2 must be either 1D or N-Dimensional array matching data1 horizontal shape."
        )

    return res_cols


def _jax_vert_interp_log_linear(
    x1: jnp.ndarray,
    data1: jnp.ndarray,
    x2: jnp.ndarray,
    left: str | None = None,
    right: str | None = None
) -> jnp.ndarray:
    return _jax_vert_interp_linear(
        x1=jnp.log(x1),
        data1=data1,
        x2=jnp.log(x2),
        left=left,
        right=right
    )
