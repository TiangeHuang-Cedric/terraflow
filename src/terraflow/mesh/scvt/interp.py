from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

    Float = typing.Float
    Array = typing.Array

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]


@partial(jax.jit, static_argnames=("logp",))
def _interp_plev_columns(
    pi_cols: jnp.ndarray,
    xi_cols: jnp.ndarray,
    po: jnp.ndarray,
    logp: bool,
) -> jnp.ndarray:
    """Vectorized column interpolation following the F90 adjacent-pair scan exactly."""
    pi_lo_all = pi_cols[:-1, :]
    pi_hi_all = pi_cols[1:, :]
    xi_lo_all = xi_cols[:-1, :]
    xi_hi_all = xi_cols[1:, :]
    po_cols = po[:, None]

    # Match the F90 condition:
    # if (pi(k-1) <= po(ko) .and. po(ko) <= pi(k)) then
    match = (
        (pi_lo_all[None, :, :] <= po_cols[:, None, :])
        & (po_cols[:, None, :] <= pi_hi_all[None, :, :])
    )
    has_match = jnp.any(match, axis=1)
    pair_idx = jnp.argmax(match, axis=1)

    pi_lo = jnp.take_along_axis(pi_lo_all, pair_idx, axis=0)
    pi_hi = jnp.take_along_axis(pi_hi_all, pair_idx, axis=0)
    xi_lo = jnp.take_along_axis(xi_lo_all, pair_idx, axis=0)
    xi_hi = jnp.take_along_axis(xi_hi_all, pair_idx, axis=0)

    if logp:
        dp1 = jnp.log(po_cols) - jnp.log(pi_lo)
        dp2 = jnp.log(pi_hi) - jnp.log(po_cols)
    else:
        dp1 = po_cols - pi_lo
        dp2 = pi_hi - po_cols

    xo_cols = (dp2 * xi_lo + dp1 * xi_hi) / (dp1 + dp2)
    return jnp.where(has_match, xo_cols, jnp.nan)


def interp_plev(
    coord: CoordSys,
    pi: Float[Array, "nlev_i nlat nlon"],
    xi: Float[Array, "nlev_i nlat nlon"],
    po: Float[Array, "nlev_o"],
    xo_vert_loc: str,
    logp: bool = False
) -> Float[Array, "nlev_o nlat nlon"]:
    k_s, k_e = coord.vert_layout.get_idx(coord, loc=xo_vert_loc, domain="domain")
    pi_src = pi[k_s:k_e+1, ...]
    xi_src = xi[k_s:k_e+1, ...]
    nlev_src = pi_src.shape[0]
    if nlev_src < 2:
        raise ValueError("interp_plev requires at least two source levels in the active vertical domain.")

    hori_shape = pi.shape[1:]
    ncol = 1
    for dim in hori_shape:
        ncol *= dim

    pi_cols = pi_src.reshape(nlev_src, ncol)
    xi_cols = xi_src.reshape(nlev_src, ncol)
    xo_cols = _interp_plev_columns(
        pi_cols=pi_cols,
        xi_cols=xi_cols,
        po=po,
        logp=logp,
    )
    return xo_cols.reshape((po.shape[0],) + hori_shape)
