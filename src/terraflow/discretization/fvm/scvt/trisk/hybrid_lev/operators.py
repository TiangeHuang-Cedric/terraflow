from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import equinox as eqx
import jax.numpy as jnp

from terraflow.mesh.scvt import operators as mesh_op


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

    IntType = typing.IntType
    Float = typing.Float
    Array = typing.Array

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]


def delta_at_full_lev(
    coord: CoordSys,
    x_lev: Float[Array, "nlev+1 n_points"],
    dx: Optional[Float[Array, "nlev n_points"]] = None,
    hori_entity: str = "polg",
    hori_end_at: str = "domain",
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.get_idx(
        entity=hori_entity,
        start_at="domain",
        end_at=hori_end_at
    )
    kds, kde = coord.vert.half_kds, coord.vert.half_kde
    res = x_lev[kds+1:kde+1, ids:ide+1] - x_lev[kds:kde, ids:ide+1]
    if dx is None:
        return res
    else:
        return dx.at[kds:kde+1, ids:ide+1].set(res)


def delta_at_half_lev(
    coord: CoordSys,
    x: Float[Array, "nlev n_points"],
    x_lev: Float[Array, "nlev+1 n_points"],
    dx_lev: Optional[Float[Array, "nlev+1 n_points"]] = None,
    hori_entity: str = "polg",
    hori_end_at: str = "domain",
) -> Float[Array, "nlev+1 n_points"]:
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.get_idx(
        entity=hori_entity,
        start_at="domain",
        end_at=hori_end_at
    )
    full_kds, full_kde = vert.full_kds, vert.full_kde
    half_kds, half_kde = vert.half_kds, vert.half_kde
    # ==========================================================
    top = (
        x[full_kds, ids:ide+1]
        - x_lev[half_kds, ids:ide+1]
    )[None, :]
    mid = (
        x[full_kds+1:full_kde+1, ids:ide+1]
        - x[full_kds:full_kde, ids:ide+1]
    )
    bottom = (
        x_lev[half_kde, ids:ide+1]
        - x[full_kde, ids:ide+1]
    )[None, :]
    res = jnp.concatenate([top, mid, bottom], axis=0)
    # ==========================================================
    if dx_lev is None:
        return res
    else:
        return dx_lev.at[half_kds:half_kde+1, ids:ide+1].set(res)


def adv_hori_at_half_lev(
    coord: CoordSys,
    para: PlanetParameterSet,
    x_lev: Float[Array, "nlev+1 n_points"],
    mfn_lev: Float[Array, "nlev+1 n_points"],
    dmg_lev: Float[Array, "nlev+1 n_points"],
    adv_hori: Optional[Float[Array, "nlev+1 n_points"]] = None
) -> Float[Array, "nlev+1 n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.half_kds, coord.vert.half_kde
    # ==========================================================
    tmp = mfn_lev.at[kds:kde+1, ids:ide+1].multiply(
        x_lev[kds:kde+1, ids:ide+1]
    )
    term1 = mesh_op.div_at_primal_cell(
        x_at_primal_edge=tmp,
        coord=coord,
        para=para,
    )
    term2 = x_lev[kds:kde+1, ids:ide+1] * mesh_op.div_at_primal_cell(
        x_at_primal_edge=mfn_lev,
        coord=coord,
        para=para,
    )
    res = (term1 - term2) / dmg_lev[kds:kde+1, ids:ide+1]
    # ===========================================================
    if adv_hori is None:
        return res
    else:
        return adv_hori.at[kds:kde+1, ids:ide+1].set(res)


def adv_vert_at_full_lev(
    coord: CoordSys,
    x: Float[Array, "nlev n_points"],
    x_lev: Float[Array, "nlev+1 n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    dmg: Float[Array, "nlev n_points"],
    adv_vert: Optional[Float[Array, "nlev n_points"]] = None,
    hori_entity: str = "polg",
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.get_idx(
        entity=hori_entity,
        start_at="domain",
        end_at="domain"
    )
    kds, kde = coord.vert.half_kds, coord.vert.half_kde
    # ==========================================================
    term1 = (
        mfz_lev[kds+1:kde+1, ids:ide+1]
        * (x_lev[kds+1:kde+1, ids:ide+1] - x[kds:kde, ids:ide+1])
    )
    term2 = (
        mfz_lev[kds:kde, ids:ide+1]
        * (x[kds:kde, ids:ide+1] - x_lev[kds:kde, ids:ide+1])
    )
    res = (term1 + term2) / dmg[kds:kde, ids:ide+1]
    # ==========================================================
    if adv_vert is None:
        return res
    else:
        return adv_vert.at[kds:kde, ids:ide+1].set(res)


class InterpMethod:
    def __init__(
        self,
    ):
        self.switch_set = {
            "edge>lev": 0,
            "cell>edge": 1
        }
        self.interp_fn = [self.edge2lev, self.cell2edge]

    def __call__(
        self,
        coord: CoordSys,
        x: ArrayField[StaggeredFieldMetaData],
        y: ArrayField[StaggeredFieldMetaData],
        **kwargs
    ) -> ArrayField[StaggeredFieldMetaData]:
        x_data = x.unwrap()
        loc = x.meta.vert_location + ">" + y.meta.vert_location
        interp_fn = self.interp_fn[self.switch_set[loc]]
        y_data = interp_fn(
            coord=coord,
            x=x_data,
            **kwargs
        )
        return eqx.tree_at(
            lambda f: f.data,
            y,
            y_data
        )

    def cell2edge(
        self,
        coord: CoordSys,
        x: Float[Array, "nlev n_cells"],
        y: Optional[Float[Array, "nlev n_edges"]] = None,
    ) -> Float[Array, "nlev n_edges"]:
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.full_kds, coord.vert.full_kde
        edd_v = coord.mesh.dual.edge.endpoints  # (nedge, 2)
        res = 0.5 * (
            x[kds:kde+1, edd_v[ids:ide+1, 0]]
            + x[kds:kde+1, edd_v[ids:ide+1, 1]]
        )
        if y is None:
            return res
        else:
            return y.at[kds:kde+1, ids:ide+1].set(res)

    def lev2edge(
        self,
        coord: CoordSys,
        x: Float[Array, "nlev+1 n_cells"],
        y: Optional[Float[Array, "nlev+1 n_edges"]] = None,
    ) -> Float[Array, "nlev+1 n_edges"]:
        ids, ide = coord.mesh.edge_ids, coord.mesh.edge_ide
        kds, kde = coord.vert.half_kds, coord.vert.half_kde
        edd_v = coord.mesh.dual.edge.endpoints  # (nedge, 2)
        res = 0.5 * (
            x[kds:kde+1, edd_v[ids:ide+1, 0]]
            + x[kds:kde+1, edd_v[ids:ide+1, 1]]
        )
        if y is None:
            return res
        else:
            return y.at[kds:kde+1, ids:ide+1].set(res)

    def edge2lev(
        self,
        coord: CoordSys,
        x: Float[Array, "nlev n_points"],
        y: Optional[Float[Array, "nlev+1 n_points"]] = None,
        extrap_opt: bool = False
    ) -> Float[Array, "nlev+1 n_points"]:
        # ------- k-1-1/2
        #
        # ===o=== k-1
        #
        # ---?--- k-1/2
        #
        # ===o=== k
        #
        # ------- k+1/2
        ids = coord.mesh.edge_ids
        ide = coord.mesh.edge_ide
        half_kds = coord.vert.half_kds
        half_kde = coord.vert.half_kde
        full_dlev = coord.vert.full_dlev[:]
        half_dlev = coord.vert.half_dlev[:]
        # From half_kds + 1 to half_kde - 1
        a = (
            full_dlev[half_kds:half_kde-1]
            / (2 * half_dlev[half_kds+1:half_kde])
        )[:, None]
        b = (
            full_dlev[half_kds+1:half_kde]
            / (2 * half_dlev[half_kds+1:half_kde])
        )[:, None]
        res = (
            a * x[half_kds+1:half_kde, ids:ide+1]
            + b * x[half_kds:half_kde-1, ids:ide+1]
        )
        if extrap_opt:
            raise NotImplementedError
        else:
            res = jnp.concatenate(
                [
                    x[half_kds, ids:ide+1][None, :],
                    res,
                    x[half_kde-1, ids:ide+1][None, :]
                ],
                axis=0
            )
        if y is None:
            return res
        else:
            return y.at[half_kds:half_kde+1, ids:ide+1].set(res)
