from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
from terraflow.discretization.fvm.scvt.trisk.common.adv import types, flux_reconstruct as fr

Float = typing.Float
Array = typing.Array


if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

    CoordSysOneLayer = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSysHybrid = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    CoordSys = TypeVar("CoordSys", CoordSysOneLayer, CoordSysHybrid)

    BLOCK_OneLayer = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]
    BLOCK_Hybrid = Block[HybridVertCoordinate, HybridVertCoordLayout]
    BLOCK = TypeVar("BLOCK", BLOCK_OneLayer, BLOCK_Hybrid)


class UpwindHorizontalAdvectionAtCell:

    def __init__(self, order: int, optimized: bool = True):
        self.order = order
        if self.order == 0:
            self.scheme = fr.CentralDifference2AtEdgeOnPrimalMesh()
        elif self.order == 1:
            self.scheme = fr.WRF1AtEdgeOnPrimalMesh()
        elif self.order == 3:
            if optimized:
                self.scheme = fr.WRFPure3AtEdgeOnPrimalMesh()
            else:
                self.scheme = fr.WRF3AtEdgeOnPrimalMesh()
        elif self.order == 5:
            self.scheme = fr.WRF5AtEdgeOnPrimalMesh()
        else:
            raise NotImplementedError(
                f"Upwind advection of order {self.order} is not implemented."
            )

    def calc_mass_hflx(
        self,
        m: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = batch.un.unwrap()
        mfn = batch.mfn.unwrap()
        # =================================================
        mfn = mfn.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                x_at_primal_cell=m,
                normal_u_at_edge=un,
                coord=coord,
                aux=aux
            ) * un[kds:kde+1, ids:ide+1]
        )
        # =================================================
        return eqx.tree_at(
            lambda b: b.mfn.data,
            batch,
            mfn
        )

    def calc_tracer_hflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ) -> batch.AdvBatchType:
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        mfn = batch.mfn.unwrap()
        un = batch.un.unwrap()
        qmfn = batch.qmfn.unwrap()
        # =================================================
        qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                x_at_primal_cell=q,
                normal_u_at_edge=un,
                coord=coord,
                aux=aux
            ) * mfn[kds:kde+1, ids:ide+1]
        )
        # =================================================
        return eqx.tree_at(
            lambda b: b.qmfn.data,
            batch,
            qmfn
        )

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.scheme.prepare_block(block)
        return block


class UpwindHorizontalAdvectionAtVtx:

    def __init__(self, order: int, use_mbs: bool = False, optimized: bool = True):
        self.order = order
        if self.order == 0:
            self.scheme = fr.CentralDifference2AtEdgeOnDualMesh()
        elif self.order == 1:
            self.scheme = fr.WRF1AtEdgeOnDualMesh()
        elif self.order == 3:
            if optimized:
                self.scheme = fr.WRFPure3AtEdgeOnDualMesh(use_mbs=use_mbs)
            else:
                self.scheme = fr.WRF3AtEdgeOnDualMesh(use_mbs=use_mbs)
        elif self.order == 5:
            self.scheme = fr.WRF5AtEdgeOnDualMesh(use_mbs=use_mbs)
        else:
            raise NotImplementedError(
                f"Upwind advection of order {self.order} is not implemented."
            )

    def calc_mass_hflx(
        self,
        m: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = batch.un.unwrap()
        mfn = batch.mfn.unwrap()
        # =================================================
        mfn = mfn.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                x_at_dual_cell=m,
                tangent_u_at_edge=un,
                coord=coord,
                aux=aux
            ) * un[kds:kde+1, ids:ide+1]
        )
        # =================================================
        return eqx.tree_at(
            lambda b: b.mfn.data,
            batch,
            mfn
        )

    def calc_tracer_hflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ) -> batch.AdvBatchType:
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = batch.un.unwrap()
        mfn = batch.mfn.unwrap()
        qmfn = batch.qmfn.unwrap()
        # =================================================
        qmfn = qmfn.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                x_at_dual_cell=q,
                tangent_u_at_edge=un,
                coord=coord,
                aux=aux
            ) * mfn[kds:kde+1, ids:ide+1]
        )
        # =================================================
        return eqx.tree_at(
            lambda b: b.qmfn.data,
            batch,
            qmfn
        )

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.scheme.prepare_block(block)
        return block


class UpwindVerticalAdvection:

    def __init__(self, order: int, wgt: float = 1.0):
        self.order = order
        if self.order == 0:
            self.adv_vflx_at_cell = upwind0_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind0_adv_vflx_at_lev
            self.wgt = wgt
        elif self.order == 1:
            self.adv_vflx_at_cell = upwind1_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind1_adv_vflx_at_lev
            self.wgt = wgt
        elif self.order == 3:
            self.adv_vflx_at_cell = upwind3_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind3_adv_vflx_at_lev
            self.wgt = wgt
        elif self.order == 5:
            self.adv_vflx_at_cell = upwind5_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind5_adv_vflx_at_lev
            self.wgt = wgt
        else:
            raise NotImplementedError(
                f"Upwind advection of order {self.order} is not implemented."
            )

    def calc_tracer_vflx(
        self,
        q: Array,
        coord: CoordSysHybrid,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        if batch.vert_loc == "cell":
            return self.calc_at_cell(
                q=q,
                coord=coord,
                batch=batch,
            )
        elif batch.vert_loc == "lev":
            return self.calc_at_lev(
                q_lev=q,
                coord=coord,
                batch=batch,
            )
        else:
            raise NotImplementedError(
                f"Vertical location {batch.vert_loc} is not implemented."
            )

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block

    def calc_at_cell(
        self,
        q: Array,
        coord: CoordSys,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        mfz_lev = batch.mfz.unwrap()
        qmfz_lev = batch.qmfz.unwrap()
        qmfz_lev = self.adv_vflx_at_cell(
            coord=coord,
            q=q,
            qmfz_lev=qmfz_lev,
            mfz_lev=mfz_lev,
            wgt=self.wgt
        ) * mfz_lev
        return eqx.tree_at(
            lambda b: b.qmfz.data,
            batch,
            qmfz_lev
        )

    def calc_at_lev(
        self,
        q_lev: Array,
        coord: CoordSys,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        mfz = batch.mfz.unwrap()
        qmfz = batch.qmfz.unwrap()
        qmfz = self.adv_vflx_at_lev(
            coord=coord,
            q_lev=q_lev,
            qmfz=qmfz,
            mfz=mfz,
            wgt=self.wgt
        ) * mfz
        return eqx.tree_at(
            lambda b: b.qmfz.data,
            batch,
            qmfz
        )


def upwind0_adv_vflx_at_cell(
    coord: CoordSys,
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float,
    qmfz_lev: Optional[Float[Array, "nlev+1 n_points"]] = None,
) -> Float[Array, "nlev+1 n_points"]:
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.half_kds, vert.half_kde
    # =================================================
    # half_kds + 1 <= k <= half_kde - 1
    mid = 0.5 * (
        q[kds:kde-1, ids:ide+1]
        + q[kds+1:kde, ids:ide+1]
    )
    # =================================================
    if qmfz_lev is None:
        return jnp.pad(
            mid,
            ((1, 1), (0, 0)),
        )
    else:
        return qmfz_lev.at[kds+1:kde, ids:ide+1].set(mid)


def upwind1_adv_vflx_at_cell(
    coord: CoordSys,
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float,
    qmfz_lev: Optional[Float[Array, "nlev+1 n_points"]] = None,
) -> Float[Array, "nlev+1 n_points"]:
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.half_kds, vert.half_kde
    # =================================================
    # half_kds + 1 <= k <= half_kde - 1
    mid_mfz_lev = mfz_lev[kds+1:kde, ids:ide+1]
    direction = jnp.sign(mid_mfz_lev)
    q_0 = q[kds:kde-1, ids:ide+1]
    q_plus_1 = q[kds+1:kde, ids:ide+1]
    mid = upwind1(
        direction=direction,
        wgt=wgt,
        q_0=q_0,
        q_plus_1=q_plus_1
    )
    # =================================================
    if qmfz_lev is None:
        return jnp.pad(
            mid,
            ((1, 1), (0, 0)),
        )
    else:
        return qmfz_lev.at[kds+1:kde, ids:ide+1].set(mid)


def upwind3_adv_vflx_at_cell(
    coord: CoordSys,
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float,
    qmfz_lev: Optional[Float[Array, "nlev+1 n_points"]] = None,
) -> Float[Array, "nlev+1 n_points"]:
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.half_kds, vert.half_kde
    # =================================================
    # k = half_kds
    top0 = jnp.zeros_like(q[kds, ids:ide+1])
    # =================================================
    # k = half_kds + 1
    top1 = 0.5 * (
        q[kds, ids:ide+1] + q[kds+1, ids:ide+1]
    )
    # =================================================
    # half_kds + 2 <= k <= half_kde - 2
    q_minus_1 = q[kds:kde-3, ids:ide+1]
    q_0 = q[kds+1:kde-2, ids:ide+1]
    q_plus_1 = q[kds+2:kde-1, ids:ide+1]
    q_plus_2 = q[kds+3:kde, ids:ide+1]
    mid_mfz_lev = mfz_lev[kds+2:kde-1, ids:ide+1]
    direction = jnp.sign(mid_mfz_lev)
    mid = upwind3(
        direction=direction,
        wgt=wgt,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2
    )
    # =================================================
    # k = half_kde - 1
    bottom1 = 0.5 * (
        q[kde-2, ids:ide+1] + q[kde-1, ids:ide+1]
    )
    # =================================================
    # k = half_kde
    bottom0 = jnp.zeros_like(q[kde-1, :])
    # =================================================
    if qmfz_lev is None:
        return jnp.concatenate(
            [
                top0[None, :],
                top1[None, :],
                mid,
                bottom1[None, :],
                bottom0[None, :]
            ],
            axis=0
        )
    else:
        qmfz_lev = qmfz_lev.at[kds, ids:ide+1].set(top0)
        qmfz_lev = qmfz_lev.at[kds+1, ids:ide+1].set(top1)
        qmfz_lev = qmfz_lev.at[kds+2:kde-1, ids:ide+1].set(mid)
        qmfz_lev = qmfz_lev.at[kde-1, ids:ide+1].set(bottom1)
        qmfz_lev = qmfz_lev.at[kde, ids:ide+1].set(bottom0)
        return qmfz_lev


def upwind5_adv_vflx_at_cell(
    coord: CoordSys,
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float,
    qmfz_lev: Optional[Float[Array, "nlev+1 n_points"]] = None,
) -> Float[Array, "nlev+1 n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.half_kds, coord.vert.half_kde
    # =================================================
    # k = half_kds
    top0 = jnp.zeros_like(q[kds, ids:ide+1])
    # =================================================
    # half_kds + 1 <= k <= half_kds + 2
    top1 = 0.5 * (
        q[kds:kds+2, ids:ide+1] + q[kds+1:kds+3, ids:ide+1]
    )
    # =================================================
    # half_kds + 3 <= k <= half_kde - 3
    q_minus_2 = q[kds:kde-5, ids:ide+1]
    q_minus_1 = q[kds+1:kde-4, ids:ide+1]
    q_0 = q[kds+2:kde-3, ids:ide+1]
    q_plus_1 = q[kds+3:kde-2, ids:ide+1]
    q_plus_2 = q[kds+4:kde-1, ids:ide+1]
    q_plus_3 = q[kds+5:kde, ids:ide+1]
    mid_mfz_lev = mfz_lev[3:-3, ids:ide+1]
    direction = jnp.sign(mid_mfz_lev)
    mid = upwind5(
        direction=direction,
        wgt=wgt,
        q_minus_2=q_minus_2,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2,
        q_plus_3=q_plus_3
    )
    # =================================================
    # half_kde - 2 <= k <= half_kde - 1
    bottom1 = 0.5 * (
        q[kde-3:kde-1, ids:ide+1] + q[kde-2:kde, ids:ide+1]
    )
    # =================================================
    # k = half_kde
    bottom0 = jnp.zeros_like(q[kde-1, ids:ide+1])
    # =================================================
    if qmfz_lev is None:
        return jnp.concatenate(
            [
                top0[None, :],
                top1,
                mid,
                bottom1,
                bottom0[None, :]
            ],
            axis=0
        )
    else:
        qmfz_lev = qmfz_lev.at[kds, ids:ide+1].set(top0)
        qmfz_lev = qmfz_lev.at[kds+1:kds+3, ids:ide+1].set(top1)
        qmfz_lev = qmfz_lev.at[kds+3:kde-2, ids:ide+1].set(mid)
        qmfz_lev = qmfz_lev.at[kde-2:kde, ids:ide+1].set(bottom1)
        qmfz_lev = qmfz_lev.at[kde, ids:ide+1].set(bottom0)
        return qmfz_lev


def upwind0_adv_vflx_at_lev(
    coord: CoordSys,
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float,
    qmfz: Optional[Float[Array, "nlev n_points"]] = None,
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.full_kds, coord.vert.full_kde
    # =================================================
    # full_kds <= k <= full_kde
    res = 0.5 * (
        q_lev[kds:kde+1, ids:ide+1] + q_lev[kds+1:kde+2, ids:ide+1]
    )
    # =================================================
    if qmfz is None:
        return res
    else:
        return qmfz.at[kds:kde+1, ids:ide+1].set(res)


def upwind1_adv_vflx_at_lev(
    coord: CoordSys,
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float,
    qmfz: Optional[Float[Array, "nlev n_points"]] = None,
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.full_kds, coord.vert.full_kde
    # =================================================
    # full_kds <= k <= full_kde
    q_0 = q_lev[kds:kde+1, ids:ide+1]
    q_plus_1 = q_lev[kds+1:kde+2, ids:ide+1]
    mfz_tmp = mfz[kds:kde+1, ids:ide+1]
    direction = jnp.sign(mfz_tmp)
    res = upwind1(
        direction=direction,
        wgt=wgt,
        q_0=q_0,
        q_plus_1=q_plus_1
    )
    # =================================================
    if qmfz is None:
        return res
    else:
        return qmfz.at[kds:kde+1, ids:ide+1].set(res)


def upwind3_adv_vflx_at_lev(
    coord: CoordSys,
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float,
    qmfz: Optional[Float[Array, "nlev n_points"]] = None,
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.full_kds, coord.vert.full_kde
    # =================================================
    # k = full_kds
    top = 0.5 * (
        q_lev[kds, ids:ide+1] + q_lev[kds+1, ids:ide+1]
    )
    # =================================================
    # full_kds + 1 <= k <= full_kde - 1
    q_minus_1 = q_lev[kds:kde-1, ids:ide+1]
    q_0 = q_lev[kds+1:kde, ids:ide+1]
    q_plus_1 = q_lev[kds+2:kde+1, ids:ide+1]
    q_plus_2 = q_lev[kds+3:kde+2, ids:ide+1]
    mid_mfz = mfz[kds+1:kde, ids:ide+1]
    direction = jnp.sign(mid_mfz)
    mid = upwind3(
        direction=direction,
        wgt=wgt,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2
    )
    # =================================================
    # k = full_kde
    bottom = 0.5 * (
        q_lev[kde, ids:ide+1] + q_lev[kde+1, ids:ide+1]
    )
    # =================================================
    if qmfz is None:
        return jnp.concatenate([top[None, :], mid, bottom[None, :]], axis=0)
    else:
        qmfz = qmfz.at[kds, ids:ide+1].set(top)
        qmfz = qmfz.at[kds+1:kde, ids:ide+1].set(mid)
        qmfz = qmfz.at[kde, ids:ide+1].set(bottom)
        return qmfz


def upwind5_adv_vflx_at_lev(
    coord: CoordSys,
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float,
    qmfz: Optional[Float[Array, "nlev n_points"]] = None,
) -> Float[Array, "nlev n_points"]:
    ids, ide = coord.mesh.polg_ids, coord.mesh.polg_ide
    kds, kde = coord.vert.full_kds, coord.vert.full_kde
    # =================================================
    # full_kds <= k <= full_kds + 1
    top = 0.5 * (
        q_lev[kds:kds+2, ids:ide+1] + q_lev[kds+1:kds+3, ids:ide+1]
    )
    # =================================================
    # full_kds + 2 <= k <= full_kde - 2
    q_minus_2 = q_lev[kds:kde-3, ids:ide+1]
    q_minus_1 = q_lev[kds+1:kde-2, ids:ide+1]
    q_0 = q_lev[kds+2:kde-1, ids:ide+1]
    q_plus_1 = q_lev[kds+3:kde, ids:ide+1]
    q_plus_2 = q_lev[kds+4:kde+1, ids:ide+1]
    q_plus_3 = q_lev[kds+5:kde+2, ids:ide+1]
    mid_mfz = mfz[kds+2:kde-1, ids:ide+1]
    direction = jnp.sign(mid_mfz)
    mid = upwind5(
        direction=direction,
        wgt=wgt,
        q_minus_2=q_minus_2,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2,
        q_plus_3=q_plus_3
    )
    # =================================================
    # full_kde - 1 <= k <= full_kde
    bottom_mid = 0.5 * (
        q_lev[kde-1:kde+1, ids:ide+1] + q_lev[kde:kde+2, ids:ide+1]
    )
    # =================================================
    if qmfz is None:
        return jnp.concatenate([top, mid, bottom_mid], axis=0)
    else:
        qmfz = qmfz.at[kds:kds+2, ids:ide+1].set(top)
        qmfz = qmfz.at[kds+2:kde-1, ids:ide+1].set(mid)
        qmfz = qmfz.at[kde-1:kde+1, ids:ide+1].set(bottom_mid)
        return qmfz


def upwind1(
    direction: Float[Array, "z, n_points"],
    wgt: float,
    q_0: Float[Array, "z, n_points"],
    q_plus_1: Float[Array, "z, n_points"],
) -> Float[Array, "z, n_points"]:
    c11 = 0.5
    c12 = -0.5
    return c11 * (q_plus_1 + q_0) + c12 * (q_plus_1 - q_0) * wgt * direction


def upwind3(
    direction: Float[Array, "z, n_points"],
    wgt: float,
    q_minus_1: Float[Array, "z, n_points"],
    q_0: Float[Array, "z, n_points"],
    q_plus_1: Float[Array, "z, n_points"],
    q_plus_2: Float[Array, "z, n_points"],
) -> Float[Array, "z, n_points"]:
    c31 = 7.0 / 12.0
    c32 = -1.0 / 12.0
    c33 = 1.0 / 12.0
    return (
        c31 * (q_plus_1 + q_0)
        + c32 * (q_plus_2 + q_minus_1)
        + c33 * (
            q_plus_2 - q_minus_1 - 3 * (q_plus_1 - q_0)
        ) * wgt * direction
    )


def upwind5(
    direction: Float[Array, "z, n_points"],
    wgt: float,
    q_minus_2: Float[Array, "z, n_points"],
    q_minus_1: Float[Array, "z, n_points"],
    q_0: Float[Array, "z, n_points"],
    q_plus_1: Float[Array, "z, n_points"],
    q_plus_2: Float[Array, "z, n_points"],
    q_plus_3: Float[Array, "z, n_points"],
) -> Float[Array, "z, n_points"]:
    c41 = 37.0 / 60.0
    c42 = -2.0 / 15.0
    c43 = 1.0 / 60.0
    c44 = -1.0 / 60.0
    return (
        c41 * (q_plus_1 + q_0)
        + c42 * (q_plus_2 + q_minus_1)
        + c43 * (q_plus_3 + q_minus_2)
        + c44 * (
            q_plus_3 - q_minus_2
            - 5 * (q_plus_2 - q_minus_1)
            + 10 * (q_plus_1 - q_0)
        ) * wgt * direction
    )
