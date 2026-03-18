from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.discretization.fvm.odm.trisk.common.adv import batch
from terraflow.discretization.fvm.odm.trisk.common.adv import types, flux_reconstruct as fr

Float = typing.Float
Array = typing.Array

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


class UpwindHorizontalAdvection:

    def __init__(self, order: int):
        self.order = order
        if self.order == 0:
            self.adv_hflx_at_cell = fr.CentralDifference2AtEdgeOnPrimalMesh()
            self.adv_hflx_at_vtx = fr.CentralDifference2AtEdgeOnDualMesh()
        elif self.order == 3:
            self.adv_hflx_at_cell = fr.WRF3AtEdgeOnPrimalMesh()
            self.adv_hflx_at_vtx = fr.WRF3AtEdgeOnDualMesh()
        elif self.order == 5:
            self.adv_hflx_at_cell = fr.WRF5AtEdgeOnPrimalMesh()
            self.adv_hflx_at_vtx = fr.WRF5AtEdgeOnDualMesh()
        else:
            raise NotImplementedError(
                f"Upwind advection of order {self.order} is not implemented."
            )

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ) -> batch.AdvBatchType:
        if batch.hori_loc == "cell":
            return self.calc_at_cell(
                q=q,
                batch=batch,
                coord=coord,
                aux=aux
            )
        elif batch.hori_loc == "vtx":
            return self.calc_at_vtx(
                q=q,
                batch=batch,
                coord=coord,
                aux=aux
            )
        else:
            raise NotImplementedError(
                f"Horizontal location {batch.hori_loc} is not implemented."
            )

    def prepare_block(self, block: Block) -> Block:
        block = self.adv_hflx_at_cell.prepare_block(block)
        block = self.adv_hflx_at_vtx.prepare_block(block)
        return block

    def calc_at_cell(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ) -> batch.AdvBatchType:
        mfn = batch.mfn.unwrap()
        un = batch.un.unwrap()
        qmfn = self.adv_hflx_at_cell(
            x_at_primal_cell=q,
            normal_u_at_edge=un,
            coord=coord,
            aux=aux
        ) * mfn
        return eqx.tree_at(
            lambda b: b.qmfn.data,
            batch,
            qmfn
        )

    def calc_at_vtx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ) -> batch.AdvBatchType:
        ut = batch.ut.unwrap()
        mft = batch.mft.unwrap()
        qmfn = self.adv_hflx_at_vtx(
            x_at_dual_cell=q,
            tangent_u_at_edge=ut,
            coord=coord,
            aux=aux
        ) * mft
        return eqx.tree_at(
            lambda b: b.qmfn.data,
            batch,
            qmfn
        )


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

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        if batch.vert_loc == "cell":
            return self.calc_at_cell(
                q=q,
                batch=batch,
            )
        elif batch.vert_loc == "lev":
            return self.calc_at_lev(
                q_lev=q,
                batch=batch,
            )
        else:
            raise NotImplementedError(
                f"Vertical location {batch.vert_loc} is not implemented."
            )

    def prepare_block(self, block: Block) -> Block:
        return block

    def calc_at_cell(
        self,
        q: Array,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        mfz = batch.mfz.unwrap()
        qmfz = self.adv_vflx_at_cell(
            q=q,
            mfz_lev=mfz,
            wgt=self.wgt
        )
        return eqx.tree_at(
            lambda b: b.qmfz.data,
            batch,
            qmfz
        )

    def calc_at_lev(
        self,
        q_lev: Array,
        batch: batch.AdvBatchType,
    ) -> batch.AdvBatchType:
        mfz = batch.mfz.unwrap()
        qmfz = self.adv_vflx_at_lev(
            q_lev=q_lev,
            mfz=mfz,
            wgt=self.wgt
        )
        return eqx.tree_at(
            lambda b: b.qmfz.data,
            batch,
            qmfz
        )


def upwind0_adv_vflx_at_cell(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float
) -> Float[Array, "nlev+1 n_points"]:
    # =================================================
    # half_kds + 1 <= k <= half_kde - 1
    mfz_lev = mfz_lev[1:-1, :]
    mid = 0.5 * (q[:-1, :] + q[1:, :]) * mfz_lev
    # =================================================
    return jnp.pad(
        mid,
        ((1, 1), (0, 0)),
    )


def upwind1_adv_vflx_at_cell(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float
) -> Float[Array, "nlev+1 n_points"]:
    # =================================================
    # half_kds + 1 <= k <= half_kde - 1
    mid_mfz_lev = mfz_lev[1:-1, :]
    direction = jnp.sign(mid_mfz_lev)
    q_0 = q[:-1, :]
    q_plus_1 = q[1:, :]
    mid = upwind1(
        direction=direction,
        wgt=wgt,
        q_0=q_0,
        q_plus_1=q_plus_1
    ) * mid_mfz_lev
    # =================================================
    return jnp.pad(
        mid,
        ((1, 1), (0, 0)),
    )


def upwind3_adv_vflx_at_cell(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float
) -> Float[Array, "nlev+1 n_points"]:
    # =================================================
    # k = half_kds
    top0 = jnp.zeros_like(q[0, :])
    # =================================================
    # k = half_kds + 1
    top1 = 0.5 * (q[0, :] + q[1, :]) * mfz_lev[1, :]
    # =================================================
    # half_kds + 2 <= k <= half_kde - 2
    q_minus_1 = q[:-3, :]
    q_0 = q[1:-2, :]
    q_plus_1 = q[2:-1, :]
    q_plus_2 = q[3:, :]
    mid_mfz_lev = mfz_lev[2:-2, :]
    direction = jnp.sign(mid_mfz_lev)
    mid = upwind3(
        direction=direction,
        wgt=wgt,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2
    ) * mid_mfz_lev
    # =================================================
    # k = half_kde - 1
    bottom1 = 0.5 * (q[-2, :] + q[-1, :]) * mfz_lev[-2, :]
    # =================================================
    # k = half_kde
    bottom0 = jnp.zeros_like(q[-1, :])
    # =================================================
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


def upwind5_adv_vflx_at_cell(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float
) -> Float[Array, "nlev+1 n_points"]:
    # =================================================
    # k = half_kds
    top0 = jnp.zeros_like(q[0, :])
    # =================================================
    # half_kds + 1 <= k <= half_kds + 2
    top1 = 0.5 * (q[:2, :] + q[1:3, :]) * mfz_lev[1:3, :]
    # =================================================
    # half_kds + 3 <= k <= half_kde - 3
    q_minus_2 = q[:-5, :]
    q_minus_1 = q[1:-4, :]
    q_0 = q[2:-3, :]
    q_plus_1 = q[3:-2, :]
    q_plus_2 = q[4:-1, :]
    q_plus_3 = q[5:, :]
    mid_mfz_lev = mfz_lev[3:-3, :]
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
    ) * mid_mfz_lev
    # =================================================
    # half_kde - 2 <= k <= half_kde - 1
    bottom1 = 0.5 * (q[-3:-1, :] + q[-2:, :]) * mfz_lev[-2:, :]
    # =================================================
    # k = half_kde
    bottom0 = jnp.zeros_like(q[-1, :])
    # =================================================
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


def upwind0_adv_vflx_at_lev(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float
) -> Float[Array, "nlev n_points"]:
    # full_kds <= k <= full_kde
    return 0.5 * (q_lev[:-1, :] + q_lev[1:, :]) * mfz


def upwind1_adv_vflx_at_lev(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float
) -> Float[Array, "nlev n_points"]:
    # full_kds <= k <= full_kde
    q_0 = q_lev[:-1, :]
    q_plus_1 = q_lev[1:, :]
    direction = jnp.sign(mfz)
    return upwind1(
        direction=direction,
        wgt=wgt,
        q_0=q_0,
        q_plus_1=q_plus_1
    ) * mfz


def upwind3_adv_vflx_at_lev(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float
) -> Float[Array, "nlev n_points"]:
    # =================================================
    # k = full_kds
    top = 0.5 * (q_lev[0, :] + q_lev[1, :]) * mfz[0, :]
    # =================================================
    # full_kds + 1 <= k <= full_kde - 1
    q_minus_1 = q_lev[:-3, :]
    q_0 = q_lev[1:-2, :]
    q_plus_1 = q_lev[2:-1, :]
    q_plus_2 = q_lev[3:, :]
    mid_mfz = mfz[2:-2, :]
    direction = jnp.sign(mid_mfz)
    mid = upwind3(
        direction=direction,
        wgt=wgt,
        q_minus_1=q_minus_1,
        q_0=q_0,
        q_plus_1=q_plus_1,
        q_plus_2=q_plus_2
    ) * mid_mfz
    # =================================================
    # k = full_kde
    bottom = 0.5 * (q_lev[-2, :] + q_lev[-1, :]) * mfz[-1, :]
    # =================================================
    return jnp.concatenate([top[None, :], mid, bottom[None, :]], axis=0)


def upwind5_adv_vflx_at_lev(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float
) -> Float[Array, "nlev n_points"]:
    # =================================================
    # full_kds <= k <= full_kds + 1
    top = 0.5 * (q_lev[0:1, :] + q_lev[1:2, :]) * mfz[0:1, :]
    # =================================================
    # full_kds + 2 <= k <= full_kde - 2
    q_minus_2 = q_lev[:-5, :]
    q_minus_1 = q_lev[1:-4, :]
    q_0 = q_lev[2:-3, :]
    q_plus_1 = q_lev[3:-2, :]
    q_plus_2 = q_lev[4:-1, :]
    q_plus_3 = q_lev[5:, :]
    mid_mfz = mfz[2:-2, :]
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
    ) * mid_mfz
    # =================================================
    # full_kde - 1 <= k <= full_kde
    bottom_mid = 0.5 * (q_lev[-2:-1, :] + q_lev[-1:, :]) * mfz[-2:, :]
    # =================================================
    return jnp.concatenate([top, mid, bottom_mid], axis=0)


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
