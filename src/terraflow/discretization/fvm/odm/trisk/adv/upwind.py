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
from terraflow.discretization.fvm.odm.trisk.adv import types, batch, flux_reconstruct as fr

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

    def prepare(self, block: Block) -> Block:
        return self.adv_hflx_at_cell.prepare(block)

    def calc_at_cell(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ):
        mfn = batch.mfn.unwrap()
        qmfn = self.adv_hflx_at_cell(
            x_at_primal_cell=q * mfn,
            normal_u_at_edge=mfn,
            coord=coord,
            aux=aux
        )
        return eqx.tree_at(
            lambda b: b.qmfn,
            batch,
            qmfn
        )

    def calc_at_vtx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset
    ):
        mfn = batch.mfn.unwrap()
        qmfn = self.adv_hflx_at_vtx(
            x_at_dual_cell=q * mfn,
            tangent_u_at_edge=mfn,
            coord=coord,
            aux=aux
        )
        return eqx.tree_at(
            lambda b: b.qmfn,
            batch,
            qmfn
        )


class UpwindVerticalAdvection:

    def __init__(self, order: int, wgt: float = 1.0):
        self.order = order
        if self.order == 0:
            self.adv_vflx_at_cell = upwind0_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind0_adv_vflx_at_lev
            self.wgt = 1.0
        elif self.order == 3:
            self.adv_vflx_at_cell = upwind3_adv_vflx_at_cell
            self.adv_vflx_at_lev = upwind3_adv_vflx_at_lev
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
            return self.calc_at_lev(
                q=q,
                batch=batch,
            )
        elif batch.vert_loc == "lev":
            return self.calc_at_cell(
                q=q,
                batch=batch,
            )
        else:
            raise NotImplementedError(
                f"Vertical location {batch.vert_loc} is not implemented."
            )

    def prepare(self, block: Block) -> Block:
        return block

    def calc_at_cell(
        self,
        q: Array,
        batch: batch.AdvBatchType,
    ):
        mfz = batch.mfz.unwrap()
        qmfz = self.adv_vflx_at_cell(
            q_lev=q,
            mfz=mfz,
            wgt=self.wgt
        )
        return eqx.tree_at(
            lambda b: b.qmfz,
            batch,
            qmfz
        )

    def calc_at_lev(
        self,
        q: Array,
        batch: batch.AdvBatchType,
    ):
        mfz = batch.mfz.unwrap()
        qmfz = self.adv_vflx_at_lev(
            q=q,
            mfz_lev=mfz,
            wgt=self.wgt
        )
        return eqx.tree_at(
            lambda b: b.qmfz,
            batch,
            qmfz
        )


def upwind0_adv_vflx_at_cell(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    *args,
    **kwargs
) -> Float[Array, "nlev n_points"]:
    return jnp.pad(
        0.5 * (q_lev[1:-2, :] + q_lev[2:-1, :]) * mfz[1:-1, :],
        ((1, 1), (0, 0)),
        before_value=0.5 * (q_lev[0, :] + q_lev[1, :]) * mfz[0, :],
        after_value=0.5 * (q_lev[-2, :] + q_lev[-1, :]) * mfz[-1, :]
    )


def upwind3_adv_vflx_at_cell(
    q_lev: Float[Array, "nlev+1 n_points"],
    mfz: Float[Array, "nlev n_points"],
    wgt: float
) -> Float[Array, "nlev n_points"]:
    c31 = 7.0 / 12.0
    c32 = -1.0 / 12.0
    c33 = 1.0 / 12.0
    direction = jnp.sign(mfz[1:-1, :])
    tmp = (
        c31 * (q_lev[1:-2, :] + q_lev[2:-1, :])
        + c32 * (q_lev[3:, :] + q_lev[:-3, :])
        + c33 * ((q_lev[3:, :] - q_lev[:-3, :])
                 - 3 * (q_lev[2:-1, :] - q_lev[1:-2, :])) * direction * mfz[1:-1, :] * wgt
    )
    return jnp.pad(
        tmp,
        ((1, 1), (0, 0)),
        before_value=0.5 * (q_lev[0, :] + q_lev[1, :]) * mfz[0, :],
        after_value=0.5 * (q_lev[-2, :] + q_lev[-1, :]) * mfz[-1, :]
    )


def upwind0_adv_vflx_at_lev(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    *args,
    **kwargs
) -> Float[Array, "nlev+1 n_points"]:
    return jnp.pad(
        0.5 * (q[1:, :] + q[:-1, :]),
        ((1, 1), (0, 0)),
    ) * mfz_lev


def upwind3_adv_vflx_at_lev(
    q: Float[Array, "nlev n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    wgt: float
) -> Float[Array, "nlev+1 n_points"]:
    c31 = 7.0 / 12.0
    c32 = -1.0 / 12.0
    c33 = 1.0 / 12.0
    direction = jnp.sign(mfz_lev[2:-2, :])
    tmp = (
        c31 * (q[2:-1, :] + q[1:-2, :])
        + c32 * q[3:, :] + q[:-3, :]
        + c33 * ((q[3:, :] - q[:-3, :])
                 - 3 * (q[2:-1, :] - q[1:-2, :])) * direction * mfz_lev[2:-2, :] * wgt
    )
    tmp = jnp.pad(
        tmp,
        ((1, 1), (0, 0)),
        before_value=0.5 * (q[0, :] + q[1, :]) * mfz_lev[1, :],
        after_value=0.5 * (q[-2, :] + q[-1, :]) * mfz_lev[-2, :]
    )
    return jnp.pad(
        tmp,
        ((1, 1), (0, 0)),
    )
