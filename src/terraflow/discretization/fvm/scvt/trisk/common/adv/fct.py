from __future__ import annotations
from typing import Callable, Optional, Any, Self, TypeVar, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.mesh.scvt.operators import div_at_primal_cell
from terraflow.discretization.abstract import AbstractEquationTermAuxiliaryData
from terraflow.layout.abstract import (
    MeshEntity,
    VertLoc,
    Basis
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.common.adv.flux_reconstruct import WRF1AtEdgeOnPrimalMesh
from terraflow.discretization.fvm.scvt.trisk.common.adv.upwind import upwind1_adv_vflx_at_cell

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.model_config import ModelConfig
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

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

    Block_OneLayer = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]
    Block_Hybrid = Block[HybridVertCoordinate, HybridVertCoordLayout]

    BLOCK = TypeVar("BLOCK", Block_OneLayer, Block_Hybrid)


class FluxCorrectedTransportData((AbstractEquationTermAuxiliaryData)):
    qmfn_lo: Array
    qmfz_lo: Array

    @classmethod
    def empty(cls) -> Self:
        return cls(
            _init=False,
            qmfn_lo=jnp.empty(0),
            qmfz_lo=jnp.empty(0),
        )

    @classmethod
    def from_coord(cls, coord: CoordSys) -> Self:
        # Horizontal fluxes of batch
        qmfn_lo = coord.zeros_like(
            hori_loc=MeshEntity.Edge,
            vert_loc=VertLoc.Cell,
            basis=Basis.Normal,
        )
        # Vertical fluxes of batch
        qmfz_lo = coord.zeros_like(
            hori_loc=MeshEntity.Cell,
            vert_loc=VertLoc.Lev,
            basis=Basis.Normal,
        )
        return cls(
            _init=True,
            qmfn_lo=qmfn_lo,
            qmfz_lo=qmfz_lo,
        )


class LowOrderHorizontalAdvectionSchemeAtCell:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
    ) -> Block:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = batch.un.unwrap()
        mfn = batch.mfn.unwrap()
        qmfn_lo = block.equaux.adv.fct.qmfn_lo
        # =============================================
        qmfn_lo = qmfn_lo.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                x_at_primal_cell=q,
                normal_u_at_edge=un,
                coord=coord,
                x_at_primal_edge=qmfn_lo
            ) * mfn[kds:kde+1, ids:ide+1]
        )
        # =============================================
        return tree_at(
            coord=coord,
            where=lambda b: b.equaux.adv.fct.qmfn_lo,
            pytree=block,
            replace=qmfn_lo,
        )

    def upwind1(self) -> WRF1AtEdgeOnPrimalMesh:
        self.scheme = WRF1AtEdgeOnPrimalMesh()
        self._cfg["fct_lo_scheme_h"] = "upwind1"
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.fct_lo_scheme_h == "upwind1":
                self.upwind1()
            else:
                raise ValueError(
                    f"Unsupported low order scheme: {cfg.fct_lo_scheme_h} for FCT horizontal limiter."
                )
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class LowOrderVerticalAdvectionSchemeAtCell:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        wgt: float = 1.0
    ) -> Block:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.half_kds, vert.half_kde
        mfz_lev = batch.mfz.unwrap()
        qmfz_lo_lev = block.equaux.adv.fct.qmfz_lo
        # =============================================
        qmfz_lo_lev = qmfz_lo_lev.at[kds:kde+1, ids:ide+1].set(
            self.scheme(
                coord=coord,
                q=q,
                mfz_lev=mfz_lev,
                qmfz_lev=qmfz_lo_lev,
                wgt=wgt
            ) * mfz_lev[kds:kde+1, ids:ide+1]
        )
        # =============================================
        return tree_at(
            coord=coord,
            where=lambda b: b.equaux.adv.fct.qmfz_lo,
            pytree=block,
            replace=qmfz_lo_lev,
        )

    def upwind1(self) -> Callable:
        self.scheme = upwind1_adv_vflx_at_cell
        self._cfg["fct_lo_scheme_v"] = "upwind1"
        self._built = True
        return self.scheme

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self._built:
            if cfg.fct_lo_scheme_v == "upwind1":
                self.upwind1()
            else:
                raise ValueError(
                    f"Unsupported low order scheme: {cfg.fct_lo_scheme_v} for FCT vertical limiter."
                )
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class FluxCorrectedTransportHorizontalLimiterAtCell:

    def __init__(self, regression: bool = True):
        self.low_order_scheme = LowOrderHorizontalAdvectionSchemeAtCell()
        self.regression = regression
        self._cfg = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        mq_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        dt: Float[Array, ""] | float
    ) -> batch.AdvBatchType:
        coord = block.coord
        qmfn_ho = batch.qmfn.unwrap()
        # =============================================
        # Calculate low order fluxes at edge
        block = self.low_order_scheme(
            q=q,
            batch=batch,
            block=block,
        )
        # Calculate limited fluxes at edge
        qmfn_lo = block.equaux.adv.fct.qmfn_lo
        limited_qmfn = fct_hori_limiter_at_cell_on_single_tracer(
            q=q,
            mq_old=mq_old,
            qmfn_lo=qmfn_lo,
            qmfn_ho=qmfn_ho,
            dmg_new=block.adv_batch_bg.m.unwrap(),
            coord=coord,
            para=block.para,
            dt=dt,
            qmfn=qmfn_ho,
            regression=self.regression
        )
        # =============================================
        return tree_at(
            coord=coord,
            where=lambda b: b.qmfn,
            pytree=batch,
            replace=limited_qmfn,
            fill_halo=True,
        )

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        self._cfg.update(
            self.low_order_scheme.config2dict()
        )
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.regression = cfg.fct_regression_h
            if not self.low_order_scheme.has_built():
                self.low_order_scheme.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        if not block.equaux.adv.fct.has_initialized():
            fct = FluxCorrectedTransportData.from_coord(coord=block.coord)
            block = eqx.tree_at(
                lambda b: b.equaux.adv.fct,
                block,
                replace=fct
            )
        return self.low_order_scheme.prepare_block(block)


class FluxCorrectedTransportVerticalLimiterAtCell:
    def __init__(self, regression: bool = True):
        self.low_order_scheme = LowOrderVerticalAdvectionSchemeAtCell()
        self.regression = regression
        self._cfg = {}
        self._built = False

    def __call__(
        self,
        q: Float[Array, "nlev ncell"],
        mq_old: Float[Array, "nlev ncell"],
        batch: batch.AdvBatchType,
        block: Block,
        dt: Float[Array, ""] | float
    ) -> batch.AdvBatchType:
        coord = block.coord
        qmfz_ho = batch.qmfz.unwrap()
        # =============================================
        # Calculate low order fluxes at half level
        block = self.low_order_scheme(
            q=q,
            batch=batch,
            block=block,
            wgt=1.0
        )
        # Calculate limited fluxes at half level
        qmfz_lo = block.equaux.adv.fct.qmfz_lo
        limted_qmfz = fct_vert_limiter_at_cell_on_single_tracer(
            q=q,
            mq_old=mq_old,
            qmfz_lo=qmfz_lo,
            qmfz_ho=qmfz_ho,
            dmg_new=block.adv_batch_bg.m.unwrap(),
            coord=coord,
            dt=dt,
            qmfz=qmfz_ho,
            regression=self.regression
        )
        # =============================================
        return tree_at(
            coord=coord,
            where=lambda b: b.qmfz,
            pytree=batch,
            replace=limted_qmfz,
            fill_halo=True,
        )

    def has_built(self) -> bool:
        return self._built

    def config2dict(self) -> dict[str, Any]:
        self._cfg.update(
            self.low_order_scheme.config2dict()
        )
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.regression = cfg.fct_regression_h
            if not self.low_order_scheme.has_built():
                self.low_order_scheme.build_with_config(cfg=cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        if not block.equaux.adv.fct.has_initialized():
            fct = FluxCorrectedTransportData.from_coord(coord=block.coord)
            block = eqx.tree_at(
                lambda b: b.equaux.adv.fct,
                block,
                replace=fct
            )
        return block


def fct_hori_limiter_at_cell_on_single_tracer(
    q: Float[Array, "nlev ncell"],
    mq_old: Float[Array, "nlev ncell"],
    qmfn_lo: Float[Array, "nlev nedge"],
    qmfn_ho: Float[Array, "nlev nedge"],
    dmg_new: Float[Array, "nlev ncell"],
    coord: CoordSys,
    para: PlanetParameterSet,
    dt: Float[Array, ""] | float,
    qmfn: Optional[Float[Array, "nlev nedge"]] = None,
    regression: bool = True,
) -> Float[Array, "nlev ncell"]:
    radius = para.radius.unwrap()
    mesh = coord.mesh
    vert = coord.vert
    p_ids, p_ihe = mesh.polg_ids, mesh.polg_ihe[0]
    e_ids, e_ihe = mesh.edge_ids, mesh.edge_ihe[0]
    kds, kde = vert.full_kds, vert.full_kde
    edp_leng = mesh._edp.length[e_ids:e_ihe+1]
    q_ = q[kds:kde+1, p_ids:p_ihe+1]
    mq_old_ = mq_old[kds:kde+1, p_ids:p_ihe+1]
    dmg_new_ = dmg_new[kds:kde+1, p_ids:p_ihe+1]
    # ========================================================
    # Calculate anti-diffusive fluxes
    ad_flux = (
        qmfn_ho[kds:kde+1, e_ids:e_ihe+1]
        - qmfn_lo[kds:kde+1, e_ids:e_ihe+1]
    ) * radius * edp_leng
    # ========================================================
    # Calculate the updated low order solution
    div = div_at_primal_cell(
        x_at_primal_edge=qmfn_lo,
        coord=coord,
        para=para,
        hori_end_at="halo0",
    )
    q_td = (mq_old_ - dt * div) / dmg_new_
    # ========================================================
    # Calculate limiter R values (nlev, ncell)
    polg_edge = mesh._plg.edge[p_ids:p_ihe+1, :]
    polg_nb = mesh._plg.neighbors[p_ids:p_ihe+1, :]
    polg_nr = mesh._plg.nr[p_ids:p_ihe+1, :]
    plg_area = mesh._plg.area[p_ids:p_ihe+1]
    # =============================================
    # Calculate qmax
    qmax = - jnp.array(1.e16, dtype=q.dtype)
    qmax = jnp.maximum(qmax, jnp.maximum(q_, q_td))

    # Filter out the invalid neighbors (with negative index) by setting their q values to very large numbers
    # to ensure they do not affect the calculation of qmax.
    q_nnb = q_[:, polg_nb]  # (nlev, ncell, nnb)
    q_td_nnb = q_td[:, polg_nb]  # (nlev, ncell, nnb)
    q_nnb = jnp.where(
        polg_nb[None, :, :] < 0,
        - jnp.array(1.e16, dtype=q.dtype),
        q_nnb
    )
    q_td_nnb = jnp.where(
        polg_nb[None, :, :] < 0,
        - jnp.array(1.e16, dtype=q.dtype),
        q_td_nnb
    )
    qmax = jnp.maximum(
        qmax,
        jnp.maximum(
            jnp.max(q_nnb, axis=-1),
            jnp.max(q_td_nnb, axis=-1)
        )
    )
    # =============================================
    # Calculate qmin
    qmin = jnp.array(1.e16, dtype=q.dtype)
    qmin = jnp.minimum(qmin, jnp.minimum(q_, q_td))

    # Filter out the invalid neighbors (with negative index) by setting their q values to very small numbers
    # to ensure they do not affect the calculation of qmin.
    q_nnb = q_[:, polg_nb]  # (nlev, ncell, nnb)
    q_td_nnb = q_td[:, polg_nb]  # (nlev, ncell, nnb)
    q_nnb = jnp.where(
        polg_nb[None, :, :] < 0,
        jnp.array(1.e16, dtype=q.dtype),
        q_nnb
    )
    q_td_nnb = jnp.where(
        polg_nb[None, :, :] < 0,
        jnp.array(1.e16, dtype=q.dtype),
        q_td_nnb
    )

    qmin = jnp.minimum(
        qmin,
        jnp.minimum(
            jnp.min(q_nnb, axis=-1),
            jnp.min(q_td_nnb, axis=-1)
        )
    )
    # =============================================
    # Calculate p_in and p_out
    tmp = polg_nr[None, :, :] * ad_flux[:, polg_edge]  # (nlev, ncell, nnb)
    tmp = jnp.where(
        polg_edge[None, :, :] < 0,
        0.0,
        tmp
    )
    p_in = jnp.minimum(
        jnp.array(0.0, dtype=q.dtype),
        tmp  # (nlev, ncell, nnb)
    ).sum(axis=-1)  # (nlev, ncell)
    p_out = jnp.maximum(
        jnp.array(0.0, dtype=q.dtype),
        tmp  # (nlev, ncell, nnb)
    ).sum(axis=-1)  # (nlev, ncell)
    # =============================================
    # Calculate m_in and m_out (nlev, ncell)
    maf_in = dmg_new_ * (qmax - q_td) * plg_area * radius**2 / dt
    maf_out = dmg_new_ * (q_td - qmin) * plg_area * radius**2 / dt
    # =============================================
    # Calculate limiter R values (nlev, ncell)
    if regression:
        r_in = jnp.where(
            p_in < 0,
            jnp.minimum(
                jnp.array(1.0, dtype=q.dtype),
                maf_in / jnp.abs(p_in)
            ),
            0.0
        )
        r_out = jnp.where(
            p_out > 0,
            jnp.minimum(
                jnp.array(1.0, dtype=q.dtype),
                maf_out / jnp.abs(p_out)
            ),
            0.0
        )
    else:
        if q.dtype == jnp.float32:
            kesi = 1.e-40
        elif q.dtype == jnp.float64:
            kesi = 1.e-80
        else:
            raise ValueError("Unsupported data type for q.")
        r_in = jnp.minimum(
            jnp.array(1.0, dtype=q.dtype),
            maf_in / (jnp.abs(p_in) + kesi)
        )
        r_out = jnp.minimum(
            jnp.array(1.0, dtype=q.dtype),
            maf_out / (jnp.abs(p_out) + kesi)
        )
    # ========================================================
    # Calculate limited mass flux at edge
    nedge = e_ihe - e_ids + 1
    nlev = kde - kds + 1
    edd_v = mesh._edd.v[e_ids:e_ihe+1, :]
    edp_nr = mesh._edp.nr[e_ids:e_ihe+1, :]
    polg_cc_cart = mesh._plg.cc.cart

    v0 = edd_v[e_ids:e_ihe+1, 0]
    v1 = edd_v[e_ids:e_ihe+1, 1]
    v0v1 = polg_cc_cart[v1, :] - polg_cc_cart[v0, :]
    flag = jnp.sign(jnp.sum(v0v1 * edp_nr, axis=-1))  # 0 or 1
    mark = jnp.rint(
        -0.5 * jnp.sign(flag * ad_flux) + 0.5
    ).astype(jnp.int32)  # (nlev, nedge)
    v_upwind = edd_v[jnp.arange(nedge), mark]  # (nlev, nedge,)
    v_dowind = edd_v[jnp.arange(nedge), 1-mark]  # (nlev, nedge,)
    # Calculate coef: (nlev, ncell) -> (nlev, nedge)
    coef = jnp.minimum(
        r_in[jnp.arange(nlev)[:, None], v_dowind],
        r_out[jnp.arange(nlev)[:, None], v_upwind]
    )
    res = qmfn_lo + coef * (qmfn_ho - qmfn_lo)  # (nlev, nedge)
    # ========================================================
    if qmfn is None:
        return res
    else:
        return qmfn.at[kds:kde+1, e_ids:e_ihe+1].set(res)


def fct_vert_limiter_at_cell_on_single_tracer(
    q: Float[Array, "nlev ncell"],
    mq_old: Float[Array, "nlev ncell"],
    qmfz_lo: Float[Array, "nlev+1 ncell"],
    qmfz_ho: Float[Array, "nlev+1 ncell"],
    dmg_new: Float[Array, "nlev+1 ncell"],
    coord: CoordSys,
    dt: Float[Array, ""] | float,
    qmfz: Optional[Float[Array, "nlev+1 ncell"]] = None,
    regression: bool = True,
) -> Float[Array, "nlev+1 ncell"]:
    mesh = coord.mesh
    vert = coord.vert
    ids, ide = mesh.polg_ids, mesh.polg_ide
    kds, kde = vert.full_kds, vert.full_kde
    q_ = q[kds:kde+1, ids:ide+1]
    mq_old_ = mq_old[kds:kde+1, ids:ide+1]
    dmg_new_ = dmg_new[kds:kde+1, ids:ide+1]
    # ========================================================
    # Calculate anti-diffusive fluxes at half levels
    # ad_flux is at half level (half_kds to half_kde, ids to ide)
    ad_flux = qmfz_ho[kds+1:kde+1, ids:ide+1] - qmfz_lo[kds+1:kde+1, ids:ide+1]
    ad_flux = jnp.pad(ad_flux, ((1, 1), (0, 0)))
    # ========================================================
    # Calculate the updated low order solution
    tmp = (qmfz_lo[kds+1:kde+2, ids:ide+1] - qmfz_lo[kds:kde+1, ids:ide+1])
    q_td = mq_old_ - dt * tmp / dmg_new_  # (nlev, ncell)
    # ========================================================
    # Calculate limiter R values
    # =============================================
    # Calculate qmax and qmin at half levels
    # At top (full_kds)
    qmax_top = jnp.maximum(
        jnp.maximum(q_[0, :], q_[1, :]),
        jnp.maximum(q_td[0, :], q_td[1, :])
    )
    qmin_top = jnp.minimum(
        jnp.minimum(q_[0, :], q_[1, :]),
        jnp.minimum(q_td[0, :], q_td[1, :])
    )
    # From full_kds + 1 to full_kde - 1
    qmax_mid = jnp.maximum(
        jnp.maximum(
            q_[1:-1, :],
            jnp.maximum(q_[:-2, :], q_[2:, :]),
        ),
        jnp.maximum(
            q_td[1:-1, :],
            jnp.maximum(q_td[:-2, :], q_td[2:, :]),
        )
    )
    qmin_mid = jnp.minimum(
        jnp.minimum(
            q_[1:-1, :],
            jnp.minimum(q_[:-2, :], q_[2:, :]),
        ),
        jnp.minimum(
            q_td[1:-1, :],
            jnp.minimum(q_td[:-2, :], q_td[2:, :]),
        )
    )
    # At bottom (full_kde)
    qmax_bot = jnp.maximum(
        jnp.maximum(q_[-1, :], q_[-2, :]),
        jnp.maximum(q_td[-1, :], q_td[-2, :])
    )
    qmin_bot = jnp.minimum(
        jnp.minimum(q_[-1, :], q_[-2, :]),
        jnp.minimum(q_td[-1, :], q_td[-2, :])
    )
    # Combine them together
    qmax = jnp.concatenate(
        [qmax_top[None, :],
         qmax_mid,
         qmax_bot[None, :]],
        axis=0
    )  # (nlev, ncell)
    qmin = jnp.concatenate(
        [qmin_top[None, :],
         qmin_mid,
         qmin_bot[None, :]],
        axis=0
    )  # (nlev, ncell)
    # =============================================
    # Calculate p_in and p_out
    zeros = jnp.array(0.0, dtype=q.dtype)
    p_in = (
        jnp.maximum(zeros, ad_flux[kds:kde+1, ids:ide+1])
        - jnp.minimum(zeros, ad_flux[kds+1:kde+2, ids:ide+1])
    )  # (nlev, ncell)
    p_out = (
        jnp.maximum(zeros, ad_flux[kds+1:kde+2, ids:ide+1])
        - jnp.minimum(zeros, ad_flux[kds:kde+1, ids:ide+1])
    )  # (nlev, ncell)
    # =============================================
    # Calculate m_in and m_out
    m_in = (qmax - q_td) * dmg_new_ / dt   # (nlev, ncell)
    m_out = (q_td - qmin) * dmg_new_ / dt  # (nlev, ncell)
    # =============================================
    ones = jnp.array(1.0, dtype=q.dtype)
    if regression:
        r_in = jnp.where(
            p_in > 0,
            jnp.minimum(ones, m_in / p_in),
            zeros
        )  # (nlev, ncell)
        r_out = jnp.where(
            p_out > 0,
            jnp.minimum(ones, m_out / p_out),
            zeros
        )  # (nlev, ncell)
    else:
        if q.dtype == jnp.float32:
            kesi = 1.e-40
        elif q.dtype == jnp.float64:
            kesi = 1.e-80
        else:
            raise ValueError("Unsupported data type for q.")
        r_in = jnp.minimum(ones, m_in / (p_in + kesi))     # (nlev, ncell)
        r_out = jnp.minimum(ones, m_out / (p_out + kesi))  # (nlev, ncell)
    # ========================================================
    # Calculate limited mass flux at half level
    cell_range = jnp.arange(ids, ide+1)
    mark = jnp.rint(
        -0.5 * jnp.sign(ad_flux[kds+1:kde+1, ids:ide+1]) + 0.5
    ).astype(jnp.int32).squeeze()  # 0 or 1 (nlev, ncell)
    ilev = jnp.arange(kds+1, kde+1)[:, None]
    v_upwind = ilev - (1 - mark)  # (nlev, ncell)
    v_dowind = ilev - mark        # (nlev, ncell)
    coef = jnp.minimum(
        r_in[v_dowind, cell_range],
        r_out[v_upwind, cell_range]
    )  # (nlev, ncell)
    qmfz_lo_ = qmfz_lo[kds+1:kde+1, ids:ide+1]
    qmfz_ho_ = qmfz_ho[kds+1:kde+1, ids:ide+1]
    res = qmfz_lo_ + coef * (qmfz_ho_ - qmfz_lo_)
    res = jnp.pad(res, ((1, 1), (0, 0)))  # Pad the top and bottom with zeros
    # ========================================================
    if qmfz is None:
        return res
    else:
        return qmfz.at[kds:kde+2, ids:ide+1].set(res)
