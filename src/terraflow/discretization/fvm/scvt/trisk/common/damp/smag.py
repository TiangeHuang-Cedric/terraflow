from __future__ import annotations
from typing import Any, Dict, Self, TypeVar, TYPE_CHECKING
import jax
import jax.numpy as jnp
import equinox as eqx

from terraflow.util.math import _project_sphere2plane
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as mesh_ops
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    EquationMode,
    AbstractDynamicalEquationOperator,
    AbstractEquationTermAuxiliaryData
)
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.util.model_config import ModelConfig

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    HybridCoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    SingleLevCoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSys = TypeVar("CoordSys", HybridCoordSys, SingleLevCoordSys)

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class SmagorinskyDampData(AbstractEquationTermAuxiliaryData):
    perot_wgt_at_cell: Array
    perot_wgt_at_vtx: Array
    un_laplacian: Array
    dundt_damp: Array
    dptdt_damp: Array

    @classmethod
    def empty(cls) -> Self:
        return cls(
            _init=False,
            perot_wgt_at_cell=jnp.array([]),
            perot_wgt_at_vtx=jnp.array([]),
            un_laplacian=jnp.array([]),
            dundt_damp=jnp.array([]),
            dptdt_damp=jnp.array([]),
        )

    @classmethod
    def from_coord(cls, coord: CoordSys) -> Self:
        mesh = coord.mesh
        cell_edges = mesh._plg.ed
        vtx_edges = mesh._dua.ed
        edd_cart = mesh._edd.midpoint.cart
        cell_area = mesh._plg.area
        dual_area = mesh._dua.area
        cell_cart = mesh._plg.cc.cart
        vtx_cart = mesh._dua.cc.cart
        edp_leng = mesh._edp.length
        edd_leng = mesh._edd.length
        edp_nr = mesh._edp.nr
        n_lev = coord.vert.nlev
        n_cell = mesh.n_polg
        n_edge = mesh.n_edge
        dev = mesh.dev()
        float_type = mesh.dev.float_type
        # =====================================================
        is_valid_cell_nnb = cell_edges >= 0
        is_valid_vtx_nnb = vtx_edges >= 0
        # Calculate Perot weights at primal cells
        # (ncell, maxnnb, 3)
        local_basis = project_sphere2plane(
            op0=edd_cart[cell_edges, :],
            op1=cell_cart,
        )
        # (ncell, maxnnb, 3)
        perot_wgt_at_cell = edp_leng[cell_edges, None] * jnp.sum(
            local_basis / jnp.linalg.norm(
                local_basis,
                axis=-1,
                keepdims=True
            ) * edp_nr[cell_edges, :],
            axis=-1
        )[:, :, None] * local_basis / cell_area[:, None, None]
        perot_wgt_at_cell = jnp.where(
            is_valid_cell_nnb[:, :, None],
            perot_wgt_at_cell,
            0.0
        )

        # Calculate Perot weights at dual cells
        # (nvtx, maxnnb_dual, 3)
        local_basis = project_sphere2plane(
            op0=edd_cart[vtx_edges, :],
            op1=vtx_cart,
        )
        # (nvtx, maxnnb_dual, 3)
        cross_product = jnp.cross(edd_cart[vtx_edges, :], local_basis, axis=-1)
        # (nvtx, maxnnb_dual)
        tmp = jnp.sum(
            edp_nr[vtx_edges, :] * cross_product / jnp.linalg.norm(
                cross_product,
                axis=-1,
                keepdims=True
            ),
            axis=-1
        )
        # (nvtx, maxnnb_dual, 3)
        perot_wgt_at_vtx = cross_product * tmp[:, :, None] * edd_leng[vtx_edges, None]
        perot_wgt_at_vtx = perot_wgt_at_vtx / dual_area[:, None, None]
        perot_wgt_at_vtx = jnp.where(
            is_valid_vtx_nnb[:, :, None],
            perot_wgt_at_vtx,
            0.0
        )

        un_laplacian = jnp.zeros((n_lev, n_edge), device=dev, dtype=float_type)
        dundt_damp = jnp.zeros((n_lev, n_edge), device=dev, dtype=float_type)
        dptdt_damp = jnp.zeros((n_lev, n_cell), device=dev, dtype=float_type)
        # =====================================================
        return cls(
            _init=True,
            perot_wgt_at_cell=perot_wgt_at_cell,
            perot_wgt_at_vtx=perot_wgt_at_vtx,
            un_laplacian=un_laplacian,
            dundt_damp=dundt_damp,
            dptdt_damp=dptdt_damp,
        )


class SmagorinskyDamp(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, itime: Int[Array, ""] | int) -> BLOCK:
        para = block.para
        radius = para.radius.unwrap()
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        kds = vert.full_kds
        cell_edges = mesh._plg.ed
        vtx_edges = mesh._dua.ed
        edd_v = mesh._edd.v
        edp_v = mesh._edp.v
        cell_cart = mesh._plg.cc.cart
        vtx_cart = mesh._dua.cc.cart
        edp_nr = mesh._edp.nr
        edp_tg = mesh._edp.tg
        edp_leng = mesh._edp.length
        edd_leng = mesh._edd.length
        perot_wgt_at_cell = block.equaux.damp.smag.perot_wgt_at_cell
        perot_wgt_at_vtx = block.equaux.damp.smag.perot_wgt_at_vtx
        un = block.state.un.unwrap()[itime, :, :]
        pt = block.state.pt.unwrap()[itime, :, :]
        dmg = block.state.dmg.unwrap()[itime, :, :]
        smag_damp_coef = self.smag_damp_coef
        # =====================================================
        # Calculate velocity vector at primal and dual cells
        # (nlev, ncell, 3)
        uv_cell = jnp.sum(un[:, cell_edges, None] * perot_wgt_at_cell[None, :, :, :], axis=-2)
        # (nlev, nvtx, 3)
        uv_vtx = jnp.sum(un[:, vtx_edges, None] * perot_wgt_at_vtx[None, :, :, :], axis=-2)
        # =====================================================
        #      pc2
        #       |
        # dc1---e---dc2
        #       |
        #      pc1
        pc1 = edd_v[:, 0]  # (nedge,)
        pc2 = edd_v[:, 1]  # (nedge,)
        dc1 = edp_v[:, 0]  # (nedge,)
        dc2 = edp_v[:, 1]  # (nedge,)
        # (nedge, 3)
        pc1pc2 = cell_cart[pc2, :] - cell_cart[pc1, :]
        dc1dc2 = vtx_cart[dc2, :] - vtx_cart[dc1, :]
        # (nedge,)
        flag1 = jnp.sign(jnp.sum(pc1pc2 * edp_nr, axis=-1))
        flag2 = jnp.sign(jnp.sum(dc1dc2 * edp_tg, axis=-1))
        # =====================================================
        # (nlev, nedge)
        un_pc1 = jnp.sum(uv_cell[:, pc1, :] * edp_nr, axis=-1)
        un_pc2 = jnp.sum(uv_cell[:, pc2, :] * edp_nr, axis=-1)
        un_dc1 = jnp.sum(uv_vtx[:, dc1, :] * edp_nr, axis=-1)
        un_dc2 = jnp.sum(uv_vtx[:, dc2, :] * edp_nr, axis=-1)

        # (nlev, nedge)
        ut_pc1 = jnp.sum(uv_cell[:, pc1, :] * edp_tg, axis=-1)
        ut_pc2 = jnp.sum(uv_cell[:, pc2, :] * edp_tg, axis=-1)
        ut_dc1 = jnp.sum(uv_vtx[:, dc1, :] * edp_tg, axis=-1)
        ut_dc2 = jnp.sum(uv_vtx[:, dc2, :] * edp_tg, axis=-1)

        # (nlev, nedge)
        un_pc1pc2 = flag1 * (un_pc2 - un_pc1) / (radius * edd_leng)
        ut_pc1pc2 = flag1 * (ut_pc2 - ut_pc1) / (radius * edd_leng)
        un_dc1dc2 = flag2 * (un_dc2 - un_dc1) / (radius * edp_leng)
        ut_dc1dc2 = flag2 * (ut_dc2 - ut_dc1) / (radius * edp_leng)

        # -----------------------------------------------------
        # Horizontal tension strain on centers
        # ∂u   ∂v
        # -- - --
        # ∂x   ∂y
        # (nlev, nedge)
        smag_t = (un_pc1pc2 - ut_dc1dc2)

        # -----------------------------------------------------
        # Horizontal shearing strain on vertices
        # ∂u   ∂v
        # -- + --
        # ∂y   ∂x
        # (nlev, nedge)
        smag_s = (un_dc1dc2 + ut_pc1pc2)

        ls2 = radius**2 * edd_leng * edp_leng
        smag_eddy_coef = smag_damp_coef * ls2 * jnp.sqrt(smag_t**2 + smag_s**2)
        # =====================================================
        # Normal wind
        # math: ((upc1-un)/(0.5*edd_leng)-(un-upc2)/(0.5*edd_leng))/(0.5*edd_leng)
        un_laplacian = (un_pc1 + un_pc2 - 2 * un) / (radius * edd_leng)**2
        un_laplacian = un_laplacian + (un_dc1 + un_dc2 - 2 * un) / (radius * edp_leng)**2
        dundt_damp = 4 * smag_eddy_coef * un_laplacian
        if self.use_sponge_layer:
            # From top to bottom
            nsponge = kds + self.sponge_layer_k0
            sponge_layer_coef = self.sponge_layer_coef
            ilev = jnp.arange(nsponge) + 1
            tmp = 4.0 * un_laplacian[kds:nsponge, :] * edp_leng * radius
            tmp = tmp * (7 - ilev)[:, None] * sponge_layer_coef / 6.0
            dundt_damp = dundt_damp.at[kds:nsponge, :].add(tmp)
        # -----------------------------------------------------
        # Potential temperature
        grad_pt = mesh_ops.grad_at_dual_edge(
            x_at_primal_cell=pt,
            coord=coord,
            para=para
        ) * smag_damp_coef
        dptdt_damp = mesh_ops.div_at_primal_cell(
            x_at_primal_edge=grad_pt,
            coord=coord,
            para=para
        ) * dmg
        # =====================================================
        return tree_at(
            coord=coord,
            where=lambda b: (
                b.equaux.damp.smag.un_laplacian,
                b.equaux.damp.smag.dundt_damp,
                b.equaux.damp.smag.dptdt_damp
            ),
            pytree=block,
            replace=(un_laplacian, dundt_damp, dptdt_damp),
        )

    def update_tend(self, block: BLOCK) -> BLOCK:
        dundt = block.tend.dundt.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        dundt_damp = block.equaux.damp.smag.dundt_damp
        dptdt_damp = block.equaux.damp.smag.dptdt_damp
        # =====================================================
        # Normal wind
        dundt = dundt + dundt_damp
        # -----------------------------------------------------
        # Potential temperature
        dptdt = dptdt + dptdt_damp
        # =====================================================
        return eqx.tree_at(
            where=lambda b: (b.tend.dundt.data, b.tend.dptdt.data),
            pytree=block,
            replace=(dundt, dptdt),
        )

    def config2dict(self) -> Dict[str, Any]:
        self._cfg.update(
            {
                "smag_damp_coef": self.smag_damp_coef,
            }
        )
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            if cfg.use_smag_damp:
                self.smag_damp_coef = cfg.smag_damp_coef
            self.use_sponge_layer = cfg.use_sponge_layer
            if cfg.use_sponge_layer:
                self.sponge_layer_k0 = cfg.sponge_layer_k0
                self.sponge_layer_coef = cfg.sponge_layer_coef
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        smag = SmagorinskyDampData.from_coord(block.coord)
        block = eqx.tree_at(
            where=lambda b: b.equaux.damp.smag,
            pytree=block,
            replace=smag,
        )
        return block


def project_sphere2plane(
    op0: Float[Array, "npoint nnb 3"],
    op1: Float[Array, "npoint 3"],
) -> Float[Array, "npoint nnb 3"]:
    return jax.vmap(
        jax.vmap(
            _project_sphere2plane,
            in_axes=(0, 0),
            out_axes=0
        ),
        in_axes=(1, None),
        out_axes=1
    )(
        op0,
        op1
    )
