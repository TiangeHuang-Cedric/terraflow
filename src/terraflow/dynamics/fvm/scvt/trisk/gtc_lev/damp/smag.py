from __future__ import annotations
from typing import Self, TYPE_CHECKING
import jax
import jax.numpy as jnp

from terraflow.util.device import AbstractDevice, CPU
from terraflow.util.sphere_geometry import _project_sphere2plane
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt import operators as mesh_ops
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryData

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.model.config import UserConfig
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    INT = typing.Int
    FLT = typing.Float
    FloatType = typing.FloatType
    Array = typing.Array


class SmagorinskyDampData(AbstractEquationAuxiliaryData):
    perot_wgt_at_cell: FLT[Array, "ncell maxnnb_cell 3"]
    perot_wgt_at_vtx : FLT[Array, "nvtx  maxnnb_vtx  3"]
    vec_at_cell      : FLT[Array, "nlev  ncell       3"]
    vec_at_vtx       : FLT[Array, "nlev  nvtx        3"]

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            perot_wgt_at_cell=jnp.array([]),
            perot_wgt_at_vtx=jnp.array([]),
            vec_at_cell=jnp.array([]),
            vec_at_vtx=jnp.array([])
        )
    
    @classmethod
    def initialize(
        cls,
        cfg: UserConfig,
        para: PlanetParameterSet, 
        mesh: SCVTMesh, 
        dev: AbstractDevice = CPU()
    ) -> Self:
        if cfg.use_smag_damp:
            return cls.from_coord(para, mesh, dev)
        else:
            return cls.empty()

    @classmethod
    def from_coord(
        cls, 
        para: PlanetParameterSet, 
        mesh: SCVTMesh, 
        dev: AbstractDevice = CPU()
    ) -> Self:
        radius        = para.radius.unwrap()
        de            = mesh.de
        le            = mesh.le
        dual_edge_xyz = mesh.dual_edge_xyz
        edpe_nr       = mesh.edge_nr
        float_type    = dev.float_type
        # =====================================================
        perot_wgt_at_cell = jnp.zeros(
            (mesh.cell_ime-mesh.cell_ims+1, mesh.maxnnb_cell, 3), 
            device=dev(), 
            dtype=float_type
        )
        perot_wgt_at_vtx  = jnp.zeros(
            (mesh.vtx_ime -mesh.vtx_ims +1, mesh.maxnnb_vtx , 3), 
            device=dev(), 
            dtype=float_type            
        )
        vec_at_cell = jnp.zeros(
            (mesh.full_nlev, mesh.cell_ime-mesh.cell_ims+1, 3),
            device=dev(),
            dtype=float_type
        )
        vec_at_vtx  = jnp.zeros(
            (mesh.full_nlev, mesh.vtx_ime -mesh.vtx_ims +1, 3),
            device=dev(),
            dtype=float_type
        )
        # ===================================================== 
        # ------ Calculate Perot weights at cell ------
        ids, ihe          = mesh.cell_ids, mesh.cell_ihe[mesh.hori_hw-2]
        cell_ed           = mesh.cell_ed  [ids:ihe+1, :]
        cell_xyz          = mesh.cell_xyz [ids:ihe+1, :] / radius
        area_cell         = mesh.area_cell[ids:ihe+1   ]
        is_valid_cell_nnb = cell_ed >= 0
        
        # (ncell, maxnnb, 3)
        local_basis = project_sphere2plane(
            R=para.radius.unwrap(),
            op0=cell_xyz,
            op1=dual_edge_xyz[cell_ed, :] / radius,
        )
        # (ncell, maxnnb, 3)
        tmp = le[cell_ed, None] * jnp.sum(
            local_basis / jnp.linalg.norm(
                local_basis,
                axis=-1,
                keepdims=True
            ) * edpe_nr[cell_ed, :],
            axis=-1
        )[:, :, None] * local_basis / area_cell[:, None, None]
        tmp = jnp.where(
            is_valid_cell_nnb[:, :, None],
            tmp,
            0.0
        )
        perot_wgt_at_cell = perot_wgt_at_cell.at[ids:ihe+1, :, :].set(tmp)

        # ------ Calculate Perot weights at vtx ------
        ids, ihe          = mesh.vtx_ids, mesh.vtx_ihe[mesh.hori_hw-1]
        vtx_ed            = mesh.vtx_ed  [ids:ihe+1, :]
        area_vtx          = mesh.area_vtx[ids:ihe+1   ]
        vtx_xyz           = mesh.vtx_xyz [ids:ihe+1, :] / radius
        is_valid_vtx_nnb  = vtx_ed >= 0
        
        # (nvtx, maxnnb_dual, 3)
        local_basis = project_sphere2plane(
            R=para.radius.unwrap(),
            op0=vtx_xyz,
            op1=dual_edge_xyz[vtx_ed, :] / radius,
        )
        
        # (nvtx, maxnnb_dual, 3)
        cross_product = jnp.cross(dual_edge_xyz[vtx_ed, :], local_basis, axis=-1)
        # (nvtx, maxnnb_dual)
        tmp = jnp.sum(
            edpe_nr[vtx_ed, :] * cross_product / jnp.linalg.norm(
                cross_product,
                axis=-1,
                keepdims=True
            ),
            axis=-1
        )
        # (nvtx, maxnnb_dual, 3)
        tmp = cross_product * tmp[:, :, None] * de[vtx_ed, None]
        tmp = tmp / area_vtx[:, None, None]
        tmp = jnp.where(
            is_valid_vtx_nnb[:, :, None],
            tmp,
            0.0
        )
        perot_wgt_at_vtx = perot_wgt_at_vtx.at[ids:ihe+1, :, :].set(tmp)
        # =====================================================
        return cls(
            initialized=True,
            perot_wgt_at_cell=perot_wgt_at_cell,
            perot_wgt_at_vtx=perot_wgt_at_vtx,
            vec_at_cell=vec_at_cell,
            vec_at_vtx=vec_at_vtx
        )
        
def smag_damp_prepare(block: Block, itime: INT[Array, ""] | int, dt: float) -> Block:
    mesh              = block.mesh
    kds               = mesh.full_kds
    cell_ed           = mesh.cell_ed
    vtx_ed            = mesh.vtx_ed
    cell_of_edge      = mesh.cell_of_edge
    vtx_of_edge       = mesh.vtx_of_edge
    cell_xyz          = mesh.cell_xyz
    vtx_xyz           = mesh.vtx_xyz
    edge_nr           = mesh.edge_nr
    edge_tg           = mesh.edge_tg
    le                = mesh.le
    de                = mesh.de
    un                = block.state.un           .unwrap()[itime, :, :]
    pt                = block.state.pt           .unwrap()[itime, :, :]
    dmg               = block.state.dmg          .unwrap()[itime, :, :]
    smag_coef         = block.aux  .smag_coef    .unwrap() # out
    smag_coef_lev     = block.aux  .smag_coef_lev.unwrap() # out
    lap_un            = block.aux  .fn_2d        .unwrap()
    grad_pt           = block.aux  .fn_2d        .unwrap()
    dundt_damp        = block.aux  .dundt_damp   .unwrap() # out
    dptdt_damp        = block.aux  .dptdt_damp   .unwrap() # out
    perot_wgt_at_cell = block.equaux.damp.smag.perot_wgt_at_cell
    perot_wgt_at_vtx  = block.equaux.damp.smag.perot_wgt_at_vtx
    vec_at_cell       = block.equaux.damp.smag.vec_at_cell
    vec_at_vtx        = block.equaux.damp.smag.vec_at_vtx
    # =====================================================
    # Calculate velocity vector at cell and vtx
    # (nlev, ncell, 3)
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ihe = mesh.cell_ids, mesh.cell_ihe[mesh.hori_hw-2]
    vec_at_cell = vec_at_cell.at[kds:kde+1, ids:ihe+1, :].set(
        jnp.sum(
            un[kds:kde+1, cell_ed[ids:ihe+1, :], None] * perot_wgt_at_cell[None, ids:ihe+1, :, :], 
            axis=-2
        )
    )
    # (nlev, nvtx, 3)
    kds, kde = mesh.full_kds, mesh.full_kde 
    ids, ihe = mesh.vtx_ids , mesh.vtx_ihe[mesh.hori_hw-1]
    vec_at_vtx = vec_at_vtx.at[kds:kde+1, ids:ihe+1, :].set(
        jnp.sum(
            un[kds:kde+1, vtx_ed [ids:ihe+1, :], None] * perot_wgt_at_vtx [None, ids:ihe+1, :, :], 
            axis=-2
        )
    )
    # =====================================================
    #      pc2
    #       |
    # dc1---e---dc2
    #       |
    #      pc1
    ids, ihe = mesh.edge_ids, mesh.edge_ihe[mesh.hori_hw-1]
    pc1 = cell_of_edge[ids:ihe+1, 0]  # (nedge,)
    pc2 = cell_of_edge[ids:ihe+1, 1]  # (nedge,)
    dc1 = vtx_of_edge [ids:ihe+1, 0]  # (nedge,)
    dc2 = vtx_of_edge [ids:ihe+1, 1]  # (nedge,)
    # (nedge, 3)
    pc1pc2 = cell_xyz[pc2, :] - cell_xyz[pc1, :]
    dc1dc2 = vtx_xyz [dc2, :] - vtx_xyz [dc1, :]
    # (nedge,)
    flag1 = jnp.sign(jnp.sum(pc1pc2 * edge_nr[ids:ihe+1, :], axis=-1))
    flag2 = jnp.sign(jnp.sum(dc1dc2 * edge_tg[ids:ihe+1, :], axis=-1))
    # =====================================================
    # (nlev, nedge)
    un_pc1 = jnp.sum(vec_at_cell[kds:kde+1, pc1, :] * edge_nr[ids:ihe+1, :], axis=-1)
    un_pc2 = jnp.sum(vec_at_cell[kds:kde+1, pc2, :] * edge_nr[ids:ihe+1, :], axis=-1)
    un_dc1 = jnp.sum(vec_at_vtx [kds:kde+1, dc1, :] * edge_nr[ids:ihe+1, :], axis=-1)
    un_dc2 = jnp.sum(vec_at_vtx [kds:kde+1, dc2, :] * edge_nr[ids:ihe+1, :], axis=-1)

    # (nlev, nedge)
    ut_pc1 = jnp.sum(vec_at_cell[kds:kde+1, pc1, :] * edge_tg[ids:ihe+1, :], axis=-1)
    ut_pc2 = jnp.sum(vec_at_cell[kds:kde+1, pc2, :] * edge_tg[ids:ihe+1, :], axis=-1)
    ut_dc1 = jnp.sum(vec_at_vtx [kds:kde+1, dc1, :] * edge_tg[ids:ihe+1, :], axis=-1)
    ut_dc2 = jnp.sum(vec_at_vtx [kds:kde+1, dc2, :] * edge_tg[ids:ihe+1, :], axis=-1)

    # (nlev, nedge)
    un_pc1pc2 = flag1 * (un_pc2 - un_pc1) / de
    ut_pc1pc2 = flag1 * (ut_pc2 - ut_pc1) / de
    un_dc1dc2 = flag2 * (un_dc2 - un_dc1) / le
    ut_dc1dc2 = flag2 * (ut_dc2 - ut_dc1) / le

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

    ls2 = de * le
    smag_coef = smag_coef.at[kds:kde+1, ids:ihe+1].set(
        block.cfg.smag_damp_coef * ls2 * jnp.sqrt(smag_t**2 + smag_s**2)
    )
    tmp = (un_pc1 + un_pc2 - 2 * un) / de**2
    lap_un = lap_un.at[kds:kde+1, ids:ihe+1].set(
        tmp + (un_dc1 + un_dc2 - 2 * un) / le**2
    )    
    # -----------------------------------------------------
    # Calculate smag lev for non-hydrostatic model
    if block.mode.is_nonhydrostatic():
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ihe = mesh.edge_ids, mesh.edge_ihe[0]
        tmp = 0.5 * (
            mesh.full_dlev[kds:kde    ] / mesh.half_dlev[kds+1:kde+1] * smag_coef[kds+1:kde+1, ids:ihe+1] + 
            mesh.full_dlev[kds+1:kde+1] / mesh.half_dlev[kds+1:kde+1] * smag_coef[kds:kde    , ids:ihe+1]
        )
        smag_coef_lev = smag_coef_lev.at[kds+1:kde+1, ids:ihe+1].set(tmp)
    # =====================================================
    # Normal wind
    # math: ((upc1-un)/(0.5*edd_leng)-(un-upc2)/(0.5*edd_leng))/(0.5*edd_leng)
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.edge_ids, mesh.edge_ide
    dundt_damp = dundt_damp.at[kds:kde+1, ids:ide+1].set(
        4 * smag_coef[kds:kde+1, ids:ide+1] * lap_un[kds:kde+1, ids:ide+1]
    )
    # -----------------------------------------------------
    # Potential temperature
    kds, kde = mesh.full_kds, mesh.full_kde
    ids, ide = mesh.cell_ids, mesh.cell_ide    
    grad_pt = mesh_ops.grad_operator_2d(
        mesh=mesh,
        f=pt,
        grad=grad_pt,
        f_loc=block.state.pt.meta.loc,
        with_halo=True,
    ) * smag_coef
    dptdt_damp = mesh_ops.div_operator_2d(
        mesh=mesh,
        f=grad_pt,
        div=dptdt_damp,
        div_loc=block.aux.dptdt_damp.meta.loc,
    )
    dptdt_damp = dptdt_damp.at[kds:kde+1, ids:ide+1].set(
        dptdt_damp[kds:kde+1, ids:ide+1] * dmg[kds:kde+1, ids:ide+1]
    )
    # =====================================================
    return tree_at(
        where=lambda b: (
            b.aux.smag_coef.d,
            b.aux.smag_coef_lev.d,
            b.aux.dundt_damp.d,
            b.aux.dptdt_damp.d
        ),
        pytree=block,
        replace=(smag_coef, smag_coef_lev, dundt_damp, dptdt_damp),
    )


def project_sphere2plane(
    R: FloatType,
    op0: FLT[Array, "npoint 3"],
    op1: FLT[Array, "npoint nnb 3"],
) -> FLT[Array, "npoint nnb 3"]:
    return jax.vmap(
        jax.vmap(
            _project_sphere2plane,
            in_axes=(None, 0, 0),
            out_axes=0
        ),
        in_axes=(None, None, 1),
        out_axes=1
    )(
        R,
        op0,
        op1
    )
