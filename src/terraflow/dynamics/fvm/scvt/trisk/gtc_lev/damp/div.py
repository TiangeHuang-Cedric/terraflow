from __future__ import annotations
from typing import Self, TYPE_CHECKING
import numpy as np


from terraflow.util.device import AbstractDevice
from terraflow.util.math import exp_two_values
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryData
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_div


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    INT = typing.Int
    FLT = typing.Float
    FloatType = typing.FloatType
    Array = typing.Array

    
class DivDampdata(AbstractEquationAuxiliaryData):
    cn: FLT[Array, "nlev nedge"]
    
    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            cn=np.array([])
        )

    @classmethod
    def initialize(
        cls,
        cfg: UserConfig,
        mesh: SCVTMesh, 
        dev: AbstractDevice
    ) -> Self:
        if cfg.use_div_damp:
            return cls.from_coord(
                cfg=cfg,
                mesh=mesh,
                dev=dev
            )
        else:
            return cls.empty()

    @classmethod
    def from_coord(
        cls,
        cfg: UserConfig,
        mesh: SCVTMesh, 
        dev: AbstractDevice
    ) -> Self:
        kms, kme = mesh.full_kms, mesh.full_kme
        ims, ime = mesh.edge_ims, mesh.edge_ime
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        bkd = dev.backend

        cn = bkd.zeros(
            (kme-kms+1, ime-ims+1), 
            device=dev(), 
            dtype=dev.float_type
        )
        
        if cfg.div_damp_order == 2:
            one            = bkd.asarray(1.0               , dtype=dev.float_type)
            div_damp_top   = bkd.asarray(cfg.div_damp_top  , dtype=dev.float_type)
            div_damp_k0    = bkd.asarray(cfg.div_damp_k0   , dtype=dev.float_type)
            div_damp_coef2 = bkd.asarray(cfg.div_damp_coef2, dtype=dev.float_type)
            k              = bkd.arange(kds, kde+1, 1)
            c1 = exp_two_values(dev.backend, div_damp_top, one, one, div_damp_k0, k)
            tmp = c1[:, None] * div_damp_coef2 * mesh.area_edge[None, ids:ide+1] / cfg.dt_dyn_seconds
            cn = bkd.slice_update(cn, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        elif cfg.div_damp_order == 4:
            div_damp_coef4 = bkd.asarray(cfg.div_damp_coef4, dtype=dev.float_type)
            tmp = div_damp_coef4 * mesh.area_edge[ids:ide+1]**2 / cfg.dt_dyn_seconds
            tmp = bkd.broadcast_to(tmp, (kde-kds+1, ide-ids+1))
            cn = bkd.slice_update(cn, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        else:
            raise ValueError(f"Unsupported div_damp_order {cfg.div_damp_order}!")
        
        return cls(
            initialized=True,
            cn=cn
        )
            

def div_damp_run(block: Block, itime: int, dt: float) -> Block:
    
    block = calc_div(block, itime)
    
    mesh = block.mesh
    bkd  = block.backend
    cn   = block.equaux.damp.div.cn
    un   = block.state.un  .unwrap()[itime, :, :]
    div  = block.aux  .div .unwrap()
    div2 = block.aux  .div2.unwrap()
    if block.cfg.div_damp_order == 2:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        tmp = (
            un[kds:kde+1, ids:ide+1] + dt * cn[kds:kde+1, ids:ide+1] * 
            (div[kds:kde+1, v2] - div[kds:kde+1, v1]) 
            * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
        )
        un = bkd.slice_update(un, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
    elif block.cfg.div_damp_order == 4:
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        v1 = mesh.cell_of_edge[ids:ide+1, 0]
        v2 = mesh.cell_of_edge[ids:ide+1, 1]
        tmp = (
            un[kds:kde+1, ids:ide+1] - dt * cn[kds:kde+1, ids:ide+1] * 
            (div2[kds:kde+1, v2] - div2[kds:kde+1, v1]) 
            * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
        )
        un = bkd.slice_update(un, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
    
    return bkd.tree_at(
        where=lambda b: b.state.un.d,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=itime
    )
