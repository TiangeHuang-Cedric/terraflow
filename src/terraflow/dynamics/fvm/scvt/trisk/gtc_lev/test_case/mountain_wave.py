from __future__ import annotations
from typing import TYPE_CHECKING
from jax import numpy as jnp

from terraflow.util.pytree import tree_at
from terraflow.util import formula
from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo
)
from terraflow.test_case import mountain_wave as mz
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_mg


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class MountainWaveTestCase(mz.MountainWaveTestCase):
    
    def set_ic(self, block: Block) -> Block:
        block = self.init_un (block)
        block = self.init_gzs(block)
        block = self.init_mgs(block)
        block = calc_mg      (block, 0)
        block = self.init_pt (block)
        return block
    
    def init_un(self, block: Block) -> Block:
        radius = block.para .radius .unwrap()
        mesh   = block.mesh
        un     = block.state.un     .unwrap()[0, :, :]
        u0     = self.u0
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cos_lat = mesh.dual_edge_cos_lat[ids:ide+1]
        u = u0 * cos_lat
        v = jnp.zeros_like(u)
        vec = vector_convert_lonlat2xyz(
            R=radius,
            u=u,
            v=v,
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        tmp = jnp.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        tmp = jnp.broadcast_to(tmp, (kde-kds+1, ide-ids+1))
        un = un.at[kds:kde+1, ids:ide+1].set(tmp)
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
        # ============================================== 
        return tree_at(
            where=lambda b: b.state.un.d,
            pytree=block,
            replace=un,
            at_dims=0,
            at_idxes=0,
        )
    
    def init_gzs(self, block: Block) -> Block:
        g      = block.para .gravity.unwrap()
        radius = block.para .radius .unwrap()
        mesh   = block.mesh
        gzs    = block.static.gzs   .unwrap()
        latc   = self.latc
        lonc   = self.lonc
        h0     = self.h0
        d      = self.d
        # ==============================================        
        ids, ide = mesh.cell_ids, mesh.cell_ide
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        cell_lon = mesh.cell_lon[ids:ide+1]
        r = radius * jnp.acos(
            jnp.sin(latc) * sin_lat + jnp.cos(latc) * cos_lat * jnp.cos(cell_lon - lonc)
        )
        gzs = gzs.at[ids:ide+1].set(g * h0 * jnp.exp(-(r / d)**2))
        gzs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs,
            meta=block.static.gzs.meta
        )
        # ==============================================
        return tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=gzs
        )
    
    def init_mgs(self, block: Block) -> Block:
        g      = block.para .gravity.unwrap()
        radius = block.para .radius .unwrap()
        omega  = block.para .omega  .unwrap()
        mesh   = block.mesh
        mgs    = block.state.mgs    .unwrap()[0, :   ]
        gzs    = block.static.gzs   .unwrap()
        u0     = self.u0
        psp    = self.psp
        N      = self.N
        kap    = self.kap
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        tmp = psp * jnp.exp(
            -0.5 * radius * N**2 * u0 / g**2 / kap * (u0 / radius + 2.0 * omega) * 
                 (sin_lat**2 - 1.0) - N**2 / g**2 / kap * gzs[ids:ide+1]
        )
        mgs = mgs.at[ids:ide+1].set(tmp)
        mgs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=mgs,
            meta=block.state.mgs.meta
        )
        # ============================================== 
        return tree_at(
            where=lambda b: b.state.mgs.d,
            pytree=block,
            replace=mgs,
            at_dims=0,
            at_idxes=0
        )
    
    def init_pt(self, block: Block) -> Block:
        mesh = block.mesh
        pt   = block.state.pt.unwrap()[0, :, :]
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        tmp = formula.modified_potential_temperature(
            t=288.0,
            p=block.state.mg.unwrap()[0, kds:kde+1, ids:ide+1],
            qv=0.0,
            para=block.para
        )
        pt = pt.at[kds:kde+1, ids:ide+1].set(tmp)
        pt = fill_field2d_array_halo(
            halo=mesh.halo,
            f=pt,
            meta=block.state.pt.meta
        )
        # ==============================================
        return tree_at(
            where=lambda b: b.state.pt.d,
            pytree=block,
            replace=pt,
            at_dims=0,
            at_idxes=0
        )