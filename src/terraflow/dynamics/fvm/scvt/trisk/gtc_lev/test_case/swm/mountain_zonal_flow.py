from __future__ import annotations
from typing import TYPE_CHECKING
from jax import numpy as jnp

from terraflow.util.pytree import tree_at
from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo
)
from terraflow.test_case.swm import mountain_zonal_flow as swm_mz

if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class MountainZonalFlowTestCase(swm_mz.MountainZonalFlowTestCase):
    
    def set_ic(self, block: Block) -> Block:
        block = self.init_gzs(block)
        block = self.init_un (block)
        block = self.init_gz (block)
        return block
        
    def init_un(self, block: Block) -> Block:
        mesh      = block.mesh
        radius    = block.para .radius.unwrap()
        un        = block.state.un    .unwrap()[0, :, :]
        cos_alpha = jnp.cos(self.alpha)
        sin_alpha = jnp.sin(self.alpha)
        u0        = self.u0
        # ==============================================        
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cos_lat = mesh.dual_edge_cos_lat[ids:ide+1]
        sin_lat = mesh.dual_edge_sin_lat[ids:ide+1]
        cos_lon = mesh.dual_edge_cos_lon[ids:ide+1]
        sin_lon = mesh.dual_edge_sin_lon[ids:ide+1]
        u = u0 * (cos_lat * cos_alpha + cos_lon * sin_lat * sin_alpha)
        v = -u0 * sin_lon * sin_alpha
        vec = vector_convert_lonlat2xyz(
            R=radius,
            u=u,
            v=v,
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        tmp = jnp.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        un = un.at[0, ids:ide+1].set(tmp)
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
            at_idxes=0
        )
        
    def init_gzs(self, block: Block) -> Block:
        mesh      = block.mesh
        gzs       = block.static.gzs.unwrap()
        R         = self.R
        lon0      = self.lon0
        lat0      = self.lat0
        gzs0      = self.gzs0
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        dlon = jnp.abs(mesh.cell_lon[ids:ide+1] - lon0)
        dlon = jnp.minimum(dlon, 2 * jnp.pi - dlon)
        d = jnp.minimum(
            R, 
            jnp.sqrt(dlon**2 + (mesh.cell_lat[ids:ide+1] - lat0)**2)
        )
        gzs = gzs.at[ids:ide+1].set(gzs0 * (1.0 - d / R))
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
    
    def init_gz(self, block: Block) -> Block:
        mesh      = block.mesh
        radius    = block.para .radius.unwrap()
        omega     = block.para .omega .unwrap()
        gz        = block.state.gz    .unwrap()[0, :, :]
        cos_alpha = jnp.cos(self.alpha)
        sin_alpha = jnp.sin(self.alpha)
        u0        = self.u0
        gz0       = self.gz0
        # ==============================================   
        ids, ide = mesh.cell_ids, mesh.cell_ide
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        cos_lon = mesh.cell_cos_lon[ids:ide+1]
        tmp = (
            gz0 - 
            (radius * omega * u0 + u0**2 * 0.5) 
            * (sin_lat * cos_alpha - cos_lon * cos_lat * sin_alpha)**2
        )
        gz = gz.at[0, ids:ide+1].set(tmp)
        gz = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        # ============================================== 
        return tree_at(
            where=lambda b: b.state.gz.d,
            pytree=block,
            replace=gz,
            at_dims=0,
            at_idxes=0
        )