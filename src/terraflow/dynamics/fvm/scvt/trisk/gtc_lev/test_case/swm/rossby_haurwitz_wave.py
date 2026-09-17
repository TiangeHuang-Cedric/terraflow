from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import fill_field2d_array_halo
from terraflow.test_case.swm import rossby_haurwitz_wave as rh


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class RossbyHaurwitzWaveTestCase(rh.RossbyHaurwitzWaveTestCase):
    
    def set_ic(self, block: Block) -> Block:
        block = self.init_gzs(block)
        block = self.init_un (block)
        block = self.init_gz (block)
        return block

    def init_un(self, block: Block) -> Block:
        mesh      = block.mesh
        bkd       = block.backend
        radius    = block.para .radius.unwrap()
        un        = block.state.un    .unwrap()[0, :, :]
        radius    = block.para.radius.unwrap()
        R         = self.R
        omg       = self.omg
        # ==============================================
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cos_lat  = mesh.dual_edge_cos_lat[ids:ide+1]
        sin_lat  = mesh.dual_edge_sin_lat[ids:ide+1]
        lon      = mesh.dual_edge_lon    [ids:ide+1]
        a = cos_lat
        b = R * cos_lat**(R - 1) * sin_lat**2 * bkd.cos(R * lon)
        c = cos_lat**(R + 1) * bkd.cos(R * lon)
        u = radius * omg * (a + b - c)
        a = R * cos_lat**(R - 1) * sin_lat * bkd.sin(R * lon)
        v = - radius * omg * a
        vec = vector_convert_lonlat2xyz(
            bkd=bkd,
            R=radius, 
            u=u, 
            v=v, 
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        un  = un.at[0, ids:ide+1].set(
            bkd.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        )
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
        # ============================================== 
        return bkd.tree_at(
            where=lambda b: b.state.un.d,
            pytree=block,
            replace=un,
            at_dims=0,
            at_idxes=0
        )

    def init_gz(self, block: Block) -> Block:
        bkd       = block.backend
        mesh      = block.mesh
        radius    = block.para .radius.unwrap()
        omega     = block.para .omega .unwrap()
        gz        = block.state.gz    .unwrap()[0, :, :]
        radius    = block.para.radius.unwrap()
        R         = self.R
        omg       = self.omg
        gz0       = self.gz0
        # ==============================================        
        ids, ide = mesh.cell_ids, mesh.cell_ide
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        a = (
            0.5 * omg * (2 * omega + omg) * cos_lat**2 + 
            0.25 * omg**2 * (
                (R + 1) * cos_lat**(2 * R + 2) + 
                (2 * R**2 - R - 2) * cos_lat**(2 * R) - 
                2 * R**2 * cos_lat**(2 * R - 2)
            )
        )
        b = (
            2 * (omega + omg) * omg * cos_lat**R * 
            (R**2 + 2 * R + 2 - (R + 1)**2 * cos_lat**2) / (R + 1) / (R + 2)
        )
        c = 0.25 * omg**2 * cos_lat**(2 * R) * ((R + 1) * cos_lat**2 - R - 2)
        lon = mesh.cell_lon[ids:ide+1]
        gz  = gz.at[0, ids:ide+1].set(
            gz0 + radius**2 * (a + b * bkd.cos(R * lon) + c * bkd.cos(2 * R * lon))
        )
        gz = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        # ============================================== 
        return bkd.tree_at(
            where=lambda b: b.state.gz.d,
            pytree=block,
            replace=gz,
            at_dims=0,
            at_idxes=0
        )
        
    def init_gzs(self, block: Block) -> Block:
        bkd = block.backend
        gzs = block.static.gzs   .unwrap()
        gzs = bkd.zeros_like(gzs)
        # ============================================== 
        return bkd.tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=gzs
        )
