from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.sphere_geometry import (
    great_circle,
    vector_convert_lonlat2xyz
)
from terraflow.mesh.scvt.parallel import (
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.test_case.adv import solid_rotation as adv_sr
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import TracerMeta


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class SolidRotationTestCase(adv_sr.SolidRotationTestCase):

    def init_tracers_meta(self, dt_adv: float) -> dict[str, TracerMeta]:
        """Initialize tracer metadata.

        Note: When calling this method, use backend compilation.
        """
        return {
            "q1": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Cosine bell tracer",
            ),
            "q2": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q2",
                long_name="Slotted cylinder tracer",
            ),
            "q3": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q3",
                long_name="Gaussian hill tracer",
            ),
            "q4": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q4",
                long_name="Cosine one tracer",
            ),
        }
        
    def set_ic(self, block: Block) -> Block:
        mesh   = block.mesh
        bkd    = block.backend
        radius = block.para   .radius.unwrap()
        m      = block.state  .dmg   .unwrap()[0, :, :   ]
        q      = block.tracers.q     .unwrap()[0, :, :, :]
        lon0   = self.lon0
        lat0   = self.lat0
        h0     = self.h0
        # ==============================================
        r0 = radius / 3.0
        
        # Background
        m = bkd.ones_like(m)
        
        ids, ide = mesh.cell_ids, mesh.cell_ide
        # Cosine bell
        lat = mesh.cell_lat[ids:ide+1]
        lon = mesh.cell_lon[ids:ide+1]
        r   = great_circle(bkd, radius, lon0, lat0, lon, lat)
        tmp = bkd.where(
            r < r0,
            h0 / 2.0 * (1 + bkd.cos(bkd.pi * r / r0)),
            0.0
        )
        q = q.at[0,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=0,
            meta=block.tracers.q.meta
        )
        
        # Slotted cylinders
        qmax = 1.0; qmin = 0.1; r = radius * 0.5
        lat = mesh.cell_lat[ids:ide+1]
        lon = mesh.cell_lon[ids:ide+1]
        r0  = great_circle(bkd, radius, lon0, lat0, lon, lat)
        cond1 = bkd.logical_and(r0 <= r, bkd.abs(lon - lon0) >= r / radius / 6)
        cond2 = bkd.logical_and(
            bkd.logical_and(r0 <= r, bkd.abs(lon - lon0) < r / radius / 6),
            (lat - lat0) < -5/12 * r / radius
        )
        tmp = bkd.where(
            cond1,
            qmax,
            bkd.where(
                cond2,
                qmax,
                qmin
            )
        )
        q = q.at[1,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=1,
            meta=block.tracers.q.meta
        )
        
        # Gaussian hill
        lat = mesh.cell_lat[ids:ide+1]
        lon = mesh.cell_lon[ids:ide+1]
        r = great_circle(bkd, radius, lon0, lat0, lon, lat)
        tmp = bkd.exp(-5 * r**2 / radius**2)
        q = q.at[2,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=2,
            meta=block.tracers.q.meta
        )
        
        # Constant one
        q = q.at[3,0,:].set(1)
        # ==============================================
        return bkd.tree_at(
            where=lambda b: (b.state.dmg.d, b.tracers.q.d),
            pytree=block,
            replace=(m, q),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        radius = block.para.radius.unwrap()
        mesh   = block.mesh
        bkd    = block.backend
        un     = block.state.un.unwrap()[itime, :, :]
        alpha  = self.alpha
        u0     = self.u0
        # ==============================================
        ids, ide = mesh.edge_ids, mesh.edge_ide
        lat = mesh.dual_edge_lat[ids:ide+1]
        lon = mesh.dual_edge_lon[ids:ide+1]
        u =  u0 * (bkd.cos(lat) * bkd.cos(alpha) + bkd.sin(lat) * bkd.cos(lon) * bkd.sin(alpha))
        v = -u0 * bkd.sin(lon) * bkd.sin(alpha)
        vec = vector_convert_lonlat2xyz(
            bkd=bkd,
            R=radius,
            u=u,
            v=v,
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        tmp = bkd.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        un = un.at[0, ids:ide+1].set(tmp)
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
            at_idxes=itime
        )
