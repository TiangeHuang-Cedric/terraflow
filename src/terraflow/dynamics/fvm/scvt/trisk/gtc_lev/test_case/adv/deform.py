from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.sphere_geometry import (
    great_circle,
    lonlat2xyz,
    vector_convert_lonlat2xyz
)
from terraflow.mesh.scvt.parallel import (
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.test_case.adv import deform as adv_dc
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import TracerMeta


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

class DeformationalFlowTestCase(adv_dc.DeformationalFlowTestCase):

    def init_tracers_meta(self, dt_adv: float) -> dict[str, TracerMeta]:
        """Initialize tracer metadata.

        Note: When calling this method, use jax.jit or equinox.filter_jit for JIT compilation.
        """
        return {
            "q1": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Cosine hills tracer",
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
                long_name="Gaussian hills tracer",
            ),
            "q4": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q4",
                long_name="Uniform one tracer",
            ),
        }
    
    def set_ic(self, block: Block) -> Block:
        mesh   = block.mesh
        bkd    = block.backend
        radius = block.para   .radius.unwrap()
        m      = block.state  .dmg   .unwrap()[0, :, :   ]
        q      = block.tracers.q     .unwrap()[0, :, :, :]        
        lon1   = self.lon1
        lat1   = self.lat1
        lon2   = self.lon2
        lat2   = self.lat2        
        # ==============================================
        x1 = lonlat2xyz(bkd, lon1, lat1, radius) / radius
        x2 = lonlat2xyz(bkd, lon2, lat2, radius) / radius
        
        # Background
        m = bkd.ones_like(m)
        
        ids, ide = mesh.cell_ids, mesh.cell_ide
        # Cosine hills
        qmax = 1.0; qmin = 0.1; c = 0.9; r = radius * 0.5
        lat  = mesh.cell_lat[ids:ide+1]
        lon  = mesh.cell_lon[ids:ide+1]
        r1 = great_circle(bkd, radius, lon1, lat1, lon, lat)
        r2 = great_circle(bkd, radius, lon2, lat2, lon, lat)
        tmp = bkd.where(
            r1 < r,
            qmin + c * qmax * 0.5 * (1 + bkd.cos(bkd.pi * r1 / r)),
            bkd.where(
                r2 < r,
                qmin + c * qmax * 0.5 * (1 + bkd.cos(bkd.pi * r2 / r)),
                qmin
            )
        )
        q = q.at[0,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=0,
            meta=block.tracers.q.meta
        )
        
        # Slotted cylinder
        qmax = 1.0; qmin = 0.1; r = radius * 0.5
        lat  = mesh.cell_lat[ids:ide+1]
        lon  = mesh.cell_lon[ids:ide+1]
        r1 = great_circle(bkd, radius, lon1, lat1, lon, lat)
        r2 = great_circle(bkd, radius, lon2, lat2, lon, lat)
        cond1 = bkd.logical_or(
            bkd.logical_and(r1 <= r, bkd.abs(lon - lon1) >= r / radius / 6),
            bkd.logical_and(r2 <= r, bkd.abs(lon - lon2) >= r / radius / 6)
        )
        cond2 = bkd.logical_and(
            bkd.logical_and(
                r1 <= r, bkd.abs(lon - lon1) < r / radius / 6
            ),
            lat - lat1 < -5.0 / 12.0 * (r / radius)
        )
        cond3 = bkd.logical_and(
            bkd.logical_and(
                r2 <= r, bkd.abs(lon - lon2) < r / radius / 6
            ),
            lat - lat2 > 5.0 / 12.0 * (r / radius)
        )
        tmp = bkd.where(
            cond1,
            qmax,
            bkd.where(
                cond2,
                qmax,
                bkd.where(
                    cond3,
                    qmax,
                    qmin
                )
            )
        )
        q = q.at[1,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=1,
            meta=block.tracers.q.meta
        )
        
        # Gaussian hills
        qmax = 0.95; c = 5.0
        lat  = mesh.cell_lat[ids:ide+1]
        lon  = mesh.cell_lon[ids:ide+1]
        x = lonlat2xyz(bkd, lon, lat, radius) / radius
        tmp = qmax * (
            bkd.exp(-c * bkd.sum((x - x1)**2, axis=-1))
            + bkd.exp(-c * bkd.sum((x - x2)**2, axis=-1))
        )
        q = q.at[2,0,ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=2,
            meta=block.tracers.q.meta
        )
        
        # Uniform one
        q = q.at[3,0,:].set(1)
        # ==============================================
        return bkd.tree_at(
            where=lambda b: (b.state.dmg.d, b.tracers.q.d),
            pytree=block,
            replace=(m, q),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )


class DeformationalFlowCase1(DeformationalFlowTestCase):
    """Deformational flow test case 1 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=1)
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        radius          = block.para.radius.unwrap()
        mesh            = block.mesh
        bkd             = block.backend
        un              = block.state.un.unwrap()[itime, :, :]
        period          = self.period
        # ==============================================
        k = 10.0 * radius / period
        cos_t = bkd.cos(bkd.pi * time_in_seconds / period)
        ids, ide = mesh.edge_ids, mesh.edge_ide
        lat = mesh.dual_edge_lat[ids:ide+1]
        lon = mesh.dual_edge_lon[ids:ide+1]
        u = k * bkd.sin(lon / 2.0)**2 * bkd.sin(2 * lat) * cos_t
        v = k / 2.0 * bkd.sin(lon) * bkd.cos(lat) * cos_t
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


class DeformationalFlowCase2(DeformationalFlowTestCase):
    """Deformational flow test case 2 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=2)
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        radius          = block.para.radius.unwrap()
        mesh            = block.mesh
        bkd             = block.backend
        un              = block.state.un.unwrap()[itime, :, :]
        period          = self.period
        # ==============================================
        k = 10.0 * radius / period
        cos_t = bkd.cos(bkd.pi * time_in_seconds / period)
        ids, ide = mesh.edge_ids, mesh.edge_ide
        lat = mesh.dual_edge_lat[ids:ide+1]
        lon = mesh.dual_edge_lon[ids:ide+1]
        u = k * bkd.sin(lon)**2 * bkd.sin(2 * lat) * cos_t
        v = k * bkd.sin(2 * lon) * bkd.cos(lat) * cos_t  
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


class DeformationalFlowCase3(DeformationalFlowTestCase):
    """Deformational flow test case 3 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=3)
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        radius          = block.para.radius.unwrap()
        mesh            = block.mesh
        bkd             = block.backend
        un              = block.state.un.unwrap()[itime, :, :]
        period          = self.period
        # ==============================================
        k = 5.0 * radius / period
        cos_t = bkd.cos(bkd.pi * time_in_seconds / period)
        ids, ide = mesh.edge_ids, mesh.edge_ide
        lat = mesh.dual_edge_lat[ids:ide+1]
        lon = mesh.dual_edge_lon[ids:ide+1]
        u = -k * bkd.sin(lon / 2.0)**2 * bkd.sin(2 * lat) * bkd.cos(lat)**2 * cos_t
        v = k / 2.0 * bkd.sin(lon) * bkd.cos(lat)**3 * cos_t
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


class DeformationalFlowCase4(DeformationalFlowTestCase):
    """Deformational flow test case 4 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(para=para, case_n=4)
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        radius          = block.para.radius.unwrap()
        mesh            = block.mesh
        bkd             = block.backend
        un              = block.state.un.unwrap()[itime, :, :]
        period          = self.period
        # ==============================================
        k = 10.0 * radius / period
        c1 = bkd.pi * 2 * time_in_seconds / period
        c2 = bkd.pi * 2 * radius / period
        cos_t = bkd.cos(bkd.pi * time_in_seconds / period)
        ids, ide = mesh.edge_ids, mesh.edge_ide
        lat = mesh.dual_edge_lat[ids:ide+1]
        lon = mesh.dual_edge_lon[ids:ide+1] - c1
        u = k * bkd.sin(lon)**2 * bkd.sin(2 * lat) * cos_t + c2 * bkd.cos(lat)
        v = k * bkd.sin(2 * lon) * bkd.cos(lat) * cos_t 
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
