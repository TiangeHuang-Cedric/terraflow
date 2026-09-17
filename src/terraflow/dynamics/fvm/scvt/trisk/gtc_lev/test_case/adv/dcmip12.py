from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field2d_array_halo,
    fill_field3d_array_halo
)
from terraflow.test_case.adv import dcmip12 as adv_dcmip12
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import (
    calc_mg,
    calc_dmg,
    calc_gz,
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import TracerMeta


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block
    
    FLT   = typing.Float
    Array = typing.Array


class DCMIP12TestCase(adv_dcmip12.DCMIP12TestCase):

    def init_tracers_meta(self, dt_adv: float) -> dict[str, TracerMeta]:
        """Initialize tracer metadata.

        Note: When calling this method, use backend compilation.
        """
        return {
            "q1": TracerMeta(
                batch_name="adv",
                dt=dt_adv,
                name="q1",
                long_name="Test tracer",
            ),
        }
        
    def update_pressure_levels(self, block: Block, itime: int) -> Block:
        bkd = block.backend
        mgs = block.state.mgs.unwrap()[itime, :]
        p0  = self.p0
        T0  = self.T0
        # ==============================================
        mgs = bkd.full_like(mgs, p0)
        block = bkd.tree_at(
            where=lambda b: b.state.mgs.d,
            pytree=block,
            replace=mgs,
            at_dims=0,
            at_idxes=itime
        )
        block = calc_mg (block, itime)
        block = calc_dmg(block, itime)
        mg_lev = block.state.mg_lev.unwrap()[itime, :, :]
        mg     = block.state.mg    .unwrap()[itime, :, :]
        tv     = block.aux  .tv    .unwrap()
        tv     = bkd.full_like(tv, T0)
        block = bkd.tree_at(
            where=lambda b: (b.state.ph_lev.d, b.state.ph.d, b.aux.tv.d),
            pytree=block,
            replace=(mg_lev, mg, tv),
            at_dims=(0, 0, None),
            at_idxes=(itime, itime, None)
        )
        block = calc_gz(block, itime)
        # ==============================================
        return block
    
    def set_ic(self, block: Block) -> Block:
        bkd = block.backend
        block = self.update_pressure_levels(block, 0)
        # ==============================================
        g    = self.para.gravity.unwrap()
        mesh = block.mesh
        gz   = block.state  .gz.unwrap()[0, :, :   ]
        q    = block.tracers.q .unwrap()[0, :, :, :]
        z1 = self.z1
        z2 = self.z2
        z0 = self.z0        
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        z = gz[kds:kde+1, ids:ide+1] / g
        tmp = bkd.where(
            bkd.logical_and(z1 < z, z < z2),
            0.5 * (1 + bkd.cos(2 * bkd.pi * (z - z0) / (z2 - z1))),
            0.0
        )
        q = q.at[0, kds:kde+1, ids:ide+1].set(tmp)
        q = fill_field3d_array_halo(
            halo=mesh.halo,
            f=q,
            i0=0,
            meta=block.tracers.q.meta
        )
        # ==============================================
        return bkd.tree_at(
            where=lambda b: b.tracers.q.d,
            pytree=block,
            replace=q,
            at_dims=0,
            at_idxes=0
        )
        
    def set_uv(self, time_in_seconds: float, block: Block, itime: int) -> Block:
        bkd   = block.backend
        cos_t = bkd.cos(bkd.pi * time_in_seconds / self.tau)
        
        block = self.update_pressure_levels(block, itime)
        # ==============================================
        g      = block.para .gravity.unwrap()
        radius = block.para .radius .unwrap()
        rd     = block.para .rd     .unwrap()
        mesh   = block.mesh
        bkd    = block.backend
        dmg    = block.state.dmg    .unwrap()[itime, :, :]
        mg     = block.state.mg     .unwrap()[itime, :, :]
        gz_lev = block.state.gz_lev .unwrap()[itime, :, :]
        gz     = block.state.gz     .unwrap()[itime, :, :]
        un     = block.state.un     .unwrap()[itime, :, :]
        we     = block.aux  .we_lev .unwrap()
        mfz    = block.aux  .mfz_lev.unwrap()
        T0     = self.T0
        u0     = self.u0
        w0     = self.w0
        rho0   = self.rho0
        K0     = self.K0
        ztop   = self.ztop
        p0     = self.p0
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        
        def _calc_un(
            mg: FLT[Array, "ncell"], 
            gz: FLT[Array, "ncell"]
        ) -> FLT[Array, "nedge"]:
            v1 = mesh.cell_of_edge[ids:ide+1, 0]
            v2 = mesh.cell_of_edge[ids:ide+1, 1]
            lat = mesh.dual_edge_lat[ids:ide+1]
            rho = 0.5 * (mg[v1] + mg[v2]) / (rd * T0)
            u   = u0 * bkd.cos(lat)
            v   = (
                -radius * w0 * bkd.pi * rho0 / (K0 * ztop * rho) * 
                bkd.cos(lat) * bkd.sin(K0 * lat) * 
                bkd.cos(
                    bkd.pi * 0.5 * (
                        gz[v1] + gz[v2]
                    ) / (g * ztop)
                ) * cos_t
            )
            vec = vector_convert_lonlat2xyz(bkd, radius, u, v, mesh.dual_edge_xyz[ids:ide+1, :])
            return bkd.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        
        un_tmp = bkd.vmap(
            _calc_un,
            in_axes=(0, 0),
            out_axes=0
        )(mg[kds:kde+1, :], gz[kds:kde+1, :])
        
        un = un.at[kds:kde+1, ids:ide+1].set(un_tmp)
        un = fill_field2d_array_halo(
            halo=mesh.halo,
            f=un,
            meta=block.state.un.meta
        )
        
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        
        def _calc_dmg(
            dmg: FLT[Array, "ncell"], 
            gz_lev: FLT[Array, "ncell"],
            half_dlev: FLT[Array, ""] 
        ) -> tuple[FLT[Array, "ncell"], FLT[Array, "ncell"]]: 
            lat = mesh.cell_lat[ids:ide+1]
            we = -g * w0 * rho0 / K0 * (
                -2 * bkd.sin(K0 * lat) * bkd.sin(lat) + K0 * bkd.cos(lat) * bkd.cos(K0 * lat)
            ) * bkd.sin(bkd.pi * gz_lev / (g * ztop)) * cos_t / p0
            mfz = we * dmg / half_dlev
            return we, mfz
        
        we_tmp, mfz_tmp = bkd.vmap(
            _calc_dmg,
            in_axes=(0, 0, 0),
            out_axes=(0, 0)
        )(dmg[kds+1:kde, ids:ide+1], gz_lev[kds+1:kde, ids:ide+1], mesh.half_dlev[kds+1:kde])
        
        we  = we .at[kds+1:kde, ids:ide+1].set(we_tmp )
        mfz = mfz.at[kds+1:kde, ids:ide+1].set(mfz_tmp)
        # ==============================================
        return bkd.tree_at(
            where=lambda b: (b.state.un.d, b.aux.we_lev.d, b.aux.mfz_lev.d),
            pytree=block,
            replace=(un, we, mfz),
            at_dims=(0, None, None),
            at_idxes=(itime, None, None)
        )
