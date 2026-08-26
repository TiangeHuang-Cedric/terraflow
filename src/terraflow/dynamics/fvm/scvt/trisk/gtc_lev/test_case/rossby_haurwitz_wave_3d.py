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
from terraflow.test_case import rossby_haurwitz_wave_3d as rh
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_mg


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class RossbyHaurwitzWaveTestCase(rh.RossbyHaurwitzWaveTestCase):
    
    def set_ic(self, block: Block) -> Block:
        block = self.init_gzs(block)
        block = self.init_un (block)
        block = self.init_mgs(block)
        block = calc_mg      (block, 0)
        block = self.init_pt (block)
        block = self.init_gz (block)
        return block

    def init_un(self, block: Block) -> Block:
        radius = block.para.radius.unwrap()
        mesh   = block.mesh
        un     = block.state.un.unwrap()[0, :, :]
        n      = self.n
        M      = self.M
        a      = self.para.radius.unwrap()
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        cos_lat = mesh.dual_edge_cos_lat[ids:ide+1]
        sin_lat = mesh.dual_edge_sin_lat[ids:ide+1]
        lon     = mesh.dual_edge_lon[ids:ide+1]
        u = (
            a * M * cos_lat
            + a * M * cos_lat**(n - 1) * jnp.cos(n * lon) *
            (n * sin_lat**2 - cos_lat**2)
        )
        v= - a * M * n * cos_lat**(n - 1) * sin_lat * jnp.sin(n * lon)
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
            at_idxes=0
        )

    def init_gzs(self, block: Block) -> Block:
        gzs = block.static.gzs.unwrap()
        # ==============================================
        return tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=jnp.zeros_like(gzs)
        )

    def init_mgs(self, block: Block) -> Block:
        g     = block.para .gravity.unwrap()
        omega = block.para .omega  .unwrap()
        a     = block.para .radius .unwrap()
        rd    = block.para .rd     .unwrap()
        mesh  = block.mesh
        mgs   = block.state.mgs    .unwrap()[0, :]
        n     = self.n
        t0    = self.t0
        pref  = self.pref
        gamma = self.gamma
        M     = self.M
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        lon     = mesh.cell_lon[ids:ide+1]
        a_lat = (
            M * (2 * omega + M) / 2 * cos_lat**2
            + M**2 / 4 * cos_lat**(2 * n) * (
                (n + 1) * cos_lat**2 + (2 * n**2 - n - 2)
            )
            - n**2 * M**2 / 2 * cos_lat**(2 * (n - 1))
        )
        b_lat = (
            2 * (omega + M) * M / (n + 1) / (n + 2) * cos_lat**n *
            ((n**2 + 2 * n + 2) - (n + 1)**2 * cos_lat**2)
        )
        c_lat = M**2 / 4 * cos_lat**(2 * n) * ((n + 1) * cos_lat**2 - (n + 2))
        phi_p = a**2 * (
            a_lat + b_lat * jnp.cos(n * lon)
            + c_lat * jnp.cos(2 * n * lon)
        )
        tmp = pref * (1 + gamma / g / t0 * phi_p)**(g / gamma / rd)
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
        mesh  = block.mesh
        g     = block.para .gravity.unwrap()
        rd    = block.para .rd     .unwrap()
        mg    = block.state.mg     .unwrap()[0, :, :]
        pt    = block.state.pt     .unwrap()[0, :, :]
        t0    = self.t0
        gamma = self.gamma
        pref  = self.pref
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        mg_ = mg[ids:ide+1]
        t = t0 * (mg_ / pref)**(gamma * rd / g)
        tmp = formula.dry_potential_temperature(
            t=t,
            p=mg_,
            para=self.para,
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

    def init_gz(self, block: Block) -> Block:
        g      = block.para .gravity.unwrap()
        rd     = block.para .rd     .unwrap()
        mesh   = block.mesh
        mg     = block.state.mg     .unwrap()[0, :, :]
        mg_lev = block.state.mg_lev .unwrap()[0, :, :]
        gz     = block.state.gz     .unwrap()[0, :, :]
        gz_lev = block.state.gz_lev .unwrap()[0, :, :]
        t0     = self.t0
        gamma  = self.gamma
        pref   = self.pref
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        mg_ = mg[kds:kde+1, ids:ide+1]
        tmp = g * t0 / gamma * (1 - (mg_ / pref) ** (gamma * rd / g))
        gz  = gz.at[kds:kde+1, ids:ide+1].set(tmp)
        gz  = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        
        kds, kde = mesh.half_kds, mesh.half_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        mg_lev_  = mg_lev[kds:kde+1, ids:ide+1]
        tmp      = g * t0 / gamma * (1 - (mg_lev_ / pref) ** (gamma * rd / g))
        gz_lev   = gz_lev.at[kds:kde+1, ids:ide+1].set(tmp)
        gz_lev  = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz_lev,
            meta=block.state.gz_lev.meta
        )
        # ==============================================
        return tree_at(
            where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
            pytree=block,
            replace=(gz, gz_lev),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )