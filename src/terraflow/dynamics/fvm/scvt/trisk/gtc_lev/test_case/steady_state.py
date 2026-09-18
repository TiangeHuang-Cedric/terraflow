from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import formula
from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo
)
from terraflow.test_case import steady_state as ss
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_mg


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    FLT   = typing.Float
    Array = typing.Array

class SCVTSteadyStateTestCase(ss.SteadyStateTestCase):

    def set_ic(self, block: Block) -> Block:
        block = self.init_mgs(block)
        block = calc_mg      (block, 0)
        block = self.init_un (block)
        block = self.init_pt (block)
        block = self.init_gz (block)
        block = self.init_gzs(block)
        return block

    def init_mgs(self, block: Block) -> Block:
        bkd = block.backend
        mgs = block.state.mgs.unwrap()[0, :]
        mgs = bkd.full_like(mgs, 1.0e5)
        # ==============================================
        return bkd.tree_at(
            where=lambda b: b.state.mgs.d,
            pytree=block,
            replace=mgs,
            at_dims=0,
            at_idxes=0
        )

    def init_un(self, block: Block) -> Block:
        bkd    = block.backend
        radius = block.para.radius.unwrap()
        mesh   = block.mesh
        un     = block.state.un.unwrap()[0, :, :]
        eta0   = self.eta0
        u0     = self.u0
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        eta  = mesh.full_lev[kds:kde+1]
        etav = (eta - eta0) * bkd.pi / 2
        lat = mesh.dual_edge_lat[ids:ide+1]

        def _wrap(etav: FLT[Array, ""]) -> FLT[Array, "nedge"]:
            return u0 * bkd.cos(etav) ** 1.5 * bkd.sin(2 * lat) ** 2

        u = bkd.vmap(_wrap, in_axes=0, out_axes=0)(etav)
        v = bkd.zeros_like(u)
        vec = bkd.vmap(
            vector_convert_lonlat2xyz,
            in_axes=(None, 0, 0, None),
            out_axes=0
        )(
            radius,
            u,
            v,
            mesh.dual_edge_xyz[ids:ide+1, :]
        )
        tmp = bkd.sum(vec * mesh.edge_nr[None, ids:ide+1, :], axis=-1)
        un = bkd.slice_update(un, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
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

    def init_pt(self, block: Block) -> Block:
        bkd    = block.backend
        rd     = block.para .rd     .unwrap()
        g      = block.para .gravity.unwrap()
        radius = block.para .radius .unwrap()
        omega  = block.para .omega  .unwrap()
        mesh   = block.mesh
        mg     = block.state.mg     .unwrap()[0, :, :]
        pt     = block.state.pt     .unwrap()[0, :, :]
        t      = block.aux  .t      .unwrap()
        etat   = self.etat
        eta0   = self.eta0
        t0     = self.t0
        gamma  = self.gamma
        dt     = self.dt
        u0     = self.u0
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        eta     = mesh.full_lev[kds:kde+1]

        def _wrap(eta: FLT[Array, ""]) -> FLT[Array, "nedge"]:
            mask = (etat <= eta) & (eta <= 1.0)
            etav = (eta - eta0) * bkd.pi / 2
            t_bar = bkd.where(
                mask,
                t0 * eta ** (rd * gamma / g),
                (t0 * eta ** (rd * gamma / g)
                 + dt * (etat - eta) ** 5)
            )
            return t_bar + 3.0 / 4.0 * eta * bkd.pi * u0 / rd * bkd.sin(etav) * bkd.sqrt(bkd.cos(etav)) * (
                (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
                * 2 * u0 * bkd.cos(etav) ** 1.5
                + (8 / 5 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - bkd.pi / 4)
                * radius * omega
            )

        t = bkd.slice_update(t, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.vmap(_wrap, in_axes=0, out_axes=0)(eta)
        )
        pt = bkd.slice_update(pt, (slice(kds, kde+1), slice(ids, ide+1)),
            formula.modified_potential_temperature(
                t=t[kds:kde+1, ids:ide+1],
                p=mg[kds:kde+1, ids:ide+1],
                qv=0.0,
                para=block.para
            )
        )
        pt = fill_field2d_array_halo(
            halo=mesh.halo,
            f=pt,
            meta=block.state.pt.meta
        )
        # ==============================================
        return bkd.tree_at(
            where=lambda b: (b.aux.t.d, b.state.pt.d),
            pytree=block,
            replace=(t, pt),
            at_dims=(None, 0),
            at_idxes=(None, 0)
        )

    def init_gz(self, block: Block) -> Block:
        bkd    = block.backend
        rd     = block.para.rd     .unwrap()
        radius = block.para.radius .unwrap()
        omega  = block.para.omega  .unwrap()
        g      = block.para.gravity.unwrap()
        mesh   = block.mesh
        gz     = block.state.gz    .unwrap()[0, :, :]
        gz_lev = block.state.gz_lev.unwrap()[0, :, :]
        eta0   = self.eta0
        etat   = self.etat
        t0     = self.t0
        gamma  = self.gamma
        dt     = self.dt
        u0     = self.u0
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        cos_lat = mesh.cell_cos_lat[ids:ide+1]

        def _wrap(eta: FLT[Array, ""]) -> FLT[Array, "ncell"]:
            etav = (eta - eta0) * bkd.pi / 2
            mask = (etat <= eta) & (eta <= 1.0)
            gz_bar = bkd.where(
                mask,
                t0 * g / gamma * (1 - eta ** (rd * gamma / g)),
                t0 * g / gamma * (1 - eta ** (rd * gamma / g)) - rd * dt * (
                    (bkd.log(eta / etat) + 137 / 60) * etat ** 5 - 5 * etat ** 4 * eta +
                    5 * etat ** 3 * eta ** 2 - 10 / 3 * etat ** 2 * eta ** 3 +
                    5 / 4 * etat * eta ** 4 - 1 / 5 * eta ** 5
                )
            )
            return gz_bar + u0 * bkd.cos(etav) ** 1.5 * (
                (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
                * u0 * bkd.cos(etav) ** 1.5
                + (8 / 15 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - bkd.pi / 4)
                * radius * omega
            )
        kds, kde = mesh.half_kds, mesh.half_kde
        eta = mesh.half_lev[kds:kde+1]
        gz_lev = bkd.slice_update(gz_lev, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.vmap(_wrap, in_axes=0, out_axes=0)(eta)
        )
        gz_lev = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz_lev,
            meta=block.state.gz_lev.meta
        )

        kds, kde = mesh.full_kds, mesh.full_kde
        eta = mesh.full_lev[kds:kde+1]
        gz = bkd.slice_update(gz, (slice(kds, kde+1), slice(ids, ide+1)),
            bkd.vmap(_wrap, in_axes=0, out_axes=0)(eta)
        )
        gz = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        # ==============================================
        return bkd.tree_at(
            where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
            pytree=block,
            replace=(gz, gz_lev),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )


    def init_gzs(self, block: Block) -> Block:
        bkd    = block.backend
        radius = block.para  .radius.unwrap()
        omega  = block.para  .omega .unwrap()
        mesh   = block.mesh
        gzs    = block.static.gzs   .unwrap()
        eta0   = self.eta0
        u0     = self.u0
        # ==============================================
        etav = (1 - eta0) * bkd.pi / 2
        ids, ide = mesh.cell_ids, mesh.cell_ide
        sin_lat = mesh.cell_sin_lat[ids:ide+1]
        cos_lat = mesh.cell_cos_lat[ids:ide+1]
        tmp = u0 * bkd.cos(etav) ** 1.5 * (
            (-2 * sin_lat ** 6 * (cos_lat ** 2 + 1 / 3) + 10 / 63)
            * u0 * bkd.cos(etav) ** 1.5
            + (8 / 5 * cos_lat ** 3 * (sin_lat ** 2 + 2 / 3) - bkd.pi / 4)
            * radius * omega
        )
        gzs = bkd.slice_update(gzs, slice(ids, ide+1), tmp)
        gzs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs,
            meta=block.static.gzs.meta
        )
        # ==============================================
        return bkd.tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=bkd.zeros_like(gzs)
        )