from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo
)
from terraflow.test_case import dcmip31
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_mg


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class DCMIP31TestCase(dcmip31.DCMIP31TestCase):

    def set_ic(self, block: Block) -> Block:
        block = self.init_un    (block)
        block = self.init_w_lev (block)
        block = self.init_gzs   (block)
        block = self.init_mgs   (block)
        block = calc_mg         (block, 0)
        block = self.init_gz    (block)
        block = self.init_pt    (block)
        return block

    def init_un(self, block: Block) -> Block:
        radius = block.para.radius.unwrap()
        mesh = block.mesh
        bkd = block.backend
        un = block.state.un.unwrap()[0, :, :]
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        u = self.u0 * mesh.dual_edge_cos_lat[ids:ide+1]
        v = bkd.zeros_like(u)
        vec = vector_convert_lonlat2xyz(
            bkd=bkd,
            R=radius,
            u=u,
            v=v,
            xyz=mesh.dual_edge_xyz[ids:ide+1, :]
        )
        tmp = bkd.sum(vec * mesh.edge_nr[ids:ide+1, :], axis=-1)
        tmp = bkd.broadcast_to(tmp, (kde-kds+1, ide-ids+1))
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

    def init_w_lev(self, block: Block) -> Block:
        if not block.mode.is_nonhydrostatic():
            return block
        bkd = block.backend
        w_lev = bkd.zeros_like(block.state.w_lev.unwrap()[0, :, :])
        return bkd.tree_at(
            where=lambda b: b.state.w_lev.d,
            pytree=block,
            replace=w_lev,
            at_dims=0,
            at_idxes=0
        )

    def init_gzs(self, block: Block) -> Block:
        bkd = block.backend
        mesh = block.mesh
        gzs = bkd.zeros_like(block.static.gzs.unwrap())
        gzs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs,
            meta=block.static.gzs.meta
        )
        return bkd.tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=gzs
        )

    def init_mgs(self, block: Block) -> Block:
        bkd = block.backend
        radius = block.para.radius.unwrap()
        omega = block.para.omega.unwrap()
        rd = block.para.rd.unwrap()
        rd_o_cpd = block.para.rd_o_cpd.unwrap()
        mesh = block.mesh
        mgs = block.state.mgs.unwrap()[0, :]
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        cos_2lat = bkd.cos(2.0 * mesh.cell_lat[ids:ide+1])
        ts = self._surface_temperature(block, cos_2lat)
        tmp = (
            self.peq
            * bkd.exp(
                self.u0 / (4.0 * self.t0 * rd)
                * (self.u0 + 2.0 * omega * radius)
                * (cos_2lat - 1.0)
            )
            * (ts / self.teq) ** (1.0 / rd_o_cpd)
        )
        mgs = bkd.slice_update(mgs, slice(ids, ide+1), tmp)
        mgs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=mgs,
            meta=block.state.mgs.meta
        )
        return bkd.tree_at(
            where=lambda b: b.state.mgs.d,
            pytree=block,
            replace=mgs,
            at_dims=0,
            at_idxes=0
        )

    def init_gz(self, block: Block) -> Block:
        bkd = block.backend
        g = block.para.gravity.unwrap()
        rd_o_cpd = block.para.rd_o_cpd.unwrap()
        mesh = block.mesh
        mgs = block.state.mgs.unwrap()[0, :]
        mg_lev = block.state.mg_lev.unwrap()[0, :, :]
        gz_lev = block.state.gz_lev.unwrap()[0, :, :]
        gz = block.state.gz.unwrap()[0, :, :]
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        kds, kde = mesh.half_kds, mesh.half_kde
        cos_2lat = bkd.cos(2.0 * mesh.cell_lat[ids:ide+1])
        ts = self._surface_temperature(block, cos_2lat)
        denom = (
            ts[None, :] / self.t0
            * (
                (mg_lev[kds:kde+1, ids:ide+1] / mgs[None, ids:ide+1])
                ** rd_o_cpd
                - 1.0
            )
            + 1.0
        )
        tmp = -g ** 2 / self.N2 * bkd.log(denom)
        gz_lev = bkd.slice_update(
            gz_lev,
            (slice(kds, kde+1), slice(ids, ide+1)),
            tmp
        )
        gz_lev = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz_lev,
            meta=block.state.gz_lev.meta
        )

        kds, kde = mesh.full_kds, mesh.full_kde
        tmp = 0.5 * (
            gz_lev[kds:kde+1, ids:ide+1]
            + gz_lev[kds+1:kde+2, ids:ide+1]
        )
        gz = bkd.slice_update(gz, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        gz = fill_field2d_array_halo(
            halo=mesh.halo,
            f=gz,
            meta=block.state.gz.meta
        )
        return bkd.tree_at(
            where=lambda b: (b.state.gz.d, b.state.gz_lev.d),
            pytree=block,
            replace=(gz, gz_lev),
            at_dims=(0, 0),
            at_idxes=(0, 0)
        )

    def init_pt(self, block: Block) -> Block:
        bkd = block.backend
        radius = block.para.radius.unwrap()
        p0 = block.para.p0.unwrap()
        g = block.para.gravity.unwrap()
        rd_o_cpd = block.para.rd_o_cpd.unwrap()
        mesh = block.mesh
        mgs = block.state.mgs.unwrap()[0, :]
        mg = block.state.mg.unwrap()[0, :, :]
        gz = block.state.gz.unwrap()[0, :, :]
        pt = block.state.pt.unwrap()[0, :, :]
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        kds, kde = mesh.full_kds, mesh.full_kde
        cos_2lat = bkd.cos(2.0 * mesh.cell_lat[ids:ide+1])
        ts = self._surface_temperature(block, cos_2lat)
        denom = (
            ts[None, :] / self.t0
            * (
                (mg[kds:kde+1, ids:ide+1] / mgs[None, ids:ide+1])
                ** rd_o_cpd
                - 1.0
            )
            + 1.0
        )
        arg = (
            bkd.sin(self.latc) * mesh.cell_sin_lat[ids:ide+1]
            + bkd.cos(self.latc)
            * mesh.cell_cos_lat[ids:ide+1]
            * bkd.cos(mesh.cell_lon[ids:ide+1] - self.lonc)
        )
        arg = bkd.clip(arg, -1.0, 1.0)
        r = radius * bkd.acos(arg)
        tmp = (
            ts[None, :]
            * (p0 / mgs[None, ids:ide+1]) ** rd_o_cpd
            / denom
            + self.dpt
            * self.d ** 2
            / (self.d ** 2 + r[None, :] ** 2)
            * bkd.sin(2.0 * bkd.pi * gz[kds:kde+1, ids:ide+1] / g / self.lz)
        )
        pt = bkd.slice_update(pt, (slice(kds, kde+1), slice(ids, ide+1)), tmp)
        pt = fill_field2d_array_halo(
            halo=mesh.halo,
            f=pt,
            meta=block.state.pt.meta
        )
        return bkd.tree_at(
            where=lambda b: b.state.pt.d,
            pytree=block,
            replace=pt,
            at_dims=0,
            at_idxes=0
        )

    def _surface_temperature(self, block: Block, cos_2lat):
        bkd = block.backend
        radius = block.para.radius.unwrap()
        omega = block.para.omega.unwrap()
        g = block.para.gravity.unwrap()
        return self.t0 + (self.teq - self.t0) * bkd.exp(
            -self.u0 * self.N2 / (4.0 * g ** 2)
            * (self.u0 + 2.0 * omega * radius)
            * (cos_2lat - 1.0)
        )
