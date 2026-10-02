from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import formula
from terraflow.util.sphere_geometry import vector_convert_lonlat2xyz
from terraflow.mesh.scvt.parallel import (
    fill_field1d_array_halo,
    fill_field2d_array_halo
)
from terraflow.test_case import ksp15
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.space_operators import calc_mg


if TYPE_CHECKING:
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


class KSP15TestCase(ksp15.KSP15TestCase):

    def set_ic(self, block: Block) -> Block:
        block = self.init_un   (block)
        block = self.init_w_lev(block)
        block = self.init_gzs  (block)
        block = self.init_mgs  (block)
        block = calc_mg        (block, 0)
        block = self.init_pt   (block)
        block = self.init_gz   (block)
        return block

    def init_un(self, block: Block) -> Block:
        radius = block.para.radius.unwrap()
        mesh = block.mesh
        bkd = block.backend
        un = block.state.un.unwrap()[0, :, :]
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.edge_ids, mesh.edge_ide
        u = self.ueq * mesh.dual_edge_cos_lat[ids:ide+1]
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
        gzs = block.static.gzs.unwrap()
        gzs_calc = self.calc_gzs(block)
        ids, ide = mesh.cell_ids, mesh.cell_ide
        gzs = bkd.slice_update(gzs, slice(ids, ide+1), gzs_calc)
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
        rd = block.para.rd.unwrap()
        mesh = block.mesh
        gzs = block.static.gzs.unwrap()
        mgs = block.state.mgs.unwrap()[0, :]
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        tmp = self.peq * bkd.exp(
            -0.5 * self.ueq ** 2 / rd / self.teq
            * mesh.cell_sin_lat[ids:ide+1] ** 2
            - gzs[ids:ide+1] / rd / self.teq
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

    def init_pt(self, block: Block) -> Block:
        bkd = block.backend
        mesh = block.mesh
        mg = block.state.mg.unwrap()[0, :, :]
        pt = block.state.pt.unwrap()[0, :, :]
        # ==============================================
        kds, kde = mesh.full_kds, mesh.full_kde
        ids, ide = mesh.cell_ids, mesh.cell_ide
        tmp = formula.modified_potential_temperature(
            t=self.teq,
            p=mg[kds:kde+1, ids:ide+1],
            qv=0.0,
            para=block.para
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

    def init_gz(self, block: Block) -> Block:
        bkd = block.backend
        rd = block.para.rd.unwrap()
        mesh = block.mesh
        mg_lev = block.state.mg_lev.unwrap()[0, :, :]
        gz_lev = block.state.gz_lev.unwrap()[0, :, :]
        gz = block.state.gz.unwrap()[0, :, :]
        # ==============================================
        ids, ide = mesh.cell_ids, mesh.cell_ide
        kds, kde = mesh.half_kds, mesh.half_kde
        tmp = (
            rd * self.teq * bkd.log(self.peq / mg_lev[kds:kde+1, ids:ide+1])
            - 0.5 * self.ueq ** 2 * mesh.cell_sin_lat[None, ids:ide+1] ** 2
        )
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

    def calc_gzs(self, block: Block):
        raise NotImplementedError


class KSP1501TestCase(KSP15TestCase, ksp15.KSP1501TestCase):

    def __init__(self, para):
        ksp15.KSP1501TestCase.__init__(self, para)

    def calc_gzs(self, block: Block):
        bkd = block.backend
        g = block.para.gravity.unwrap()
        radius = block.para.radius.unwrap()
        mesh = block.mesh
        ids, ide = mesh.cell_ids, mesh.cell_ide
        dlon = bkd.abs(mesh.cell_lon[ids:ide+1] - self.lonc)
        dlon = bkd.minimum(2.0 * bkd.pi - dlon, dlon)
        r0 = radius * dlon
        return (
            g * self.h0
            * bkd.exp(-r0 ** 2 / self.d0 ** 2)
            * bkd.cos(bkd.pi * r0 / self.xi0) ** 2
            * mesh.cell_cos_lat[ids:ide+1]
        )


class KSP1502TestCase(KSP15TestCase, ksp15.KSP1502TestCase):

    def __init__(self, para):
        ksp15.KSP1502TestCase.__init__(self, para)

    def calc_gzs(self, block: Block):
        bkd = block.backend
        g = block.para.gravity.unwrap()
        radius = block.para.radius.unwrap()
        mesh = block.mesh
        ids, ide = mesh.cell_ids, mesh.cell_ide
        arg = (
            bkd.sin(self.latc) * mesh.cell_sin_lat[ids:ide+1]
            + bkd.cos(self.latc)
            * mesh.cell_cos_lat[ids:ide+1]
            * bkd.cos(mesh.cell_lon[ids:ide+1] - self.lonc)
        )
        arg = bkd.clip(arg, -1.0, 1.0)
        r = radius * bkd.acos(arg)
        return (
            g * self.h0
            * bkd.exp(-r ** 2 / self.d0 ** 2)
            * bkd.cos(bkd.pi * r / self.xi0) ** 2
        )
