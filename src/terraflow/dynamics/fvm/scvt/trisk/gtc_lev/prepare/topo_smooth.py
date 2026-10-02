from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util import log
from terraflow.util.math import sign
from terraflow.mesh.scvt.parallel import global_max, fill_field1d_array_halo
from terraflow.mesh.scvt.operators import (
    grad_operator_1d,
    div_operator_1d
)

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.prepare.readers.grist import GRISTStaticDataReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as Block

    Float = typing.Float
    Array = typing.Array

    TOPOREADER = AbstractTopographyReader | GRISTStaticDataReader


def topo_smooth(block: Block, reader: TOPOREADER) -> Block:
    bkd = block.backend
    comm = block.dev.proc.comm_model
    g = block.para.gravity.unwrap()
    # ================================================
    tmp = global_max(bkd, comm, bkd.max(block.static.gzs.unwrap() / g))
    if block.dev.is_root(): log.logger.info(f"Maximum zs is {tmp:.2f} m.")
    tmp = global_max(bkd, comm, bkd.rad2deg(bkd.arctan(bkd.max(block.static.dzsdn.unwrap()))))
    if block.dev.is_root(): log.logger.info(f"Maximum topography slope angle before grad filter is {tmp:.2f} deg.")
    block = zs_grad_filter(block=block, reader=reader)
    tmp = global_max(bkd, comm, bkd.rad2deg(bkd.arctan(bkd.max(block.static.dzsdn.unwrap()))))
    if block.dev.is_root(): log.logger.info(f"Maximum topography slope angle after grad filter is {tmp:.2f} deg.")
    tmp = global_max(bkd, comm, bkd.max(block.static.gzs.unwrap() / g))
    if block.dev.is_root(): log.logger.info(f"Maximum zs is {tmp:.2f} m")
    # ================================================
    return block


def zs_grad_filter(block: Block, reader: TOPOREADER) -> Block:
    filter_type = reader.cfg.zs_grad_filter_type
    if filter_type == "laplacian1":
        block = _laplacian1_zs_grad_filter(block, reader)
    elif filter_type == "laplacian2":
        block = _laplacian2_zs_grad_filter(block, reader)
    else:
        raise ValueError(f"Unsupported zs grad filter type: {filter_type}")
    return block


def _laplacian1_zs_grad_filter(block: Block, reader: TOPOREADER) -> Block:
    bkd = block.backend
    # Do Laplacian damping on target grid with terrain slope larger than a threshold.
    mesh = block.mesh
    cell_v  = mesh.cell_of_edge
    cell_v0 = mesh.cell_of_edge[:, 0]
    cell_v1 = mesh.cell_of_edge[:, 1]
    edpnr_eddtg = mesh.edpnr_eddtg
    lnd = block.static.landmask.unwrap()
    gzs = block.static.gzs.unwrap()
    landmask = block.static.landmask.unwrap()
    dzsdn = block.static.dzsdn.unwrap()
    grad_f = block.aux.fn_1d.unwrap()
    df = block.aux.df_1d.unwrap()
    zs_grad_filter_order = reader.cfg.zs_grad_filter_order
    zs_grad_filter_coef = reader.cfg.zs_grad_filter_coef
    zs_grad_filter_cycles = reader.cfg.zs_grad_filter_cycles
    topo_max_slope = reader.cfg.topo_max_slope
    # =========================================================
    gzs_ = gzs
    c0 = (-1)**(zs_grad_filter_order / 2) * zs_grad_filter_coef
    for _ in range(zs_grad_filter_cycles):
        f = gzs_
        for _ in range(int((zs_grad_filter_order - 2) / 2)):
            grad_f = grad_operator_1d(
                mesh=mesh,
                f=f,
                grad=grad_f,
                f_loc=block.static.gzs.meta.loc
            )
            f = div_operator_1d(
                mesh=mesh,
                f=grad_f,
                div=f,
                div_loc=block.static.gzs.meta.loc
            )
            f = fill_field1d_array_halo(
                halo=mesh.halo,
                f=f,
                meta=block.static.gzs.meta
            )
        grad_f = grad_operator_1d(
            mesh=mesh,
            f=f,
            grad=grad_f,
            f_loc=block.static.gzs.meta.loc
        )
        grad_f = bkd.where(
            bkd.abs(dzsdn) < topo_max_slope,
            bkd.abs(dzsdn) / topo_max_slope * grad_f,
            grad_f,
        )
        grad_f = c0 * bkd.maximum(
            0.0,
            bkd.min(lnd[cell_v], axis=-1)
        ) * grad_f
        if zs_grad_filter_order > 2:
            grad_f = grad_f * bkd.maximum(
                0.0,
                sign(bkd, -grad_f * (gzs_[cell_v1] - gzs_[cell_v0]) * edpnr_eddtg)
            )
        df = div_operator_1d(
            mesh=mesh,
            f=grad_f,
            div=df,
            div_loc=block.static.gzs.meta.loc
        )
        df = bkd.where(landmask > 0.5, df, 0.0)
        gzs_ = gzs_ - df
        gzs_ = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs_,
            meta=block.static.gzs.meta
        )
        dzsdn = calc_zs_slope(block, gzs_, dzsdn)
    gzs = bkd.where(landmask > 0.5, gzs_, gzs)
    gzs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=gzs,
        meta=block.static.gzs.meta
    )
    dzsdn = calc_zs_slope(block, gzs, dzsdn)
    # =========================================================
    return bkd.tree_at(
        where=lambda b: (b.static.gzs.d, b.static.dzsdn.d),
        pytree=block,
        replace=(gzs, dzsdn),
    )


def _laplacian2_zs_grad_filter(block: Block, reader: TOPOREADER) -> Block:
    bkd = block.backend
    mesh = block.mesh
    gzs = block.static.gzs.unwrap()
    landmask = block.static.landmask.unwrap()
    dzsdn = block.static.dzsdn.unwrap()
    grad_f = block.aux.fn_1d.unwrap()
    df = block.aux.df_1d.unwrap()
    zs_grad_filter_cycles = reader.cfg.zs_grad_filter_cycles
    dt_dyn = block.dt_dyn
    # =========================================================
    gzs_ = gzs
    for _ in range(zs_grad_filter_cycles):
        f = gzs_
        grad_f = grad_operator_1d(
            mesh=mesh,
            f=f,
            grad=grad_f,
            f_loc=block.static.gzs.meta.loc
        )
        df = div_operator_1d(
            mesh=mesh,
            f=grad_f,
            div=df,
            div_loc=block.static.gzs.meta.loc
        )
        gzs_ = gzs_ + dt_dyn * df
        gzs_ = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs_,
            meta=block.static.gzs.meta
        )
    gzs = bkd.where(landmask > 0.5, gzs_, gzs)
    gzs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=gzs,
        meta=block.static.gzs.meta
    )
    dzsdn = calc_zs_slope(block, gzs, dzsdn)
    # =========================================================
    return bkd.tree_at(
        where=lambda b: (b.static.gzs.d, b.static.dzsdn.d),
        pytree=block,
        replace=(gzs, dzsdn),
    )


def calc_zs_slope(
    block: Block,
    gzs: Float[Array, "npoints"],
    dzsdn: Float[Array, "npoints"]
) -> Float[Array, "npoints"]:
    mesh = block.mesh
    g = block.para.gravity.unwrap()
    # ================================
    dzsdn = grad_operator_1d(
        mesh=mesh,
        f=gzs,
        grad=dzsdn,
        f_loc=block.static.gzs.meta.loc
    )
    # ================================
    dzsdn = dzsdn / g
    dzsdn = fill_field1d_array_halo(
        halo=mesh.halo,
        f=dzsdn,
        meta=block.static.dzsdn.meta
    )
    return dzsdn
