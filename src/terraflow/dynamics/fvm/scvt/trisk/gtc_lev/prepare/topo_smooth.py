from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import log
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt.parallel import fill_field1d_array_halo
from terraflow.mesh.scvt.operators import (
    grad_operator_1d,
    div_operator_1d
)

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.prepare.readers.grist import GRISTStaticDataReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as BLOCK

    Float = typing.Float
    Array = typing.Array

    TOPOREADER = AbstractTopographyReader | GRISTStaticDataReader


def topo_smooth(block: BLOCK, reader: TOPOREADER) -> BLOCK:
    g = block.para.gravity.unwrap()
    # ================================================
    tmp = jnp.max(block.static.gzs.unwrap() / g)
    log.logger.info(f"Maximum zs is {tmp:.2f} m.")
    tmp = jnp.rad2deg(jnp.atan(jnp.max(block.static.dzsdn.unwrap())))
    log.logger.info(f"Maximum topography slope angle before grad filter is {tmp:.2f} deg.")
    block = zs_grad_filter(block=block, reader=reader)
    tmp = jnp.rad2deg(jnp.atan(jnp.max(block.static.dzsdn.unwrap())))
    log.logger.info(f"Maximum topography slope angle after grad filter is {tmp:.2f} deg.")
    tmp = jnp.max(block.static.gzs.unwrap() / g)
    log.logger.info(f"Maximum zs is {tmp:.2f} m")
    # ================================================
    return block


def zs_grad_filter(block: BLOCK, reader: TOPOREADER) -> BLOCK:
    filter_type = reader.cfg.zs_grad_filter_type
    if filter_type == "laplacian1":
        block = _laplacian1_zs_grad_filter(block, reader)
    elif filter_type == "laplacian2":
        block = _laplacian2_zs_grad_filter(block, reader)
    else:
        raise ValueError(f"Unsupported zs grad filter type: {filter_type}")
    return block


def _laplacian1_zs_grad_filter(block: BLOCK, reader: TOPOREADER) -> BLOCK:
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
        grad_f = jnp.where(
            jnp.abs(dzsdn) < topo_max_slope,
            jnp.abs(dzsdn) / topo_max_slope * grad_f,
            grad_f,
        )
        grad_f = c0 * jnp.maximum(
            0.0,
            jnp.min(lnd[cell_v], axis=-1)
        ) * grad_f
        if zs_grad_filter_order > 2:
            grad_f = grad_f * jnp.maximum(
                0.0,
                jnp.sign(-grad_f * (gzs_[cell_v1] - gzs_[cell_v0]) * edpnr_eddtg)
            )
        df = div_operator_1d(
            mesh=mesh,
            f=grad_f,
            div=df,
            div_loc=block.static.gzs.meta.loc
        )
        # TODO: Maybe add filter here?
        gzs_ = gzs_ - df
        gzs_ = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs_,
            meta=block.static.gzs.meta
        )
        dzsdn = calc_zs_slope(block, gzs_, dzsdn)
    gzs = jnp.where(landmask > 0.5, gzs_, gzs)
    gzs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=gzs,
        meta=block.static.gzs.meta
    )
    dzsdn = calc_zs_slope(block, gzs, dzsdn)
    # =========================================================
    return tree_at(
        where=lambda b: (b.static.gzs.d, b.static.dzsdn.d),
        pytree=block,
        replace=(gzs, dzsdn),
    )


def _laplacian2_zs_grad_filter(block: BLOCK, reader: TOPOREADER) -> BLOCK:
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
    gzs = jnp.where(landmask > 0.5, gzs_, gzs)
    gzs = fill_field1d_array_halo(
        halo=mesh.halo,
        f=gzs,
        meta=block.static.gzs.meta
    )
    dzsdn = calc_zs_slope(block, gzs, dzsdn)
    # =========================================================
    return tree_at(
        where=lambda b: (b.static.gzs.d, b.static.dzsdn.d),
        pytree=block,
        replace=(gzs, dzsdn),
    )


def calc_zs_slope(
    block: BLOCK,
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
