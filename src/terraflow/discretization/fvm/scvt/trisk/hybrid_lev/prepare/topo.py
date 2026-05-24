from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

from terraflow.util import log
from terraflow.mesh.scvt.operators import (
    grad_at_dual_edge,
    div_at_primal_cell
)
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.prepare.scvt.topo import fill_grid
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK

    Float = typing.Float
    Array = typing.Array


def regrid_topo(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    coord = block.coord
    log.logger.info(f"Regrid topography.")
    # =================================================================
    regrid_type = reader.cfg.regrid_type
    if regrid_type == "fill":
        res = fill_grid(
            reader=reader,
            mesh=coord.mesh,
            loc="cell",
            entity="primal"
        )
        gzs = res["gzs"]
        lnd = res["lnd"]
        block = tree_at(
            coord=coord,
            where=lambda b: (b.static.gzs, b.static.landmask),
            pytree=block,
            replace=(gzs, lnd),
        )
    elif regrid_type == "interp":
        log.logger.info("Build regridder for topography data.")
        src = Regridder.create_latlon_info(
            lon=reader.lon,
            lat=reader.lat
        )
        dst = Regridder.create_scvt_info(
            mesh=coord.mesh,
            loc="cell",
            entity="primal"
        )
        regridder = Regridder(
            src=src,
            dst=dst,
            backend=reader.cfg.interp_backend,
            method=reader.cfg.interp_method,
            extrap=True,
            conserve_opt=reader.cfg.interp_conserve_opt,
            precompute_weights=False,
        )
        gzs = regridder(reader.gzs)
        block = tree_at(
            coord=coord,
            where=lambda b: b.static.gzs,
            pytree=block,
            replace=gzs,
        )
    else:
        raise ValueError(
            f"Unsupported horizontal interpolation method: {regrid_type}"
        )
    gzs = block.static.gzs.unwrap().reshape(1, -1)
    grad_zs = block.static.grad_zs.unwrap().reshape(1, -1)
    grad_zs = calc_zs_slope(block, gzs, grad_zs).reshape(-1)
    block = tree_at(
        coord=coord,
        where=lambda b: b.static.grad_zs,
        pytree=block,
        replace=grad_zs,
    )
    # =================================================================
    return block


def topo_smooth(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    g = block.para.gravity.unwrap()
    # ================================================
    tmp = jnp.max(block.static.gzs.unwrap() / g)
    log.logger.info(f"Maximum zs is {tmp:.2f} m.")
    tmp = jnp.rad2deg(jnp.atan(jnp.max(block.static.grad_zs.unwrap())))
    log.logger.info(f"Maximum topography slope angle before grad filter is {tmp:.2f} deg.")
    block = zs_grad_filter(block=block, reader=reader)
    tmp = jnp.rad2deg(jnp.atan(jnp.max(block.static.grad_zs.unwrap())))
    log.logger.info(f"Maximum topography slope angle after grad filter is {tmp:.2f} deg.")
    tmp = jnp.max(block.static.gzs.unwrap() / g)
    log.logger.info(f"Maximum zs is {tmp:.2f} m")
    # ================================================
    return block


def zs_grad_filter(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    filter_type = reader.cfg.zs_grad_filter_type
    if filter_type == "laplacian1":
        block = _laplacian1_zs_grad_filter(block, reader)
    elif filter_type == "laplacian2":
        block = _laplacian2_zs_grad_filter(block, reader)
    else:
        raise ValueError(f"Unsupported zs grad filter type: {filter_type}")
    return block


def _laplacian1_zs_grad_filter(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    # Do Laplacian damping on target grid with terrain slope larger than a threshold.
    para = block.para
    coord = block.coord
    mesh = coord.mesh
    cell_v = mesh._edd.v
    cell_v0 = cell_v[:, 0]
    cell_v1 = cell_v[:, 1]
    edpnr_eddtg = mesh.aux.edpnr_eddtg
    lnd = block.static.landmask.unwrap().reshape(1, -1)
    gzs = block.static.gzs.unwrap().reshape(1, -1)
    grad_zs = block.static.grad_zs.unwrap().reshape(1, -1)
    grad_f = block.aux.f_1d.unwrap().reshape(1, -1)
    df = block.aux.df_1d.unwrap().reshape(1, -1)
    zs_grad_filter_order = reader.cfg.zs_grad_filter_order
    zs_grad_filter_coef = reader.cfg.zs_grad_filter_coef
    zs_grad_filter_cycles = reader.cfg.zs_grad_filter_cycles
    topo_max_slope = reader.cfg.topo_max_slope
    # =========================================================
    c0 = (-1)**(zs_grad_filter_order / 2) * zs_grad_filter_coef
    for _ in range(zs_grad_filter_cycles):
        f = gzs
        for _ in range(int((zs_grad_filter_order - 2) / 2)):
            grad_f = grad_at_dual_edge(
                x_at_primal_cell=f,
                x_at_dual_edge=grad_f,
                coord=coord,
                para=para,
            )
            f = div_at_primal_cell(
                x_at_primal_edge=grad_f,
                x_at_primal_cell=f,
                coord=coord,
                para=para,
            )
        grad_f = grad_at_dual_edge(
            x_at_primal_cell=f,
            x_at_dual_edge=grad_f,
            coord=coord,
            para=para,
        )
        grad_f = jnp.where(
            jnp.abs(grad_zs) < topo_max_slope,
            jnp.abs(grad_zs) / topo_max_slope * grad_f,
            grad_f,
        )
        grad_f = c0 * jnp.maximum(
            0.0,
            jnp.min(lnd[:, cell_v], axis=-1)
        ) * grad_f
        if zs_grad_filter_order > 2:
            grad_f = grad_f * jnp.maximum(
                0.0,
                jnp.sign(-grad_f * (gzs[:, cell_v1] - gzs[:, cell_v0]) * edpnr_eddtg)
            )
        df = div_at_primal_cell(
            x_at_primal_edge=grad_f,
            x_at_primal_cell=df,
            coord=coord,
            para=para,
        )
        # TODO: Maybe add filter here?
        gzs = gzs - df
        grad_zs = calc_zs_slope(block, gzs, grad_zs)
    gzs = gzs.reshape(-1)
    grad_zs = grad_zs.reshape(-1)
    # =========================================================
    return tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.grad_zs),
        pytree=block,
        replace=(gzs, grad_zs),
    )


def _laplacian2_zs_grad_filter(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    para = block.para
    coord = block.coord
    gzs = block.static.gzs.unwrap().reshape(1, -1)
    grad_zs = block.static.grad_zs.unwrap().reshape(1, -1)
    grad_f = block.aux.f_1d.unwrap().reshape(1, -1)
    df = block.aux.df_1d.unwrap().reshape(1, -1)
    zs_grad_filter_cycles = reader.cfg.zs_grad_filter_cycles
    dt_dyn = block.dt_dyn
    # =========================================================
    for _ in range(zs_grad_filter_cycles):
        f = gzs
        grad_f = grad_at_dual_edge(
            x_at_primal_cell=f,
            x_at_dual_edge=grad_f,
            coord=coord,
            para=para,
        )
        df = div_at_primal_cell(
            x_at_primal_edge=grad_f,
            x_at_primal_cell=df,
            coord=coord,
            para=para,
        )
        gzs = gzs + dt_dyn * df
    grad_zs = calc_zs_slope(block, gzs, grad_zs)
    gzs = gzs.reshape(-1)
    grad_zs = grad_zs.reshape(-1)
    # =========================================================
    return tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.grad_zs),
        pytree=block,
        replace=(gzs, grad_zs),
    )


def calc_zs_slope(
    block: BLOCK,
    gzs: Float[Array, "npoints"],
    grad_zs: Float[Array, "npoints"]
) -> Float[Array, "npoints"]:
    g = block.para.gravity.unwrap()
    # ================================
    grad_zs = grad_at_dual_edge(
        x_at_primal_cell=gzs,
        x_at_dual_edge=grad_zs,
        coord=block.coord,
        para=block.para,
    )
    # ================================
    return grad_zs / g
