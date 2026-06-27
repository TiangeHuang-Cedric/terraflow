from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util import log
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.prepare.scvt.topo import fill_grid
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from . import ref
from .topo_smooth import topo_smooth, calc_zs_slope

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK

    Float = typing.Float
    Array = typing.Array


def prepare_topo(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    # Regrid topo
    block = regrid_topo(block=block, reader=reader)
    if reader.cfg.use_zs_grad_filter:
        block = topo_smooth(block=block, reader=reader)
    block = ref.ref_calc_ps(block=block)
    return block


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
            layout="triskC"
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
            coord=coord,
            loc="cell",
        )
        regridder = Regridder(
            src=src,
            dst=dst,
            backend=reader.cfg.interp_backend,
            method=reader.cfg.interp_method,
            extrap=True,
            esmf_conserve_opt=reader.cfg.interp_conserve_opt,
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
