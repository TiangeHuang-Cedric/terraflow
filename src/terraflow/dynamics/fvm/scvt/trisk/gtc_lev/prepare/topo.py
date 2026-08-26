from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util import log
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt.parallel import fill_field1d_array_halo
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.prepare.scvt.topo import fill_grid
from . import ref
from .topo_smooth import topo_smooth, calc_zs_slope

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.prepare.readers.topo import AbstractTopographyReader
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block as BLOCK

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
    para = block.para
    mesh = block.mesh
    log.logger.info(f"Regrid topography.")
    # =================================================================
    regrid_type = reader.cfg.regrid_type
    if regrid_type == "fill":
        res = fill_grid(
            reader=reader,
            mesh=mesh,
            loc="cell",
            layout="triskc"
        )
        gzs = res["gzs"]
        lnd = res["lnd"]
        gzs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs,
            meta=block.static.gzs.meta
        )
        lnd = fill_field1d_array_halo(
            halo=mesh.halo,
            f=lnd,
            meta=block.static.landmask.meta
        )
        block = tree_at(
            where=lambda b: (b.static.gzs.d, b.static.landmask.d),
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
            para=para,
            mesh=mesh,
            loc="cell",
        )
        regridder = Regridder(
            src=src,
            dst=dst,
            layout="triskc",
            backend=reader.cfg.interp_backend,
            method=reader.cfg.interp_method,
            extrap=True,
            esmf_conserve_opt=reader.cfg.interp_conserve_opt,
            precompute_weights=False,
        )
        gzs = regridder(reader.gzs)
        gzs = fill_field1d_array_halo(
            halo=mesh.halo,
            f=gzs,
            meta=block.static.gzs.meta
        )
        block = tree_at(
            where=lambda b: b.static.gzs.d,
            pytree=block,
            replace=gzs,
        )
    else:
        raise ValueError(
            f"Unsupported horizontal interpolation method: {regrid_type}"
        )
    gzs = block.static.gzs.unwrap()
    dzsdn = block.static.dzsdn.unwrap()
    dzsdn = calc_zs_slope(block, gzs, dzsdn)
    block = tree_at(
        where=lambda b: b.static.dzsdn.d,
        pytree=block,
        replace=dzsdn,
    )
    # =================================================================
    return block
