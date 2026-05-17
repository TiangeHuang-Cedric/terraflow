from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util import log
from terraflow.mesh.scvt.regridder import Regridder

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import (
        AbstractBackgroundReader,
        AbstractTopographyReader
    )
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


from . import bkg
from . import topo
from . import ref


def prepare_topo(block: BLOCK, reader: AbstractTopographyReader) -> BLOCK:
    # Regrid topo
    block = topo.regrid_topo(block=block, reader=reader)
    block = ref.ref_calc_ps(block=block)
    return block


def prepare_bkg(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    prepare_regrid_gz: bool = True
) -> BLOCK:
    # Regridder for background data
    src = Regridder.create_latlon_info(
        lon=reader.lon,
        lat=reader.lat
    )
    dst_at_cell = Regridder.create_scvt_info(
        mesh=block.coord.mesh,
        loc="cell",
        entity="primal"
    )
    dst_at_edge = Regridder.create_scvt_info(
        mesh=block.coord.mesh,
        loc="edge",
        entity="dual"
    )
    log.logger.info("Build regridder for background data at cell center.")
    regridder_at_cell = Regridder(
        src=src,
        dst=dst_at_cell,
        backend=reader.cfg.hori_interp_backend,
        ndbounds=(reader.lev.shape[0],),
        method=reader.cfg.hori_interp_method,
        extrap=reader.cfg.hori_extrap,
        conserve_opt=reader.cfg.hori_conserve_opt,
        precompute_weights=True,
    )
    log.logger.info("Build regridder for background data at edge midpoint.")
    regridder_at_edge = Regridder(
        src=src,
        dst=dst_at_edge,
        backend=reader.cfg.hori_interp_backend,
        ndbounds=(reader.lev.shape[0],),
        method=reader.cfg.hori_interp_method_at_edge,
        extrap=reader.cfg.hori_extrap,
        conserve_opt=reader.cfg.hori_conserve_opt,
        precompute_weights=True,
    )
    # Regrid background data and calculate derived variables
    # Stage 1:
    block = bkg.regrid_phs(
        block=block,
        reader=reader,
        regridder=regridder_at_cell
    )
    block = bkg.bkg_calc_ph(block=block)
    for cyc in range(3):
        block = bkg.regrid_wet_qv(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qc(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qi(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qr(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qs(
            block=block,
            reader=reader,
            regridder=regridder_at_cell,
            mute=cyc != 0,
        )
        block = block.calc_qm(itime=0)
        # Stage 2:
        block = bkg.bkg_calc_mgs(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_mg(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_dry_qv(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_dry_qc(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_dry_qi(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_dry_qr(block=block, mute=cyc != 0)
        block = bkg.bkg_calc_dry_qs(block=block, mute=cyc != 0)
        block = block.calc_qm(itime=0)
        block = bkg.bkg_calc_ph(block=block)
        # Stage 3:
        block = bkg.regrid_un_from_uv(
            block=block,
            reader=reader,
            regridder=regridder_at_edge
        )
        if prepare_regrid_gz:
            block = bkg.regrid_gz(
                block=block,
                reader=reader,
                regridder=regridder_at_cell
            )
            block = bkg.bkg_calc_t(block=block)
        else:
            block = bkg.bkg_calc_t(block=block)
        block = bkg.bkg_calc_tv(block=block)
        block = bkg.bkg_calc_pt(block=block)
    return block
