from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.prepare.readers.abstract import (
        AbstractBackgroundReader,
        AbstractTopographyReader
    )
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK


from . import bkg
from . import topo
from . import ref


def prepare_topo(
    block: BLOCK,
    reader: AbstractTopographyReader,
) -> BLOCK:
    # Regrid topo
    block = topo.regrid_topo(block=block, reader=reader)
    block = ref.ref_calc_ps(block=block)
    return block


def prepare_bkg(
    block: BLOCK,
    reader: AbstractBackgroundReader,
    prepare_regrid_gz: bool = True
) -> BLOCK:
    # Regrid background data and calculate derived variables
    # Stage 1:
    block = bkg.regrid_phs(block=block, reader=reader)
    block = bkg.bkg_calc_ph(block=block)
    for cyc in range(3):
        block = bkg.regrid_wet_qv(
            block=block,
            reader=reader,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qc(
            block=block,
            reader=reader,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qi(
            block=block,
            reader=reader,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qr(
            block=block,
            reader=reader,
            mute=cyc != 0,
        )
        block = bkg.regrid_wet_qs(
            block=block,
            reader=reader,
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
        block = bkg.regrid_un_from_uv(block=block, reader=reader)
        if prepare_regrid_gz:
            block = bkg.regrid_gz(block=block, reader=reader)
            block = bkg.bkg_calc_t(block=block)
        else:
            block = bkg.bkg_calc_t(block=block)
        block = bkg.bkg_calc_tv(block=block)
        block = bkg.bkg_calc_pt(block=block)
    return block
