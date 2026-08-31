from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util.device import AbstractDevice, CPU
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.damp.smag import SmagorinskyDampData, smag_damp_prepare
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.damp.div import DivDampdata, div_damp_run

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


    Float = typing.Float
    Array = typing.Array


class DampDataset(AbstractEquationAuxiliaryDataset):
    smag: SmagorinskyDampData
    div: DivDampdata

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            smag=SmagorinskyDampData.empty(),
            div=DivDampdata.empty()
        )
        
    @classmethod
    def initialize(
        cls,
        cfg: UserConfig,
        para: PlanetParameterSet, 
        mesh: SCVTMesh, 
        dev: AbstractDevice = CPU()
    ) -> Self:
        smag = SmagorinskyDampData.initialize(
            cfg=cfg,
            para=para,
            mesh=mesh,
            dev=dev
        )
        div = DivDampdata.initialize(
            cfg=cfg,
            mesh=mesh,
            dev=dev
        )
        return cls(
            initialized=True,
            smag=smag,
            div=div
        )


def damp_prepare(block: Block, itime: int, dt: float) -> Block:
    if block.cfg.use_smag_damp:
        return smag_damp_prepare(block, itime, dt)
    return block


def damp_run(block: Block, itime: int, dt: float) -> Block:
    if block.cfg.use_div_damp:
        block = div_damp_run(block, itime, dt)
    return block