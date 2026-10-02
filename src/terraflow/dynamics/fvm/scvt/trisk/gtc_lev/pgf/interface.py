from __future__ import annotations
from typing import TYPE_CHECKING, Self


from terraflow.util.device import AbstractDevice
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.pgf.swm import pgf_swm_run
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.pgf.ptb import RefProfileData, pgf_ptb_run
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.pgf.lin97 import pgf_lin97_run


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.abstract import ModelMode
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


class PressureGradientForceDataset(AbstractEquationAuxiliaryDataset):
    pro: RefProfileData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            pro=RefProfileData.empty()
        )
        
    @classmethod
    def initialize(
        cls,
        mode: ModelMode,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        ref_ps: Float[Array, "ncell"],
        dev: AbstractDevice
    ) -> Self:
        if mode.is_baroclinic():
            pro = RefProfileData.initialize(
                cfg=cfg,
                para=para,
                mesh=mesh,
                vert_coord=vert_coord,
                ref_ps=ref_ps,
                dev=dev
            )
        else:
            pro = RefProfileData.empty()
        return cls(
            initialized=True,
            pro=pro
        )
        

def pgf_run(block: Block, star_idx: Int[Array, ""] | int) -> Block:
    if block.mode.is_baroclinic():
        if block.cfg.pgf_scheme == "ptb":
            return pgf_ptb_run(block, star_idx)
        elif block.cfg.pgf_scheme == "lin97":
            return pgf_lin97_run(block, star_idx)
        else:
            raise ValueError(f"pgf_run: unsupported pgf_scheme for baroclinic model: {block.cfg.pgf_scheme}!")
    else:
        return pgf_swm_run(block, star_idx)
