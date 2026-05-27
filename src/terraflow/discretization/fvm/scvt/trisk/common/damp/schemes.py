from __future__ import annotations
from typing import Any, Dict, Self, TypeVar, TYPE_CHECKING

from terraflow.operator import FiniteVolumeMethod
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    EquationMode,
    AbstractDynamicalEquationOperator
)
from terraflow.discretization.fvm.scvt.trisk.common.damp.smag import SmagorinskyDamp
from terraflow.util.model_config import ModelConfig

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    HybridCoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    SingleLevCoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSys = TypeVar("CoordSys", HybridCoordSys, SingleLevCoordSys)

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)

DynOperator = AbstractDynamicalEquationOperator[FiniteVolumeMethod]


class Damp(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: EquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.smag = SmagorinskyDamp(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, itime: Int[Array, ""] | int) -> BLOCK:
        if self.use_smag_damp:
            block = self.smag(block, itime)
        return block

    def update_tend(self, block: BLOCK) -> BLOCK:
        if self.use_smag_damp:
            block = self.smag.update_tend(block)
        return block

    def config2dict(self) -> Dict[str, Any]:
        self._cfg.update(self.smag.config2dict())
        return self._cfg

    def build_with_config(self, cfg: ModelConfig) -> Self:
        if not self.has_built():
            self.use_smag_damp = cfg.use_smag_damp
            self.smag.build_with_config(cfg)
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        if self.use_smag_damp:
            block = self.smag.prepare_block(block)
        return block
