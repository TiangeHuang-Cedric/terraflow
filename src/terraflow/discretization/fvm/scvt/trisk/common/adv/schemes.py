from __future__ import annotations
from typing import Self, Any, Dict, TYPE_CHECKING, TypeVar

from terraflow.util import typing
from terraflow.discretization.fvm.scvt.trisk.common.adv import batch
from terraflow.discretization.fvm.scvt.trisk.common.adv import types, upwind

Array = typing.Array

if TYPE_CHECKING:
    from terraflow.util.model_config import ModelConfig
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    CoordSysOneLayer = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSysHybrid = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]

    CoordSys = TypeVar("CoordSys", CoordSysOneLayer, CoordSysHybrid)

    BLOCK = TypeVar("BLOCK", SingleLevBlock, HybridLevBlock)


class HorizontalAdvectionSchemeAtCell:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def has_built(self) -> bool:
        return self._built

    def calc_mass_hflx(
        self,
        m: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ):
        return self.scheme.calc_mass_hflx(
            m=m,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def calc_tracer_hflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def upwind(
        self,
        order: int = 3,
        optimized: bool = True
    ) -> upwind.UpwindHorizontalAdvectionAtCell:
        self.scheme = upwind.UpwindHorizontalAdvectionAtCell(
            order=order, optimized=optimized
        )
        self.scheme_name = "upwind"
        self._cfg["adv_scheme"] = "upwind"
        self._cfg["upwind_order"] = order
        self._cfg["upwind_optimized"] = int(optimized)
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, target: str, cfg: ModelConfig) -> Self:
        if not self._built:
            if target == "bg":
                if cfg.bg_adv_scheme == "upwind":
                    self.upwind(
                        order=cfg.upwind_order_bg,
                        optimized=cfg.upwind_optimized_bg,
                    )
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.bg_adv_scheme}"
                    )
            if target == "pt":
                if cfg.pt_adv_scheme == "upwind":
                    self.upwind(
                        order=cfg.upwind_order_h,
                        optimized=cfg.upwind_optimized,
                    )
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.pt_adv_scheme}"
                    )
            if target == "tracer":
                if cfg.adv_scheme == "upwind":
                    self.upwind(
                        order=cfg.upwind_order_h,
                        optimized=cfg.upwind_optimized,
                    )
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.adv_scheme}"
                    )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class HorizontalAdvectionSchemeAtVtx:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def has_built(self) -> bool:
        return self._built

    def __call__(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSys,
        aux: types.AdvectionAuxiliaryDataset,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )

    def upwind(
        self,
        order: int = 3,
        use_mbs: bool = True,
        optimized: bool = True
    ) -> upwind.UpwindHorizontalAdvectionAtVtx:
        self.scheme = upwind.UpwindHorizontalAdvectionAtVtx(
            order=order,
            use_mbs=use_mbs,
            optimized=optimized
        )
        self.scheme_name = "upwind"
        self._cfg["adv_scheme"] = "upwind"
        self._cfg["upwind_order"] = order
        self._cfg["upwind_use_mbs"] = int(use_mbs)
        self._cfg["upwind_optimized"] = int(optimized)
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, target: str, cfg: ModelConfig) -> Self:
        if not self._built:
            if target == "tracer":
                if cfg.adv_scheme == "upwind":
                    self.upwind(
                        order=cfg.upwind_order_h,
                        use_mbs=cfg.upwind_use_mbs,
                        optimized=cfg.upwind_optimized,
                    )
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.bg_adv_scheme}"
                    )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


class VerticalAdvectionScheme:

    def __init__(self):
        self._cfg: dict[str, Any] = {}
        self._built = False

    def has_built(self) -> bool:
        return self._built

    def calc_tracer_vflx(
        self,
        q: Array,
        batch: batch.AdvBatchType,
        coord: CoordSysHybrid,
    ) -> batch.AdvBatchType:
        return self.scheme.calc_tracer_vflx(
            q=q,
            coord=coord,
            batch=batch,
        )

    def upwind(self, order: int = 3, wgt: float = 1) -> upwind.UpwindVerticalAdvection:
        self.scheme = upwind.UpwindVerticalAdvection(order=order, wgt=wgt)
        self.scheme_name = "upwind"
        self._cfg["adv_scheme"] = "upwind"
        self._cfg["upwind_order"] = order
        self._cfg["upwind_wgt"] = wgt
        self._built = True
        return self.scheme

    def config2dict(self) -> Dict[str, Any]:
        return self._cfg

    def build_with_config(self, target: str, cfg: ModelConfig) -> Self:
        if not self._built:
            if target == "bg":
                if cfg.bg_adv_scheme == "upwind":
                    self.upwind(order=cfg.upwind_order_bg, wgt=cfg.upwind_wgt)
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.bg_adv_scheme}"
                    )
            if target == "pt":
                if cfg.pt_adv_scheme == "upwind":
                    self.upwind(order=cfg.upwind_order_v, wgt=cfg.upwind_wgt)
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.pt_adv_scheme}"
                    )
            if target == "tracer":
                if cfg.adv_scheme == "upwind":
                    self.upwind(order=cfg.upwind_order_v, wgt=cfg.upwind_wgt)
                else:
                    raise ValueError(
                        f"Unsupported advection scheme: {cfg.adv_scheme}"
                    )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)
