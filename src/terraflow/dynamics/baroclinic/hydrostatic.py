from __future__ import annotations
from typing import (
    Any,
    Dict,
    TypeVar,
    Generic,
    Self,
    overload,
    Callable,
    TYPE_CHECKING
)

from terraflow.util import log, model_config
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.abstract import AbstractUnstructuredMesh
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.dynamics.baroclinic import types
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    import terraflow.discretization.fvm.scvt.trisk.hybrid_lev.term as fvm_scvt_trisk_terms
    import terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types as fvm_scvt_trisk_vars


HM = TypeVar("HM", bound=AbstractUnstructuredMesh)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)


class ExVifHydrostaticEquationSet4UGRID(
    types.AbstractHydrostaticEquationSet[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        ideal_dry_core: bool = False,
        **kwargs
    ):
        super().__init__(
            disc=disc, 
            method=method, 
            ideal_dry_core=ideal_dry_core, 
            **kwargs
        )

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return types.AbstractHydrostaticEquationSet.alias_name()

    @staticmethod
    def form_name() -> str:
        """Return the form name of the dynamics set."""
        return "vif"

    @staticmethod
    def ode_kind_alias_name() -> str:
        """Return the ODE kind alias name of the dynamics set."""
        return "explicit"

    @overload
    def build_block_cls(
        self,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
    ) -> type[fvm_scvt_trisk_vars.Block]:
        ...

    @overload
    def build_block_cls(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
    ) -> type[disc_types.AbstractBlock]:
        ...

    def build_block_cls(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
    ) -> type[disc_types.AbstractBlock]:
        from terraflow.operator import FiniteVolumeMethod
        from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
        from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
        from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
        from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
        if (
            isinstance(self.method, FiniteVolumeMethod) and
            isinstance(coord.mesh, SphericalCentroidVoronoiTessellations) and
            isinstance(coord.vert, HybridVertCoordinate) and
            isinstance(coord.hori_layout, TriSKBasedCGridStaggeringLayout) and
            isinstance(coord.vert_layout, HybridVertCoordLayout)
        ):
            from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block
            return Block
        else:
            raise NotImplementedError(
                "Unsupported combination of method and coordinate system for building block. "
            )

    def config2dict(self) -> Dict[str, Any]:
        cfg = self._space_operators.config2dict()
        self._cfg.update(cfg)
        return self._cfg

    def build_default(self) -> Self:
        cfg = model_config.ModelConfig()
        self.build_with_config(cfg=cfg)
        self._cfg = self.config2dict()
        return self

    def build_with_config(self, cfg: model_config.ModelConfig) -> Self:
        if not self.has_built():
            self.space_operators.build_with_config(cfg=cfg)
            self._cfg = cfg.to_dict()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.space_operators.prepare_block(block)
        log.logger.success("Prepared auxiliary and data.")
        return block

    def explicit_terms(self, advection: bool = False, *args, **kwargs) -> Callable:

        if not advection:
            def _wrap_dynamics(block: BLOCK, *args, **kwargs) -> BLOCK:
                return self.space_operators(
                    block=block,
                    *args,
                    **kwargs,
                    explicit=True
                )
            return _wrap_dynamics
        else:
            def _wrap_advection(
                block: BLOCK,
                *args,
                **kwargs
            ) -> BLOCK:
                return self.space_operators.advection_stepper(
                    block,
                    *args,
                    **kwargs,
                    explicit=True
                )
            return _wrap_advection


class HydrostaticEquationBuilder:
    @overload
    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        config: model_config.ModelConfig = model_config.ModelConfig(),
        output_h1_tend: bool = False,
        state_tile_depth: int = 3,
    ) -> ExVif_HDC_SCVT_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs
    ) -> types.AbstractHydrostaticEquationSet:
        ...

    @classmethod
    def build_with_config(
        cls,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        config: model_config.ModelConfig = model_config.ModelConfig(),
        *args,
        **kwargs
    ) -> types.AbstractHydrostaticEquationSet:
        equation = cls.from_model(
            coord=coord,
            para=para,
            method=method,
            form=form,
            ode_kind=ode_kind,
            ideal_dry_core=ideal_dry_core,
            *args,
            **kwargs
        )
        equation.build_with_config(cfg=config)
        return equation

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        output_h1_tend: bool = False,
        state_tile_depth: int = 3,
    ) -> ExVif_HDC_SCVT_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        *args,
        **kwargs
    ) -> types.AbstractHydrostaticEquationSet:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        ideal_dry_core: bool = False,
        *args,
        **kwargs
    ) -> types.AbstractHydrostaticEquationSet:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVifHydrostaticEquationSet4UGRID.from_model(
                coord=coord,
                para=para,
                method=method,
                ideal_dry_core=ideal_dry_core,
                *args,
                **kwargs
            )
        else:
            raise ValueError(
                f"Kwargs ode_kind={ode_kind} and form={form} not supported."
                "Please check the avaliable options."
            )


if TYPE_CHECKING:
    ExVif_HDC_SCVT_TriSKCGrid_FVM = ExVifHydrostaticEquationSet4UGRID[
        FiniteVolumeMethod,
        fvm_scvt_trisk_terms.ExSpaceOperators,
    ]
