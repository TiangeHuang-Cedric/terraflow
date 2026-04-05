from __future__ import annotations
from typing import (
    cast,
    TypeVar,
    Optional,
    Generic,
    Self,
    overload,
    TYPE_CHECKING
)

from terraflow.util import log
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.abstract import AbstractUnstructuredMesh
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.test_case.baroclinic import BaroclinicEquationTestCase
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
    import terraflow.discretization.fvm.scvt.trisk.dynamics_types as fvm_scvt_trisk_vars


HM = TypeVar("HM", bound=AbstractUnstructuredMesh)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)


class ExVifDryHydrostaticEquationSet4UGRID(
    types.AbstractDryHydrostaticEquationSet[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs
    ):
        super().__init__(disc=disc, method=method, **kwargs)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return types.AbstractDryHydrostaticEquationSet.alias_name()

    @staticmethod
    def form_name() -> str:
        """Return the form name of the dynamics set."""
        return "vif"

    @staticmethod
    def ode_kind_alias_name() -> str:
        """Return the ODE kind alias name of the dynamics set."""
        return "explicit"

    @overload
    def build_block_from_testcase(
        self,
        coord: CoordinateSystem[
            SphericalCentroidVoronoiTessellations,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        *,
        case: Optional[BaroclinicEquationTestCase] = None
    ) -> fvm_scvt_trisk_vars.Block:
        ...

    @overload
    def build_block_from_testcase(
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        *,
        case: Optional[BaroclinicEquationTestCase] = None
    ) -> disc_types.AbstractBlock:
        ...

    def build_block_from_testcase(  # type: ignore[override]
        self,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        *,
        case: Optional[BaroclinicEquationTestCase] = None
    ) -> disc_types.AbstractBlock:
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
            from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block
            block_cls = Block
            coord_specialized = cast(
                CoordinateSystem[
                    SphericalCentroidVoronoiTessellations,
                    HybridVertCoordinate,
                    TriSKBasedCGridStaggeringLayout,
                    HybridVertCoordLayout
                ],
                coord
            )
        else:
            raise NotImplementedError(
                "Unsupported combination of method and coordinate system for building block. "
            )
        return block_cls.from_testcase(
            coord=coord_specialized,
            para=para,
            equation=self,
            case=case
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.space_operators.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.space_operators.prepare_block(block)
        log.logger.success("Prepared auxiliary and data.")
        return block

    def explicit_terms(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        return self.space_operators(
            block,
            *args,
            **kwargs,
            explicit=True
        )


class DryHydrostaticEquationBuilder:

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
        output_h1_tend: bool = False
    ) -> ExVif_DHDC_SCVT_TriSKCGrid_FVM:
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
        *args,
        **kwargs
    ) -> types.AbstractDryHydrostaticEquationSet:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[UHM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs
    ) -> types.AbstractDryHydrostaticEquationSet:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVifDryHydrostaticEquationSet4UGRID.from_model(
                coord=coord,
                para=para,
                method=method,
                *args,
                **kwargs
            )
        else:
            raise ValueError(
                f"Kwargs ode_kind={ode_kind} and form={form} not supported."
                "Please check the avaliable options."
            )


if TYPE_CHECKING:
    ExVif_DHDC_SCVT_TriSKCGrid_FVM = ExVifDryHydrostaticEquationSet4UGRID[
        FiniteVolumeMethod,
        fvm_scvt_trisk_terms.ExSpaceOperators,
    ]
