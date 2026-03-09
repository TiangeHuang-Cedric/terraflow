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

from terraflow.planet import PlanetParameterSet
from terraflow.mesh.abstract import AbstractUnstructuredMesh
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.test_case.baroclinic import BaroclinicEquationTestCase
from terraflow.dynamics.baroclinic import types
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    import terraflow.discretization.fvm.odm.trisk.hybrid_lev.aux as fvm_odm_trisk_aux
    import terraflow.discretization.fvm.odm.trisk.dynamics_types as fvm_odm_trisk_vars
    import terraflow.discretization.fvm.odm.trisk.hybrid_lev.term as fvm_odm_trisk_hybridlev_builder


HM = TypeVar("HM", bound=AbstractUnstructuredMesh)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
ESAUX = TypeVar("ESAUX", bound=disc_types.AbstractEquationSetAuxiliaryTerms)


class ExVifDryHydrostaticEquationSet4UGRID(
    types.AbstractDryHydrostaticEquationSet[DM, ESAUX],
    AbstractExplicitODE,
    Generic[
        types.DMFS,
        types.DMFT, types.VAT,
        types.NCT, types.GKE, types.VAM, types.PGF,
        ESAUX,
        DM
    ],
):
    dmgsdt: types.ExContinuityEqu[
        types.DMFS,
        DM,
    ]
    dptdt: types.ExThermodynamicEqu[
        types.DMFT, types.VAT,
        DM,
    ]
    dundt: types.ExNormalMomentumEqu4UGRID[
        types.NCT, types.GKE, types.VAM, types.PGF,
        DM,
    ]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc=disc, method=method)
        self.dmgsdt = self.init_continuity_equation(
            disc=disc,
            method=method,
        )
        self.dptdt = self.init_thermodynamic_equation(
            disc=disc,
            method=method,
        )
        self.dundt = self.init_momentum_equation(
            disc=disc,
            method=method,
        )

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
            OrthogonalDualMesh,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        *,
        case: Optional[BaroclinicEquationTestCase] = None
    ) -> fvm_odm_trisk_vars.Block:
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
        from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
        from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
        from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
        from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
        if (
            isinstance(self.method, FiniteVolumeMethod) and
            isinstance(coord.mesh, OrthogonalDualMesh) and
            isinstance(coord.vert, HybridVertCoordinate) and
            isinstance(coord.hori_layout, TriSKBasedCGridStaggeringLayout) and
            isinstance(coord.vert_layout, HybridVertCoordLayout)
        ):
            from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block
            block_cls = Block
            coord_specialized = cast(
                CoordinateSystem[
                    OrthogonalDualMesh,
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
            self.aux.build_default()
            self.dmgsdt.build_default()
            self.dptdt.build_default()
            self.dundt.build_default()
            self._built = True
        return self

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> types.ExContinuityEqu[
        types.DMFS,
        DM,
    ]:
        return types.ExContinuityEqu(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def init_thermodynamic_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> types.ExThermodynamicEqu[
        types.DMFT, types.VAT,
        DM,
    ]:
        return types.ExThermodynamicEqu(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> types.ExNormalMomentumEqu4UGRID[
        types.NCT, types.GKE, types.VAM, types.PGF,
        DM,
    ]:
        return types.ExNormalMomentumEqu4UGRID(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def prepare(self, block: BLOCK) -> BLOCK:
        block = self.aux.prepare(block)
        block = self.dmgsdt.prepare(block)
        block = self.dptdt.prepare(block)
        block = self.dundt.prepare(block)
        return block

    def explicit_terms(self, block: BLOCK, pass_tag: str = "all_pass") -> BLOCK:
        block = self.aux(block, pass_tag=pass_tag)
        if pass_tag == "all_pass":
            block = self.dmgsdt(block)
            block = self.dptdt(block)
            block = self.dundt(block)
        elif pass_tag == "forward_pass":
            block = self.dmgsdt(block)
            block = self.dptdt(block)
        elif pass_tag == "backward_pass":
            block = self.dundt(block)
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return block


class DryHydrostaticEquationBuilder:

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs
    ) -> ExVif_DHDC_ODM_TriSKCGrid_FVM:
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
    ExVif_DHDC_ODM_TriSKCGrid_FVM = ExVifDryHydrostaticEquationSet4UGRID[
        fvm_odm_trisk_hybridlev_builder.TendencyOfDryAirMassAtSurface,
        fvm_odm_trisk_hybridlev_builder.TendencyOfPotentialTemperatureDueToDivergenceOfFlux,
        fvm_odm_trisk_hybridlev_builder.TendencyOfPotentialTemperatureDueToVerticalAdvection,
        fvm_odm_trisk_hybridlev_builder.NonlinearCoriolisTermAtDualEdge,
        fvm_odm_trisk_hybridlev_builder.GradientOfKineticEnergyAtDualEdge,
        fvm_odm_trisk_hybridlev_builder.TendencyOfNormalVelocityDueToVerticalAdvection,
        fvm_odm_trisk_hybridlev_builder.TendencyOfNormalVelocityDueToPressureGradientForce,
        fvm_odm_trisk_aux.AuxiliaryTerms,
        FiniteVolumeMethod,
    ]
