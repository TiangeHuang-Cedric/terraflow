from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    cast,
    Optional,
    Literal,
    Self,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Generic
)
import inspect
import equinox as eqx
from loguru import logger

from terraflow.util import typing
from terraflow.test_case.swe import ShallowWaterEquationTestCase
from terraflow.planet import PlanetParameterSet
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.mesh import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout import AbstractLayout, CoordinateSystem
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh import OrthogonalDualMesh
    from terraflow.vert_coord import OneLayerVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import terraflow.discretization.fvm.odm.trisk.dynamics_types as fvm_odm_trisk_vars
    import terraflow.discretization.fvm.odm.trisk.single_lev.aux as fvm_odm_trisk_1lev_aux
    import terraflow.discretization.fvm.odm.trisk.single_lev.term as fvm_odm_trisk_1lev_builder


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
ESAUX = TypeVar("ESAUX", bound=disc_types.AbstractEquationSetAuxiliaryTerms)

Array = typing.Array


class AbstractShallowWaterEquationSet(
    disc_types.AbstractEquationSet[DM, ESAUX],
    Generic[DM, ESAUX],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False
        )

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "swe"

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> disc_types.AbstractContinuityEquForm:
        raise NotImplementedError

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM
    ) -> disc_types.AbstractMomentumEquForm:
        raise NotImplementedError


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class AbstractShallowWaterEquationForm(
    disc_types.AbstractEquationForm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False
        )


class AbstractShallowWaterTerm(
    disc_types.AbstractEquationTerm[DM],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False
        )


class DivergenceOfMassFluxProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class NonlinearCoriolisTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class PressureGradientForceTermProtocol(
    AbstractShallowWaterTerm[DM],
    Generic[DM]
):
    pass


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractContinuityEquForm[DM],
    Generic[DMF, DM],
    ABC
):
    dmf: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc=disc, method=method)
        self.dmf = self.init_div_mfx()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.dmf(block)
        dgzdt_mfn = self.dmf.unwrap(block).unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dgzdt.data,
            block,
            dgzdt_mfn
        )

    @staticmethod
    def alias_name() -> str:
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.dmf.build_default()
            self._built = True
        return self

    def prepare(self, block: BLOCK) -> BLOCK:
        block = self.dmf.prepare(block)
        return block

    def init_div_mfx(self) -> DMF:
        factory = self.navi2term(
            group="div_of_mfx",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxProtocol):
            instance = factory(disc=self._disc, method=self.method)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractMomentumEquForm[DM],
    Generic[NCT, GKE, PGF, DM],
    ABC
):
    coriolis: NCT
    grad_ke: GKE
    pgf: PGF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> None:
        super().__init__(disc=disc, method=method)
        self.coriolis = self.init_nonlinear_coriolis_term()
        self.grad_ke = self.init_grad_kinetic_energy_term()
        self.pgf = self.init_grad_geopotential_term()

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.coriolis(block)
        block = self.grad_ke(block)
        block = self.pgf(block)
        dundt_coriolis = self.coriolis.unwrap(block).unwrap()
        dundt_grad_ke = self.grad_ke.unwrap(block).unwrap()
        dundt_pgf = self.pgf.unwrap(block).unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dundt.data,
            block,
            replace=(
                dundt_coriolis
                + dundt_grad_ke
                + dundt_pgf
            )
        )

    @staticmethod
    def alias_name() -> str:
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self._built:
            self.coriolis.build_default()
            self.grad_ke.build_default()
            self.pgf.build_default()
            self._built = True
        return self

    def prepare(self, block: BLOCK) -> BLOCK:
        block = self.coriolis.prepare(block)
        block = self.grad_ke.prepare(block)
        block = self.pgf.prepare(block)
        return block

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(disc=self._disc, method=self.method)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of NonlinearCoriolisTermProtocol."
            )

    def init_grad_kinetic_energy_term(self) -> GKE:
        factory = self.navi2term(
            group="grad_of_kinetic_energy",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, GradientOfKineticEnergyProtocol):
            instance = factory(disc=self._disc, method=self.method)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfKineticEnergyTermProtocol."
            )

    def init_grad_geopotential_term(self) -> PGF:
        factory = self.navi2term(
            group="grad_of_geopotential",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, PressureGradientForceTermProtocol):
            instance = factory(disc=self._disc, method=self._method)
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[DM, ESAUX],
    AbstractExplicitODE,
    Generic[
        DMF,
        NCT, GKE, PGF,
        ESAUX,
        DM
    ],
):
    dgzdt: ExContinuityEqu[DMF, DM]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ):
        super().__init__(disc=disc, method=method)
        self.dgzdt = self.init_continuity_equation(disc=disc, method=method)
        self.dundt = self.init_momentum_equation(disc=disc, method=method)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return AbstractShallowWaterEquationSet.alias_name()

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
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            OneLayerVertCoordLayout
        ],
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None
    ) -> fvm_odm_trisk_vars.Block:
        ...

    @overload
    def build_block_from_testcase(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None
    ) -> disc_types.AbstractBlock:
        ...

    def build_block_from_testcase(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        *,
        case: Optional[ShallowWaterEquationTestCase] = None
    ) -> disc_types.AbstractBlock:
        from terraflow.operator import FiniteVolumeMethod
        from terraflow.mesh.orthogonal_dual_mesh import OrthogonalDualMesh
        from terraflow.vert_coord import OneLayerVertCoordinate
        from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
        from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
        if (
            isinstance(self.method, FiniteVolumeMethod) and
            isinstance(coord.mesh, OrthogonalDualMesh) and
            isinstance(coord.vert, OneLayerVertCoordinate) and
            isinstance(coord.hori_layout, TriSKBasedCGridStaggeringLayout) and
            isinstance(coord.vert_layout, OneLayerVertCoordLayout)
        ):
            from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block
            block_cls = Block
            coord_specialized = cast(
                CoordinateSystem[
                    OrthogonalDualMesh,
                    OneLayerVertCoordinate,
                    TriSKBasedCGridStaggeringLayout,
                    OneLayerVertCoordLayout
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

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExContinuityEqu[DMF, DM]:
        return ExContinuityEqu(disc=disc, method=method)

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]:
        return ExNormalMomentumEqu4UGRID(disc=disc, method=method)

    def build_default(self) -> Self:
        if not self.has_built():
            self.dgzdt.build_default()
            self.dundt.build_default()
            self._built = True
        return self

    def prepare(self, block: BLOCK) -> BLOCK:
        block = self.aux.prepare(block)
        block = self.dgzdt.prepare(block)
        block = self.dundt.prepare(block)
        logger.success("Prepared auxiliary terms and data.")
        return block

    def explicit_terms(self, block: BLOCK, pass_tag: str = "all_pass") -> BLOCK:
        block = self.aux(block)
        if pass_tag == "all_pass":
            block = self.dgzdt(block)
            block = self.dundt(block)
        elif pass_tag == "forward_pass":
            block = self.dgzdt(block)
        elif pass_tag == "backward_pass":
            block = self.dundt(block)
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return block


class ShallowWaterEquationBuilder:
    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            OneLayerVertCoordLayout
        ],
        para: PlanetParameterSet,
        method: FiniteVolumeMethod,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
    ) -> ExVif_SWE_ODM_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        para: PlanetParameterSet,
        method: AbstractDiscretizationMethod,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVifShallowWaterEquationSet4UGRID.from_model(
                coord,
                para,
                method,
                *args,
                **kwargs
            )
        else:
            raise ValueError(
                f"Kwargs ode_kind={ode_kind} and form={form} not supported."
                "Please check the avaliable options."
            )


if TYPE_CHECKING:
    ExVif_SWE_ODM_TriSKCGrid_FVM = ExVifShallowWaterEquationSet4UGRID[
        fvm_odm_trisk_1lev_builder.TendencyOfDivergenceOfMassFlux,
        fvm_odm_trisk_1lev_builder.NonlinearCoriolisTermAtDualEdge,
        fvm_odm_trisk_1lev_builder.GradientOfKineticEnergyAtDualEdge,
        fvm_odm_trisk_1lev_builder.PressureGradientForceAtDualEdge,
        fvm_odm_trisk_1lev_aux.AuxiliaryTerms,
        FiniteVolumeMethod,
    ]
