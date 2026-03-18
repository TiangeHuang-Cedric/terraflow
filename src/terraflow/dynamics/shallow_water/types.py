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
    Generic,
)
import inspect
from loguru import logger

from terraflow.util import typing
from terraflow.test_case.swe import ShallowWaterEquationTestCase
from terraflow.planet import PlanetParameterSet
from terraflow.operator import AbstractDiscretizationMethod
from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc_types

if TYPE_CHECKING:
    from terraflow.operator import FiniteVolumeMethod
    from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    import terraflow.discretization.fvm.odm.trisk.dynamics_types as fvm_odm_trisk_vars
    import terraflow.discretization.fvm.odm.trisk.single_lev.term as fvm_odm_trisk_1lev_terms

    Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)
BLOCK = TypeVar("BLOCK", bound=disc_types.AbstractBlock)
STATE = TypeVar("STATE", bound=disc_types.AbstractFields)
PSO = TypeVar("PSO", bound=disc_types.AbstractPrepareSpaceOperators)
SOP = TypeVar("SOP", bound=disc_types.AbstractSpaceOperators)


class ShallowWaterEquationMode(disc_types.AbstractEquationMode):
    def __init__(self, **kwargs):
        super().__init__(
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False,
            **kwargs
        )


class AbstractShallowWaterEquationSet(
    disc_types.AbstractEquationSet[DM, ShallowWaterEquationMode, SOP],
    Generic[DM, SOP],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=ShallowWaterEquationMode(**kwargs)
        )
        factory = self.navi2space_ops()
        space_operators: SOP = factory(
            disc=disc,
            method=method,
            mode=self.mode
        )
        self._space_operators = space_operators

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "swe"


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class AbstractShallowWaterEquationForm(
    disc_types.AbstractEquationForm[DM, ShallowWaterEquationMode],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterTerm(
    disc_types.AbstractEquationTerm[DM, ShallowWaterEquationMode],
    Generic[DM],
    ABC
):
    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )


class AbstractShallowWaterSpaceOperators(
    disc_types.AbstractSpaceOperators[DM, ShallowWaterEquationMode, PSO],
    Generic[DM, PSO],
    ABC
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )

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
    disc_types.AbstractContinuityEquForm[DM, ShallowWaterEquationMode],
    Generic[DMF, DM],
    ABC
):
    dmf: DMF

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.dmf = self.init_div_mfx()

    def __call__(self, block: BLOCK, star_state: STATE) -> tuple[BLOCK, STATE]:
        block, star_state = self.dmf(block, star_state)
        return block, star_state

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

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmf.prepare_block(block)
        return block

    def init_div_mfx(self) -> DMF:
        factory = self.navi2term(
            group="div_of_mfx",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfMassFluxProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    AbstractShallowWaterEquationForm[DM],
    disc_types.AbstractMomentumEquForm[DM, ShallowWaterEquationMode],
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
        mode: ShallowWaterEquationMode
    ) -> None:
        super().__init__(disc=disc, method=method, mode=mode)
        self.coriolis = self.init_nonlinear_coriolis_term()
        self.grad_ke = self.init_grad_kinetic_energy_term()
        self.pgf = self.init_grad_geopotential_term()

    def __call__(self, block: BLOCK, star_state: STATE) -> tuple[BLOCK, STATE]:
        block, star_state = self.coriolis(block, star_state)
        block, star_state = self.grad_ke(block, star_state)
        block, star_state = self.pgf(block, star_state)
        return block, star_state

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

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.coriolis.prepare_block(block)
        block = self.grad_ke.prepare_block(block)
        block = self.pgf.prepare_block(block)
        return block

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
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
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
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
            instance = factory(
                disc=self._disc, method=self.method, mode=self.mode
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRIDSpaceOperators(
    AbstractShallowWaterSpaceOperators[DM, PSO],
    Generic[
        DMF,
        NCT, GKE, PGF,
        DM, PSO
    ],
    ABC
):
    dgzdt: ExContinuityEqu[DMF, DM]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        mode: ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.dgzdt = self.init_continuity_equation(disc=disc, method=method)
        self.dundt = self.init_momentum_equation(disc=disc, method=method)

    def init_continuity_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExContinuityEqu[DMF, DM]:
        return ExContinuityEqu(disc=disc, method=method, mode=self.mode)

    def init_momentum_equation(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, DM
    ]:
        return ExNormalMomentumEqu4UGRID(disc=disc, method=method, mode=self.mode)

    def build_default(self) -> Self:
        if not self.has_built():
            self.prepare.build_default()
            self.dgzdt.build_default()
            self.dundt.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.prepare.prepare_block(block)
        block = self.dgzdt.prepare_block(block)
        block = self.dundt.prepare_block(block)
        logger.success("Prepared auxiliary and data.")
        return block


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[DM, SOP],
    AbstractExplicitODE,
    Generic[DM, SOP],
):

    def __init__(
        self,
        disc: disc_types.DiscretizationInfo,
        method: DM,
        **kwargs,
    ):
        super().__init__(disc=disc, method=method, **kwargs)

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
        from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
        from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
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

    def build_default(self) -> Self:
        if not self.has_built():
            self._space_operators.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self._space_operators.prepare_block(block)
        logger.success("Prepared auxiliary and data.")
        return block

    def explicit_terms(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        return self.space_operators(
            block,
            *args,
            **kwargs,
            explicit=True
        )


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
        output_h1_tend: bool = False,
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
        FiniteVolumeMethod,
        fvm_odm_trisk_1lev_terms.ExSpaceOperators,
    ]
