from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self, Literal
from typing import overload, TYPE_CHECKING, TypeVar, Generic
import inspect

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coordinates import AbstractVertCoordinate
from terraflow.layout import AbstractLayout, CoordinateSystem
from terraflow.operator import AbstractOperator
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc
from terraflow.discretization.equation.shallow_water.vars.unstructure import StateArrays, StaticArrays, StaticFields

if TYPE_CHECKING:
    from terraflow.mesh import OrthogonalDualMesh
    from terraflow.vert_coordinates import OneLayerVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    from terraflow.operator import FiniteVolumeMethodOperator
    from terraflow.discretization.term.fvm.odm.builder import (
        DivergenceOfThicknessFluxAtPrimalCell,
        NonlinearCoriolisTermAtDualEdge,
        GradientOfGeopotentialAtDualEdge,
        GradientOfKineticEnergyAtDualEdge
    )


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)
OP = TypeVar("OP", bound=AbstractOperator)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)

Array = typing.Array


class AbstractShallowWaterEquationSet(
    disc.AbstractEquationSet[HM, VM, LO, OP],
    Generic[HM, VM, LO, OP],
    ABC
):
    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ):
        super().__init__(disc, coord, para, operator)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the equation set."""
        return "swe"

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ) -> disc.AbstractContinuityEquForm:
        pass

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ) -> disc.AbstractMomentumEquForm:
        pass


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class DivergenceOfThicknessTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, OP],
    Generic[HM, VM, LO, OP]
):
    pass


DTK = TypeVar("DTK", bound=DivergenceOfThicknessTermProtocol)


class NonlinearCoriolisTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, OP],
    Generic[HM, VM, LO, OP]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, OP],
    Generic[HM, VM, LO, OP]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyTermProtocol)


class GradientOfGeopotentialTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, OP],
    Generic[HM, VM, LO, OP]
):
    pass


GPHI = TypeVar("GPHI", bound=GradientOfGeopotentialTermProtocol)


class ExContinuityEqu(
    disc.AbstractContinuityEquForm[HM, VM, LO, OP],
    Generic[DTK, HM, VM, LO, OP],
    ABC
):
    div_thickness_term: DTK

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ):
        super().__init__(disc, coord, para, operator)
        self.div_thickness_term = self.init_div_thickness()

    def __call__(self, state: StateArrays) -> Array:
        return - self.div_thickness_term(state)

    @staticmethod
    def alias_name() -> str:
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.div_thickness_term.build_default()
            self._built = True
        return self

    def init_div_thickness(self) -> DTK:
        factory = self.navi2term(
            group="div_of_thickness",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfThicknessTermProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self.operator
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfThicknessTermProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    disc.AbstractMomentumEquForm[UHM, VM, LO, OP],
    Generic[NCT, GKE, GPHI, UHM, VM, LO, OP],
    ABC
):
    nonlinear_coriolis_term: NCT
    grad_kinetic_energy_term: GKE
    grad_geopotential_term: GPHI

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ) -> None:
        super().__init__(disc, coord, para, operator)
        self.nonlinear_coriolis_term = self.init_nonlinear_coriolis_term()
        self.grad_kinetic_energy_term = self.init_grad_kinetic_energy_term()
        self.grad_geopotential_term = self.init_grad_geopotential_term()

    def __call__(
        self,
        state: StateArrays,
        static: StaticArrays
    ) -> Array:
        return (
            - self.nonlinear_coriolis_term(state)
            - self.grad_kinetic_energy_term(state)
            - self.grad_geopotential_term(state, static)
        )

    @staticmethod
    def alias_name() -> str:
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.nonlinear_coriolis_term.build_default()
            self.grad_kinetic_energy_term.build_default()
            self.grad_geopotential_term.build_default()
            self._built = True
        return self

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self.operator
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
        if inspect.isclass(factory) and issubclass(factory, GradientOfKineticEnergyTermProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self._operator
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfKineticEnergyTermProtocol."
            )

    def init_grad_geopotential_term(self) -> GPHI:
        factory = self.navi2term(
            group="grad_of_geopotential",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, GradientOfGeopotentialTermProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self._operator
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[UHM, VM, LO, OP],
    Generic[
        DTK,
        NCT, GKE, GPHI,
        UHM, VM, LO, OP
    ],
    AbstractExplicitODE,
    ABC
):
    dhdt: ExContinuityEqu[DTK, UHM, VM, LO, OP]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, GPHI, UHM, VM, LO, OP
    ]
    static: StaticArrays

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
        static: StaticArrays,
    ):
        super().__init__(disc, coord, para, operator)
        self.dhdt = self.init_continuity_equation(disc, coord, para, operator)
        self.dundt = self.init_momentum_equation(disc, coord, para, operator)
        self.static = static

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the equation set."""
        return AbstractShallowWaterEquationSet.alias_name()

    @staticmethod
    def form_name() -> str:
        """Return the form name of the equation set."""
        return "vif"

    @staticmethod
    def ode_kind_alias_name() -> str:
        """Return the ODE kind alias name of the equation set."""
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.dhdt.build_default()
            self.dundt.build_default()
            self._built = True
        return self

    def init_continuity_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ) -> ExContinuityEqu[DTK, HM, VM, LO, OP]:
        return ExContinuityEqu(disc, coord, para, operator)

    def init_momentum_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, GPHI, UHM, VM, LO, OP
    ]:
        return ExNormalMomentumEqu4UGRID(disc, coord, para, operator)

    def explicit_terms(self, state: StateArrays) -> StateArrays:
        variables = {
            "h_pc": self.dhdt(state),
            "un_e": self.dundt(state, self.static)
        }
        return StateArrays(
            variables=variables,
            meta=state.meta
        )

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        operator: OP,
        static: StaticFields,
    ) -> ExVifShallowWaterEquationSet4UGRID:
        static_array = static.to_arrays()
        return cls._from_model(
            coord, para, operator, static_array
        )


class ShallowWaterEquationSetBuilder:
    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            OneLayerVertCoordinate,
            TriSKBasedCGridStaggeringLayout
        ],
        para: PlanetParameterSet,
        operator: FiniteVolumeMethodOperator,
        form: Literal["vector_invariant"],
        ode_kind: Literal["explicit"],
        static: StaticFields,
    ) -> ExVif_SWE_ODM_TriSKCGrid_FVM:
        ...

    @overload
    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: AbstractOperator,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        ...

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        operator: AbstractOperator,
        form: str,
        ode_kind: str,
        *args,
        **kwargs,
    ) -> AbstractShallowWaterEquationSet:
        if (ode_kind == "explicit" and form == "vector_invariant"):
            return ExVifShallowWaterEquationSet4UGRID.from_model(
                coord,
                para,
                operator,
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
        DivergenceOfThicknessFluxAtPrimalCell,
        NonlinearCoriolisTermAtDualEdge,
        GradientOfKineticEnergyAtDualEdge,
        GradientOfGeopotentialAtDualEdge,
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        FiniteVolumeMethodOperator
    ]
