from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self, Literal
from typing import overload, TYPE_CHECKING, TypeVar, Generic
import jax.numpy as jnp
import inspect

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.layout import AbstractLayout, CoordinateSystem
from terraflow.discretization.operator import AbstractDiscretizationMethod
from terraflow.solver.abstract import AbstractExplicitODE
import terraflow.discretization.abstract as disc
from terraflow.dynamics.shallow_water.vars.unstructure import StateArrays, StaticArrays, StaticFields

if TYPE_CHECKING:
    from terraflow.mesh import OrthogonalDualMesh
    from terraflow.vert_coord import OneLayerVertCoordinate
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    from terraflow.discretization.operator import FiniteVolumeMethod
    import terraflow.discretization.term.fvm.odm.trisk.single_lev.builder as fvm_odm_trisk_1lev_builder


HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
LO = TypeVar("LO", bound=AbstractLayout)
DM = TypeVar("DM", bound=AbstractDiscretizationMethod)
UHM = TypeVar("UHM", bound=AbstractUnstructuredMesh)

Array = typing.Array


class AbstractShallowWaterEquationSet(
    disc.AbstractEquationSet[HM, VM, LO, DM],
    Generic[HM, VM, LO, DM],
    ABC
):
    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ):
        super().__init__(disc, coord, para, method)

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the dynamics set."""
        return "swe"

    @abstractmethod
    def init_continuity_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ) -> disc.AbstractContinuityEquForm:
        pass

    @abstractmethod
    def init_momentum_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ) -> disc.AbstractMomentumEquForm:
        pass


ShallowWaterEquationSet = TypeVar(
    "ShallowWaterEquationSet",
    bound=AbstractShallowWaterEquationSet
)


class DivergenceOfMassFluxProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, DM],
    Generic[HM, VM, LO, DM]
):
    pass


DMF = TypeVar("DMF", bound=DivergenceOfMassFluxProtocol)


class NonlinearCoriolisTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, DM],
    Generic[HM, VM, LO, DM]
):
    pass


NCT = TypeVar("NCT", bound=NonlinearCoriolisTermProtocol)


class GradientOfKineticEnergyProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, DM],
    Generic[HM, VM, LO, DM]
):
    pass


GKE = TypeVar("GKE", bound=GradientOfKineticEnergyProtocol)


class PressureGradientForceTermProtocol(
    disc.AbstractEquationTerm[HM, VM, LO, DM],
    Generic[HM, VM, LO, DM]
):
    pass


PGF = TypeVar("PGF", bound=PressureGradientForceTermProtocol)


class ExContinuityEqu(
    disc.AbstractContinuityEquForm[HM, VM, LO, DM],
    Generic[DMF, HM, VM, LO, DM],
    ABC
):
    div_mfx: DMF

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[HM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ):
        super().__init__(disc, coord, para, method)
        self.div_mfx = self.init_div_thickness()

    def __call__(self, state: StateArrays) -> Array:
        return - self.div_mfx(state)

    @staticmethod
    def alias_name() -> str:
        return "continuity_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.div_mfx.build_default()
            self._built = True
        return self

    def init_div_thickness(self) -> DMF:
        factory = self.navi2term(
            group="div_of_thickness",
            choice="builder"
        )
        origin = getattr(factory, '__origin__', factory)
        if inspect.isclass(origin) and issubclass(origin, DivergenceOfMassFluxProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self.method
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of DivergenceOfThicknessTermProtocol."
            )


class ExNormalMomentumEqu4UGRID(
    disc.AbstractMomentumEquForm[UHM, VM, LO, DM],
    Generic[NCT, GKE, PGF, UHM, VM, LO, DM],
    ABC
):
    coriolis: NCT
    grad_ke: GKE
    pgf: PGF

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ) -> None:
        super().__init__(disc, coord, para, method)
        self.coriolis = self.init_nonlinear_coriolis_term()
        self.grad_ke = self.init_grad_kinetic_energy_term()
        self.pgf = self.init_grad_geopotential_term()

    def __call__(
        self,
        state: StateArrays,
        static: StaticArrays
    ) -> Array:
        return (
            - self.coriolis(state)
            - self.grad_ke(state)
            - self.pgf(state, static)
        )

    @staticmethod
    def alias_name() -> str:
        return "normal_momentum_equ"

    @staticmethod
    def ode_kind_alias_name() -> str:
        return "explicit"

    def build_default(self) -> Self:
        if not self.has_built():
            self.coriolis.build_default()
            self.grad_ke.build_default()
            self.pgf.build_default()
            self._built = True
        return self

    def init_nonlinear_coriolis_term(self) -> NCT:
        factory = self.navi2term(
            group="nonlinear_coriolis",
            choice="builder"
        )
        if inspect.isclass(factory) and issubclass(factory, NonlinearCoriolisTermProtocol):
            instance = factory(
                self._disc, self.coord, self.para, self.method
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
                self._disc, self.coord, self.para, self._method
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
                self._disc, self.coord, self.para, self._method
            )
            return instance  # type: ignore[return-value]
        else:
            raise TypeError(
                f"The factory {factory} is not a subclass of GradientOfGeopotentialTermProtocol."
            )


class ExVifShallowWaterEquationSet4UGRID(
    AbstractShallowWaterEquationSet[UHM, VM, LO, DM],
    Generic[
        DMF,
        NCT, GKE, PGF,
        UHM, VM, LO, DM
    ],
    AbstractExplicitODE,
    ABC
):
    dhdt: ExContinuityEqu[DMF, UHM, VM, LO, DM]
    dundt: ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, UHM, VM, LO, DM
    ]
    static: StaticArrays

    def __init__(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
        static: StaticArrays,
    ):
        super().__init__(disc, coord, para, method)
        self.dhdt = self.init_continuity_equation(disc, coord, para, method)
        self.dundt = self.init_momentum_equation(disc, coord, para, method)
        self.static = static

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
        method: DM,
    ) -> ExContinuityEqu[DMF, HM, VM, LO, DM]:
        return ExContinuityEqu(disc, coord, para, method)

    def init_momentum_equation(
        self,
        disc: disc.DiscretizationInfo,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
    ) -> ExNormalMomentumEqu4UGRID[
        NCT, GKE, PGF, UHM, VM, LO, DM
    ]:
        return ExNormalMomentumEqu4UGRID(disc, coord, para, method)

    def explicit_terms(self, state: StateArrays, pass_tag: str = "all_pass") -> StateArrays:
        if pass_tag == "all_pass":
            variables = {
                "h_pc": self.dhdt(state),
                "un_e": self.dundt(state, self.static)
            }
        elif pass_tag == "forward_pass":
            variables = {
                "h_pc": self.dhdt(state),
                "un_e": jnp.zeros_like(state.variables["un_e"])
            }
        elif pass_tag == "backward_pass":
            variables = {
                "h_pc": jnp.zeros_like(state.variables["h_pc"]),
                "un_e": self.dundt(state, self.static)
            }
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return StateArrays(
            variables=variables,
            meta=state.meta
        )

    @classmethod
    def from_model(
        cls,
        coord: CoordinateSystem[UHM, VM, LO],
        para: PlanetParameterSet,
        method: DM,
        static: StaticFields,
    ) -> ExVifShallowWaterEquationSet4UGRID:
        static_array = static.to_arrays()
        return cls._from_model(
            coord, para, method, static_array
        )


class ShallowWaterEquationBuilder:
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
        method: FiniteVolumeMethod,
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
        coord: CoordinateSystem[HM, VM, LO],
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
        fvm_odm_trisk_1lev_builder.DivergenceOfMassFluxAtPrimalCell,
        fvm_odm_trisk_1lev_builder.NonlinearCoriolisTermAtDualEdge,
        fvm_odm_trisk_1lev_builder.GradientOfKineticEnergyAtDualEdge,
        fvm_odm_trisk_1lev_builder.PressureGradientForceAtDualEdge,
        OrthogonalDualMesh,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        FiniteVolumeMethod
    ]
