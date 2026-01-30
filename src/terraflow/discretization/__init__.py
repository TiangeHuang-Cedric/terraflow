from __future__ import annotations
from typing import TypeVar, Generic, Callable
from typing import overload, TYPE_CHECKING, cast

from terraflow.manager.registry import navigate
from terraflow.planet import PlanetParameterSet
from terraflow.layout.abstract import CoordinateSystem

from .abstract import (
    AbstractDiagnosticTools,
    AbstractBlock,
    AbstractFields,
    AbstractVariableSet,
    AbstractMomentumEquForm,
    AbstractContinuityEquForm,
    AbstractThermodynamicEquForm,
    AbstractEquationTerm,
    AbstractEquationForm,
    AbstractEquationSet,
    AbstractEquationSetAuxiliaryTerms,
    DiscretizationInfo
)

from . import fvm

if TYPE_CHECKING:
    from terraflow.dynamics.shallow_water.types import ExVif_SWE_ODM_TriSKCGrid_FVM


DIAG = TypeVar("DIAG", bound=AbstractDiagnosticTools)


class DiagnosticTools(Generic[DIAG]):

    @classmethod
    def select(
        cls,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        equation: AbstractEquationSet,
    ) -> Callable[..., DIAG]:
        disc = DiscretizationInfo.from_model(
            coord=coord,
            para=para,
            operator=equation.method,
            equset_type=type(equation)
        )
        diag_factory = navigate(
            "diag",
            id_col="unique_id",
            id_val=disc.id(),
            attr="factory"
        )[0]
        return cast(Callable[..., DIAG], diag_factory)

    @overload
    @classmethod
    def from_equation(
        cls,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        equation: ExVif_SWE_ODM_TriSKCGrid_FVM,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> "fvm.odm.trisk.single_lev.diag.DiagnosticsTools":
        ...

    @overload
    @classmethod
    def from_equation(
        cls,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        ...

    @classmethod
    def from_equation(
        cls,
        coord: CoordinateSystem,
        para: PlanetParameterSet,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        diag_factory = cls.select(
            coord=coord, para=para, equation=equation
        )
        return diag_factory(equation, jit=jit, trajectory=trajectory, **kwargs)


__all__ = [
    "DiscretizationInfo",
    "AbstractEquationSetAuxiliaryTerms",
    "AbstractEquationSet",
    "AbstractEquationForm",
    "AbstractEquationTerm",
    "AbstractContinuityEquForm",
    "AbstractThermodynamicEquForm",
    "AbstractMomentumEquForm",
    "AbstractVariableSet",
    "AbstractBlock",
    "AbstractFields",
    "AbstractDiagnosticTools",
    "DiagnosticTools",
]
