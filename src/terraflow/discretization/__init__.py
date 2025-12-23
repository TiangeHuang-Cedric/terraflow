from __future__ import annotations
from typing import TypeVar, Generic, Callable, Any
from typing import overload, TYPE_CHECKING, cast

from terraflow.manager.registry import navigate

from .abstract import AbstractDiagnosticTools
from .abstract import AbstractArrayBasedVariableSet
from .abstract import AbstractQuantityBasedVariableSet
from .abstract import AbstractFieldBasedVariableSet
from .abstract import AbstractVariableSet
from .abstract import AbstractMomentumEquForm
from .abstract import AbstractContinuityEquForm
from .abstract import AbstractEquationTerm
from .abstract import AbstractEquationForm
from .abstract import AbstractEquationSet
from .abstract import DiscretizationInfo

from . import equation
from . import term

if TYPE_CHECKING:
    from terraflow.discretization.equation.shallow_water.base import ExVif_SWE_ODM_TriSKCGrid_FVM


DIAG = TypeVar("DIAG", bound=AbstractDiagnosticTools)


class DiagnosticToolsBuilder(Generic[DIAG]):

    @classmethod
    def select(cls, equation: AbstractEquationSet) -> Callable[..., DIAG]:
        disc = DiscretizationInfo.from_model(
            coord=equation.coord,
            para=equation.para,
            operator=equation.operator,
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
        equation: ExVif_SWE_ODM_TriSKCGrid_FVM,
        jit: bool = False,
        trajectory: bool = False,
    ) -> "term.fvm.odm.diag.DiagnosticsTools":
        ...

    @overload
    @classmethod
    def from_equation(
        cls,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
    ) -> AbstractDiagnosticTools:
        ...

    @classmethod
    def from_equation(
        cls,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
    ) -> AbstractDiagnosticTools:
        diag_factory = cls.select(equation)
        return diag_factory(equation, jit=jit, trajectory=trajectory)


__all__ = [
    "DiscretizationInfo",
    "AbstractEquationSet",
    "AbstractEquationForm",
    "AbstractEquationTerm",
    "AbstractContinuityEquForm",
    "AbstractMomentumEquForm",
    "AbstractVariableSet",
    "AbstractFieldBasedVariableSet",
    "AbstractQuantityBasedVariableSet",
    "AbstractArrayBasedVariableSet",
    "AbstractDiagnosticTools",
    "DiagnosticToolsBuilder",
]
