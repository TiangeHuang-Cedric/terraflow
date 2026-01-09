from __future__ import annotations
from typing import TypeVar, Generic, Callable
from typing import overload, TYPE_CHECKING, cast

from terraflow.manager.registry import navigate

from .abstract import AbstractOperator
from .abstract import AbstractDiscretizationMethod
from .abstract import FiniteVolumeMethod
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

from . import fvm

if TYPE_CHECKING:
    from terraflow.dynamics.shallow_water.types import ExVif_SWE_ODM_TriSKCGrid_FVM


DIAG = TypeVar("DIAG", bound=AbstractDiagnosticTools)


class DiagnosticTools(Generic[DIAG]):

    @classmethod
    def select(cls, equation: AbstractEquationSet) -> Callable[..., DIAG]:
        disc = DiscretizationInfo.from_model(
            coord=equation.coord,
            para=equation.para,
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
        equation: ExVif_SWE_ODM_TriSKCGrid_FVM,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> "fvm.odm.trisk.single_lev.term.diag.DiagnosticsTools":
        ...

    @overload
    @classmethod
    def from_equation(
        cls,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        ...

    @classmethod
    def from_equation(
        cls,
        equation: AbstractEquationSet,
        jit: bool = False,
        trajectory: bool = False,
        **kwargs,
    ) -> AbstractDiagnosticTools:
        diag_factory = cls.select(equation)
        return diag_factory(equation, jit=jit, trajectory=trajectory, **kwargs)


__all__ = [
    "AbstractOperator",
    "AbstractDiscretizationMethod",
    "FiniteVolumeMethod",
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
    "DiagnosticTools",
]
