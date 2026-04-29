from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.solver.abstract import compiled_with_block
    from terraflow.solver.base import ExEulerSolver as ExEulerSolver
    from terraflow.solver.base import ExRungeKutta3Solver as ExRungeKutta3Solver


_LAZY_OBJ = {
    "compiled_with_block": "terraflow.solver.abstract.compiled_with_block",
    "ExEulerSolver": "terraflow.solver.base.ExEulerSolver",
    "ExRungeKutta3Solver": "terraflow.solver.base.ExRungeKutta3Solver",
}

__all__ = [
    "compiled_with_block",
    "ExEulerSolver",
    "ExRungeKutta3Solver"
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_OBJ:
        module_path, obj_name = _LAZY_OBJ[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
