from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import manager
    from . import discretization as discretization
    from . import dynamics as dynamics
    from . import layout as layout
    from . import mesh as mesh
    from . import operator as operator
    from . import planet as planet
    from . import solver as solver
    from . import test_case as test_case
    from . import util as util
    from . import vert_coord as vert_coord


_LAZY_SUBMODULES = {
    "manager": "terraflow.core.manager",
    "util": "terraflow.core.util",
    "planet": "terraflow.core.planet",
    "mesh": "terraflow.core.mesh",
    "vert_coord": "terraflow.core.vert_coord",
    "layout": "terraflow.core.layout",
    "operator": "terraflow.core.operator",
    "test_case": "terraflow.core.test_case",
    "dynamics": "terraflow.core.dynamics",
    "discretization": "terraflow.core.discretization",
    "solver": "terraflow.core.solver",
}

__all__ = [
    "manager",
    "util",
    "planet",
    "mesh",
    "vert_coord",
    "layout",
    "operator",
    "test_case",
    "dynamics",
    "discretization",
    "solver",
]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
