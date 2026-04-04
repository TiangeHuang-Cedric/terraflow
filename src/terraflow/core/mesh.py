from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations


_LAZY_CLASS = {
    "SphericalCentroidVoronoiTessellations": "terraflow.mesh.scvt.construct.SphericalCentroidVoronoiTessellations",
}

__all__ = [
    "SphericalCentroidVoronoiTessellations",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_CLASS:
        module_path, class_name = _LAZY_CLASS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
