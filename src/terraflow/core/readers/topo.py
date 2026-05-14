from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.prepare.readers.topo import ETOPO1Reader, GMETED2010Reader


_LAZY_CLASS = {
    "ETOPO1Reader": "terraflow.prepare.readers.topo.ETOPO1Reader",
    "GMETED2010Reader": "terraflow.prepare.readers.topo.GMETED2010Reader",
}

__all__ = [
    "ETOPO1Reader",
    "GMETED2010Reader"
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
