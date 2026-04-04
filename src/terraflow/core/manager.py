from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.manager.registry import catalog, load_info, navigate


_LAZY_CLASS = {
    "catalog": "terraflow.manager.registry.catalog",
    "load_info": "terraflow.manager.registry.load_info",
    "navigate": "terraflow.manager.registry.navigate",
}


__all__ = [
    "catalog",
    "load_info",
    "navigate"
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_CLASS:
        module_path, method_name = _LAZY_CLASS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        method = getattr(module, method_name)
        globals()[name] = method
        return method
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
