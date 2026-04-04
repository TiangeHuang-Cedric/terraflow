from __future__ import annotations
from typing import TYPE_CHECKING, Any
import importlib

if TYPE_CHECKING:
    from terraflow.util.history.abstract import UGRIDHistoryset as UGRIDHistoryset
    from terraflow.util.history.abstract import UnstructuredHistoryset as UnstructuredHistoryset


_LAZY_CLASS = {
    "UGRIDHistoryset": "terraflow.util.history.abstract.UGRIDHistoryset",
    "UnstructuredHistoryset": "terraflow.util.history.abstract.UnstructuredHistoryset",
}

__all__ = [
    "UGRIDHistoryset",
    "UnstructuredHistoryset"
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
