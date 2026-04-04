from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate as HybridVertCoordinate
    from terraflow.vert_coord.hybrid_coord_interp import interp2plev as hybrid_interp2plev
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate as OneLayerVertCoordinate


_LAZY_SYMBOLS = {
    "OneLayerVertCoordinate": "terraflow.vert_coord.onelayer_coord.OneLayerVertCoordinate",
    "HybridVertCoordinate": "terraflow.vert_coord.hybrid_coord.HybridVertCoordinate",
    "hybrid_interp2plev": "terraflow.vert_coord.hybrid_coord_interp.interp2plev",
}

__all__ = [
    "OneLayerVertCoordinate",
    "HybridVertCoordinate",
    "hybrid_interp2plev"
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SYMBOLS:
        module_path, symbol_name = _LAZY_SYMBOLS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        symbol = getattr(module, symbol_name)
        globals()[name] = symbol
        return symbol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
