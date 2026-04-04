from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        DiagnosticToolsBuilder as DiagnosticTools,
    )

__all__ = [
    "DiagnosticTools"
]


def __getattr__(name: str) -> Any:
    if name == "DiagnosticTools":
        from terraflow.discretization.abstract import (
            DiagnosticToolsBuilder as DiagnosticToolsBuilder,
        )
        globals()[name] = DiagnosticToolsBuilder
        return DiagnosticToolsBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
