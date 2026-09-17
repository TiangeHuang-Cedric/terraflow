from __future__ import annotations
from typing import Any

from terraflow.util.backend.abstract import AbstractBackend


def valid_backend() -> tuple[str, str]:
    return ("jax", "torch")


def __getattr__(name: str) -> Any:
    if name == "JaxBackend":
        from terraflow.util.backend.jax_backend import JaxBackend 
        return JaxBackend
    if name == "TorchBackend":
        from terraflow.util.backend.torch_backend import TorchBackend
        return TorchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractBackend",
    "valid_backend",
]
