from __future__ import annotations

from typing import Any

from terraflow.util.backend.abstract import AbstractBackend


def valid_backend() -> tuple[str, str]:
    return ("jax", "torch")


def require_supported_backend(backend: str) -> str:
    backend_name = backend.lower()
    if backend_name not in valid_backend():
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: {', '.join(valid_backend())}"
        )
    return backend_name


def import_jax_backend():
    from terraflow.util.backend.jax_backend import JaxBackend

    return JaxBackend


def import_torch_backend():
    from terraflow.util.backend.torch_backend import TorchBackend

    return TorchBackend


def infer_backend_from_array(value: Any) -> AbstractBackend | None:
    try:
        import jax
    except ModuleNotFoundError:
        jax = None

    if jax is not None and isinstance(value, jax.Array):
        JaxBackend = import_jax_backend()
        device_ref = value.device
        if callable(device_ref):
            device_ref = device_ref()
        return JaxBackend(getattr(device_ref, "platform", "jax"), device_ref)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        TorchBackend = import_torch_backend()
        return TorchBackend(str(value.device), value.device)

    return None


def __getattr__(name: str) -> Any:
    if name == "JaxBackend":
        return import_jax_backend()
    if name == "TorchBackend":
        return import_torch_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractBackend",
    "JaxBackend",
    "TorchBackend",
    "import_jax_backend",
    "import_torch_backend",
    "infer_backend_from_array",
    "require_supported_backend",
    "valid_backend",
]
