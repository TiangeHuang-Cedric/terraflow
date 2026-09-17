from __future__ import annotations
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


class SupportsProfileScope(Protocol):
    def profile_scope(self, name: str) -> AbstractContextManager[Any]:
        ...


def get_profile_scope(
    name: str,
    backend: SupportsProfileScope | None = None,
) -> AbstractContextManager[Any]:
    if backend is None:
        return nullcontext()
    return backend.profile_scope(name)


def profile_scope(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fun: Callable[P, R]) -> Callable[P, R]:
        @wraps(fun)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            backend = _infer_backend(args, kwargs)
            with get_profile_scope(name, backend):
                return fun(*args, **kwargs)

        return wrapper

    return decorator


def _infer_backend(args: tuple[Any, ...], kwargs: dict[str, Any]) -> SupportsProfileScope | None:
    block = kwargs.get("block")
    if block is None and args:
        block = args[0]
    dev = getattr(block, "dev", None)
    return getattr(dev, "backend", None)
