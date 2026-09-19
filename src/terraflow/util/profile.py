from __future__ import annotations
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from functools import wraps
import os
import sys
import time
from typing import Any, ParamSpec, Protocol, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


class SupportsProfileScope(Protocol):
    def profile_scope(self, name: str) -> AbstractContextManager[Any]:
        ...


@dataclass
class ProfileStat:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)


class TimerScope(AbstractContextManager[None]):
    def __init__(self, name: str):
        self.name = name
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        record(self.name, time.perf_counter() - self._start)
        return None


_STATS: dict[str, ProfileStat] = {}


def enabled() -> bool:
    return os.environ.get("ENABLE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}


def record(name: str, elapsed_s: float) -> None:
    if not enabled():
        return
    stat = _STATS.get(name)
    if stat is None:
        stat = ProfileStat()
        _STATS[name] = stat
    stat.add(elapsed_s)


def timed(name: str) -> AbstractContextManager[Any]:
    if not enabled() or is_torch_compiling():
        return nullcontext()
    return TimerScope(name)


def reset() -> None:
    _STATS.clear()


def snapshot() -> dict[str, ProfileStat]:
    return {
        name: ProfileStat(
            count=stat.count,
            total_s=stat.total_s,
            max_s=stat.max_s,
        )
        for name, stat in _STATS.items()
    }


def get_profile_scope(
    name: str,
    backend: SupportsProfileScope | None = None,
) -> AbstractContextManager[Any]:
    stack = ExitStack()
    stack.enter_context(timed(name))
    if backend is not None:
        stack.enter_context(backend.profile_scope(name))
    return stack


def summary_lines(
    *,
    comm: Any | None = None,
    top: int = 30,
) -> list[str]:
    local_stats = snapshot()
    rank_stats = gather_stats(local_stats, comm)
    if rank_stats is None:
        return []

    names = sorted({name for stats in rank_stats for name in stats})
    rows = []
    for name in names:
        counts = [stats.get(name, ProfileStat()).count for stats in rank_stats]
        totals = [stats.get(name, ProfileStat()).total_s for stats in rank_stats]
        maxes = [stats.get(name, ProfileStat()).max_s for stats in rank_stats]
        rows.append(
            (
                max(totals),
                sum(totals) / len(totals),
                min(totals),
                max(counts),
                max(maxes),
                name,
            )
        )

    rows.sort(reverse=True)
    if top > 0:
        rows = rows[:top]

    lines = [
        "TerraFlow timing summary (seconds, aggregated across ranks)",
        "rank_max_total  rank_mean_total  rank_min_total  max_count  max_call  name",
    ]
    lines.extend(
        (
            f"{rank_max_total:14.6f}  {rank_mean_total:15.6f}  "
            f"{rank_min_total:14.6f}  {max_count:9d}  {max_call:8.6f}  {name}"
        )
        for (
            rank_max_total,
            rank_mean_total,
            rank_min_total,
            max_count,
            max_call,
            name,
        ) in rows
    )
    return lines


def print_summary(
    *,
    comm: Any | None = None,
    top: int = 30,
    logger: Any | None = None,
) -> None:
    lines = summary_lines(comm=comm, top=top)
    if not lines:
        return
    message = "\n".join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def gather_stats(
    local_stats: dict[str, ProfileStat],
    comm: Any | None,
) -> list[dict[str, ProfileStat]] | None:
    if comm is None:
        return [local_stats]
    if getattr(comm, "size", 1) <= 1:
        return [local_stats]
    if hasattr(comm, "gather"):
        return comm.gather(local_stats, root=0)
    return [local_stats]


def is_torch_compiling() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is None:
        return False
    try:
        return bool(is_compiling())
    except Exception:
        return False


def profile_scope(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fun: Callable[P, R]) -> Callable[P, R]:
        @wraps(fun)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            backend = infer_backend(args, kwargs)
            with get_profile_scope(name, backend):
                return fun(*args, **kwargs)

        return wrapper

    return decorator


def infer_backend(args: tuple[Any, ...], kwargs: dict[str, Any]) -> SupportsProfileScope | None:
    block = kwargs.get("block")
    if block is None and args:
        block = args[0]
    dev = getattr(block, "dev", None)
    return getattr(dev, "backend", None)
