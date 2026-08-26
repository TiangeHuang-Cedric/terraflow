from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.test_case.mountain_wave import MountainWaveTestCase
    from terraflow.test_case.rossby_haurwitz_wave_3d import RossbyHaurwitzWaveTestCase
    from terraflow.test_case.steady_state import SteadyStateTestCase


_LAZY_CLASS = {
    "MountainWaveTestCase": "terraflow.test_case.mountain_wave.MountainWaveTestCase",
    "RossbyHaurwitzWaveTestCase": "terraflow.test_case.rossby_haurwitz_wave_3d.RossbyHaurwitzWaveTestCase",
    "SteadyStateTestCase": "terraflow.test_case.steady_state.SteadyStateTestCase",
}

__all__ = [
    "MountainWaveTestCase",
    "RossbyHaurwitzWaveTestCase",
    "SteadyStateTestCase",
]


def __getattr__(name: str):
    if name in _LAZY_CLASS:
        module_path, class_name = _LAZY_CLASS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
