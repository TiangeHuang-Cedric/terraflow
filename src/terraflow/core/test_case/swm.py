from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.test_case.swm.mountain_zonal_flow import MountainZonalFlowTestCase
    from terraflow.test_case.swm.rossby_haurwitz_wave import RossbyHaurwitzWaveTestCase
    from terraflow.test_case.swm.jet_zonal_flow import JetZonalFlowTestCase


_LAZY_CLASS = {
    "MountainZonalFlowTestCase": "terraflow.test_case.swm.mountain_zonal_flow.MountainZonalFlowTestCase",
    "RossbyHaurwitzWaveTestCase": "terraflow.test_case.swm.rossby_haurwitz_wave.RossbyHaurwitzWaveTestCase",
    "JetZonalFlowTestCase": "terraflow.test_case.swm.jet_zonal_flow.JetZonalFlowTestCase"
}

__all__ = [
    "MountainZonalFlowTestCase",
    "RossbyHaurwitzWaveTestCase",
    "JetZonalFlowTestCase"
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
