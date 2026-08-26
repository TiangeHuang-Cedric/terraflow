from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terraflow.test_case.adv.solid_rotation import SolidRotationTestCase
    from terraflow.test_case.adv.deform import (
        DeformationalFlowCase1,
        DeformationalFlowCase2,
        DeformationalFlowCase3,
        DeformationalFlowCase4
    )
    from terraflow.test_case.adv.dcmip12 import DCMIP12TestCase


_LAZY_CLASS = {
    "SolidRotationTestCase": "terraflow.test_case.adv.solid_rotation.SolidRotationTestCase",
    "DeformationalFlowCase1": "terraflow.test_case.adv.deform.DeformationalFlowCase1",
    "DeformationalFlowCase2": "terraflow.test_case.adv.deform.DeformationalFlowCase2",
    "DeformationalFlowCase3": "terraflow.test_case.adv.deform.DeformationalFlowCase3", 
    "DeformationalFlowCase4": "terraflow.test_case.adv.deform.DeformationalFlowCase4",
    "DCMIP12TestCase": "terraflow.test_case.adv.dcmip12.DCMIP12TestCase"
}

__all__ = [
    "SolidRotationTestCase",
    "DeformationalFlowCase1",
    "DeformationalFlowCase2",
    "DeformationalFlowCase3",
    "DeformationalFlowCase4",
    "DCMIP12TestCase" 
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
