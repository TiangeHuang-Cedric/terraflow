from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from terraflow.model.abstract import compiled_with_model, block_until_ready
    from terraflow.model.config import UserConfig
    from terraflow.model.shallow_water import build_shallow_water_model
    from terraflow.model.hydrostatic import build_hydrostatic_model
    from terraflow.model.adv import build_adv_2d_model
    from terraflow.model.adv import build_adv_3d_model


_LAZY_CLASS = {
    "block_until_ready": "terraflow.model.abstract.block_until_ready",
    "compiled_with_model": "terraflow.model.abstract.compiled_with_model",
    "UserConfig": "terraflow.model.config.UserConfig",
    "build_shallow_water_model": "terraflow.model.shallow_water.build_shallow_water_model",
    "build_hydrostatic_model": "terraflow.model.hydrostatic.build_hydrostatic_model",
    "build_adv_2d_model": "terraflow.model.adv.build_adv_2d_model",
    "build_adv_3d_model": "terraflow.model.adv.build_adv_3d_model"
}


__all__ = [
    "block_until_ready",
    "compiled_with_model",
    "UserConfig",
    "build_shallow_water_model",
    "build_hydrostatic_model",
    "build_adv_2d_model",
    "build_adv_3d_model",
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
