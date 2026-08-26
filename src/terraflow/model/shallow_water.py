from __future__ import annotations
from typing import (
    Literal,
    overload,
    cast,
    TYPE_CHECKING,
)


from terraflow.model.abstract import ModelMode


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.planet import PlanetParameterSet
    from terraflow.model.abstract import AbstractModel
    from terraflow.model.config import UserConfig
    from terraflow.mesh.abstract import AbstractMesh
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.test_case.abstract import AbstractShallowWaterModelTestCase
    import terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.swm as fvm_scvt_trisk_gtc_model


@overload
def build_shallow_water_model(
    config: UserConfig,
    para:PlanetParameterSet,
    mesh: SCVTMesh,
    method: Literal["finite_volume_method"],
    form: Literal["vector_invariant"],
    staggered_kind: Literal["triskc"],
    test_case: AbstractShallowWaterModelTestCase,
    dev: AbstractDevice,
    restart: bool = False,
) -> fvm_scvt_trisk_gtc_model.ShallowWaterModel:
    ...


@overload
def build_shallow_water_model(
    config: UserConfig,
    para: PlanetParameterSet, 
    mesh: AbstractMesh,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractShallowWaterModelTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    **kwargs,
) -> AbstractModel:
    ...


def build_shallow_water_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractShallowWaterModelTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    **kwargs,
) -> AbstractModel:
    if mesh.alias_name() == "scvt" and method == "finite_volume_method":
        if staggered_kind == "triskc":
            if form == "vector_invariant":
                from terraflow.mesh.scvt.mesh_type import SCVTMesh
                from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.swm import ShallowWaterModel
                mesh = cast(SCVTMesh, mesh)
                return ShallowWaterModel(
                    mode=ModelMode.from_swm(restart=restart), 
                    cfg=config,
                    para=para,
                    mesh=mesh,
                    test_case=test_case,
                    dev=dev,
                )
            else:
                raise ValueError(
                    f"Equation form {form} not supported. Please check the avaliable options."
                )
        else:
            raise ValueError(f"Unsuppored layout {staggered_kind} for scvt mesh!")
    else:
        raise ValueError("Unsupported model!")