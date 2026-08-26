from __future__ import annotations
from typing import (
    Literal,
    overload,
    cast,
    TYPE_CHECKING,
)


from terraflow.model.abstract import ModelMode
from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate


if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.util.device import AbstractDevice
    from terraflow.model.config import UserConfig
    from terraflow.model.abstract import AbstractModel
    from terraflow.mesh.abstract import AbstractMesh
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractVertCoordinate
    from terraflow.test_case.abstract import (
        AbstractAdvectionModel2DTestCase,
        AbstractAdvectionModel3DTestCase
    )
    import terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.adv as fvm_scvt_trisk_gtc_model


@overload
def build_adv_2d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    method: Literal["finite_volume_method"],
    form: Literal["vector_invariant"],
    staggered_kind: Literal["triskc"],
    test_case: AbstractAdvectionModel2DTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    *args,
    **kwargs,
) -> fvm_scvt_trisk_gtc_model.Adv2DModel:
    ...


@overload
def build_adv_2d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractAdvectionModel2DTestCase, 
    dev: AbstractDevice,
    restart: bool = False,
    *args,
    **kwargs,
) -> AbstractModel:
    ...


def build_adv_2d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractAdvectionModel2DTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    *args,
    **kwargs,
) -> AbstractModel:
    if mesh.alias_name() == "scvt" and method == "finite_volume_method":
        if staggered_kind == "triskc":
            if form == "vector_invariant":
                from terraflow.mesh.scvt.mesh_type import SCVTMesh
                from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.adv import Adv2DModel
                mesh = cast(SCVTMesh, mesh)
                return Adv2DModel(
                    mode=ModelMode.from_adv2d(restart=restart), 
                    cfg=config,
                    para=para,
                    mesh=mesh,
                    test_case=test_case,
                    dev=dev
                )
            else:
                raise ValueError(
                    f"Equation form {form} not supported. Please check the avaliable options."
                )
        else:
            raise ValueError(f"Unsuppored layout {staggered_kind} for scvt mesh!")
    else:
        raise ValueError("Unsupported model!")


@overload
def build_adv_3d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
    method: Literal["finite_volume_method"],
    form: Literal["vector_invariant"],
    staggered_kind: Literal["triskc"],
    test_case: AbstractAdvectionModel3DTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    *args,
    **kwargs,
) -> fvm_scvt_trisk_gtc_model.Adv3DModel:
    ...


@overload
def build_adv_3d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    vert_coord: AbstractVertCoordinate,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractAdvectionModel3DTestCase,
    dev: AbstractDevice,
    restart: bool = False, 
    *args,
    **kwargs,
) -> AbstractModel:
    ...


def build_adv_3d_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    vert_coord: AbstractVertCoordinate,
    method: str,
    form: str,
    staggered_kind: str,
    test_case: AbstractAdvectionModel3DTestCase,
    dev: AbstractDevice,
    restart: bool = False,
    *args,
    **kwargs,
) -> AbstractModel:
    if (
        mesh.alias_name() == "scvt" and 
        isinstance(vert_coord, AbstractGeneralizedTerrainFollowingCoordinate) and 
        method == "finite_volume_method"
    ):
        if staggered_kind == "triskc":
            if form == "vector_invariant":
                from terraflow.mesh.scvt.mesh_type import SCVTMesh
                from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.adv import Adv3DModel
                mesh = cast(SCVTMesh, mesh)
                if mesh.full_nlev == 1 or mesh.half_nlev == 2:
                    raise ValueError(
                        f"SCVTMesh has been initialized with one single layer, but vert_coord input has {vert_coord.full_lev} levels!"
                    )
                vert_coord = cast(AbstractGeneralizedTerrainFollowingCoordinate, vert_coord)
                return Adv3DModel(
                    mode=ModelMode.from_adv3d(restart=restart), 
                    cfg=config,
                    para=para,
                    mesh=mesh,
                    vert_coord=vert_coord,
                    test_case=test_case,
                    dev=dev
                )
            else:
                raise ValueError(
                    f"Equation form {form} not supported. Please check the avaliable options."
                )           
        else:
            raise ValueError(f"Unsuppored layout {staggered_kind} for scvt mesh!")
    else:
        raise ValueError("Unsupported model!")
