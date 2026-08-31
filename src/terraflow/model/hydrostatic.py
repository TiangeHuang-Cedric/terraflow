from __future__ import annotations
from typing import TYPE_CHECKING, overload, cast, Optional, Literal

from terraflow.model.abstract import ModelMode
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate


if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.model.abstract import AbstractModel
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractVertCoordinate
    from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase
    from terraflow.prepare.readers.abstract import (
        AbstractBackgroundReader, 
        AbstractTopographyReader
    )
    import terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.baroclinic as fvm_scvt_trisk_gtc_model


@overload
def build_hydrostatic_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
    method: Literal["finite_volume_method"],
    form: Literal["vector_invariant"],
    staggered_kind: Literal["triskc"],
    dev: AbstractDevice,
    test_case: Optional[AbstractBaroclinicModelTestCase] = None,
    bkg_reader: Optional[AbstractBackgroundReader] = None,
    topo_reader: Optional[AbstractTopographyReader] = None,
    ideal_dry_core: bool = False,
    restart: bool = False
) -> fvm_scvt_trisk_gtc_model.BaroclinicModel:
    ...

@overload
def build_hydrostatic_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    vert_coord: AbstractVertCoordinate,
    method: str,
    form: str,
    staggered_kind: str,
    dev: AbstractDevice,
    test_case: Optional[AbstractBaroclinicModelTestCase] = None,
    bkg_reader: Optional[AbstractBackgroundReader] = None,
    topo_reader: Optional[AbstractTopographyReader] = None,
    ideal_dry_core: bool = False,
    restart: bool = False,
    *args,
    **kwargs
) -> AbstractModel:
    ...


def build_hydrostatic_model(
    config: UserConfig,
    para: PlanetParameterSet,
    mesh: AbstractMesh,
    vert_coord: AbstractVertCoordinate,
    method: str,
    form: str,
    staggered_kind: str,
    dev: AbstractDevice,
    test_case: Optional[AbstractBaroclinicModelTestCase] = None,
    bkg_reader: Optional[AbstractBackgroundReader] = None,
    topo_reader: Optional[AbstractTopographyReader] = None,
    ideal_dry_core: bool = False,
    restart: bool = False,
    *args,
    **kwargs
) -> AbstractModel:
    if (
        mesh.alias_name() == "scvt" and 
        isinstance(vert_coord, AbstractGeneralizedTerrainFollowingCoordinate) and 
        method == "finite_volume_method"
    ):
        if staggered_kind == "triskc":
            if form == "vector_invariant":
                from terraflow.mesh.scvt.mesh_type import SCVTMesh
                from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.baroclinic import BaroclinicModel
                mesh = cast(SCVTMesh, mesh)
                if mesh.full_nlev == 1 or mesh.half_nlev == 2:
                    raise ValueError(
                        f"SCVTMesh has been initialized with one single layer, but vert_coord input has {vert_coord.full_lev} levels!"
                    )
                vert_coord = cast(AbstractGeneralizedTerrainFollowingCoordinate, vert_coord)
                return BaroclinicModel(
                    mode=ModelMode.from_hdc(
                        ideal_dry_core=ideal_dry_core,
                        restart=restart
                    ), 
                    cfg=config,
                    para=para,
                    mesh=mesh,
                    vert_coord=vert_coord,
                    dev=dev,
                    test_case=test_case,
                    bkg_reader=bkg_reader,
                    topo_reader=topo_reader
                )
            else:
                raise ValueError(
                    f"Equation form {form} not supported. Please check the avaliable options."
                )           
        else:
            raise ValueError(f"Unsuppored layout {staggered_kind} for scvt mesh!")
    else:
        raise ValueError("Unsupported model!")
