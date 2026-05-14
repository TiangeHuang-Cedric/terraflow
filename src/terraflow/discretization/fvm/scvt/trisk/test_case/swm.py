from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

from terraflow.layout.field import StaggeredFieldMetaData
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from ..element_meta import default_state_meta, default_static_meta

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.abstract import EquationMode
    from terraflow.solver.abstract import AbstractSolver
    from terraflow.test_case.swe import AbstractShallowWaterEquationTestCase
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as BLOCK

    Array = typing.Array

    VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
    VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)

    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        VM,
        TriSKBasedCGridStaggeringLayout,
        VLO
    ]

    CoordSysOneLayer = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    CoordSysHybrid = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        HybridVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        HybridVertCoordLayout
    ]


def barotropic_case_init_fn(
    coord: CoordSysOneLayer,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: AbstractShallowWaterEquationTestCase,
) -> BLOCK:
    from ..single_lev.dynamics_types import Block
    g = para.gravity.unwrap()
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    un = init_un_in_barotropic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    gz = init_z_in_barotropic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    ) * g
    gzs = init_zs_in_barotropic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    ) * g
    block = tree_at(
        coord=coord,
        where=lambda b: (b.state.un, b.state.gz),
        pytree=block,
        replace=(un, gz),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )
    block = tree_at(
        coord=coord,
        where=lambda b: b.static.gzs,
        pytree=block,
        replace=gzs
    )
    return block


def init_z_in_barotropic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for gz variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_geopotential_height(
        lon, lat
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_un_in_barotropic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["un"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for un variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for un variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        lon=lon,
        lat=lat,
        cartesian=cart,
        edge_normal=nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_zs_in_barotropic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractShallowWaterEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for gzs variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_topography(lon, lat)
