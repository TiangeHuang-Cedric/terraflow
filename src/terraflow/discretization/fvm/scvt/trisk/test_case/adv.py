from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

from terraflow.layout.field import StaggeredFieldMetaData
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from terraflow.discretization.fvm.scvt.trisk.tracer import Tracer, default_tracer_meta
from ..element_meta import default_state_meta, default_auxiliary_meta

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
    from terraflow.test_case.adv import (
        AbstractAdvectionEquation2DTestCase,
        AbstractAdvectionEquation3DTestCase,
        DCMIP12TestCase
    )
    from terraflow.solver.abstract import AbstractSolver
    from terraflow.discretization.abstract import (
        EquationMode,
        AbstractEquationSet,
    )
    from terraflow.discretization.fvm.scvt.trisk.single_lev.dynamics_types import Block as SingleLevBlock
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as HybridLevBlock

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array

    VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)
    VLO = TypeVar("VLO", OneLayerVertCoordLayout, HybridVertCoordLayout)
    ES = TypeVar("ES", bound=AbstractEquationSet)

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


def advection2d_case_init_fn(
    time_in_seconds: Float[Array, ""],
    coord: CoordSysOneLayer,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: AbstractAdvectionEquation2DTestCase,
) -> SingleLevBlock:
    from ..single_lev.dynamics_types import Block
    mode = solver.equation.mode
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        tracers=Tracer.add_tracers(
            mode=mode,
            coord=coord,
            tracers_meta=test_case.init_tracers_meta(
                dt_adv=solver.dt_adv
            )
        ),
        test_case=test_case
    )
    dmg = init_background_in_advection2d_case(
        coord=coord, mode=mode, test_case=test_case
    )
    un = init_un_in_advection2d_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    q = init_tracers_in_advection2d_case(
        coord=coord, mode=mode, test_case=test_case
    )
    return tree_at(
        coord=coord,
        where=lambda b: (b.state.dmg, b.state.un, b.tracers.q),
        pytree=block,
        replace=(dmg, un, q),
        at_dims=(0, 0, 0),
        at_idxes=(0, 0, 0)
    )


def advection2d_case_set_uv_fn(
    block: SingleLevBlock,
    test_case: AbstractAdvectionEquation2DTestCase,
    itime: int | Int[Array, ""]
) -> SingleLevBlock:
    time_in_seconds = block.elapsed_seconds
    coord = block.coord
    mode = block.mode
    un = init_un_in_advection2d_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    return tree_at(
        coord=coord,
        where=lambda b: b.state.un,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=itime
    )


def dcmip12_case_init_fn(
    time_in_seconds: Float[Array, ""],
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver[ES],
    *,
    test_case: DCMIP12TestCase,
) -> HybridLevBlock:
    from ..hybrid_lev.dynamics_types import Block
    mode = solver.equation.mode
    space_operators = solver.equation.space_operators
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        tracers=Tracer.add_tracers(
            mode=mode,
            coord=coord,
            tracers_meta=test_case.init_tracers_meta(
                dt_adv=solver.dt_adv
            )
        ),
        test_case=test_case
    )
    block = space_operators.update_pressure_levels(block, block.old)
    mg = block.state.mg.unwrap()[block.old, ...]
    dmg = block.state.dmg.unwrap()[block.old, ...]
    gz = block.state.gz.unwrap()[block.old, ...]
    gz_lev = block.state.gz_lev.unwrap()[block.old, ...]
    un = init_un_in_dcmip12_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=mg,
        gz=gz,
    )
    q = init_tracers_in_advection3d_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        gz=gz,
    )
    we_lev = init_we_lev_in_advection3d_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case,
        gz_lev=gz_lev,
    )
    mfz_lev = init_mfz_lev_in_advection3d_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        we_lev=we_lev,
        dmg=dmg,
        deta=coord.vert.half_dlev
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.state.un, b.tracers.q),
        pytree=block,
        replace=(un, q),
        at_dims=(0, 0),
        at_idxes=(0, 0)
    )
    return tree_at(
        coord=coord,
        where=lambda b: (b.aux.we_lev, b.aux.mfz_lev),
        pytree=block,
        replace=(we_lev, mfz_lev)
    )


def dcmip12_case_set_uv_fn(
    block: HybridLevBlock,
    test_case: DCMIP12TestCase,
    itime: int | Int[Array, ""]
) -> HybridLevBlock:
    time_in_seconds = block.elapsed_seconds
    coord = block.coord
    mode = block.mode
    un = init_un_in_dcmip12_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=block.state.mg.unwrap()[itime, :],
        gz=block.state.gz.unwrap()[itime, :],
    )
    we_lev = init_we_lev_in_advection3d_case(
        time_in_seconds=time_in_seconds,
        coord=coord,
        mode=mode,
        test_case=test_case,
        gz_lev=block.state.gz_lev.unwrap()[itime, :],
    )
    mfz_lev = init_mfz_lev_in_advection3d_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        deta=coord.vert.half_dlev,
        we_lev=we_lev,
        dmg=block.state.dmg.unwrap()[itime, :],
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.aux.we_lev, b.aux.mfz_lev),
        pytree=block,
        replace=(we_lev, mfz_lev)
    )
    return tree_at(
        coord=coord,
        where=lambda b: b.state.un,
        pytree=block,
        replace=un,
        at_dims=0,
        at_idxes=itime
    )


def init_background_in_advection2d_case(
    coord: CoordSysOneLayer,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation2DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["dmg"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for dmg variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_background(lon, lat, *args, **kwargs)


def init_tracers_in_advection2d_case(
    coord: CoordSysOneLayer,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation2DTestCase,
    *args,
    **kwargs
) -> Array:
    # set to 1 just for return metadata
    meta = default_tracer_meta(mode, ntracers=1)["q"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for tracer variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    q = test_case.init_tracers(lon, lat, *args, **kwargs)
    ntracers = q.shape[0]
    return q.reshape((ntracers, 1,) + lon.shape)


def init_un_in_advection2d_case(
    time_in_seconds: Float[Array, ""],
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation2DTestCase,
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
        time_in_seconds=time_in_seconds,
        lon=lon,
        lat=lat,
        cartesian=cart,
        edge_normal=nr
    ).reshape((1,) + lon.shape)  # add leading axis for vmap


def init_background_in_advection3d_case(
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation3DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["dmg"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for dmg variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_background(lon, lat, eta, *args, **kwargs)


def init_tracers_in_advection3d_case(
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation3DTestCase,
    *args,
    **kwargs
) -> Array:
    # set to 1 just for return metadata
    meta = default_tracer_meta(mode, ntracers=1)["q"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for tracer variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    q = test_case.init_tracers(lon, lat, eta, *args, **kwargs)
    ntracers = q.shape[0]
    return q.reshape((ntracers,) + eta.shape + lon.shape)


def init_un_in_advection3d_case(
    time_in_seconds: Float[Array, ""],
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation3DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["un"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for un variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for un variable."
        )
    if vertical_coord is None:
        raise ValueError(
            "Vertical coordinate information is missing for un variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        time_in_seconds=time_in_seconds,
        lon=lon,
        lat=lat,
        eta=eta,
        cartesian=cart,
        edge_normal=nr,
        *args,
        **kwargs
    )


def init_un_in_dcmip12_case(
    time_in_seconds: Float[Array, ""],
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: DCMIP12TestCase,
    mg: Float[Array, "nlev ncell"],
    gz: Float[Array, "nlev ncell"],
    *args,
) -> Array:
    edd_v = coord.mesh._edd.v
    cell_1 = edd_v[:, 0]
    cell_2 = edd_v[:, 1]
    mg_e = (mg[:, cell_1] + mg[:, cell_2]) * 0.5
    gz_e = (gz[:, cell_1] + gz[:, cell_2]) * 0.5
    meta = default_state_meta(mode=mode)["un"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for un variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for un variable."
        )
    if vertical_coord is None:
        raise ValueError(
            "Vertical coordinate information is missing for un variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        time_in_seconds=time_in_seconds,
        lon=lon,
        lat=lat,
        eta=eta,
        cartesian=cart,
        edge_normal=nr,
        mg_e=mg_e,
        gz_e=gz_e
    )


def init_we_lev_in_advection3d_case(
    time_in_seconds: Float[Array, ""],
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation3DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_auxiliary_meta(mode=mode)["we_lev"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for we variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            "Horizontal coordinate information is missing for we variable."
        )
    if vertical_coord is None:
        raise ValueError(
            "Vertical coordinate information is missing for we variable."
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_we_lev(
        time_in_seconds=time_in_seconds,
        lon=lon,
        lat=lat,
        eta=eta,
        *args,
        **kwargs
    )


def init_mfz_lev_in_advection3d_case(
    coord: CoordSysHybrid,
    mode: EquationMode,
    test_case: AbstractAdvectionEquation3DTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_auxiliary_meta(mode=mode)["mfz_lev"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for mfz_lev variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    vertical_coord = meta.get_vert_position(coord).vertical_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    return test_case.init_mfz_lev(lon, lat, eta, *args, **kwargs)
