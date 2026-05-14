from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

from terraflow.util import typing, formula
from terraflow.layout.field import StaggeredFieldMetaData
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at
from ..element_meta import default_state_meta, default_static_meta, default_auxiliary_meta

if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.test_case.baroclinic import (
        AbstractBaroclinicEquationTestCase,
        SteadyStateTestCase,
        MountainWaveTestCase,
        RossbyHaurwitzWaveTestCase
    )
    from terraflow.solver.abstract import AbstractSolver
    from terraflow.discretization.abstract import EquationMode
    from terraflow.discretization.fvm.scvt.trisk.hybrid_lev.dynamics_types import Block as BLOCK

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


def baroclinic_steady_state_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: SteadyStateTestCase,
) -> BLOCK:
    from ..hybrid_lev.dynamics_types import Block
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    gzs = init_gzs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    gz_lev = init_gz_lev_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs, ks=full_kds, ke=full_kde)
    t = init_t_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    pt = formula.dry_potential_temperature(t=t, p=mg, para=para)
    un = init_un_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg,
            b.state.gz, b.state.gz_lev,
            b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, gz, gz_lev, un, pt),
        at_dims=(0,) * 6,
        at_idxes=(0,) * 6
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps),
    )
    block = tree_at(
        coord=coord,
        where=lambda b: b.aux.t,
        pytree=block,
        replace=t,
    )
    return block


def baroclinic_mountain_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: MountainWaveTestCase,
) -> BLOCK:
    from ..hybrid_lev.dynamics_types import Block
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    gzs = init_gzs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        gzs=gzs
    )
    mg = coord.vert.calc_mg(
        mgs=mgs,
        ks=full_kds,
        ke=full_kde
    )
    pt = init_pt_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=mg
    )
    un = init_un_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg, b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, un, pt),
        at_dims=(0,) * 4,
        at_idxes=(0,) * 4
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps)
    )
    return block


def baroclinic_rossby_wave_case_init_fn(
    coord: CoordSysHybrid,
    para: PlanetParameterSet,
    solver: AbstractSolver,
    *,
    test_case: RossbyHaurwitzWaveTestCase,
) -> BLOCK:
    from ..hybrid_lev.dynamics_types import Block
    full_kds = coord.vert.full_kds
    full_kde = coord.vert.full_kde
    half_kds = coord.vert.half_kds
    half_kde = coord.vert.half_kde
    block = Block.from_model(
        coord=coord,
        para=para,
        solver=solver,
        test_case=test_case
    )
    mode = solver.equation.mode
    un = init_un_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    gzs = init_gzs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case
    )
    mgs = init_mgs_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case, gzs=gzs
    )
    mg = coord.vert.calc_mg(mgs=mgs, ks=full_kds, ke=full_kde)
    mg_lev = coord.vert.calc_mg_lev(mgs=mgs, ks=half_kds, ke=half_kde)
    pt = init_pt_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=mg
    )
    gz = init_gz_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=mg
    )
    gz_lev = init_gz_lev_in_baroclinic_case(
        coord=coord,
        mode=mode,
        test_case=test_case,
        mg=mg_lev
    )
    ref_ps = mgs
    block = tree_at(
        coord=coord,
        where=lambda b: (
            b.state.mgs, b.state.mg, b.state.mg_lev,
            b.state.gz, b.state.gz_lev,
            b.state.un, b.state.pt
        ),
        pytree=block,
        replace=(mgs, mg, mg_lev, gz, gz_lev, un, pt),
        at_dims=(0,) * 7,
        at_idxes=(0,) * 7
    )
    block = tree_at(
        coord=coord,
        where=lambda b: (b.static.gzs, b.static.ref_ps),
        pytree=block,
        replace=(gzs, ref_ps)
    )
    return block


def init_mgs_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    gzs: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["mgs"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for mgs variable: {type(meta)}"
        )
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_mgs(lon, lat, gzs)


def init_t_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_auxiliary_meta(mode=mode)["t"]
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
    return test_case.init_t(lon, lat, eta)


def init_pt_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    mg: Array,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["pt"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for pt variable: {type(meta)}"
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
    return test_case.init_pt(lon, lat, eta, mg)


def init_un_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
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
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    if vertical_coord is None:
        raise ValueError(
            f"Vertical coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    eta = vertical_coord
    cart = horizontal_coord["cart"]
    nr = horizontal_coord["nr_orient"]
    return test_case.init_normal_velocity(
        lon=lon,
        lat=lat,
        eta=eta,
        cartesian=cart,
        edge_normal=nr
    )


def init_gzs_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_static_meta(mode=mode)["gzs"]
    horizontal_coord = meta.get_hori_position(coord).horizontal_coord
    if horizontal_coord is None:
        raise ValueError(
            f"Horizontal coordinate information is missing for meta: {meta}"
        )
    lon = horizontal_coord["lon"]
    lat = horizontal_coord["lat"]
    return test_case.init_gzs(lon, lat)


def init_gz_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz variable: {type(meta)}"
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
    return test_case.init_gz(lon, lat, eta, *args, **kwargs)


def init_gz_lev_in_baroclinic_case(
    coord: CoordSys,
    mode: EquationMode,
    test_case: AbstractBaroclinicEquationTestCase,
    *args,
    **kwargs
) -> Array:
    meta = default_state_meta(mode=mode)["gz_lev"]
    if not isinstance(meta, StaggeredFieldMetaData):
        raise ValueError(
            f"Invalid meta type for gz_lev variable: {type(meta)}"
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
    return test_case.init_gz(lon, lat, eta, *args, **kwargs)
