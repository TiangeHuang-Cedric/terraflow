from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.layout.abstract import (
    Basis,
    MeshEntity,
    VertLoc,
)
from terraflow.layout.field import StaggeredFieldMetaData, NumericMetaData

if TYPE_CHECKING:
    from terraflow.discretization.abstract import AbstractEquationMode


NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


def default_state_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData | NumericMetaData]:
    meta: dict[str, StaggeredFieldMetaData | NumericMetaData] = {}
    if mode.is_barotropic() or mode.is_advection2d():
        meta["un"] = StaggeredFieldMetaData(
            name="un",
            dims=(NEDGE,),
            long_name="Normal velocity at edge",
            description="Normal velocity at edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["un"] = StaggeredFieldMetaData(
            name="un",
            dims=(NLEV, NEDGE,),
            long_name="Normal velocity at edge",
            description="Normal velocity at edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_barotropic() or mode.is_advection2d():
        meta["gz"] = StaggeredFieldMetaData(
            name="gz",
            dims=(NCELL,),
            long_name="Geopotential at primal cell center",
            description="Geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_barotropic() else ""
        )
    elif mode.is_baroclinic() or mode.is_advection3d():
        meta["gz"] = StaggeredFieldMetaData(
            name="gz",
            dims=(NLEV, NCELL,),
            long_name="Geopotential at primal cell center",
            description="Geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["gz_lev"] = StaggeredFieldMetaData(
            name="gz_lev",
            dims=(NLEVP, NCELL,),
            long_name="Geopotential at primal cell center and vertical level",
            description="Geopotential at primal cell center and vertical level",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["mg"] = StaggeredFieldMetaData(
            name="mg",
            dims=(NLEV, NCELL,),
            long_name="Dry air weight at primal cell center and full level",
            description="Dry air weight at primal cell center and full level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["mg_lev"] = StaggeredFieldMetaData(
            name="mg_lev",
            dims=(NLEVP, NCELL,),
            long_name="Dry air weight at primal cell center and half level",
            description="Dry air weight at primal cell center and half level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["mgs"] = StaggeredFieldMetaData(
            name="mgs",
            dims=(NCELL,),
            long_name="Dry air weight on surface at primal cell center and full level",
            description="Dry air weight on surface at primal cell center and full level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["dmg"] = StaggeredFieldMetaData(
            name="dmg",
            dims=(NLEV, NCELL,),
            long_name="Difference of dry air weight at primal cell center between the neighboring half levels",
            description="Difference of dry air weight at primal cell center between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NLEV, NEDGE,),
            long_name="Difference of dry air weight at edge between the neighboring half levels",
            description="Difference of dry air weight at edge between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["dmg_vtx"] = StaggeredFieldMetaData(
            name="dmg_vtx",
            dims=(NLEV, NVTX,),
            long_name="Difference of dry air weight at dual cell center between the neighboring half levels",
            description="Difference of dry air weight at dual cell center between the neighboring half levels",
            units="Pa",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["dmg_lev"] = StaggeredFieldMetaData(
            name="dmg_lev",
            dims=(NLEVP, NCELL,),
            long_name="Difference of dry air weight at primal cell center between the neighboring full levels",
            description="Difference of dry air weight at primal cell center between the neighboring full levels",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
    if mode.is_barotropic() or mode.is_advection2d():
        meta["dmg"] = StaggeredFieldMetaData(
            name="dmg",
            dims=(NCELL,),
            long_name="Fluid thickness at primal cell",
            description="Fluid thickness at primal cell",
            units="m",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1"
        )
        meta["dmg_e"] = StaggeredFieldMetaData(
            name="dmg_e",
            dims=(NEDGE,),
            long_name="Fluid thickness at edge",
            description="Fluid thickness at edge",
            units="m",
            hori_location=MeshEntity.Edge,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_barotropic() else ""
        )
        meta["dmg_vtx"] = StaggeredFieldMetaData(
            name="dmg_vtx",
            dims=(NVTX,),
            long_name="Fluid thickness at dual cell center",
            description="Fluid thickness at dual cell center",
            units="m",
            hori_location=MeshEntity.Vtx,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_barotropic() else ""
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["pt"] = StaggeredFieldMetaData(
            name="pt",
            dims=(NLEV, NCELL,),
            long_name="Potential temperature at primal cell center and vertical level",
            description="Potential temperature at primal cell center and vertical level",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["ph"] = StaggeredFieldMetaData(
            name="ph",
            dims=(NLEV, NCELL,),
            long_name="Hydrostatic pressure at primal cell center",
            description="Hydrostatic pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["ph_lev"] = StaggeredFieldMetaData(
            name="ph_lev",
            dims=(NLEVP, NCELL,),
            long_name="Hydrostatic pressure at primal cell center and vertical level",
            description="Hydrostatic pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["phs"] = StaggeredFieldMetaData(
            name="phs",
            dims=(NCELL,),
            long_name="Surface hydrostatic pressure at primal cell center",
            description="Surface hydrostatic pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["p"] = StaggeredFieldMetaData(
            name="p",
            dims=(NLEV, NCELL,),
            long_name="Pressure at primal cell center and vertical level",
            description="Pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.nonhydrostatic else ""
        )
        meta["p_lev"] = StaggeredFieldMetaData(
            name="p_lev",
            dims=(NLEVP, NCELL),
            long_name="Pressure at primal cell center and vertical level",
            description="Pressure at primal cell center and vertical level",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0" if mode.nonhydrostatic else ""
        )
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        meta["tm"] = NumericMetaData(
            name="tm",
            long_name="Total mass",
            description="Total mass",
            units="kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["tpe"] = NumericMetaData(
            name="tpe",
            long_name="Total potential vorticity enstrophy",
            description="Total potential vorticity enstrophy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["te"] = NumericMetaData(
            name="te",
            long_name="Total energy",
            description="Total energy",
            is_static=False,
            is_expand_dims=True,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
        meta["te_ke"] = NumericMetaData(
            name="te_ke",
            long_name="Total energy due to kinetic energy",
            description="Total energy due to kinetic energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
        meta["te_pe"] = NumericMetaData(
            name="te_pe",
            long_name="Total energy due to potential energy",
            description="Total energy due to potential energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
    if mode.is_baroclinic():
        meta["te_ie"] = NumericMetaData(
            name="te_ie",
            long_name="Total energy due to internal energy",
            description="Total energy due to internal energy",
            units="J/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output=""
        )
        meta["tpt"] = NumericMetaData(
            name="tpt",
            long_name="Total potential temperature",
            description="Total potential temperature",
            units="K",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    if mode.is_advection():
        meta["tq"] = NumericMetaData(
            name="tq",
            long_name="Total tracer dry mixing ratio",
            description="Total tracer dry mixing ratio",
            units="m3 kg/kg",
            is_static=False,
            is_expand_dims=True,
            expand_axis=0,
            expand_axis_length=mode.state_tile_depth,
            output="h0"
        )
    return meta


def default_tendency_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic():
        meta["dgzdt"] = StaggeredFieldMetaData(
            name="dgzdt",
            dims=(NCELL,),
            long_name="Tendency of geopotential",
            description="Tendency of geopotential calculated from state variables",
            units="m2 / s3",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dgzdt_dmf"] = StaggeredFieldMetaData(
            name="dgzdt_dmf",
            dims=(NCELL,),
            long_name="Tendency of geopotential due to mass flux divergence",
            description="Tendency of geopotential due to mass flux divergence calculated from state variables",
            units="m2 / s3",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_baroclinic():
        meta["dmgsdt"] = StaggeredFieldMetaData(
            name="dmgsdt",
            dims=(NCELL,),
            long_name="Tendency of dry air weight at primal cell center and full level",
            description="Tendency of dry air weight at primal cell center and full level",
            units="Pa / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dmgsdt_dmf"] = StaggeredFieldMetaData(
            name="dmgsdt_dmf",
            dims=(NCELL,),
            long_name="Tendency of dry air weight at primal cell center and full level due to mass flux divergence",
            description="Tendency of dry air weight at primal cell center and full level due to mass flux divergence calculated from state variables",
            units="Pa / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dptdt"] = StaggeredFieldMetaData(
            name="dptdt",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level",
            description="Tendency of potential temperature at primal cell center and vertical level",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dptdt_dptfn"] = StaggeredFieldMetaData(
            name="dptdt_dptfn",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level due to horizontal advection",
            description="Tendency of potential temperature at primal cell center and vertical level due to horizontal advection",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dptdt_ptvf"] = StaggeredFieldMetaData(
            name="dptdt_ptvf",
            dims=(NLEV, NCELL,),
            long_name="Tendency of potential temperature at primal cell center and vertical level due to vertical advection",
            description="Tendency of potential temperature at primal cell center and vertical level due to vertical advection",
            units="K / s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["dundt"] = StaggeredFieldMetaData(
            name="dundt",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge",
            description="Tendency of normal velocity at edge",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
        meta["dundt_coriolis"] = StaggeredFieldMetaData(
            name="dundt_coriolis",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to Coriolis force",
            description="Tendency of normal velocity at edge due to Coriolis force",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dundt_grad_ke"] = StaggeredFieldMetaData(
            name="dundt_grad_ke",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to gradient of kinetic energy",
            description="Tendency of normal velocity at edge due to gradient of kinetic energy",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
        meta["dundt_pgf"] = StaggeredFieldMetaData(
            name="dundt_pgf",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tendency of normal velocity at edge due to pressure gradient force",
            description="Tendency of normal velocity at edge due to pressure gradient force",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    if mode.is_baroclinic():
        meta["dundt_wedundeta"] = StaggeredFieldMetaData(
            name="dundt_wedundeta",
            dims=(NLEV, NEDGE,),
            long_name="Tendency of normal velocity at edge due to vertical advection",
            description="Tendency of normal velocity at edge due to vertical advection",
            units="m / s2",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.output_h1_tend else ""
        )
    return meta


def default_static_meta(mode: AbstractEquationMode,) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        meta["gzs"] = StaggeredFieldMetaData(
            name="gzs",
            dims=(NCELL,),
            long_name="Surface geopotential at primal cell center",
            description="Surface geopotential at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_static=True,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["ref_ps"] = StaggeredFieldMetaData(
            name="ref_ps",
            dims=(NCELL,),
            long_name="Reference surface pressure at primal cell center",
            description="Reference surface pressure at primal cell center",
            units="Pa",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Sfc,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            is_static=False,
            output="h1"
        )
    return meta


def default_auxiliary_meta(mode: AbstractEquationMode) -> dict[str, StaggeredFieldMetaData]:
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_barotropic() or mode.is_advection2d():
        meta["u"] = StaggeredFieldMetaData(
            name="u",
            dims=(NCELL,),
            long_name="Zonal velocity at primal cell center",
            description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["v"] = StaggeredFieldMetaData(
            name="v",
            dims=(NCELL,),
            long_name="Meridional velocity at primal cell center",
            description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location="",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["u"] = StaggeredFieldMetaData(
            name="u",
            dims=(NLEV, NCELL,),
            long_name="Zonal velocity at primal cell center",
            description="Zonal velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
        meta["v"] = StaggeredFieldMetaData(
            name="v",
            dims=(NLEV, NCELL,),
            long_name="Meridional velocity at primal cell center",
            description="Meridional velocity at primal cell center reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["un_lev"] = StaggeredFieldMetaData(
            name="un_lev",
            dims=(NLEVP, NEDGE,),
            long_name="Normal velocity at edge and half level",
            description="Normal velocity at edge and half level",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Lev,
            basis=Basis.Normal,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=True,
            output=""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["ut"] = StaggeredFieldMetaData(
            name="ut",
            dims=(NLEV, NEDGE,) if mode.is_baroclinic() else (NEDGE,),
            long_name="Tangential velocity at edge",
            description="Tangential velocity at edge reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    if mode.is_advection():
        meta["ut"] = StaggeredFieldMetaData(
            name="ut",
            dims=(NLEV, NEDGE,) if mode.is_advection3d() else (NEDGE,),
            long_name="Tangential velocity at edge",
            description="Tangential velocity at edge reconstructed from normal velocity at dual edge",
            units="m/s",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_advection3d() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output=""
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["we_lev"] = StaggeredFieldMetaData(
            name="we_lev",
            dims=(NLEVP, NCELL,),
            long_name="Vertical coordinate velocity speed",
            description="Vertical coordinate velocity speed at primal cell center and half level",
            units="m/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output=""
        )
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        meta["div"] = StaggeredFieldMetaData(
            name="div",
            dims=(NLEV, NCELL,) if mode.is_3d_model() else (NCELL,),
            long_name="Divergence at primal cell center",
            description="Divergence at primal cell center",
            units="1/s",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0" if not mode.is_advection() else ""
        )
        meta["vor"] = StaggeredFieldMetaData(
            name="vor",
            dims=(NLEV, NVTX,) if mode.is_3d_model() else (NVTX,),
            long_name="Relative vorticity at dual cell center",
            description="Relative vorticity at dual cell center",
            units="1/s",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0" if not mode.is_advection() else ""
        )
        meta["pv_vtx"] = StaggeredFieldMetaData(
            name="pv_vtx",
            dims=(NLEV, NVTX,) if mode.is_3d_model() else (NVTX,),
            long_name="Potential vorticity at dual cell center",
            description="Potential vorticity at dual cell center",
            units="Pa-1 s-1",
            hori_location=MeshEntity.Vtx,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_vtx lat_vtx",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection() else ""
        )
        meta["pv_e"] = StaggeredFieldMetaData(
            name="pv_e",
            dims=(NLEV, NEDGE,) if mode.is_3d_model() else (NEDGE,),
            long_name="Potential vorticity at edge",
            description="Potential vorticity at edge",
            units="Pa-1 s-1",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection() else ""
        )
        meta["ke"] = StaggeredFieldMetaData(
            name="ke",
            dims=(NLEV, NCELL,) if mode.is_3d_model() else (NCELL,),
            long_name="Kinetic energy at primal cell center",
            description="Kinetic energy at primal cell center",
            units="m2 / s2",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection() else ""
        )
        meta["mfn"] = StaggeredFieldMetaData(
            name="mfn",
            dims=(NLEV, NEDGE,) if mode.is_3d_model() else (NEDGE,),
            long_name="Mass flux at edge midpoint",
            description="Mass flux at edge midpoint",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell if mode.is_3d_model() else "",
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection() else ""
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["mft"] = StaggeredFieldMetaData(
            name="mft",
            dims=(NLEV, NEDGE,),
            long_name="Mass flux at edge midpoint reconstructed from tangential velocity",
            description="Mass flux at edge midpoint reconstructed from tangential velocity",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection3d() else ""
        )
        meta["mfz_lev"] = StaggeredFieldMetaData(
            name="mfz_lev",
            dims=(NLEVP, NCELL,),
            long_name="Vertical mass flux at primal cell center and half level",
            description="Vertical mass flux at primal cell center and half level",
            units="kg / (m s)",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if not mode.is_advection3d() else ""
        )
        meta["mfz_e_lev"] = StaggeredFieldMetaData(
            name="mfz_e_lev",
            dims=(NLEVP, NEDGE,),
            long_name="Vertical mass flux at edge and half level",
            description="Vertical mass flux at edge and half level",
            units="kg / (m s)",
            hori_location=MeshEntity.Edge,
            vert_location=VertLoc.Lev,
            basis=Basis.Scalar,
            coordinates="lon_edge lat_edge",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output=""
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        meta["dmf"] = StaggeredFieldMetaData(
            name="dmf",
            dims=(NLEV, NCELL,) if mode.is_baroclinic() else (NCELL,),
            long_name="Divergence of mass flux at primal cell center",
            description="Divergence of mass flux at primal cell center",
            units="kg / (m s)",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_baroclinic() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1"
        )
    if mode.is_advection():
        meta["dmf"] = StaggeredFieldMetaData(
            name="dmf",
            dims=(NLEV, NCELL,) if mode.is_advection3d() else (NCELL,),
            long_name="Divergence of mass flux at primal cell center",
            description="Divergence of mass flux at primal cell center",
            units="kg / (m s)",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell if mode.is_advection3d() else "",
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output=""
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["t"] = StaggeredFieldMetaData(
            name="t",
            dims=(NLEV, NCELL,),
            long_name="Temperature at primal cell center",
            description="Temperature at primal cell center",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h0" if mode.is_baroclinic() else ""
        )
        meta["tv"] = StaggeredFieldMetaData(
            name="tv",
            dims=(NLEV, NCELL,),
            long_name="Virtual temperature at primal cell center",
            description="Virtual temperature at primal cell center",
            units="K",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.is_baroclinic() else ""
        )
        meta["rhod"] = StaggeredFieldMetaData(
            name="rhod",
            dims=(NLEV, NCELL,),
            long_name="Density of dry air at primal cell center",
            description="Density of dry air at primal cell center",
            units="kg / m3",
            hori_location=MeshEntity.Cell,
            vert_location=VertLoc.Cell,
            basis=Basis.Scalar,
            coordinates="lon_cell lat_cell",
            staggered_kind="C",
            mesh_name="primal",
            is_prognostic=False,
            output="h1" if mode.is_baroclinic() else ""
        )
    return meta
