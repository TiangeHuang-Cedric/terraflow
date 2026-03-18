from typing import Optional
import equinox as eqx

from terraflow.util import typing
from terraflow.layout.abstract import MeshEntity, VertLoc, Basis
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.discretization.abstract import AbstractAdvBatchType, AbstractEquationMode


FDT = ArrayField[StaggeredFieldMetaData]

Array = typing.Array

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


def default_adv_batch_meta(
    name: str,
    batch_hori_loc: str,
    batch_vert_loc: str,
    mode: AbstractEquationMode,
) -> dict[str, StaggeredFieldMetaData]:
    if batch_hori_loc == MeshEntity.Edge:
        raise NotImplementedError(
            "Batch horizontal location at edge is not implemented yet."
        )
    if batch_hori_loc == MeshEntity.Vtx and batch_vert_loc == VertLoc.Lev:
        raise NotImplementedError(
            "Batch at vertex-level is not implemented yet."
        )
    if batch_hori_loc == MeshEntity.Cell and batch_vert_loc == VertLoc.Cell:
        loc0 = {"hori": MeshEntity.Cell, "vert": VertLoc.Cell}
        dim0 = (NLEV, NCELL,)
        coordinates0 = "lon_cell lat_cell"
        locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
        dimn = (NLEV, NEDGE,)
        coordinatesn = "lon_edge lat_edge"
        locz = {"hori": MeshEntity.Cell, "vert": VertLoc.Lev}
        dimz = (NLEVP, NCELL,)
        coordinatesz = "lon_cell lat_cell"
    elif batch_hori_loc == MeshEntity.Cell and batch_vert_loc == VertLoc.Lev:
        loc0 = {"hori": MeshEntity.Cell, "vert": VertLoc.Lev}
        dim0 = (NLEVP, NCELL,)
        coordinates0 = "lon_cell lat_cell"
        locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Lev}
        dimn = (NLEVP, NEDGE,)
        coordinatesn = "lon_edge lat_edge"
        locz = {"hori": MeshEntity.Cell, "vert": VertLoc.Cell}
        dimz = (NLEV, NCELL,)
        coordinatesz = "lon_cell lat_cell"
    elif batch_hori_loc == MeshEntity.Vtx and batch_vert_loc == VertLoc.Cell:
        loc0 = {"hori": MeshEntity.Vtx, "vert": VertLoc.Cell}
        dim0 = (NLEV, NVTX,)
        coordinates0 = "lon_vtx lat_vtx"
        locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
        dimn = (NLEV, NEDGE,)
        coordinatesn = "lon_edge lat_edge"
        locz = {"hori": MeshEntity.Vtx, "vert": VertLoc.Lev}
        dimz = (NLEVP, NVTX,)
        coordinatesz = "lon_vtx lat_vtx"
    else:
        raise NotImplementedError(
            f"Invalid grid location {batch_hori_loc}, {batch_vert_loc}."
        )
    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.baroclinic:
        meta["m"] = StaggeredFieldMetaData(
            name=name + "_m",
            dims=dim0,
            long_name=name + "Dry air weight",
            description=name + "Dry air weight",
            units="Pa",
            hori_location=loc0["hori"],
            vert_location=loc0["vert"],
            basis=Basis.Scalar,
            coordinates=coordinates0,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["mfn"] = StaggeredFieldMetaData(
            name=name + "_mfn",
            dims=dimn,
            long_name=name + "Dry air weight at edge",
            description=name + "Dry air weight at edge",
            units="Pa",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["mft"] = StaggeredFieldMetaData(
            name=name + "_mft",
            dims=dimn,
            long_name=name + "Dry air weight at edge",
            description=name + "Dry air weight at edge",
            units="Pa",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Tangent,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["mfz"] = StaggeredFieldMetaData(
            name=name + "_mfz",
            dims=dimz,
            long_name=name + "Dry air weight at level",
            description=name + "Dry air weight at level",
            units="Pa",
            hori_location=locz["hori"],
            vert_location=locz["vert"],
            basis=Basis.Scalar,
            coordinates=coordinatesz,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["un"] = StaggeredFieldMetaData(
            name=name + "_un",
            dims=dimn,
            long_name=name + " normal wind at edge",
            description=name + " normal wind at edge",
            units="m s-1",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["ut"] = StaggeredFieldMetaData(
            name=name + "_ut",
            dims=dimn,
            long_name=name + " tangent wind at edge",
            description=name + " tangent wind at edge",
            units="m s-1",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Tangent,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["w"] = StaggeredFieldMetaData(

            name=name + "_w",
            dims=dimz,
            long_name=name + " vertical coordinate velocity",
            description=name + " vertical coordinate velocity",
            units="s-1",
            hori_location=locz["hori"],
            vert_location=locz["vert"],
            basis=Basis.Scalar,
            coordinates=coordinatesz,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.advection else "",
        )
        meta["qmfn"] = StaggeredFieldMetaData(
            name=name + "_qmfn",
            dims=dimn,
            long_name=name + " normal mass flux",
            description=name + " normal mass flux",
            units="Pa kg kg-1 m s-1",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h1",
        )
        meta["qmfz"] = StaggeredFieldMetaData(
            name=name + "_qmfz",
            dims=dimz,
            long_name=name + " vertical mass flux",
            description=name + " vertical mass flux",
            units="Pa kg kg-1 m s-1",
            hori_location=locz["hori"],
            vert_location=locz["vert"],
            basis=Basis.Scalar,
            coordinates=coordinatesz,
            staggered_kind="C",
            mesh_name="primal",
            output="h1",
        )
    return meta


class AdvBatchType(AbstractAdvBatchType[StaggeredFieldMetaData]):
    name: str = eqx.field(static=True)
    hori_loc: str = eqx.field(static=True)
    vert_loc: str = eqx.field(static=True)
    m: FDT
    mfn: FDT
    mft: FDT
    mfz: FDT
    un: FDT
    ut: FDT
    w: FDT
    qmfn: FDT
    qmfz: FDT

    @staticmethod
    def default_meta(
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        mode: AbstractEquationMode,
    ) -> dict[str, StaggeredFieldMetaData]:
        return default_adv_batch_meta(
            name=name,
            batch_hori_loc=batch_hori_loc,
            batch_vert_loc=batch_vert_loc,
            mode=mode,
        )

    def set_wind(
        self,
        un: Array,
        ut: Array,
        w: Optional[Array] = None,
        mfn: Optional[Array] = None,
        mft: Optional[Array] = None,
        mfz: Optional[Array] = None,
        m: Optional[Array] = None
    ) -> "AdvBatchType":
        self = eqx.tree_at(
            lambda b: (b.un.data, b.ut.data),
            self,
            (un, ut)
        )
        if w is not None:
            self = eqx.tree_at(
                lambda b: b.w.data,
                self,
                w
            )
        if mfn is not None:
            self = eqx.tree_at(
                lambda b: b.mfn.data,
                self,
                mfn
            )
        if mft is not None:
            self = eqx.tree_at(
                lambda b: b.mft.data,
                self,
                mft
            )
        if mfz is not None:
            self = eqx.tree_at(
                lambda b: b.mfz.data,
                self,
                mfz
            )
        if m is not None:
            self = eqx.tree_at(
                lambda b: b.m.data,
                self,
                m
            )
        return self
