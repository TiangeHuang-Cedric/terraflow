import stat
from typing import Optional
import equinox as eqx
from matplotlib.pyplot import cla

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout import CoordinateSystem, MeshEntity, VertLoc, Basis
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.layout.field import Field, ArrayField, StaggeredFieldMetaData


FDT = ArrayField[StaggeredFieldMetaData]

Array = typing.Array


class AdvBatchType(eqx.Module):
    name: str = eqx.field(static=True)
    coord: CoordinateSystem
    hori_loc: str = eqx.field(static=True)
    vert_loc: str = eqx.field(static=True)
    m: FDT
    mfn: FDT
    mfz: FDT
    un: FDT
    ut: FDT
    w: FDT
    qmfn: FDT
    qmfz: FDT

    def __init__(
        self,
        name: str,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        hori_loc: str,
        vert_loc: str,
        m: FDT,
        mfn: FDT,
        mfz: FDT,
        un: FDT,
        ut: FDT,
        w: FDT,
        qmfn: FDT,
        qmfz: FDT,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coord", coord)
        object.__setattr__(self, "hori_loc", hori_loc)
        object.__setattr__(self, "vert_loc", vert_loc)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "mfn", mfn)
        object.__setattr__(self, "mfz", mfz)
        object.__setattr__(self, "un", un)
        object.__setattr__(self, "ut", ut)
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "qmfn", qmfn)
        object.__setattr__(self, "qmfz", qmfz)

    @classmethod
    def initialize(
        cls,
        name: str,
        batch_hori_loc: str,
        batch_vert_loc: str,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            HybridVertCoordinate,
            TriSKBasedCGridStaggeringLayout,
            HybridVertCoordLayout
        ],
        advection: bool = False,
    ):
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
            dim0 = ("nlev", "ncell",)
            coordinates0 = "lon_cell lat_cell"
            locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
            dimn = ("nlev", "nedge",)
            coordinatesn = "lon_edge lat_edge"
            locz = {"hori": MeshEntity.Cell, "vert": VertLoc.Lev}
            dimz = ("nlevp", "ncell",)
            coordinatesz = "lon_cell lat_cell"
        elif batch_hori_loc == MeshEntity.Cell and batch_vert_loc == VertLoc.Lev:
            loc0 = {"hori": MeshEntity.Cell, "vert": VertLoc.Lev}
            dim0 = ("nlevp", "ncell",)
            coordinates0 = "lon_cell lat_cell"
            locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Lev}
            dimn = ("nlevp", "nedge",)
            coordinatesn = "lon_edge lat_edge"
            locz = {"hori": MeshEntity.Cell, "vert": VertLoc.Cell}
            dimz = ("nlev", "ncell",)
            coordinatesz = "lon_cell lat_cell"
        elif batch_hori_loc == MeshEntity.Vtx and batch_vert_loc == VertLoc.Cell:
            loc0 = {"hori": MeshEntity.Vtx, "vert": VertLoc.Cell}
            dim0 = ("nlev", "nvtx",)
            coordinates0 = "lon_vtx lat_vtx"
            locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
            dimn = ("nlev", "nedge",)
            coordinatesn = "lon_edge lat_edge"
            locz = {"hori": MeshEntity.Vtx, "vert": VertLoc.Lev}
            dimz = ("nlevp", "nvtx",)
            coordinatesz = "lon_vtx lat_vtx"
        else:
            raise NotImplementedError(
                f"Invalid grid location {batch_hori_loc}, {batch_vert_loc}."
            )
        m = Field.create(
            data=coord.zeros_like(loc0["hori"], loc0["vert"], Basis.Scalar),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        mfn = Field.create(
            data=coord.zeros_like(locn["hori"], locn["vert"], Basis.Normal),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        mfz = Field.create(
            data=coord.zeros_like(locz["hori"], locz["vert"], Basis.Scalar),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        un = Field.create(
            data=coord.zeros_like(locn["hori"], locn["vert"], Basis.Normal),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        ut = Field.create(
            data=coord.zeros_like(locn["hori"], locn["vert"], Basis.Tangent),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        w = Field.create(
            data=coord.zeros_like(locz["hori"], locz["vert"], Basis.Scalar),
            meta=StaggeredFieldMetaData(
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
                output="h0" if advection else "",
            ),
            as_quantity=False
        )
        qfn = Field.create(
            data=coord.zeros_like(locn["hori"], locn["vert"], Basis.Normal),
            meta=StaggeredFieldMetaData(
                name=name + "_qfn",
                dims=dimn,
                long_name=name + " normal mass flux at edge",
                description=name + " normal mass flux at edge",
                units="Pa kg kg-1 m s-1",
                hori_location=locn["hori"],
                vert_location=locn["vert"],
                basis=Basis.Normal,
                coordinates=coordinatesn,
                staggered_kind="C",
                mesh_name="primal",
                output="h1",
            ),
            as_quantity=False
        )
        qmfz = Field.create(
            data=coord.zeros_like(locz["hori"], locz["vert"], Basis.Scalar),
            meta=StaggeredFieldMetaData(
                name=name + "_qmfz",
                dims=dimz,
                long_name=name + " vertical mass flux at level",
                description=name + " vertical mass flux at level",
                units="Pa kg kg-1 m s-1",
                hori_location=locz["hori"],
                vert_location=locz["vert"],
                basis=Basis.Scalar,
                coordinates=coordinatesz,
                staggered_kind="C",
                mesh_name="primal",
                output="h1",
            ),
            as_quantity=False
        )
        return cls(
            name=name,
            coord=coord,
            hori_loc=batch_hori_loc,
            vert_loc=batch_vert_loc,
            m=m,
            mfn=mfn,
            mfz=mfz,
            un=un,
            ut=ut,
            w=w,
            qmfn=qfn,
            qmfz=qmfz
        )

    @classmethod
    def empty(cls, coord: CoordinateSystem) -> "AdvBatchType":
        return cls(
            name="",
            coord=coord,
            hori_loc="",
            vert_loc="",
            m=Field.empty(),
            mfn=Field.empty(),
            mfz=Field.empty(),
            un=Field.empty(),
            ut=Field.empty(),
            w=Field.empty(),
            qmfn=Field.empty(),
            qmfz=Field.empty()
        )

    def set_wind(
        self,
        un: Array,
        ut: Array,
        w: Optional[Array] = None,
        mfn: Optional[Array] = None,
        mfz: Optional[Array] = None,
        m: Optional[Array] = None
    ):
        self = eqx.tree_at(
            lambda b: (b.un, b.ut),
            self,
            (un, ut)
        )
        if w is not None:
            self = eqx.tree_at(
                lambda b: b.w,
                self,
                w
            )
        if mfn is not None:
            self = eqx.tree_at(
                lambda b: b.mfn,
                self,
                mfn
            )
        if mfz is not None:
            self = eqx.tree_at(
                lambda b: b.mfz,
                self,
                mfz
            )
        if m is not None:
            self = eqx.tree_at(
                lambda b: b.m,
                self,
                m
            )
