from __future__ import annotations
from typing import Optional, Self, TYPE_CHECKING
import equinox as eqx
import jax

from terraflow.util import typing
from terraflow.layout.abstract import MeshEntity, VertLoc, Basis
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.discretization.abstract import AbstractAdvBatchType, AbstractEquationMode

if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block

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

    CoordSys = CoordSysOneLayer | CoordSysHybrid


FDT = ArrayField[StaggeredFieldMetaData]

Int = typing.Int
Float = typing.Float
Array = typing.Array

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


class AdvBatchType(AbstractAdvBatchType[StaggeredFieldMetaData]):
    m: FDT     # Dry-air weight at n time level
    mstp: FDT  # Dry-air weight for averaging in substeps
    mfn: FDT   # Mass flux in normal direction
    mfz: FDT   # Mass flux in z direction
    un: FDT
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

    def associate(
        self,
        un: Optional[Array] = None,
        w: Optional[Array] = None,
        mfn: Optional[Array] = None,
        mfz: Optional[Array] = None,
        m: Optional[Array] = None,
    ) -> Self:
        if un is not None:
            self = eqx.tree_at(
                lambda b: b.un.data,
                self,
                un
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

    def set_new_mass(
        self,
        block: Block,
        substep: int,
    ) -> Self:
        def on_last_substep() -> Self:
            m_new = block.adv_batch_bg.m.unwrap()
            return eqx.tree_at(
                lambda b: b.m.data,
                self,
                m_new
            )

        def not_on_last_substep() -> Self:
            return self

        return jax.lax.cond(
            pred=substep == self.nsubsteps - 1,
            true_fun=on_last_substep,
            false_fun=not_on_last_substep
        )

    def set_wind(
        self,
        un: Float[Array, "nlev nedge"],
        w: Optional[Float[Array, "nlevp ncell"]] = None,
        mfn: Optional[Float[Array, "nlev nedge"]] = None,
        mfz: Optional[Float[Array, "nlevp ncell"]] = None,
        m: Optional[Float[Array, "nlev ncell"]] = None,
    ) -> Self:
        return self.associate(
            un=un,
            w=w,
            mfn=mfn,
            mfz=mfz,
            m=m
        )

    def accum_wind(
        self,
        coord: CoordSys,
        m: Float[Array, "nlev ncell"],
        mfn: Float[Array, "nlev nedge"],
        mfz: Optional[Float[Array, "nlevp ncell"]] = None,
    ) -> Self:

        def istep_at_reset(batch: Self) -> Self:
            # istep == 0
            if mfz is not None:
                return eqx.tree_at(
                    lambda b: (b.mstp.data, b.mfn.data, b.mfz.data),
                    batch,
                    (m, mfn, mfz)
                )
            else:
                return eqx.tree_at(
                    lambda b: (b.mstp.data, b.mfn.data),
                    batch,
                    (m, mfn)
                )

        def istep_at_end(batch: Self) -> Self:
            # istep == nsteps -1
            edd_v = coord.mesh._edd.v
            cell_0 = edd_v[:, 0]
            cell_1 = edd_v[:, 1]
            nsteps = batch.nsteps

            m_ = (batch.mstp.data + m) / nsteps
            mfn_ = (batch.mfn.data + mfn) / nsteps
            m_e_ = (m_[:, cell_1] + m_[:, cell_0]) / 2
            un_ = mfn_ / m_e_

            if mfz is not None:
                mfz_ = (batch.mfz.data + mfz) / nsteps
                return eqx.tree_at(
                    lambda b: (b.un.data, b.mstp.data, b.mfn.data, b.mfz.data),
                    batch,
                    (un_, m_, mfn_, mfz_)
                )
            else:
                return eqx.tree_at(
                    lambda b: (b.un.data, b.mstp.data, b.mfn.data),
                    batch,
                    (un_, m_, mfn_)
                )

        def istep_at_accumulating(batch: Self) -> Self:
            # 0 < istep < nsteps -1
            m_ = batch.mstp.data + m
            mfn_ = batch.mfn.data + mfn
            if mfz is not None:
                mfz_ = batch.mfz.data + mfz
                return eqx.tree_at(
                    lambda b: (b.mstp.data, b.mfn.data, b.mfz.data),
                    batch,
                    (m_, mfn_, mfz_)
                )
            else:
                return eqx.tree_at(
                    lambda b: (b.mstp.data, b.mfn.data),
                    batch,
                    (m_, mfn_)
                )

        return jax.lax.cond(
            pred=self.istep == 0,
            true_fun=istep_at_reset,
            false_fun=lambda batch: jax.lax.cond(
                pred=batch.istep == batch.nsteps - 1,
                true_fun=istep_at_end,
                false_fun=istep_at_accumulating,
                operand=batch
            ),
            operand=self
        )


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

    if mode.is_barotropic() or mode.is_advection2d():
        # For barotropic mode
        if batch_hori_loc == MeshEntity.Cell and batch_vert_loc == "":
            loc0 = {"hori": MeshEntity.Cell, "vert": VertLoc.Cell}
            dim0 = (NCELL,)
            coordinates0 = "lon_cell lat_cell"
            locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
            dimn = (NEDGE,)
            coordinatesn = "lon_edge lat_edge"
            locz = {}
            dimz = ()
            coordinatesz = ""
        elif batch_hori_loc == MeshEntity.Vtx and batch_vert_loc == "":
            loc0 = {"hori": MeshEntity.Vtx, "vert": VertLoc.Cell}
            dim0 = (NLEV, NVTX,)
            coordinates0 = "lon_vtx lat_vtx"
            locn = {"hori": MeshEntity.Edge, "vert": VertLoc.Cell}
            dimn = (NLEV, NEDGE,)
            coordinatesn = "lon_edge lat_edge"
            locz = {}
            dimz = ()
            coordinatesz = ""
        else:
            raise NotImplementedError(
                f"Invalid batch location {batch_hori_loc}, {batch_vert_loc}."
            )
    elif mode.is_baroclinic() or mode.is_advection3d():
        # For baroclinic mode
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
                f"Invalid batch location {batch_hori_loc}, {batch_vert_loc}."
            )
    else:
        raise NotImplementedError(
            f"Invalid equation mode {mode} for batch metadata generation."
        )

    meta: dict[str, StaggeredFieldMetaData] = {}
    if mode.is_baroclinic() or mode.is_advection():
        meta["m"] = StaggeredFieldMetaData(
            name=name + "_m",
            dims=dim0,
            long_name="Dry air weight",
            description="Dry air weight",
            units="Pa",
            hori_location=loc0["hori"],
            vert_location=loc0["vert"],
            basis=Basis.Scalar,
            coordinates=coordinates0,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.is_advection() else "",
        )
        meta["mstp"] = StaggeredFieldMetaData(
            name=name + "_mstp",
            dims=dim0,
            long_name="Dry air weight",
            description="Dry air weight",
            units="Pa",
            hori_location=loc0["hori"],
            vert_location=loc0["vert"],
            basis=Basis.Scalar,
            coordinates=coordinates0,
            staggered_kind="C",
            mesh_name="primal",
            output="",
        )
        meta["mfn"] = StaggeredFieldMetaData(
            name=name + "_mfn",
            dims=dimn,
            long_name="Dry air weight at edge",
            description="Dry air weight at edge",
            units="Pa",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.is_advection() else "",
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["mfz"] = StaggeredFieldMetaData(
            name=name + "_mfz",
            dims=dimz,
            long_name="Dry air weight at level",
            description="Dry air weight at level",
            units="Pa",
            hori_location=locz["hori"],
            vert_location=locz["vert"],
            basis=Basis.Scalar,
            coordinates=coordinatesz,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.is_advection() else "",
        )
    if mode.is_baroclinic() or mode.is_advection():
        meta["un"] = StaggeredFieldMetaData(
            name=name + "_un",
            dims=dimn,
            long_name="Normal wind at edge",
            description="Normal wind at edge",
            units="m s-1",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.is_advection() else "",
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["w"] = StaggeredFieldMetaData(
            name=name + "_w",
            dims=dimz,
            long_name="Vertical coordinate velocity",
            description="Vertical coordinate velocity",
            units="s-1",
            hori_location=locz["hori"],
            vert_location=locz["vert"],
            basis=Basis.Scalar,
            coordinates=coordinatesz,
            staggered_kind="C",
            mesh_name="primal",
            output="h0" if mode.is_advection() else "",
        )
    if mode.is_baroclinic() or mode.is_advection():
        meta["qmfn"] = StaggeredFieldMetaData(
            name=name + "_qmfn",
            dims=dimn,
            long_name="Tracer mass flux in normal direction",
            description="Tracer mass flux in normal direction",
            units="Pa kg kg-1 m s-1",
            hori_location=locn["hori"],
            vert_location=locn["vert"],
            basis=Basis.Normal,
            coordinates=coordinatesn,
            staggered_kind="C",
            mesh_name="primal",
            output="h1",
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        meta["qmfz"] = StaggeredFieldMetaData(
            name=name + "_qmfz",
            dims=dimz,
            long_name="Tracer mass flux in z direction",
            description="Tracer mass flux in z direction",
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
