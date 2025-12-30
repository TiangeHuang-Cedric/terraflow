from typing import TypeVar
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.vert_coord import LorenzCoordinate
from terraflow.mesh import OrthogonalDualMesh
from terraflow.layout.abstract import AbstractCGridStaggeringLayout
from terraflow.layout.abstract import Basis, MeshEntity, VertLoc
from terraflow.layout.abstract import CoordinateInfo, CoordinateSystem
from terraflow.vert_coord.abstract import OneLayerVertCoordinate

Array = typing.Array

VM = TypeVar("VM", bound=AbstractVertCoordinate)


class AbstractCGridStaggeringLayout4OrthogonalDualMesh(
    AbstractCGridStaggeringLayout
):
    pass


class TriSKBasedCGridStaggeringLayout(
    AbstractCGridStaggeringLayout4OrthogonalDualMesh
):
    @staticmethod
    def alias_name() -> str:
        return "triskC"

    def get_layout(
        self,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            VM,
            "TriSKBasedCGridStaggeringLayout",
        ],
        hori_loc: str,
        vert_loc: str,
        basis: str,
    ) -> CoordinateInfo:
        if hori_loc not in [MeshEntity.Cell, MeshEntity.Edge]:
            raise ValueError(
                f"Unsupported horizontal location for trisk layout: {hori_loc}"
            )
        elif hori_loc == MeshEntity.Cell and basis != Basis.Scalar:
            raise ValueError(
                f"Incompatible basis {basis} for horizontal location {hori_loc}"
            )
        elif hori_loc == MeshEntity.Edge and basis != Basis.Normal:
            raise ValueError(
                f"Incompatible basis {basis} for horizontal location {hori_loc}"
            )
        elif hori_loc == MeshEntity.Cell and basis == Basis.Scalar:
            hori_point = {
                "cart": coord.mesh.primal.cell.centroid.cartesian,
                "lat": coord.mesh.primal.cell.centroid.lat,
                "lon": coord.mesh.primal.cell.centroid.lon,
            }
        elif hori_loc == MeshEntity.Edge and basis == Basis.Normal:
            hori_point = {
                "cart": coord.mesh.dual.edge.midpoint.cartesian,
                "lat": coord.mesh.dual.edge.midpoint.lat,
                "lon": coord.mesh.dual.edge.midpoint.lon,
                "nr_orient": coord.mesh.primal.edge.normal,
                "tg_orient": coord.mesh.primal.edge.tangent,
            }
        else:
            raise ValueError(
                "Unsupported combination of horizontal location"
                f"{hori_loc} and basis {basis} for trisk layout"
            )
        if isinstance(coord.vert, LorenzCoordinate):
            if vert_loc == VertLoc.FullLev:
                vert_point = coord.vert.get_full_lev()
            elif vert_loc == VertLoc.HalfLev:
                vert_point = coord.vert.get_half_lev()
            else:
                raise ValueError(
                    f"Unsupported vertical location for trisk layout: {vert_loc}"
                )
        elif isinstance(coord.vert, OneLayerVertCoordinate):
            vert_point = jnp.array([])
        else:
            raise ValueError(
                f"Unsupported vertical coordinate type: {type(coord.vert)}"
            )
        return CoordinateInfo(
            horizontal_coord=hori_point,
            vertical_coord=vert_point,
        )
