from typing import TypeVar
import jax.numpy as jnp
from loguru import logger

from terraflow.util import typing
from terraflow.vert_coord import AbstractVertCoordinate
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.layout.abstract import (
    AbstractLayout,
    AbstractCGridStaggeringLayout,
    Basis,
    MeshEntity,
    CoordinateInfo,
    CoordinateSystem
)

Array = typing.Array

VM = TypeVar("VM", bound=AbstractVertCoordinate)
VLO = TypeVar("VLO", bound="AbstractLayout")


class AbstractCGridStaggeringLayout4OrthogonalDualMesh(
    AbstractCGridStaggeringLayout
):
    pass


class TriSKBasedCGridStaggeringLayout(
    AbstractCGridStaggeringLayout4OrthogonalDualMesh
):
    def __init__(self):
        logger.success("Initialized TriSKBasedCGridStaggeringLayout.")

    @staticmethod
    def alias_name() -> str:
        return "triskC"

    def shape(
        self,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            VM,
            "TriSKBasedCGridStaggeringLayout",
            VLO
        ],
        loc: str,
        basis: str,
    ) -> tuple[int, ...]:
        coord_info = self.get_layout(coord, loc, basis).horizontal_coord
        if coord_info is None:
            raise ValueError(
                f"Horizontal coordinate information is missing for location: {loc}"
                f" and basis: {basis}"
            )
        return coord_info["lon"].shape

    def get_layout(
        self,
        coord: CoordinateSystem[
            OrthogonalDualMesh,
            VM,
            "TriSKBasedCGridStaggeringLayout",
            VLO
        ],
        loc: str,
        basis: str,
    ) -> CoordinateInfo:
        if loc not in [MeshEntity.Cell, MeshEntity.Vtx, MeshEntity.Edge]:
            raise ValueError(
                f"Unsupported horizontal location for trisk layout: {loc}"
            )
        elif loc == MeshEntity.Cell:
            hori_point = {
                "cart": coord.mesh.primal.cell.centroid.cartesian,
                "lat": coord.mesh.primal.cell.centroid.lat,
                "lon": coord.mesh.primal.cell.centroid.lon,
            }
        elif loc == MeshEntity.Vtx:
            hori_point = {
                "cart": coord.mesh.dual.cell.centroid.cartesian,
                "lat": coord.mesh.dual.cell.centroid.lat,
                "lon": coord.mesh.dual.cell.centroid.lon,
            }
        elif loc == MeshEntity.Edge and (basis == Basis.Normal or basis == Basis.Tangent):
            hori_point = {
                "cart": coord.mesh.dual.edge.midpoint.cartesian,
                "lat": coord.mesh.dual.edge.midpoint.lat,
                "lon": coord.mesh.dual.edge.midpoint.lon,
                "nr_orient": coord.mesh.primal.edge.normal,
                "tg_orient": coord.mesh.primal.edge.tangent,
            }
        elif loc == MeshEntity.Edge and basis == Basis.Scalar:
            hori_point = {
                "cart": coord.mesh.dual.edge.midpoint.cartesian,
                "lat": coord.mesh.dual.edge.midpoint.lat,
                "lon": coord.mesh.dual.edge.midpoint.lon,
            }
        else:
            raise ValueError(
                "Unsupported combination of horizontal location"
                f" {loc} and basis {basis} for trisk layout"
            )
        return CoordinateInfo(
            horizontal_coord=hori_point,
        )
