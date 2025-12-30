from jax import numpy as jnp

from terraflow.util import typing
from terraflow.discretization.operator.abstract import AbstractOperator
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.abstract import OneLayerVertCoordinate
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.abstract import CoordinateSystem


FloatType = typing.FloatType
Int = typing.Int
Float = typing.Float
Array = typing.Array

FVM_Op_ODM_OneLev_TriSK = AbstractOperator[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout
]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout
]


class DivAtPrimalCell(FVM_Op_ODM_OneLev_TriSK):
    """Divergence operator at primal cell center."""

    def __call__(
        self,
        x_at_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nPrimal_cell"]:
        plg_ed = self.mesh.primal.cell.edges  # (nplg, maxnnb)
        plg_nr = self.mesh.primal.cell.orient  # (nplg, maxnnb)
        plg_area = self.mesh.primal.cell.area  # (nplg,)
        edp_length = self.mesh.primal.edge.length  # (nedg,)
        radius = self.para.radius.unwrap()
        ed_idx = plg_ed  # (nplg, maxnnb)
        leng_tile = edp_length[ed_idx]  # (nplg, maxnnb)
        x_tile = x_at_edge[ed_idx]  # (nplg, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(plg_nr * leng_tile * x_tile, axis=-1) / (plg_area * radius)


class GradAtDualEdge(FVM_Op_ODM_OneLev_TriSK):
    """Gradient operator at edge."""

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nPrimal_cell"],
    ) -> Float[Array, "nEdge"]:
        edd_v = self.mesh.dual.edge.endpoints  # (nedg, 2)
        edd_length = self.mesh.dual.edge.length  # (nedg,)
        edpnr_eddtg = self.mesh.aux.edpnr_eddtg  # (nedg,)
        radius = self.para.radius.unwrap()
        x_at_vertex = x_at_primal_cell[edd_v]  # (nedg, 2)
        q0 = x_at_vertex[:, 0]
        q1 = x_at_vertex[:, 1]
        return (q1 - q0) * edpnr_eddtg / (radius * edd_length)


class CurlAtDualCell(FVM_Op_ODM_OneLev_TriSK):
    """Curl operator at dual cell center."""

    def __call__(
        self,
        x_at_dual_edge: Float[Array, "nEdge"],
    ) -> Float[Array, "nDual_cell"]:
        dua_ed = self.mesh.dual.cell.edges  # (ndua, maxnnb)
        dua_nr = self.mesh.dual.cell.orient  # (ndua, maxnnb)
        dua_area = self.mesh.dual.cell.area  # (ndua,)
        edd_length = self.mesh.dual.edge.length  # (nedg,)
        edptg_eddnr = self.mesh.aux.edptg_eddnr  # (nedg,)
        radius = self.para.radius.unwrap()
        ed_idx = dua_ed  # (nDual_cell, maxnnb)
        x_tile = x_at_dual_edge[ed_idx]  # (nDual_cell, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        leng_tile = edd_length[ed_idx]  # (nDual_cell, maxnnb)
        edpTg_eddNr_tile = edptg_eddnr[ed_idx]  # (nDual_cell, maxnnb)
        return jnp.sum(
            x_tile * edpTg_eddNr_tile * dua_nr * leng_tile, axis=-1
        ) / (dua_area * radius)
