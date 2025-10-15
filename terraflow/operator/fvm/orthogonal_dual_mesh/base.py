from jax import numpy as jnp

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh import PrimeCell, DualCell
from terraflow.mesh.orthogonal_dual_mesh import PrimeEdge, DualEdge
from terraflow.operator.fvm import Divergence, Gradient, Curl

Int = typing.Int
Float = typing.Float
Array = typing.Array

class DivergenceAtPrimeCell(Divergence):
    def __call__(
        self,
        x_at_edge: Array,
        radius: Float,
        plg: PrimeCell,
        edp: PrimeEdge
    ) -> Array:
        ed_idx = plg.edge()  # (nplg, maxnnb)
        leng_tile = edp.length[ed_idx]  # (nplg, maxnnb)
        x_tile = x_at_edge[ed_idx]  # (nplg, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(plg.nr * leng_tile * x_tile, axis=-1) / (plg.area * radius)


class GradientAtEdge(Gradient):
    def __call__(
        self,
        x_at_prime_cell: Array,
        radius: Float,
        edd: DualEdge,
        edpNr_edtTg: Array
    ) -> Array:
        vertex = edd.vertex()  # (nedg, 2)
        x_at_vertex = x_at_prime_cell[vertex]  # (nedg, 2)
        q0 = x_at_vertex[:, 0]
        q1 = x_at_vertex[:, 1]
        return (q1 - q0) * edpNr_edtTg / (radius * edd.length)


class CurlAtDualCell(Curl):
    def __call__(
        self,
        x_at_edge: Array,
        radius: Float,
        dua: DualCell,
        edd: DualEdge,
        edpTg_edtNr: Array
    ) -> Array:
        ed_idx = dua.edge()  # (ndual, maxnnb)
        x_tile = x_at_edge[ed_idx]  # (ndual, maxnnb)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        leng_tile = jnp.take(edd.length, ed_idx)  # (ndual, maxnnb)
        return jnp.sum(x_tile * edpTg_edtNr
                       * dua.nr * leng_tile, axis=-1) / (dua.area * radius)

