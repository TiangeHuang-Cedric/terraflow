import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operators as mesh_op
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout


CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]


Float = typing.Float
Array = typing.Array


def delta_at_full_lev(
    x_lev: Float[Array, "nlev+1 n_points"],
) -> Float[Array, "nlev n_points"]:
    return x_lev[1:, :] - x_lev[:-1, :]


def delta_at_half_lev(
    x: Float[Array, "nlev n_points"],
    x_lev: Float[Array, "nlev+1 n_points"],
) -> Float[Array, "nlev+1 n_points"]:
    top = (x[0, :] - x_lev[0, :])[None, :]
    mid = x[1:, :] - x[:-1, :]
    bottom = (x_lev[-1, :] - x[-1, :])[None, :]
    return jnp.concatenate([top, mid, bottom], axis=0)


def adv_hori_at_half_lev(
    coord: CoordSys,
    para: PlanetParameterSet,
    x_lev: Float[Array, "nlev+1 n_points"],
    un_lev: Float[Array, "nlev+1 n_points"],
    dmg_lev: Float[Array, "nlev+1 n_points"]
) -> Float[Array, "nlev+1 n_points"]:
    term1 = mesh_op.div_at_primal_cell(
        x_at_edge=x_lev * un_lev * dmg_lev,
        coord=coord,
        para=para,

    )
    term2 = x_lev * mesh_op.div_at_primal_cell(
        x_at_edge=un_lev * dmg_lev,
        coord=coord,
        para=para,
    )
    return (term1 - term2) / dmg_lev


def adv_vert_at_full_lev(
    x: Float[Array, "nlev n_points"],
    x_lev: Float[Array, "nlev+1 n_points"],
    mfz_lev: Float[Array, "nlev+1 n_points"],
    dmg: Float[Array, "nlev n_points"]
) -> Float[Array, "nlev n_points"]:
    term1 = mfz_lev[1:, :] * (x_lev[1:, :] - x[:, :])
    term2 = mfz_lev[:-1, :] * (x[:, :] - x_lev[:-1, :])
    return (term1 + term2) / dmg


class InterpMethod:
    def __init__(
        self,
    ):
        self.switch_set = {
            "edge>lev": 0,
            "cell>edge": 1
        }
        self.interp_fn = [self.edge2lev, self.cell2edge]

    def __call__(
        self,
        coord: CoordSys,
        x: ArrayField[StaggeredFieldMetaData],
        y: ArrayField[StaggeredFieldMetaData],
        **kwargs
    ) -> ArrayField[StaggeredFieldMetaData]:
        x_data = x.unwrap()
        loc = x.meta.vert_location + ">" + y.meta.vert_location
        interp_fn = self.interp_fn[self.switch_set[loc]]
        y_data = interp_fn(
            coord=coord,
            x=x_data,
            **kwargs
        )
        return eqx.tree_at(
            lambda f: f.data,
            y,
            y_data
        )

    def cell2edge(
        self,
        x: Float[Array, "nlev n_cells"],
        coord: CoordSys,
    ) -> Float[Array, "nlev n_edges"]:
        edd_v = coord.mesh.dual.edge.endpoints  # (nedge, 2)
        return 0.5 * (x[:, edd_v[:, 0]] + x[:, edd_v[:, 1]])

    def lev2edge(
        self,
        x: Float[Array, "nlev+1 n_cells"],
        coord: CoordSys,
    ) -> Float[Array, "nlev+1 n_edges"]:
        edd_v = coord.mesh.dual.edge.endpoints  # (nedge, 2)
        return 0.5 * (x[:, edd_v[:, 0]] + x[:, edd_v[:, 1]])

    def edge2lev(
        self,
        x: Float[Array, "nlev n_points"],
        coord: CoordSys,
        extrap_opt: bool = False
    ) -> Float[Array, "nlev+1 n_points"]:
        # ------- k-1-1/2
        #
        # ===o=== k-1
        #
        # ---?--- k-1/2
        #
        # ===o=== k
        #
        # ------- k+1/2
        nlev = x.shape[0]
        n_points = x.shape[1]
        full_dlev = coord.vert.full_dlev[:, None]
        half_dlev = coord.vert.half_dlev[:, None]
        dtype = x.dtype
        x_at_half_lev = jnp.zeros((nlev + 1, n_points), dtype=dtype)
        a = full_dlev[:-1, :] / (2 * half_dlev[1:-1, :])
        b = full_dlev[1:, :] / (2 * half_dlev[1:-1, :])
        tmp = a * x[1:, :] + b * x[:-1, :]
        x_at_half_lev = x_at_half_lev.at[1:-1, :].set(tmp)
        if extrap_opt:
            raise NotImplementedError
        else:
            tmp = x[0, :]
            x_at_half_lev = x_at_half_lev.at[0, :].set(tmp)
            tmp = x[-1, :]
            x_at_half_lev = x_at_half_lev.at[-1, :].set(tmp)
        return x_at_half_lev


def exner_at_full_lev(
    coord: CoordSys,
    para: PlanetParameterSet,
    mg: Float[Array, "nlev n_points"],
) -> Float[Array, "nlev n_points"]:
    rd = para.rd.unwrap()
    cp = para.cp.unwrap()
    p0 = coord.vert.p0
    return (mg / p0) ** (rd / cp)
