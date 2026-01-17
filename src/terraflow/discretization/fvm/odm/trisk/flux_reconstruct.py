from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Tuple
from functools import partial
import jax.numpy as jnp
import jax
import equinox as eqx

from terraflow.util import typing, jax_tools, math
from terraflow.planet import PlanetParameterSet
from terraflow.operator import FiniteVolumeMethodOperator
from terraflow.mesh.orthogonal_dual_mesh.construct import (
    AbstractMeshEntity,
    PrimalMeshEntity,
    DualMeshEntity,
    OrthogonalDualMesh
)
from terraflow.vert_coord.abstract import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout

Int = typing.Int
Float = typing.Float
IntType = typing.IntType
FloatType = typing.FloatType
Array = typing.Array

VM = TypeVar("VM", OneLayerVertCoordinate, HybridVertCoordinate)

FVM_ODM_TriSK = FiniteVolumeMethodOperator[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout
]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    VM,
    TriSKBasedCGridStaggeringLayout
]


class CentralDifference2AtPrimalEdge(FVM_ODM_TriSK):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para)

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nlev nCell"],
        *args,
    ) -> Float[Array, "nlev nEdge"]:
        return jax.vmap(
            central_difference_2nd_order,
            in_axes=(0, None),
            out_axes=0
        )(x_at_primal_cell, self.mesh.dual.edge.endpoints)


class CentralDifference2AtDualEdge(FVM_ODM_TriSK):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para)

    def __call__(
        self,
        x_at_dual_cell: Float[Array, "nlev nCell"],
        *args,
        **kwargs,
    ) -> Float[Array, "nlev nEdge"]:
        return jax.vmap(
            central_difference_2nd_order,
            in_axes=(0, None),
            out_axes=0
        )(x_at_dual_cell, self.mesh.primal.edge.endpoints)


class AbstractWRF34AtPrimalEdge(FVM_ODM_TriSK):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType
    ):
        super().__init__(coord, para)
        stencil_idxes, edd_v_fcoe = init_primal_bmatrix_at_dual_edge(
            coord.mesh, 2
        )
        self.beta = beta
        self.stencil_2nd_idxes = stencil_idxes
        self.edd_2nd_v_fcoe = edd_v_fcoe

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nlev nCell"],
        normal_u_at_edge: Float[Array, "nlev nEdge"]
    ):
        return jax.vmap(
            wrf_3rd4th_order,
            in_axes=(0, 0, None, None, None, None, None, None),
            out_axes=0
        )(
            x_at_primal_cell,
            normal_u_at_edge,
            self.mesh.dual.edge.endpoints,
            self.mesh.dual.edge.length,
            self.mesh.aux.edpnr_eddtg,
            self.stencil_2nd_idxes,
            self.edd_2nd_v_fcoe,
            self.beta
        )


class WRF3AtPrimalEdge(AbstractWRF34AtPrimalEdge):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para, beta=1.0)


class WRF4AtPrimalEdge(AbstractWRF34AtPrimalEdge):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para, beta=0.0)


class HybridWRF34AtPrimalEdge(AbstractWRF34AtPrimalEdge):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType
    ):
        super().__init__(coord, para, beta)


class AbstractWRF56AtPrimalEdge(FVM_ODM_TriSK):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType
    ):
        super().__init__(coord, para)
        stencil_2nd_idxes, edd_2nd_v_fcoe = init_primal_bmatrix_at_dual_edge(
            coord.mesh, 2
        )
        stencil_4th_idxes, edd_4th_v_fcoe = init_primal_bmatrix_at_dual_edge(
            coord.mesh, 4
        )
        self.stencil_2nd_idxes = stencil_2nd_idxes
        self.edd_2nd_v_fcoe = edd_2nd_v_fcoe
        self.stencil_4th_idxes = stencil_4th_idxes
        self.edd_4th_v_fcoe = edd_4th_v_fcoe
        self.beta = beta

    def __call__(
        self,
        x_at_primal_cell: Float[Array, "nlev nCell"],
        normal_u_at_edge: Float[Array, "nlev nEdge"]
    ):
        return jax.vmap(
            wrf_5th6th_order,
            in_axes=(0, 0, None, None, None, None, None, None, None, None),
            out_axes=0
        )(
            x_at_primal_cell,
            normal_u_at_edge,
            self.mesh.dual.edge.endpoints,
            self.mesh.dual.edge.length,
            self.mesh.aux.edpnr_eddtg,
            self.stencil_2nd_idxes,
            self.edd_2nd_v_fcoe,
            self.stencil_4th_idxes,
            self.edd_4th_v_fcoe,
            self.beta
        )


class WRF5AtPrimalEdge(AbstractWRF56AtPrimalEdge):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para, beta=1.0)


class WRF6AtPrimalEdge(AbstractWRF56AtPrimalEdge):

    def __init__(self, coord: CoordSys, para: PlanetParameterSet):
        super().__init__(coord, para, beta=0.0)


class HybridWRF56AtPrimalEdge(AbstractWRF56AtPrimalEdge):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType
    ):
        super().__init__(coord, para, beta=beta)


class AbstractWRF34AtDualEdge(FVM_ODM_TriSK):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType,
        use_mbs: bool
    ):
        super().__init__(coord, para)
        stencil_idxes, edp_v_fcoe = init_dual_bmatrix_at_primal_edge(
            coord.mesh, 2, use_mbs=use_mbs
        )
        self.beta = beta
        self.stencil_2nd_idxes = stencil_idxes
        self.edp_2nd_v_fcoe = edp_v_fcoe

    def __call__(
        self,
        x_at_dual_cell: Float[Array, "nlev nCell"],
        tangent_u_at_edge: Float[Array, "nlev nEdge"]
    ):
        return jax.vmap(
            wrf_3rd4th_order,
            in_axes=(0, 0, None, None, None, None, None, None),
            out_axes=0
        )(
            x_at_dual_cell,
            tangent_u_at_edge,
            self.mesh.primal.edge.endpoints,
            self.mesh.primal.edge.length,
            self.mesh.aux.edp12_edptg,
            self.stencil_2nd_idxes,
            self.edp_2nd_v_fcoe,
            self.beta
        )


class WRF3AtDualEdge(AbstractWRF34AtDualEdge):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=1.0, use_mbs=use_mbs)


class WRF4AtDualEdge(AbstractWRF34AtDualEdge):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=0.0, use_mbs=use_mbs)


class HybridWRF34AtDualEdge(AbstractWRF34AtDualEdge):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=beta, use_mbs=use_mbs)


class AbstractWRF56AtDualEdge(FVM_ODM_TriSK):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType,
        use_mbs: bool
    ):
        super().__init__(coord, para)
        stencil_2nd_idxes, edp_2nd_v_fcoe = init_dual_bmatrix_at_primal_edge(
            coord.mesh, 2, use_mbs=use_mbs
        )
        stencil_4th_idxes, edp_4th_v_fcoe = init_dual_bmatrix_at_primal_edge(
            coord.mesh, 4, use_mbs=use_mbs
        )
        self.stencil_2nd_idxes = stencil_2nd_idxes
        self.edp_2nd_v_fcoe = edp_2nd_v_fcoe
        self.stencil_4th_idxes = stencil_4th_idxes
        self.edp_4th_v_fcoe = edp_4th_v_fcoe
        self.beta = beta

    def __call__(
        self,
        x_at_dual_cell: Float[Array, "nlev nCell"],
        tangent_u_at_edge: Float[Array, "nlev nEdge"]
    ):
        return jax.vmap(
            wrf_5th6th_order,
            in_axes=(0, 0, None, None, None, None, None, None, None, None),
            out_axes=0
        )(
            x_at_dual_cell,
            tangent_u_at_edge,
            self.mesh.primal.edge.endpoints,
            self.mesh.primal.edge.length,
            self.mesh.aux.edp12_edptg,
            self.stencil_2nd_idxes,
            self.edp_2nd_v_fcoe,
            self.stencil_4th_idxes,
            self.edp_4th_v_fcoe,
            self.beta
        )


class WRF5AtDualEdge(AbstractWRF56AtDualEdge):
    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=1.0, use_mbs=use_mbs)


class WRF6AtDualEdge(AbstractWRF56AtDualEdge):
    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=0.0, use_mbs=use_mbs)


class HybridWRF56AtDualEdge(AbstractWRF56AtDualEdge):
    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
        beta: FloatType,
        use_mbs: bool = True
    ):
        super().__init__(coord, para, beta=beta, use_mbs=use_mbs)


def central_difference_2nd_order(
    x_at_cell: Float[Array, "nCell"],
    edge_v: Int[Array, "nEdge 2"]
) -> Float[Array, "nEdge"]:
    vertex = edge_v
    v1 = vertex[:, 0]
    v2 = vertex[:, 1]
    x1 = x_at_cell[v1]
    x2 = x_at_cell[v2]
    return 0.5 * (x1 + x2)


def wrf_3rd4th_order(
    x_at_cell: Float[Array, "nCell"],
    normal_u: Float[Array, "nEdge"],
    edd_v: Int[Array, "nEdge 2"],
    edd_leng: Float[Array, "nEdge"],
    direction_marker: Int[Array, "nEdge"],
    stencil_2nd_idx: Int[Array, "nCell max_m"],
    edd_2nd_v_fcoe: Float[Array, "nEdge 2 max_m"],
    beta: FloatType = 0.75,
) -> Float[Array, "nEdge"]:
    wind = normal_u * direction_marker  # (nEdge, )
    v1_idx = edd_v[:, 0]  # (nEdge, )
    v2_idx = edd_v[:, 1]  # (nEdge, )
    x1 = x_at_cell[v1_idx]  # (nEdge, )
    x2 = x_at_cell[v2_idx]  # (nEdge, )
    dx1_2nd = calc_2nd_derivative(
        v1_idx, 0, x_at_cell, edd_leng, stencil_2nd_idx, edd_2nd_v_fcoe
    )
    dx2_2nd = calc_2nd_derivative(
        v2_idx, 1, x_at_cell, edd_leng, stencil_2nd_idx, edd_2nd_v_fcoe
    )
    return operator_3rd4th_order_formular(
        x1, x2, dx1_2nd, dx2_2nd, wind, beta
    )


def wrf_5th6th_order(
    x_at_cell: Float[Array, "nCell"],
    normal_u: Float[Array, "nEdge"],
    edd_v: Int[Array, "nEdge 2"],
    edd_leng: Float[Array, "nEdge"],
    direction_marker: Int[Array, "nEdge"],
    stencil_2nd_idx: Int[Array, "nCell max_m"],
    edd_2nd_v_fcoe: Float[Array, "nEdge 2 max_m"],
    stencil_4th_idx: Int[Array, "nCell max_m"],
    edd_4th_v_fcoe: Float[Array, "nEdge 2 max_m"],
    beta: FloatType = 0.75,
) -> Float[Array, "nEdge"]:
    wind = normal_u * direction_marker  # (nEdge, )
    v1_idx = edd_v[:, 0]  # (nEdge, )
    v2_idx = edd_v[:, 1]  # (nEdge, )
    x1 = x_at_cell[v1_idx]  # (nEdge, )
    x2 = x_at_cell[v2_idx]  # (nEdge, )
    dx1_2nd = calc_2nd_derivative(
        v1_idx, 0, x_at_cell, edd_leng, stencil_2nd_idx, edd_2nd_v_fcoe
    )
    dx2_2nd = calc_2nd_derivative(
        v2_idx, 1, x_at_cell, edd_leng, stencil_2nd_idx, edd_2nd_v_fcoe
    )
    dx1_4th = calc_4th_derivative(
        v1_idx, 0, x_at_cell, edd_leng, stencil_4th_idx, edd_4th_v_fcoe
    )
    dx2_4th = calc_4th_derivative(
        v2_idx, 1, x_at_cell, edd_leng, stencil_4th_idx, edd_4th_v_fcoe
    )
    return operator_5th6th_order_formular(
        x1, x2, dx1_2nd, dx2_2nd, dx1_4th, dx2_4th, wind, beta
    )


ME = TypeVar("ME", bound=AbstractMeshEntity)


class AbstractFluxReconstructInfo(eqx.Module, Generic[ME], ABC):
    n_coeff: int = eqx.field(static=True)
    polynomial_order: int = eqx.field(static=True)
    recon_polynomial_fn: Callable = eqx.field(static=True)
    spec_loc: int = eqx.field(static=True)
    polynomial_coef: int = eqx.field(static=True)
    mesh_type: str = eqx.field(static=True)
    max_member: int = eqx.field(static=True)
    stencil_fn: Callable = eqx.field(static=True)

    def __init__(self, polynomial_order: int):
        if polynomial_order == 2:
            self.n_coeff = 5
            self.polynomial_order = polynomial_order
            self.recon_polynomial_fn = recon_2nd_polynomial
            self.spec_loc = 3 - 1
            self.polynomial_coef = 2
        elif polynomial_order == 4:
            self.n_coeff = 14
            self.polynomial_order = polynomial_order
            self.recon_polynomial_fn = recon_4th_polynomial
            self.spec_loc = 10 - 1
            self.polynomial_coef = 24
        else:
            raise ValueError("Only 2nd and 4th order least square supported.")

    @abstractmethod
    def calc_stencil(
        self,
        mesh_entity: ME
    ) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
        raise NotImplementedError

    def recon(
        self,
        local_x: Int[Array, "max_m"],
        local_y: Int[Array, "max_m"],
    ) -> Float[Array, "n_coeff max_m"]:
        b_matrix = self.recon_polynomial_fn(
            local_x, local_y
        )  # (max_m, n_coeff)
        recon_b_matrix = b_matrix.T @ b_matrix  # (n_coeff, n_coeff)
        recon_b_matrix = math.pinv_svd(recon_b_matrix)  # (n_coeff, n_coeff)
        return recon_b_matrix @ b_matrix.T  # (n_coeff, max_m)


class FluxReconstructInfoAtPrimalMesh(AbstractFluxReconstructInfo[PrimalMeshEntity]):
    def __init__(self, polynomial_order: int):
        super().__init__(polynomial_order)
        self.mesh_type = "primal"
        if self.polynomial_order == 2:
            self.max_member = 6
            self.stencil_fn = hx_stencil_member_list_2nd_order
        elif self.polynomial_order == 4:
            self.max_member = 18
            self.stencil_fn = hx_stencil_member_list_4th_order

    def calc_stencil(
        self, mesh_entity: PrimalMeshEntity
    ) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
        return self.stencil_fn(mesh_entity)


class FluxReconstructInfoAtDualMesh(AbstractFluxReconstructInfo[DualMeshEntity]):
    def __init__(self, polynomial_order: int, use_mbs: bool = True):
        super().__init__(polynomial_order)
        self.mesh_type = "dual"
        if self.polynomial_order == 2:
            if use_mbs:
                self.max_member = 12
                self.stencil_fn = tr_stencil_member_list_2nd_order_mbs
            else:
                self.max_member = 9
                self.stencil_fn = tr_stencil_member_list_2nd_order
        elif self.polynomial_order == 4:
            if use_mbs:
                self.max_member = 21
                self.stencil_fn = tr_stencil_member_list_4th_order_mbs
            else:
                self.max_member = 18
                self.stencil_fn = tr_stencil_member_list_4th_order

    def calc_stencil(
        self,
        mesh_entity: DualMeshEntity
    ) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
        return self.stencil_fn(mesh_entity)


def calc_2nd_derivative(
    global_iv: Int[Array, "nEdge"],
    local_iv: IntType,
    x_at_cell: Float[Array, "nCell"],
    leng: Float[Array, "nEdge"],
    stencil_idx: Int[Array, "nCell max_m"],
    edd_v_fcoe: Float[Array, "nEdge 2 max_m"],
) -> Float[Array, "nEdge"]:
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = edd_v_fcoe[:, local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_2nd = 2 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_2nd = dqdx_2nd * leng ** 2  # (nEdge)
    return dq_2nd


def operator_3rd4th_order_formular(
    q1: Float[Array, "nEdge"],
    q2: Float[Array, "nEdge"],
    dq1_2nd: Float[Array, "nEdge 2"],
    dq2_2nd: Float[Array, "nEdge 2"],
    normal_u: Float[Array, "nEdge"],
    beta: FloatType,
) -> Float[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - (dq1_2nd + dq2_2nd) / 12.0
    qe = qe + jnp.sign(normal_u) * beta * (dq2_2nd - dq1_2nd) / 12.0
    return qe


def operator_5th6th_order_formular(
    q1: Float[Array, "nEdge"],
    q2: Float[Array, "nEdge"],
    dq1_2nd: Float[Array, "nEdge 2"],
    dq2_2nd: Float[Array, "nEdge 2"],
    dq1_4th: Float[Array, "nEdge 2"],
    dq2_4th: Float[Array, "nEdge 2"],
    normal_u: Float[Array, "nEdge"],
    beta: FloatType,
) -> Float[Array, "nEdge"]:
    qe = 0.5 * (q1 + q2)
    qe = qe - (dq1_2nd + dq2_2nd) / 12.0 + (dq1_4th + dq2_4th) / 60.0
    qe = qe - jnp.sign(normal_u) * beta * (dq2_4th - dq1_4th) / 60.0
    return qe


def calc_4th_derivative(
    global_iv: Int[Array, "nEdge"],
    local_iv: IntType,
    x_at_cell: Float[Array, "nCell"],
    leng: Float[Array, "nEdge"],
    stencil_idx: Int[Array, "nCell max_m"],
    edd_v_fcoe: Float[Array, "nEdge 2 max_m"],
) -> Float[Array, "nEdge"]:
    q0 = x_at_cell[global_iv]  # (nEdge)
    fcoe = edd_v_fcoe[:, local_iv, :]  # (nEdge, max_m)
    idx = stencil_idx[global_iv]  # (nEdge, max_m)
    q = x_at_cell[idx]  # (nEdge, max_m)
    s = q - q0[:, None]  # (nEdge, max_m)
    dqdx_4th = 24 * jnp.sum(fcoe * s, axis=-1)  # (nEdge)
    dq_4th = dqdx_4th * leng ** 4  # (nEdge)
    return dq_4th


def hx_stencil_member_list_2nd_order(
    primal: PrimalMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """5/6 point stencil for 2nd order reconstruction on hexagonal mesh."""
    stencil_num = primal.cell.n_adjacency
    stencil_idx = primal.cell.adjacency
    return stencil_num, stencil_idx


def hx_stencil_member_list_4th_order(
    primal: PrimalMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """15/17/18 point stencil for 4th order reconstruction on hexagonal mesh."""
    plg_nb = primal.cell.adjacency  # (ncell, maxnnb)
    ncell = plg_nb.shape[0]
    maxnnb = plg_nb.shape[1]
    max_m: int = 18

    def cell_body(_, cell_idx):
        nb = plg_nb[cell_idx, :]  # (maxnnb,)
        valid_nb = nb >= 0  # (maxnnb,)
        # padding invalid index value using -1
        nb_safe = jnp.where(valid_nb, nb, -1)  # (maxnnb,)
        nb_of_nb = plg_nb[nb_safe]  # (maxnnb, maxnnb)
        # Mask out the home cell contributions
        nb_of_nb = jnp.where(
            nb_of_nb == cell_idx, -1, nb_of_nb
        )  # (maxnnb, maxnnb)
        # Mask out contributions from invalid neighbors
        # filter out horizontal invalid neighbors
        nb_of_nb = jnp.where(
            valid_nb[:, None], nb_of_nb, -1
        )  # (maxnnb, maxnnb)
        # filter out vertical invalid neighbors
        nb_of_nb = jnp.where(
            valid_nb[None, :], nb_of_nb, -1
        )  # (maxnnb, maxnnb)
        stencil_idx = nb_of_nb.reshape((maxnnb * maxnnb,))  # (maxnnb*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=max_m, fill_value=-1
        )  # (max_m,)
        return None, stencil_idx

    _, stencil_idx = jax.lax.scan(cell_body, init=None, xs=jnp.arange(ncell))
    stencil_num = jnp.sum(stencil_idx >= 0, axis=-1)
    return stencil_num, stencil_idx


def tr_stencil_member_list_2nd_order(
    dual: DualMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """9/10 point stencil for 2nd order reconstruction on triangular mesh."""
    tri_nb = dual.cell.adjacency  # (nDual_cell, 3)
    ncell = tri_nb.shape[0]
    maxnnb = tri_nb.shape[1]

    def cell_body(_, cell_idx):
        nb = tri_nb[cell_idx, :]  # (3,)
        nb_of_nb = tri_nb[nb, :]  # (3, 3)
        # padding invalid index value using -1
        nb_of_nb = jnp.where(
            nb_of_nb >= 0, nb_of_nb, -1
        )  # (3, maxnnb)
        # Mask out the home cell contributions
        nb_of_nb = jnp.where(
            nb_of_nb == cell_idx, -1, nb_of_nb
        )  # (3, maxnnb)
        # Gather all neighboring cells
        nb_of_nb = nb_of_nb.reshape((3 * maxnnb, ))  # (3, maxnnb)
        stencil_idx = jnp.concatenate(
            [nb, nb_of_nb],
            axis=0
        )  # (3 + 3*maxnnb,)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=9, fill_value=-1
        )  # (9,)
        return None, stencil_idx

    _, stencil_idx = jax.lax.scan(cell_body, None, xs=jnp.arange(ncell))
    stencil_num = jnp.sum(stencil_idx >= 0, axis=-1)
    return stencil_num, stencil_idx


def tr_stencil_member_list_2nd_order_mbs(
    dual: DualMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """11/12 point stencil for 2nd order reconstruction on triangular mesh."""
    plg_vtx = dual.node.adjacent_cells  # (nPrimal_cell, maxnnb)
    tri_nb = dual.cell.adjacency  # (nDual_cell, 3)
    tri_vtx = dual.cell.nodes  # (nDual_cell, 3)
    ncell = tri_nb.shape[0]
    maxnnb = plg_vtx.shape[1]

    def cell_body(_, cell_idx):
        vtx = tri_vtx[cell_idx, :]  # (3,)
        nbtr_of_vtx = plg_vtx[vtx, :]  # (3, maxnnb)
        # padding invalid index value using -1
        nbtr_of_vtx = jnp.where(
            nbtr_of_vtx >= 0, nbtr_of_vtx, -1
        )  # (3, maxnnb)
        # Mask out the home cell contributions
        nbtr_of_vtx = jnp.where(
            nbtr_of_vtx == cell_idx, -1, nbtr_of_vtx
        )  # (3, maxnnb)
        stencil_idx = nbtr_of_vtx.reshape((3 * maxnnb,))  # (3*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=12, fill_value=-1
        )  # (12,)
        return None, stencil_idx

    _, stencil_idx = jax.lax.scan(cell_body, None, xs=jnp.arange(ncell))
    stencil_num = jnp.sum(stencil_idx >= 0, axis=-1)
    return stencil_num, stencil_idx


def tr_stencil_member_list_4th_order(
    dual: DualMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """17/18 point stencil for 4th order reconstruction on triangular mesh."""
    tri_nb = dual.cell.adjacency  # (nDual_cell, 3)
    tri_vtx = dual.cell.nodes  # (nDual_cell, 3)
    plg_vtx = dual.node.adjacent_cells  # (nPrimal_cell, maxnnb)
    ncell = tri_nb.shape[0]
    maxnnb = plg_vtx.shape[1]

    def cell_body(_, cell_idx):
        cell_vtxs = tri_vtx[cell_idx, :]  # (3,)
        num_bp = 0
        base_point = jnp.zeros((3, ), dtype=cell_vtxs.dtype)

        def tri_nb_body(carry, inb):
            base_point, num_bp = carry
            tri = tri_nb[cell_idx, inb]

            def vtx_body(carry, iv):
                base_point, num_bp = carry
                vtx = tri_vtx[tri, iv]
                mask = vtx == cell_vtxs  # (3,)

                def false_fn(carry):
                    return carry

                def true_fn(carry):
                    base_point, num_bp = carry
                    base_point = base_point.at[num_bp].set(vtx)
                    return base_point, num_bp + 1

                base_point, num_bp = jax.lax.cond(
                    jnp.sum(mask) >= 1,
                    false_fn,
                    true_fn,
                    operand=(base_point, num_bp)
                )
                return (base_point, num_bp), None

            return jax.lax.scan(
                vtx_body,
                (base_point, num_bp),
                xs=jnp.arange(3)
            )

        base_point, num_bp = jax.lax.scan(
            tri_nb_body,
            (base_point, num_bp),
            xs=jnp.arange(3)
        )[0]

        # Now get neighboring triangles of these vertexes
        nb_vtx = plg_vtx[base_point, :]  # (3, maxnnb)
        stencil_idx = nb_vtx.reshape((3 * maxnnb,))  # (3*maxnnb)
        return None, stencil_idx

    _, stencil_idx = jax.lax.scan(cell_body, None, xs=jnp.arange(ncell))
    stencil_num = jnp.sum(stencil_idx >= 0, axis=-1)
    return stencil_num, stencil_idx


def tr_stencil_member_list_4th_order_mbs(
    dual: DualMeshEntity
) -> Tuple[Int[Array, "ncell"], Int[Array, "ncell max_m"]]:
    """20/21 point stencil for 4th order reconstruction on triangular mesh."""
    tri_nb = dual.cell.adjacency  # (nDual_cell, 3)
    tri_vtx = dual.cell.nodes  # (nDual_cell, 3)
    plg_vtx = dual.node.adjacent_cells  # (nPrimal_cell, maxnnb)
    ncell = tri_nb.shape[0]
    maxnnb = plg_vtx.shape[1]
    max_m: int = 21

    def cell_body(_, cell_idx):
        nbtri = tri_nb[cell_idx, :]  # (3,)
        vtx_of_nbtr = tri_vtx[nbtri, :]  # (3, 3)
        nb_vtx = plg_vtx[vtx_of_nbtr, :]  # (3, 3, maxnnb)
        # padding invalid index value using -1
        nb_vtx = jnp.where(nb_vtx >= 0, nb_vtx, -1)  # (3, 3, maxnnb)
        # Mask out the home cell contributions
        nb_vtx = jnp.where(
            nb_vtx == cell_idx, -1, nb_vtx
        )  # (3, 3, maxnnb)
        stencil_idx = nb_vtx.reshape((3 * 3 * maxnnb,))  # (3*3*maxnnb)
        stencil_idx = jax_tools.filter_ordered_unique(
            stencil_idx, size=max_m, fill_value=-1
        )
        return None, stencil_idx

    _, stencil_idx = jax.lax.scan(cell_body, None, xs=jnp.arange(ncell))
    stencil_num = jnp.sum(stencil_idx >= 0, axis=-1)
    return stencil_num, stencil_idx


def recon_2nd_polynomial(
    x: Int[Array, "max_m"],
    y: Int[Array, "max_m"]
) -> Float[Array, "max_m n_coeff"]:
    return jnp.stack(
        [x, y, x ** 2, x * y, y ** 2], axis=-1
    )  # (max_m, 5)


def recon_4th_polynomial(
    x: Int[Array, "max_m"],
    y: Int[Array, "max_m"]
) -> Float[Array, "max_m n_coeff"]:
    return jnp.stack(
        [
            x, y,
            x ** 2, x * y, y ** 2,
            x ** 3, (x ** 2) * y, x * (y ** 2), y ** 3,
            x ** 4, (x ** 3) * y, (x ** 2) * (y ** 2), x * (y ** 3), y ** 4
        ],
        axis=-1
    )  # (max_m, 14)


def set_lob_xy(
    op0: Float[Array, "3"],
    op1: Float[Array, "3"],
) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
    eps = jnp.finfo(op0.dtype).eps
    p0p1 = math._project_sphere2plane(op0, op1)  # (3, )
    # Avoid division by zero
    norm1 = jnp.linalg.norm(p0p1, axis=-1, keepdims=True)
    norm1 = jnp.maximum(norm1, eps*100)
    tx = p0p1 / norm1  # (3,)

    tmp = jnp.cross(-tx, op0)  # (3,)
    # Avoid division by zero
    norm2 = jnp.linalg.norm(tmp, axis=-1, keepdims=True)
    norm2 = jnp.maximum(norm2, eps*100)
    ty = tmp / norm2  # (3,)
    return tx, ty


def get_lob_xy(
    op0: Float[Array, "3"],
    opi: Float[Array, "max_m 3"],
    tx: Float[Array, "3"],
    ty: Float[Array, "3"],
    is_valid_stencil: Float[Array, "max_m"]
) -> Tuple[Float[Array, "max_m"], Float[Array, "max_m"]]:
    p0pi = math.project_sphere2plane(op0, opi)  # (max_m, 3)
    local_x = jnp.sum(
        p0pi * tx[None, :],
        axis=-1
    )  # (max_m,)
    local_y = jnp.sum(
        p0pi * ty[None, :],
        axis=-1
    )  # (max_m,)
    # Mask out invalid stencil members
    local_x = jnp.where(is_valid_stencil, local_x, 0.0)  # (max_m,)
    local_y = jnp.where(is_valid_stencil, local_y, 0.0)  # (max_m,)
    return local_x, local_y


def _calc_b_matrix_in_a_direction_of_cell(
    op0: Float[Array, "3"],
    op1: Float[Array, "3"],
    opi: Float[Array, "max_m 3"],
    is_valid_stencil: Int[Array, "max_m"],
    recon_builder: AbstractFluxReconstructInfo
) -> Float[Array, "n_coeff max_m"]:
    tx, ty = set_lob_xy(op0, op1)  # (3,), (3,)
    local_x, local_y = get_lob_xy(
        op0, opi, tx, ty, is_valid_stencil
    )  # (max_m,), (max_m,)
    return recon_builder.recon(
        local_x, local_y
    )  # (n_coeff, max_m)


def _calc_b_matrix_at_single_cell(
    op0: Float[Array, "3"],
    op1: Int[Array, "maxnnb 3"],
    opi: Int[Array, "max_m 3"],
    is_valid_stencil: Int[Array, "max_m"],
    is_valid_nb: Int[Array, "maxnnb"],
    recon_builder: AbstractFluxReconstructInfo
) -> Float[Array, "maxnnb n_coeff max_m"]:
    b_matrix = jax.vmap(
        _calc_b_matrix_in_a_direction_of_cell,
        in_axes=(None, 0, None, None, None),
        out_axes=0
    )(
        op0,
        op1,
        opi,
        is_valid_stencil,
        recon_builder
    )  # (maxnnb, n_coeff, max_m,)

    # Mask out invalid neighbors
    return jnp.where(
        is_valid_nb[:, None, None],
        b_matrix,
        0.0
    )


@jax.jit
def _calc_b_matrix_at_cells(
    op0: Float[Array, "ncell 3"],
    op1: Int[Array, "ncell maxnnb 3"],
    opi: Int[Array, "ncell max_m 3"],
    is_valid_stencil: Int[Array, "ncell max_m"],
    is_valid_nb: Int[Array, "ncell maxnnb"],
    recon_builder: AbstractFluxReconstructInfo
) -> Float[Array, "ncell maxnnb n_coeff max_M"]:
    ncell = op0.shape[0]

    def cell_body(_, cell_idx):

        b_matrix = _calc_b_matrix_at_single_cell(
            op0[cell_idx, :],
            op1[cell_idx, :, :],
            opi[cell_idx, :, :],
            is_valid_stencil[cell_idx, :],
            is_valid_nb[cell_idx, :],
            recon_builder
        )

        return None, b_matrix

    _, b_matrix = jax.lax.scan(
        cell_body, init=None, xs=jnp.arange(ncell)
    )

    return b_matrix


def calc_b_matrix_at_dual_cells(
    mesh: OrthogonalDualMesh,
    polynomial_order: int,
    use_mbs: bool = True
) -> Float[Array, "ncell maxnnb n_coeff max_m"]:
    op0 = mesh.dual.cell.centroid.cartesian  # (ncell, 3)
    dual_nb = mesh.dual.cell.adjacency  # (ncell, maxnnb)
    dual_cart = mesh.dual.cell.centroid.cartesian  # (ncell, 3)
    recon_builder = FluxReconstructInfoAtDualMesh(polynomial_order, use_mbs)
    stencil_idxes = recon_builder.calc_stencil(mesh.dual)[1]  # (ncell, max_m)

    op1 = jnp.where(
        dual_nb[..., None] >= 0,
        dual_cart[dual_nb],
        0.0
    )  # (ncell, maxnnb, 3)
    opi = jnp.where(
        stencil_idxes[..., None] >= 0,
        dual_cart[stencil_idxes],
        0.0
    )  # (ncell, max_m, 3)
    return _calc_b_matrix_at_cells(
        op0,
        op1,
        opi,
        stencil_idxes >= 0,
        dual_nb >= 0,
        recon_builder
    )


def calc_b_matrix_at_primal_cells(
    mesh: OrthogonalDualMesh,
    polynomial_order: int,
) -> Float[Array, "ncell maxnnb n_coeff max_m"]:
    op0 = mesh.primal.cell.centroid.cartesian  # (ncell, 3)
    primal_nb = mesh.primal.cell.adjacency  # (ncell, maxnnb)
    primal_cart = mesh.primal.cell.centroid.cartesian  # (ncell, 3)
    recon_builder = FluxReconstructInfoAtPrimalMesh(polynomial_order)
    stencil_idxes = recon_builder.calc_stencil(mesh.primal)[1]

    op1 = jnp.where(
        primal_nb[..., None] >= 0,
        primal_cart[primal_nb],
        0.0
    )  # (ncell, maxnnb, 3)
    opi = jnp.where(
        stencil_idxes[..., None] >= 0,
        primal_cart[stencil_idxes],
        0.0
    )  # (ncell, max_m, 3)
    return _calc_b_matrix_at_cells(
        op0,
        op1,
        opi,
        stencil_idxes >= 0,
        primal_nb >= 0,
        recon_builder
    )


@partial(jax.jit, static_argnums=(1, 2))
def _init_bmatrix_at_edge(
    mesh: OrthogonalDualMesh,
    recon_builder: AbstractFluxReconstructInfo,
    location: str,
) -> Tuple[Int[Array, "nedge max_m"], Float[Array, "nedge 2 max_m"]]:
    if location == "primal":
        primal_entity = mesh.primal
        dual_entity = mesh.dual
    else:
        primal_entity = mesh.dual
        dual_entity = mesh.primal
    cell_centroid_cart = primal_entity.cell.centroid.cartesian  # (ncell, 3)
    cell_adjacency = primal_entity.cell.adjacency  # (ncell, maxnnb)
    maxnnb = primal_entity.cell.adjacency.shape[1]
    nedge = dual_entity.edge.endpoints.shape[0]
    max_m = recon_builder.max_member
    spec_loc = recon_builder.spec_loc
    stencil_idxes = recon_builder.calc_stencil(primal_entity)[1]

    def edge_body(_, ie):

        def vtx_body(_, iv):
            v_idx = dual_entity.edge.endpoints[ie, iv]

            def nb_body(_, inb):
                globel_idx_ed = primal_entity.cell.edges[v_idx, inb]

                def do_nothing():
                    return jnp.zeros((max_m,), dtype=cell_centroid_cart.dtype)

                def get_value():
                    op0 = cell_centroid_cart[v_idx, :]
                    op1 = cell_centroid_cart[cell_adjacency[v_idx, inb], :]
                    opi = cell_centroid_cart[stencil_idxes[v_idx, :], :]
                    is_valid_stencil = stencil_idxes[v_idx, :] >= 0
                    return _calc_b_matrix_in_a_direction_of_cell(
                        op0=op0,
                        op1=op1,
                        opi=opi,
                        is_valid_stencil=is_valid_stencil,
                        recon_builder=recon_builder
                    )[spec_loc, :]

                return None, jax.lax.cond(
                    globel_idx_ed == ie,
                    get_value,
                    do_nothing,
                )

            _, recon_matrix = jax.lax.scan(
                nb_body, init=None, xs=jnp.arange(maxnnb)
            )
            recon_matrix = jnp.sum(recon_matrix, axis=0)  # (max_m,)
            return None, recon_matrix

        _, recon_matrix = jax.lax.scan(
            vtx_body, init=None, xs=jnp.arange(2)
        )  # (2, max_m,)
        return None, recon_matrix

    _, recon_matrix = jax.lax.scan(
        edge_body, init=None, xs=jnp.arange(nedge)
    )  # (nedge, 2, max_m)

    return stencil_idxes, recon_matrix


def init_primal_bmatrix_at_dual_edge(
    mesh: OrthogonalDualMesh,
    polynomial_order: int,
) -> Tuple[Int[Array, "nedge max_m"], Float[Array, "nedge 2 max_m"]]:
    recon_builder = FluxReconstructInfoAtPrimalMesh(polynomial_order)
    return _init_bmatrix_at_edge(
        mesh,
        recon_builder,
        location="primal"
    )


def init_dual_bmatrix_at_primal_edge(
    mesh: OrthogonalDualMesh,
    polynomial_order: int,
    use_mbs: bool
) -> Tuple[Int[Array, "nedge max_m"], Float[Array, "nedge 2 max_m"]]:
    recon_builder = FluxReconstructInfoAtDualMesh(polynomial_order, use_mbs)
    return _init_bmatrix_at_edge(
        mesh,
        recon_builder,
        location="dual"
    )
