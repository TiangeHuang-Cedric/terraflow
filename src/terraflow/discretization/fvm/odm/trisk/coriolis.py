from typing import TypeVar
from jax import numpy as jnp
from jax import lax, debug
import jax
import equinox as eqx

from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.operator import FiniteVolumeMethodOperator
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.vert_coord.abstract import OneLayerVertCoordinate
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.abstract import CoordinateSystem
from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import BlockArrays

Int = typing.Int
Float = typing.Float
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


class CoriolisForceAtDualEdge(FVM_ODM_TriSK):

    def __init__(
        self,
        coord: CoordSys,
        para: PlanetParameterSet,
    ):
        super().__init__(coord, para)
        collection = trisk_weighted_interpolation_operator(self.mesh)
        self.edp_numb_on_edge = collection[0]
        self.edp_edge_on_edge = collection[1]
        self.edp_trsk_on_edge = collection[2]
        self.edp_edpl_on_edge = collection[3]

    def __call__(
        self,
        block: BlockArrays
    ) -> BlockArrays:
        """Compute nonlinear Coriolis term at dual edges."""
        edd_length = self.mesh.dual.edge.length  # (nedg,)
        ep_idx = self.edp_edge_on_edge  # (nedg, maxnnb*2)
        weight = self.edp_trsk_on_edge
        lenp = self.edp_edpl_on_edge
        un_e = block.state.un_e
        h_e = block.aux.h_e
        pv_e = block.aux.pv_e

        def _wrap(
            un_e: Float[Array, "nEdge"],
            h_e: Float[Array, "nEdge"],
            pv_e: Float[Array, "nEdge"],
        ) -> Float[Array, "nEdge"]:
            pv_iee = pv_e
            pv_iep = jnp.take(
                pv_e,
                ep_idx,
                axis=0
            )  # (nedge, max_numb*2)
            pv_eep = (pv_iee[:, None] + pv_iep) / 2  # (nedge, max_numb*2)
            mfx_at_primal_edge = un_e * h_e

            mfx_tile = mfx_at_primal_edge[ep_idx]  # (nedg, maxnnb*2)
            mfx_tile = jnp.where(ep_idx < 0, 0.0, mfx_tile)
            mfx_tile = mfx_tile * pv_eep
            return jnp.sum(weight * lenp * mfx_tile, axis=-1) / edd_length

        res = - jax.vmap(_wrap, in_axes=(0, 0, 0), out_axes=0)(
            un_e, h_e, pv_e
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_coriolis,
            block,
            replace=res
        )

    def recon_tangent_velocity(
        self,
        block: BlockArrays
    ) -> BlockArrays:
        """Reconstruct tangent velocity at dual edges from normal velocity at edges."""
        edd_length = self.mesh.dual.edge.length  # (nedg,)
        ep_idx = self.edp_edge_on_edge  # (nedg, maxnnb*2)
        weight = self.edp_trsk_on_edge
        lenp = self.edp_edpl_on_edge
        un_e = block.state.un_e

        def _wrap(
            un_e: Float[Array, "nEdge"],
        ) -> Float[Array, "nEdge"]:
            wind_tile = un_e[ep_idx]  # (nedg, maxnnb*2)
            wind_tile = jnp.where(ep_idx < 0, 0.0, wind_tile)
            return jnp.sum(weight * lenp * wind_tile, axis=-1) / edd_length

        res = jax.vmap(_wrap, in_axes=0, out_axes=0)(un_e)
        return eqx.tree_at(
            lambda b: b.aux.ut_e,
            block,
            replace=res
        )


def calc_trsk_weight(
    mesh: OrthogonalDualMesh
) -> Float[Array, "nPrimal_cell maxnnb maxnnb"]:
    """
    Compute TRiSK weights with nested ``lax.scan`` loops.
    Args:
        mesh (OrthogonalDualMesh):
            AbstractMesh container with zero-based indexing and ``-1`` padding.
    Returns:
        jnp.ndarray:
            Array of shape ``(n_cell, maxnnb, maxnnb)``.
    """
    plg = mesh._plg
    dua = mesh._dua
    dtype = plg.kite_area.dtype

    n_polg = mesh.shape["n_polg"]
    maxnnb = plg.maxnnb
    alpha = jnp.array(0.5, dtype=dtype)

    def polg_body(_, ic):
        riv = plg.kite_area[ic] / plg.area[ic]  # (maxnnb,)
        valid_neighbors = plg.valid_neighbors()[ic]  # (maxnnb,)

        def edge_body(_, iee):

            def edge_pair_body(_, iep):
                mask = jnp.arange(maxnnb) * valid_neighbors  # (maxnnb,)

                def condition_equal(_):
                    """When iep == iee, return 0.0."""
                    return jnp.array(0.0, dtype=dtype)

                def condition_greater(_):
                    """When iep > iee, return sum(riv[iep:nnb-1]) + sum(riv[0:iee])."""
                    mask1 = mask >= iep
                    mask2 = (mask <= iee - 1) & (mask >= 0)
                    return jnp.sum(riv * mask1) + jnp.sum(riv * mask2)

                def condition_less(_):
                    """When iep < iee, return sum(riv[iep:iee])."""
                    mask3 = (mask >= iep) & (mask <= iee - 1)
                    return jnp.sum(riv * mask3)

                riv_sum = lax.cond(
                    iep > iee,
                    condition_greater,
                    lambda _: jnp.array(0.0, dtype=dtype),
                    None
                )
                riv_sum = lax.cond(
                    iep < iee,
                    condition_less,
                    lambda _: riv_sum,
                    None
                )
                riv_sum = lax.cond(
                    iep == iee,
                    condition_equal,
                    lambda _: riv_sum,
                    None
                )
                nei = plg.nr[ic, iep]
                # --------------------------------------------------------------
                #                       Calculate tev2
                # v2 is the polygon vertice corresponding to riv[iee-1],
                # which has the same index order as tri[iee-1]
                # tev2 belongs to triangle at dual cell v2
                # so we need to know:
                # (1) dcell_structure at v2
                # (2) local index of iee within triangle v2;
                #     the number of v2=iee-1
                # --------------------------------------------------------------
                global_idx_trv2 = lax.cond(
                    iee - 1 != -1,
                    lambda _: plg.vertex[ic, iee - 1],
                    lambda _: plg.vertex[ic, plg.nnb[ic] - 1],
                    None
                )
                global_idx_iee = plg.edge[ic, iee]

                # (maxnnb,)
                mask = dua.edge[global_idx_trv2] == global_idx_iee

                local_idx_tev2 = jnp.where(
                    jnp.any(mask),
                    jnp.argmax(mask),  # return the first True index
                    -999  # if no True index found
                )

                tev2 = (dua.nr[global_idx_trv2, local_idx_tev2]
                        * mesh._aux.edptg_eddnr[global_idx_iee])

                signator = tev2 * nei

                def raise_error(_):
                    debug.print(
                        "Errors in calc_trsk_weight: abs(signator) != 1, "
                        "signator = {}", signator
                    )
                    return None

                def do_nothing(_):
                    return None

                lax.cond(
                    (jnp.abs(signator) != 1) & (nei != 0),
                    raise_error,
                    do_nothing,
                    None
                )

                def is_iee_equal_iep(_):
                    return jnp.array(0.0, dtype=dtype)

                def is_iee_not_equal_iep(_):
                    return (riv_sum - alpha) * signator

                return None, lax.cond(
                    iee == iep,
                    is_iee_equal_iep,
                    is_iee_not_equal_iep,
                    None
                )

            _, row = lax.scan(edge_pair_body, None, jnp.arange(maxnnb))
            return None, row

        _, weights = lax.scan(edge_body, None, jnp.arange(maxnnb))
        weights = jnp.where(
            valid_neighbors[:, None],
            weights,
            jnp.zeros(maxnnb, dtype=weights.dtype)
        )
        return None, weights

    _, trsk_weight = lax.scan(polg_body, None, jnp.arange(n_polg))
    return jnp.array(trsk_weight, dtype=dtype)


def trisk_weighted_interpolation_operator(
    mesh: OrthogonalDualMesh
) -> tuple[
    Float[Array, "n_edge"],
    Int[Array, "n_edge maxnnb*2"],
    Float[Array, "n_edge maxnnb*2"],
    Float[Array, "n_edge maxnnb*2"]
]:
    """
    Assemble TRiSK operator tables using ``lax.scan`` for edge traversal.
    Args:
        mesh : OrthogonalDualMesh
            AbstractMesh container with zero-based indexing and ``-1`` padding.
    Returns: Tuple of four arrays:
        edp_numb_on_edge : jnp.ndarray
            Number of ``edp`` entries associated with each edge.
            Shape: ``(n_edge,)``.
        edp_edge_on_edge : jnp.ndarray
            Indices of ``edp`` entries associated with each edge.
            Shape: ``(n_edge, maxnnb*2)``.
        edp_trsk_on_edge : jnp.ndarray
            TRiSK weights for each associated ``edp`` entry.
            Shape: ``(n_edge, maxnnb*2)``.
        edp_edpl_on_edge : jnp.ndarray
            Lengths of each associated ``edp`` entry.
            Shape: ``(n_edge, maxnnb*2)``.
    """
    plg = mesh._plg
    edp = mesh._edp
    edd = mesh._edd
    n_edge = mesh.shape["n_edge"]
    maxnnb = mesh._plg.maxnnb

    trsk_weight = calc_trsk_weight(mesh)

    edp_numb_on_edge = jnp.zeros((n_edge,), dtype=plg.v.dtype)
    edp_edge_on_edge = -jnp.ones((n_edge, maxnnb * 2), dtype=edp.v.dtype)
    edp_trsk_on_edge = jnp.zeros((n_edge, maxnnb * 2), dtype=trsk_weight.dtype)
    edp_edpl_on_edge = jnp.zeros((n_edge, maxnnb * 2), dtype=edp.length.dtype)

    def edge_body(carry, ie):
        numb_arr, edge_arr, trsk_arr, edpl_arr = carry

        edge_out = edge_arr[ie]
        trsk_out = trsk_arr[ie]
        edpl_out = edpl_arr[ie]

        kk = jnp.array(0, dtype=edp.v.dtype)

        def iv_body(carry_iv, iv):
            kk, edge_out, trsk_out, edpl_out = carry_iv
            idx_cell = edd.vertex[ie, iv]

            def iee_body(carry_iee, iee):
                kk, edge_out, trsk_out, edpl_out = carry_iee
                is_valid_iee = iee < plg.nnb[idx_cell]
                is_target_edge = plg.edge[idx_cell, iee] == ie

                def iep_body(carry_iep, iep):
                    kk, edge_out, trsk_out, edpl_out = carry_iep
                    is_valid_iep = iep < plg.nnb[idx_cell]
                    e_idx = plg.edge[idx_cell, iep]

                    def update_arrays(_):
                        nonlocal edge_out, trsk_out, edpl_out
                        new_edge_out = edge_out.at[kk].set(e_idx)
                        new_trsk_out = trsk_out.at[kk].set(
                            trsk_weight[idx_cell, iee, iep]
                        )
                        new_edpl_out = edpl_out.at[kk].set(edp.length[e_idx])
                        return kk + 1, new_edge_out, new_trsk_out, new_edpl_out

                    def no_update(_):
                        return kk, edge_out, trsk_out, edpl_out

                    new_kk, new_edge_out, new_trsk_out, new_edpl_out = lax.cond(
                        is_valid_iep,
                        update_arrays,
                        no_update,
                        None
                    )
                    return (new_kk, new_edge_out, new_trsk_out, new_edpl_out), None

                def run_iep_loop(_):
                    (final_kk, final_edge, final_trsk, final_edpl), _ = lax.scan(
                        iep_body,
                        (kk, edge_out, trsk_out, edpl_out),
                        jnp.arange(maxnnb)
                    )
                    return final_kk, final_edge, final_trsk, final_edpl

                def skip_iep_loop(_):
                    return kk, edge_out, trsk_out, edpl_out

                new_kk, new_edge_out, new_trsk_out, new_edpl_out = lax.cond(
                    is_valid_iee & is_target_edge,
                    run_iep_loop,
                    skip_iep_loop,
                    None
                )

                return (new_kk, new_edge_out, new_trsk_out, new_edpl_out), None

            (final_kk, final_edge, final_trsk, final_edpl), _ = lax.scan(
                iee_body,
                (kk, edge_out, trsk_out, edpl_out),
                jnp.arange(maxnnb)
            )
            return (final_kk, final_edge, final_trsk, final_edpl), None

        (final_kk, final_edge, final_trsk, final_edpl), _ = lax.scan(
            iv_body,
            (kk, edge_out, trsk_out, edpl_out),
            jnp.arange(2)
        )

        new_numb_arr = numb_arr.at[ie].set(final_kk)
        new_edge_arr = edge_arr.at[ie].set(final_edge)
        new_trsk_arr = trsk_arr.at[ie].set(final_trsk)
        new_edpl_arr = edpl_arr.at[ie].set(final_edpl)

        return (new_numb_arr, new_edge_arr, new_trsk_arr, new_edpl_arr), None

    (final_numb, final_edge, final_trsk, final_edpl), _ = lax.scan(
        edge_body,
        (edp_numb_on_edge, edp_edge_on_edge, edp_trsk_on_edge, edp_edpl_on_edge),
        jnp.arange(n_edge)
    )

    return final_numb, final_edge, final_trsk, final_edpl
