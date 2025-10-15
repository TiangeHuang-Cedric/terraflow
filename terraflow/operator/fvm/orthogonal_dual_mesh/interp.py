from tkinter import N
from jax import numpy as jnp
from jax import lax, jit, debug
from functools import partial

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import PrimeCell, DualCell
from terraflow.mesh.orthogonal_dual_mesh import PrimeEdge, DualEdge
from terraflow.operator.fvm import Interpolation

Int = typing.Int
Float = typing.Float
Array = typing.Array


class InterpWeightedAddEdp2Plg(Interpolation):
    def __call__(self, x_at_edge: Array, plg: PrimeCell,
                 edp: PrimeEdge, edd: DualEdge) -> Array:
        ed_idx = plg.edge()  # (nplg, maxnnb)
        Ae = jnp.take(edp.length, ed_idx) * jnp.take(edd.length, ed_idx)
        x_tile = jnp.take(x_at_edge, ed_idx)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(Ae * x_tile, axis=-1) / (4.0 * plg.area)


class InterpWeightedAddEdd2Dua(Interpolation):
    def __call__(self, x_at_edge: Array, dua: DualCell,
                 edd: DualEdge, edp: PrimeEdge) -> Array:
        ed_idx = dua.edge()  # (ndual, maxnnb)
        Ae = jnp.take(edd.length, ed_idx) * jnp.take(edd.length, ed_idx)
        x_tile = jnp.take(x_at_edge, ed_idx)
        x_tile = jnp.where(ed_idx < 0, 0.0, x_tile)
        return jnp.sum(Ae * x_tile, axis=-1) / (4.0 * dua.area)


class InterpWeightedAddDua2Plg(Interpolation):
    def __call__(self, x_at_dual_cell: Array, plg: PrimeCell) -> Array:
        dua_idx = plg.vertex()  # (nplg, maxnnb)
        x_tile = jnp.take(x_at_dual_cell, dua_idx)  # (nplg, maxnnb)
        x_tile = jnp.where(dua_idx < 0, 0.0, x_tile)  # (nplg, maxnnb)
        Adv = plg.kite_area
        return jnp.sum(Adv * x_tile, axis=-1) / plg.area


class InterpWeightedAddPairMeetingEdges2Edge(Interpolation):
    def __call__(self, x_at_edge: Array,
                 edge_on_edge: Array, weight_on_edge: Array,
                 edpl_on_edge: Array,
                 edge_dual: DualEdge) -> Array:
        ep_idx = edge_on_edge  # (nedg, maxnnb*2)
        weight = jnp.take(weight_on_edge, ep_idx, axis=0)
        lenp = jnp.take(edpl_on_edge, ep_idx, axis=0)
        x_tile = jnp.take(x_at_edge, ep_idx)
        x_tile = jnp.where(edge_on_edge < 0, 0.0, x_tile)
        return -jnp.sum(weight * lenp * x_tile, axis=-1) / edge_dual.length


def calc_trsk_weight(mesh: OrthogonalDualMesh) -> jnp.ndarray:
    """
    Compute TRiSK weights with nested ``lax.scan`` loops.
    Args:
        mesh (OrthogonalDualMesh):
            Mesh container with zero-based indexing and ``-1`` padding.
    Returns:
        jnp.ndarray:
            Array of shape ``(n_cell, maxnnb, maxnnb)``.
    """
    plg = mesh.plg
    dua = mesh.dua
    dtype = plg.kite_area.dtype

    n_polg = mesh.shape["n_polg"]
    maxnnb = plg.maxnnb
    alpha = jnp.array(0.5, dtype=dtype)

    def polg_body(_, ic):
        riv = plg.kite_area[ic]/plg.area[ic]
        valid_neighbors = plg.valid_neighbors()[ic]

        def edge_body(_, iee):

            def edge_pair_body(_, iep):
                mask = jnp.arange(maxnnb)

                def condition_equal(_):
                    """When iep == iee, return 0.0."""
                    return jnp.array(0.0, dtype=dtype)

                def condition_greater(_):
                    """When iep > iee, return sum(riv[iep:nnb]) + sum(riv[0:iee])."""
                    mask1 = (mask >= iep).astype(riv.dtype)
                    mask2 = (mask < iee).astype(riv.dtype)
                    return jnp.sum(riv * mask1) + jnp.sum(riv * mask2)

                def condition_less(_):
                    """When iep < iee, return sum(riv[iep:iee])."""
                    mask3 = ((mask >= iep) & (mask < iee)).astype(riv.dtype)
                    return jnp.sum(riv * mask3)

                riv_sum = lax.cond(
                    iep > iee, condition_greater, lambda _: 0.0, None)
                riv_sum = lax.cond(
                    iep < iee, condition_less, lambda _: riv_sum, None)
                riv_sum = lax.cond(
                    iep == iee, condition_equal, lambda _: riv_sum, None)
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
                    lambda _: plg.vertex()[ic, iee - 1],
                    lambda _: plg.vertex()[ic, plg.nnb[ic] - 1],
                    None
                )
                global_idx_iee = plg.edge()[ic, iee]

                # (maxnnb,)
                mask = dua.edge()[global_idx_trv2] == global_idx_iee

                local_idx_tev2 = jnp.where(
                    jnp.any(mask),
                    jnp.argmax(mask),  # return the first True index
                    -999  # if no True index found
                )

                tev2 = (dua.nr[global_idx_trv2, local_idx_tev2]
                        * mesh.aux.edptg_eddnr[global_idx_iee])

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

                return None, (riv_sum - alpha) * signator

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


def trisk_weighted_interpolation_operator(mesh: OrthogonalDualMesh):
    """
    Assemble TRiSK operator tables using ``lax.scan`` for edge traversal.
    Args:
        mesh : OrthogonalDualMesh
            Mesh container with zero-based indexing and ``-1`` padding.
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
    plg = mesh.plg
    edp = mesh.edp
    edd = mesh.edd
    n_edge = mesh.shape["n_edge"]
    maxnnb = mesh.plg.maxnnb

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
            idx_cell = edd.vertex()[ie, iv]

            def iee_body(carry_iee, iee):
                kk, edge_out, trsk_out, edpl_out = carry_iee
                is_valid_iee = iee < plg.nnb[idx_cell]
                is_target_edge = plg.edge()[idx_cell, iee] == ie

                def iep_body(carry_iep, iep):
                    kk, edge_out, trsk_out, edpl_out = carry_iep
                    is_valid_iep = iep < plg.nnb[idx_cell]
                    e_idx = plg.edge()[idx_cell, iep]

                    def update_arrays(_):
                        nonlocal edge_out, trsk_out, edpl_out
                        new_edge_out = edge_out.at[kk].set(e_idx)
                        new_trsk_out = trsk_out.at[kk].set(
                            trsk_weight[idx_cell, iee, iep])
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


class InterpTriSKWeightedAdd(Interpolation):
    edp_numb_on_edge: Array
    edp_edge_on_edge: Array
    edp_trsk_on_edge: Array
    edp_edpl_on_edge: Array

    def __init__(self, mesh: OrthogonalDualMesh):
        collect = trisk_weighted_interpolation_operator(mesh)
        numb, edge, trsk, edpl = collect
        self.edp_numb_on_edge = numb
        self.edp_edge_on_edge = edge
        self.edp_trsk_on_edge = trsk
        self.edp_edpl_on_edge = edpl

    def __call__(self, x_at_edge: Array, edd: DualEdge) -> Array:
        op = InterpWeightedAddPairMeetingEdges2Edge()
        return op(
            x_at_edge,
            self.edp_edge_on_edge,
            self.edp_trsk_on_edge,
            self.edp_edpl_on_edge,
            edd
        )
