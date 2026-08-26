from __future__ import annotations
from typing import TYPE_CHECKING
import functools
from jax import numpy as jnp
from jax import lax, debug
import jax
import equinox as eqx


from terraflow.util import device
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryData


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array


class TriSKCoriolisForceData(AbstractEquationAuxiliaryData):
    max_ne_of_edge  : int = eqx.field(static=True)
    ne_edge_of_edge : INT[Array, "n_edge"]
    edge_of_edge    : INT[Array, "n_edge maxnnb*2"]
    wgt_edge_of_edge: FLT[Array, "n_edge maxnnb*2"]
    le_edge_of_edge : FLT[Array, "n_edge maxnnb*2"]

    @classmethod
    def empty(cls) -> "TriSKCoriolisForceData":
        max_ne_of_edge   = 0
        ne_edge_of_edge  = jnp.empty(0)
        edge_of_edge     = jnp.empty(0)
        wgt_edge_of_edge = jnp.empty(0)
        le_edge_of_edge  = jnp.empty(0)
        return cls(
            initialized=False,
            max_ne_of_edge=max_ne_of_edge,
            ne_edge_of_edge=ne_edge_of_edge,
            edge_of_edge=edge_of_edge,
            wgt_edge_of_edge=wgt_edge_of_edge,
            le_edge_of_edge=le_edge_of_edge
        )

    @classmethod
    def from_coord(cls, mesh: SCVTMesh, dev: device.AbstractDevice = device.CPU()) -> "TriSKCoriolisForceData":
        collection = calc_trsk_wgt(mesh, dev)
        ne_edge_of_edge  = jax.device_put(collection[0], device=dev())
        edge_of_edge     = jax.device_put(collection[1], device=dev())
        wgt_edge_of_edge = jax.device_put(collection[2], device=dev())
        le_edge_of_edge  = jax.device_put(collection[3], device=dev())
        return cls(
            initialized=True,
            max_ne_of_edge=mesh.maxnnb_cell * 2,
            ne_edge_of_edge=ne_edge_of_edge,
            edge_of_edge=edge_of_edge,
            wgt_edge_of_edge=wgt_edge_of_edge,
            le_edge_of_edge=le_edge_of_edge
        )


@functools.partial(jax.jit, static_argnames="dev", backend="cpu")
def calc_trsk_wgt_stage_1(
    mesh: SCVTMesh, 
    dev: device.AbstractDevice = device.CPU()
) -> FLT[Array, "nPrimal_cell maxnnb maxnnb"]:
    dtype = dev.float_type
    maxnnb = mesh.maxnnb_cell
    alpha = jnp.array(0.5, dtype=dtype)

    def polg_body(_, ic):
        riv = mesh.kite_area_cell[ic] / mesh.area_cell[ic]  # (maxnnb,)
        valid_neighbors = (mesh.cell_nb >= 0)[ic]  # (maxnnb,)

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
                nei = mesh.cell_nr[ic, iep]
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
                    lambda _: mesh.vtx_of_cell[ic, iee - 1],
                    lambda _: mesh.vtx_of_cell[ic, mesh.cell_nnb[ic] - 1],
                    None
                )
                global_idx_iee = mesh.cell_ed[ic, iee]

                # (maxnnb,)
                mask = mesh.vtx_ed[global_idx_trv2] == global_idx_iee

                local_idx_tev2 = jnp.where(
                    jnp.any(mask),
                    jnp.argmax(mask),  # return the first True index
                    -999  # if no True index found
                )

                tev2 = (mesh.vtx_nr[global_idx_trv2, local_idx_tev2]
                        * mesh.edptg_eddnr[global_idx_iee])

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

    _, trsk_weight = lax.scan(polg_body, None, jnp.arange(mesh.cell_ids, mesh.cell_ihe[0]+1, 1))

    trsk_weight = jnp.array(trsk_weight, dtype=dtype)

    return trsk_weight


@functools.partial(jax.jit, static_argnames="dev", backend="cpu")
def calc_trsk_wgt(
    mesh: SCVTMesh, 
    dev: device.AbstractDevice = device.CPU()
) -> tuple[
    FLT[Array, "n_edge"],
    INT[Array, "n_edge maxnnb*2"],
    FLT[Array, "n_edge maxnnb*2"],
    FLT[Array, "n_edge maxnnb*2"]
]:
    ids, ide = mesh.edge_ids, mesh.edge_ide
    n_edge = mesh.edge_ime - mesh.edge_ims + 1
    maxnnb = mesh.maxnnb_cell

    trsk_weight = calc_trsk_wgt_stage_1(mesh, dev)

    edp_numb_on_edge = jnp.zeros((n_edge,)           , dtype=dev.int_type  )
    edp_edge_on_edge = -jnp.ones((n_edge, maxnnb * 2), dtype=dev.int_type  )
    edp_trsk_on_edge = jnp.zeros((n_edge, maxnnb * 2), dtype=dev.float_type)
    edp_edpl_on_edge = jnp.zeros((n_edge, maxnnb * 2), dtype=dev.float_type)

    def edge_body(carry, ie):
        numb_arr, edge_arr, trsk_arr, edpl_arr = carry

        edge_out = edge_arr[ie]
        trsk_out = trsk_arr[ie]
        edpl_out = edpl_arr[ie]

        kk = jnp.array(0, dtype=dev.int_type)

        def iv_body(carry_iv, iv):
            kk, edge_out, trsk_out, edpl_out = carry_iv
            idx_cell = mesh.cell_of_edge[ie, iv]

            def iee_body(carry_iee, iee):
                kk, edge_out, trsk_out, edpl_out = carry_iee
                is_valid_iee = iee < mesh.cell_nnb[idx_cell]
                is_target_edge = mesh.cell_ed[idx_cell, iee] == ie

                def iep_body(carry_iep, iep):
                    kk, edge_out, trsk_out, edpl_out = carry_iep
                    is_valid_iep = iep < mesh.cell_nnb[idx_cell]
                    e_idx = mesh.cell_ed[idx_cell, iep]

                    def update_arrays(_):
                        nonlocal edge_out, trsk_out, edpl_out
                        new_edge_out = edge_out.at[kk].set(e_idx)
                        new_trsk_out = trsk_out.at[kk].set(
                            trsk_weight[idx_cell, iee, iep]
                        )
                        new_edpl_out = edpl_out.at[kk].set(mesh.le[e_idx])
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

        new_numb_arr = numb_arr.at[ie].set(final_kk  )
        new_edge_arr = edge_arr.at[ie].set(final_edge)
        new_trsk_arr = trsk_arr.at[ie].set(final_trsk)
        new_edpl_arr = edpl_arr.at[ie].set(final_edpl)

        return (new_numb_arr, new_edge_arr, new_trsk_arr, new_edpl_arr), None

    (final_numb, final_edge, final_trsk, final_edpl), _ = lax.scan(
        edge_body,
        (edp_numb_on_edge, edp_edge_on_edge, edp_trsk_on_edge, edp_edpl_on_edge),
        jnp.arange(ids, ide+1)
    )

    return final_numb, final_edge, final_trsk, final_edpl
