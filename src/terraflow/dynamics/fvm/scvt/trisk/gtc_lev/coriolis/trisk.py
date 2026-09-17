from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit


from terraflow.util import device
from terraflow.util import equinox as eqx
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
        ne_edge_of_edge  = None 
        edge_of_edge     = None 
        wgt_edge_of_edge = None 
        le_edge_of_edge  = None 
        return cls(
            initialized=False,
            max_ne_of_edge=max_ne_of_edge,
            ne_edge_of_edge=ne_edge_of_edge,
            edge_of_edge=edge_of_edge,
            wgt_edge_of_edge=wgt_edge_of_edge,
            le_edge_of_edge=le_edge_of_edge
        )

    @classmethod
    def from_coord(cls, mesh: SCVTMesh, dev: device.AbstractDevice) -> "TriSKCoriolisForceData":
        collection = calc_trsk_wgt(mesh, dev)
        ne_edge_of_edge  = dev.backend.asarray(collection[0], device=dev())
        edge_of_edge     = dev.backend.asarray(collection[1], device=dev())
        wgt_edge_of_edge = dev.backend.asarray(collection[2], device=dev())
        le_edge_of_edge  = dev.backend.asarray(collection[3], device=dev())
        return cls(
            initialized=True,
            max_ne_of_edge=mesh.maxnnb_cell * 2,
            ne_edge_of_edge=ne_edge_of_edge,
            edge_of_edge=edge_of_edge,
            wgt_edge_of_edge=wgt_edge_of_edge,
            le_edge_of_edge=le_edge_of_edge
        )


@njit(cache=True)
def _calc_trsk_wgt_stage_1_numba(
    cell_ids: int,
    cell_ihe0: int,
    maxnnb: int,
    kite_area_cell: np.ndarray,
    area_cell: np.ndarray,
    cell_nb: np.ndarray,
    cell_nr: np.ndarray,
    vtx_of_cell: np.ndarray,
    cell_ed: np.ndarray,
    cell_nnb: np.ndarray,
    vtx_ed: np.ndarray,
    vtx_nr: np.ndarray,
    edptg_eddnr: np.ndarray,
) -> np.ndarray:
    alpha = 0.5
    n_cell = cell_ihe0 - cell_ids + 1
    trsk_weight = np.zeros((n_cell, maxnnb, maxnnb), dtype=kite_area_cell.dtype)

    for local_ic in range(n_cell):
        ic = cell_ids + local_ic
        riv = kite_area_cell[ic] / area_cell[ic]

        for iee in range(maxnnb):
            for iep in range(maxnnb):
                if iep == iee:
                    weight = 0.0
                else:
                    riv_sum = 0.0
                    for imask in range(maxnnb):
                        mask_value = imask if cell_nb[ic, imask] >= 0 else 0
                        if iep > iee:
                            if mask_value >= iep:
                                riv_sum += riv[imask]
                            if mask_value <= iee - 1 and mask_value >= 0:
                                riv_sum += riv[imask]
                        elif iep < iee:
                            if mask_value >= iep and mask_value <= iee - 1:
                                riv_sum += riv[imask]

                    nei = cell_nr[ic, iep]
                    if iee - 1 != -1:
                        global_idx_trv2 = vtx_of_cell[ic, iee - 1]
                    else:
                        global_idx_trv2 = vtx_of_cell[ic, cell_nnb[ic] - 1]
                    global_idx_iee = cell_ed[ic, iee]

                    local_idx_tev2 = -1
                    for local_idx in range(vtx_ed.shape[1]):
                        if vtx_ed[global_idx_trv2, local_idx] == global_idx_iee:
                            local_idx_tev2 = local_idx
                            break

                    if local_idx_tev2 < 0:
                        signator = 0.0
                    else:
                        tev2 = vtx_nr[global_idx_trv2, local_idx_tev2] * edptg_eddnr[global_idx_iee]
                        signator = tev2 * nei
                    weight = (riv_sum - alpha) * signator

                if cell_nb[ic, iee] >= 0:
                    trsk_weight[local_ic, iee, iep] = weight

    return trsk_weight


def calc_trsk_wgt_stage_1(
    mesh: SCVTMesh,
    dev: device.AbstractDevice
) -> FLT[Array, "nPrimal_cell maxnnb maxnnb"]:
    bkd = dev.backend
    trsk_weight = _calc_trsk_wgt_stage_1_numba(
        int(mesh.cell_ids),
        int(mesh.cell_ihe[0]),
        int(mesh.maxnnb_cell),
        bkd.to_numpy(mesh.kite_area_cell),
        bkd.to_numpy(mesh.area_cell),
        bkd.to_numpy(mesh.cell_nb),
        bkd.to_numpy(mesh.cell_nr),
        bkd.to_numpy(mesh.vtx_of_cell),
        bkd.to_numpy(mesh.cell_ed),
        bkd.to_numpy(mesh.cell_nnb),
        bkd.to_numpy(mesh.vtx_ed),
        bkd.to_numpy(mesh.vtx_nr),
        bkd.to_numpy(mesh.edptg_eddnr),
    )
    return trsk_weight.astype(bkd.to_numpy_dtype(dev.float_type), copy=False)


@njit(cache=True)
def _calc_trsk_wgt_numba(
    ids: int,
    ide: int,
    edge_ims: int,
    edge_ime: int,
    maxnnb: int,
    cell_of_edge: np.ndarray,
    cell_nnb: np.ndarray,
    cell_ed: np.ndarray,
    le: np.ndarray,
    trsk_weight: np.ndarray,
    int_dtype_ref: np.ndarray,
    float_dtype_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_edge = edge_ime - edge_ims + 1

    edp_numb_on_edge = np.zeros((n_edge,), dtype=int_dtype_ref.dtype)
    edp_edge_on_edge = -np.ones((n_edge, maxnnb * 2), dtype=int_dtype_ref.dtype)
    edp_trsk_on_edge = np.zeros((n_edge, maxnnb * 2), dtype=float_dtype_ref.dtype)
    edp_edpl_on_edge = np.zeros((n_edge, maxnnb * 2), dtype=float_dtype_ref.dtype)

    for ie in range(ids, ide + 1):
        kk = 0
        for iv in range(2):
            idx_cell = cell_of_edge[ie, iv]
            for iee in range(maxnnb):
                if iee < cell_nnb[idx_cell] and cell_ed[idx_cell, iee] == ie:
                    for iep in range(maxnnb):
                        if iep < cell_nnb[idx_cell]:
                            e_idx = cell_ed[idx_cell, iep]
                            if kk < maxnnb * 2:
                                edp_edge_on_edge[ie, kk] = e_idx
                                edp_trsk_on_edge[ie, kk] = trsk_weight[idx_cell, iee, iep]
                                edp_edpl_on_edge[ie, kk] = le[e_idx]
                            kk += 1
        edp_numb_on_edge[ie] = kk

    return edp_numb_on_edge, edp_edge_on_edge, edp_trsk_on_edge, edp_edpl_on_edge


def calc_trsk_wgt(
    mesh: SCVTMesh,
    dev: device.AbstractDevice
) -> tuple[
    FLT[Array, "n_edge"],
    INT[Array, "n_edge maxnnb*2"],
    FLT[Array, "n_edge maxnnb*2"],
    FLT[Array, "n_edge maxnnb*2"]
]:
    bkd         = dev.backend
    int_dtype   = bkd.to_numpy_dtype(dev.int_type)
    float_dtype = bkd.to_numpy_dtype(dev.float_type)
    trsk_weight = calc_trsk_wgt_stage_1(mesh, dev)

    return _calc_trsk_wgt_numba(
        int(mesh.edge_ids),
        int(mesh.edge_ide),
        int(mesh.edge_ims),
        int(mesh.edge_ime),
        int(mesh.maxnnb_cell),
        bkd.to_numpy(mesh.cell_of_edge),
        bkd.to_numpy(mesh.cell_nnb),
        bkd.to_numpy(mesh.cell_ed),
        bkd.to_numpy(mesh.le),
        bkd.to_numpy(trsk_weight),
        np.empty(0, dtype=int_dtype),
        np.empty(0, dtype=float_dtype)
    )
