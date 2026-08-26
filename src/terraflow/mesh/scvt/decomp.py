from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast
import jax.numpy as jnp
import numpy as np
import pymetis
from mpi4py import MPI
from numba import njit

from terraflow.util import log
from terraflow.mesh.scvt.parallel_types import (
    CommonSite,
    Domain,
    SCVTProc,
    Proc
)

if TYPE_CHECKING:
    from terraflow.mesh.scvt.mesh_type import SCVTMesh


@njit
def build_cell_csr_from_dense(
    cell_nb: np.ndarray,
    cell_nnb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_cell = cell_nb.shape[0]
    maxnnb = cell_nb.shape[1]

    indptr = np.empty(n_cell + 1, dtype=np.int64)
    indices = np.empty(n_cell * maxnnb, dtype=np.int64)

    indptr[0] = 0
    cursor = 0
    for i in range(n_cell):
        for k in range(int(cell_nnb[i])):
            nb = int(cell_nb[i, k])
            if 0 <= nb < n_cell:
                indices[cursor] = nb
                cursor += 1
        indptr[i + 1] = cursor
    
    return indptr, indices[:cursor]


@njit
def build_bdry_halo_layers(
    n_cell: int,
    cell_nnb: np.ndarray,
    cell_nb: np.ndarray,
    cell_proc_idmap: np.ndarray,
    rank: int,
    hw: int,
) -> tuple[np.ndarray, np.ndarray]:
    bdry_layer = np.zeros(n_cell, dtype=np.int32)
    halo_layer = np.zeros(n_cell, dtype=np.int32)

    if hw <= 0:
        return bdry_layer, halo_layer

    bdry_curr = np.empty(n_cell, dtype=np.int32)
    bdry_next = np.empty(n_cell, dtype=np.int32)
    halo_curr = np.empty(n_cell, dtype=np.int32)
    halo_next = np.empty(n_cell, dtype=np.int32)

    bdry_count = 0
    halo_count = 0
    for cell in range(n_cell):
        if int(cell_proc_idmap[cell]) != rank:
            continue

        is_bdry = False
        for inb in range(int(cell_nnb[cell])):
            nb = int(cell_nb[cell, inb])
            if int(cell_proc_idmap[nb]) != rank:
                is_bdry = True
                if halo_layer[nb] == 0:
                    halo_layer[nb] = 1
                    halo_curr[halo_count] = nb
                    halo_count += 1

        if is_bdry and bdry_layer[cell] == 0:
            bdry_layer[cell] = 1
            bdry_curr[bdry_count] = cell
            bdry_count += 1

    for layer in range(2, hw + 1):
        next_bdry_count = 0
        for pos in range(bdry_count):
            cell = int(bdry_curr[pos])
            for inb in range(int(cell_nnb[cell])):
                nb = int(cell_nb[cell, inb])
                if int(cell_proc_idmap[nb]) == rank and bdry_layer[nb] == 0:
                    bdry_layer[nb] = layer
                    bdry_next[next_bdry_count] = nb
                    next_bdry_count += 1

        next_halo_count = 0
        for pos in range(halo_count):
            cell = int(halo_curr[pos])
            for inb in range(int(cell_nnb[cell])):
                nb = int(cell_nb[cell, inb])
                if int(cell_proc_idmap[nb]) != rank and halo_layer[nb] == 0:
                    halo_layer[nb] = layer
                    halo_next[next_halo_count] = nb
                    next_halo_count += 1

        tmp = bdry_curr
        bdry_curr = bdry_next
        bdry_next = tmp
        bdry_count = next_bdry_count

        tmp = halo_curr
        halo_curr = halo_next
        halo_next = tmp
        halo_count = next_halo_count

    return bdry_layer, halo_layer


@njit
def build_inner_cells(
    cmpt_cell_id: np.ndarray,
    bdry_layer: np.ndarray
) -> np.ndarray:
    inner_cells = np.empty(cmpt_cell_id.size, dtype=np.int32)
    n = 0
    for i in range(cmpt_cell_id.size):
        cell = cmpt_cell_id[i]
        if bdry_layer[cell] == 0:
            inner_cells[n] = cell
            n += 1
    return inner_cells[:n]


@njit
def check_append(
    values: np.ndarray,
    g2l_id: np.ndarray,
    l2g_id: np.ndarray,
    cursor: int
) -> tuple[int, int]:
    before = cursor
    for i in range(values.size):
        g = int(values[i])
        if g >= 0 and g2l_id[g] < 0:
            g2l_id[g] = cursor
            l2g_id[cursor] = g
            cursor += 1
    return cursor, cursor - before


@njit
def domain_entities_from_cells(
    cells: np.ndarray,
    cell_vtx: np.ndarray,
    cell_edge: np.ndarray,
    cell_nnb: np.ndarray,
    n_vtx: int,
    n_edge: int,
) -> tuple[np.ndarray, np.ndarray]:
    vtx_seen = np.zeros(n_vtx, dtype=np.bool_)
    edge_seen = np.zeros(n_edge, dtype=np.bool_)
    vtx_id = np.empty(n_vtx, dtype=np.int64)
    edge_id = np.empty(n_edge, dtype=np.int64)

    n_domain_vtx = 0
    n_domain_edge = 0
    for i in range(cells.size):
        cell = int(cells[i])
        for k in range(int(cell_nnb[cell])):
            vtx = int(cell_vtx[cell, k])
            if 0 <= vtx < n_vtx and not vtx_seen[vtx]:
                vtx_seen[vtx] = True
                vtx_id[n_domain_vtx] = vtx
                n_domain_vtx += 1

            edge = int(cell_edge[cell, k])
            if 0 <= edge < n_edge and not edge_seen[edge]:
                edge_seen[edge] = True
                edge_id[n_domain_edge] = edge
                n_domain_edge += 1

    return vtx_id[:n_domain_vtx], edge_id[:n_domain_edge]


def domain_list_search_index(
    domain: Domain,
    mesh: SCVTMesh,
) -> None:
    cell_vtx = np.asarray(mesh.vtx_of_cell, dtype=np.int64)
    cell_edge = np.asarray(mesh.cell_ed, dtype=np.int64)
    cell_nnb = np.asarray(mesh.cell_nnb, dtype=np.int64)
    n_vtx = int(mesh.n_vtx)
    n_edge = int(mesh.n_edge)
    cell_id = np.asarray(domain.cell_id, dtype=np.int64)
    vtx_id, edge_id = domain_entities_from_cells(
        cell_id,
        cell_vtx,
        cell_edge,
        cell_nnb,
        n_vtx,
        n_edge,
    )
    domain.cell_id = cell_id
    domain.vtx_id = vtx_id
    domain.edge_id = edge_id
    domain.n_cell = len(domain.cell_id)
    domain.n_vtx = len(domain.vtx_id)
    domain.n_edge = len(domain.edge_id)


def metis_run(proc: SCVTProc, mesh: SCVTMesh) -> None:
    if proc.is_root(): log.logger.info("Run METIS for mesh decomposition.")
    np_model = int(proc.np_model)
    n_cell = int(mesh.n_cell)
    if np_model == 1:
        proc.cell_proc_idmap = np.zeros(n_cell, dtype=np.int32)
        return

    cell_proc_idmap = None
    comm = proc.comm_model
    if int(proc.id_model) == 0:
        adj_starts, adjacent = build_cell_csr_from_dense(
            cell_nb=np.asarray(mesh.cell_nb, dtype=np.int64),
            cell_nnb=np.asarray(mesh.cell_nnb, dtype=np.int64),
        )
        csr_adj = pymetis.CSRAdjacency(adj_starts, adjacent)
        graph_partition = pymetis.part_graph(
            np_model,
            adjacency=csr_adj,
            contiguous=True,
        )
        cell_proc_idmap = np.asarray(graph_partition.vertex_part, dtype=np.int32)

    cell_proc_idmap = comm.bcast(cell_proc_idmap, root=0)
    proc.cell_proc_idmap = np.asarray(cell_proc_idmap, dtype=np.int32)


def find_cmpt_indices(
    proc: SCVTProc,
    mesh: SCVTMesh,
) -> None:
    if proc.is_root(): log.logger.info("Find computation domain indices.")
    cell_proc_idmap = np.asarray(proc.cell_proc_idmap, dtype=np.int32)
    proc.cmpt.cell_id = np.flatnonzero(
        cell_proc_idmap == proc.id_model
    ).astype(np.int32)
    domain_list_search_index(proc.cmpt, mesh)


def find_bdry_halo_indices(
    proc: SCVTProc,
    mesh: SCVTMesh,
) -> tuple[np.ndarray, np.ndarray]:
    if proc.is_root(): log.logger.info("Find boundary and halo domain indices.")
    cell_proc_idmap = proc.cell_proc_idmap
    n_cell = int(mesh.n_cell)
    cell_nnb = np.asarray(mesh.cell_nnb, dtype=np.int64)
    cell_nb = np.asarray(mesh.cell_nb, dtype=np.int64)

    bdry_layer, halo_layer = build_bdry_halo_layers(
        n_cell,
        cell_nnb,
        cell_nb,
        cell_proc_idmap,
        proc.id_model,
        proc.hori_hw,
    )
    for layer in range(1, proc.hori_hw + 1):
        bdry = proc.bdry[layer - 1]
        halo = proc.halo[layer - 1]
        bdry.cell_id = np.flatnonzero(bdry_layer == layer).astype(np.int64)
        halo.cell_id = np.flatnonzero(halo_layer == layer).astype(np.int64)
        domain_list_search_index(bdry, mesh)
        domain_list_search_index(halo, mesh)
    return bdry_layer, halo_layer


def find_inner_indices(
    proc: SCVTProc,
    mesh: SCVTMesh,
    bdry_layer: np.ndarray
) -> None:
    if proc.is_root(): log.logger.info("Find inner domain indices.")
    proc.inner.cell_id = build_inner_cells(
        cmpt_cell_id=proc.cmpt.cell_id,
        bdry_layer=bdry_layer
    )
    domain_list_search_index(proc.inner, mesh)


def process_full_indices(proc: SCVTProc, mesh: SCVTMesh) -> None:
    if proc.is_root(): log.logger.info("Process full domain indices.")
    n_cell = mesh.n_cell
    n_vtx  = mesh.n_vtx
    n_edge = mesh.n_edge
    proc.g2l_cell_id = -np.ones(n_cell, dtype=np.int32)
    proc.l2g_cell_id = -np.ones(n_cell, dtype=np.int32)
    proc.g2l_vtx_id  = -np.ones(n_vtx, dtype=np.int32)
    proc.l2g_vtx_id  = -np.ones(n_vtx, dtype=np.int32)
    proc.g2l_edge_id = -np.ones(n_edge, dtype=np.int32)
    proc.l2g_edge_id = -np.ones(n_edge, dtype=np.int32)

    cell_cursor = 0
    vtx_cursor  = 0
    edge_cursor = 0
    cell_cursor, _ = check_append(
        values=proc.inner.cell_id,
        g2l_id=proc.g2l_cell_id,
        l2g_id=proc.l2g_cell_id,
        cursor=cell_cursor
    )
    vtx_cursor, _ = check_append(
        values=proc.inner.vtx_id,
        g2l_id=proc.g2l_vtx_id,
        l2g_id=proc.l2g_vtx_id,
        cursor=vtx_cursor
    )
    edge_cursor, _ = check_append(
        values=proc.inner.edge_id,
        g2l_id=proc.g2l_edge_id,
        l2g_id=proc.l2g_edge_id,
        cursor=edge_cursor
    )

    proc.n_cell_inner = cell_cursor
    proc.n_vtx_inner = vtx_cursor
    proc.n_edge_inner = edge_cursor

    for layer in range(proc.hori_hw, 0, -1):
        domain = proc.bdry[layer - 1]
        cell_cursor, proc.n_cell_bdry[layer - 1] = check_append(
            values=domain.cell_id,
            g2l_id=proc.g2l_cell_id,
            l2g_id=proc.l2g_cell_id,
            cursor=cell_cursor
        )
        vtx_cursor , proc.n_vtx_bdry [layer - 1] = check_append(
            values=domain.vtx_id,
            g2l_id=proc.g2l_vtx_id,
            l2g_id=proc.l2g_vtx_id,
            cursor=vtx_cursor
        )
        edge_cursor, proc.n_edge_bdry[layer - 1] = check_append(
            values=domain.edge_id,
            g2l_id=proc.g2l_edge_id,
            l2g_id=proc.l2g_edge_id,
            cursor=edge_cursor
        )

    proc.n_cell_cmpt = cell_cursor
    proc.n_vtx_cmpt = vtx_cursor
    proc.n_edge_cmpt = edge_cursor

    for layer in range(1, proc.hori_hw + 1):
        domain = proc.halo[layer - 1]
        cell_cursor, proc.n_cell_halo[layer - 1] = check_append(
            values=domain.cell_id,
            g2l_id=proc.g2l_cell_id,
            l2g_id=proc.l2g_cell_id,
            cursor=cell_cursor
        )
        vtx_cursor , proc.n_vtx_halo [layer - 1] = check_append(
            values=domain.vtx_id,
            g2l_id=proc.g2l_vtx_id,
            l2g_id=proc.l2g_vtx_id,
            cursor=vtx_cursor
        )
        edge_cursor, proc.n_edge_halo[layer - 1] = check_append(
            values=domain.edge_id,
            g2l_id=proc.g2l_edge_id,
            l2g_id=proc.l2g_edge_id,
            cursor=edge_cursor
        )

    proc.n_cell = cell_cursor
    proc.n_vtx  = vtx_cursor
    proc.n_edge = edge_cursor
    proc.l2g_cell_id = proc.l2g_cell_id[:cell_cursor].copy()
    proc.l2g_vtx_id  = proc.l2g_vtx_id [:vtx_cursor ].copy()
    proc.l2g_edge_id = proc.l2g_edge_id[:edge_cursor].copy()


def set_sendrecv_sites(
    proc: SCVTProc,
    mesh: SCVTMesh,
    halo_layer: np.ndarray
) -> None:
    if proc.is_root(): log.logger.info("Set send/recv sites for halo exchange.")
    cell_proc_idmap = np.asarray(proc.cell_proc_idmap, dtype=np.int32)
    cell_vtx  = np.asarray(mesh.vtx_of_cell, dtype=np.int64)
    cell_edge = np.asarray(mesh.cell_ed, dtype=np.int64)
    cell_nnb  = np.asarray(mesh.cell_nnb, dtype=np.int64)
    np_model  = int(proc.np_model)
    rank   = int(proc.id_model)
    n_vtx  = int(mesh.n_vtx)
    n_edge = int(mesh.n_edge)

    comm  = cast(MPI.Intracomm, proc.comm_model)
    empty = np.empty(0, dtype=np.int32)

    # ----------Set neighboring processes for communication----------
    halo_cell_parts: list[np.ndarray] = []
    for hw in range(1, proc.hori_hw + 1):
        cell_id = np.asarray(proc.halo[hw - 1].cell_id, dtype=np.int32)
        if cell_id.size:
            halo_cell_parts.append(cell_id)

    if halo_cell_parts:
        halo_cell_id = np.concatenate(halo_cell_parts).astype(np.int32, copy=False)
        halo_cell_id = halo_cell_id[halo_layer[halo_cell_id] > 0]
        halo_owner = cell_proc_idmap[halo_cell_id]
        proc_idmap_in_halo = np.unique(halo_owner[halo_owner != rank]).astype(np.int32)
    else:
        halo_cell_id = empty
        halo_owner = empty
        proc_idmap_in_halo = empty

    proc.recv_nb = proc_idmap_in_halo.astype(int).tolist()
    proc.send_nb = proc.recv_nb.copy()
    proc.indegree = len(proc.recv_nb)
    proc.outdegree = len(proc.send_nb)
    proc.nngb = proc.indegree

    proc.comm_graph = comm.Create_dist_graph_adjacent(
        sources=proc.recv_nb,
        destinations=proc.send_nb,
        reorder=False,
    )
    proc.comm_model = proc.comm_graph
    comm = proc.comm_graph

    # ----------Recv Msg direction: halo<-bdry----------
    recv_cell_list = [empty for _ in range(np_model)]
    recv_vtx_list  = [empty for _ in range(np_model)]
    recv_edge_list = [empty for _ in range(np_model)]
    cmpt_vtx = np.zeros(n_vtx, dtype=bool)
    cmpt_edge = np.zeros(n_edge, dtype=bool)
    cmpt_vtx[np.asarray(proc.cmpt.vtx_id, dtype=np.int64)] = True
    cmpt_edge[np.asarray(proc.cmpt.edge_id, dtype=np.int64)] = True

    for nb in proc.recv_nb:
        recv_cell_id = halo_cell_id[halo_owner == nb].astype(np.int32, copy=False)
        recv_vtx_id, recv_edge_id = domain_entities_from_cells(
            recv_cell_id.astype(np.int64, copy=False),
            cell_vtx,
            cell_edge,
            cell_nnb,
            n_vtx,
            n_edge,
        )
        recv_vtx_id = recv_vtx_id[~cmpt_vtx[recv_vtx_id]].astype(np.int32)
        recv_edge_id = recv_edge_id[~cmpt_edge[recv_edge_id]].astype(np.int32)
        recv_cell_list[nb] = recv_cell_id
        recv_vtx_list[nb] = recv_vtx_id
        recv_edge_list[nb] = recv_edge_id

    proc.recv = []
    for nb in proc.recv_nb:
        global_cell_id = recv_cell_list[nb]
        global_vtx_id = recv_vtx_list[nb]
        global_edge_id = recv_edge_list[nb]
        local_cell_id = proc.g2l_cell_id[global_cell_id]
        local_vtx_id = proc.g2l_vtx_id[global_vtx_id]
        local_edge_id = proc.g2l_edge_id[global_edge_id]
        if (
            np.any(local_cell_id < 0)
            or np.any(local_vtx_id < 0)
            or np.any(local_edge_id < 0)
        ):
            raise ValueError("Recv sites contain indices outside the local domain.")
        proc.recv.append(
            CommonSite(
                n_cell=int(global_cell_id.size),
                n_vtx=int(global_vtx_id.size),
                n_edge=int(global_edge_id.size),
                global_cell_id=jnp.asarray(global_cell_id, dtype=jnp.int32),
                global_vtx_id=jnp.asarray(global_vtx_id, dtype=jnp.int32),
                global_edge_id=jnp.asarray(global_edge_id, dtype=jnp.int32),
                local_cell_id=jnp.asarray(local_cell_id, dtype=jnp.int32),
                local_vtx_id=jnp.asarray(local_vtx_id, dtype=jnp.int32),
                local_edge_id=jnp.asarray(local_edge_id, dtype=jnp.int32),
            )
        )

    # ----------Send Msg direction: bdry->halo----------
    recv_payload = [
        (recv_cell_list[nb], recv_vtx_list[nb], recv_edge_list[nb])
        for nb in range(np_model)
    ]
    send_payload = comm.alltoall(recv_payload)
    send_cell_list = [
        np.asarray(send_payload[nb][0], dtype=np.int32)
        for nb in range(np_model)
    ]
    send_vtx_list = [
        np.asarray(send_payload[nb][1], dtype=np.int32)
        for nb in range(np_model)
    ]
    send_edge_list = [
        np.asarray(send_payload[nb][2], dtype=np.int32)
        for nb in range(np_model)
    ]

    proc.send_nb = [
        nb for nb in range(np_model)
        if (
            send_cell_list[nb].size
            or send_vtx_list[nb].size
            or send_edge_list[nb].size
        )
    ]
    proc.outdegree = len(proc.send_nb)

    proc.send = []
    for nb in proc.send_nb:
        global_cell_id = send_cell_list[nb]
        global_vtx_id = send_vtx_list[nb]
        global_edge_id = send_edge_list[nb]
        local_cell_id = proc.g2l_cell_id[global_cell_id]
        local_vtx_id = proc.g2l_vtx_id[global_vtx_id]
        local_edge_id = proc.g2l_edge_id[global_edge_id]
        if (
            np.any(local_cell_id < 0)
            or np.any(local_vtx_id < 0)
            or np.any(local_edge_id < 0)
        ):
            raise ValueError("Send sites contain indices outside the local domain.")
        proc.send.append(
            CommonSite(
                n_cell=int(global_cell_id.size),
                n_vtx=int(global_vtx_id.size),
                n_edge=int(global_edge_id.size),
                global_cell_id=jnp.asarray(global_cell_id, dtype=jnp.int32),
                global_vtx_id=jnp.asarray(global_vtx_id, dtype=jnp.int32),
                global_edge_id=jnp.asarray(global_edge_id, dtype=jnp.int32),
                local_cell_id=jnp.asarray(local_cell_id, dtype=jnp.int32),
                local_vtx_id=jnp.asarray(local_vtx_id, dtype=jnp.int32),
                local_edge_id=jnp.asarray(local_edge_id, dtype=jnp.int32),
            )
        )

    proc.ngb = sorted(set(proc.recv_nb) | set(proc.send_nb))
    proc.nngb = len(proc.ngb)


def scvt_decomp_run(
    mesh: SCVTMesh,
    *,
    proc: Proc = Proc(),
    hori_hw: int | None = None
) -> SCVTProc:
    hori_hw = int(mesh.hori_hw if hori_hw is None else hori_hw)
    scvt_proc = SCVTProc(
        proc=proc,
        hori_hw=hori_hw
    )

    # ---------- Decompose the mesh using METIS ----------
    metis_run(scvt_proc, mesh)

    # ---------- Search for computation domain indices ----------
    find_cmpt_indices(scvt_proc, mesh)

    # ---------- Search bdry and halo domain indices ----------
    bdry_layer, halo_layer = find_bdry_halo_indices(scvt_proc, mesh)

    # ---------- Search inner domain indices ----------
    find_inner_indices(scvt_proc, mesh, bdry_layer)

    # ---------- Process full domain indices ----------
    process_full_indices(scvt_proc, mesh)

    # ---------- Set Send/Recv sites for halo exchange -------------
    set_sendrecv_sites(scvt_proc, mesh, halo_layer)
    
    proc.comm_model = scvt_proc.comm_model

    return scvt_proc


__all__ = ["scvt_decomp_run"]
