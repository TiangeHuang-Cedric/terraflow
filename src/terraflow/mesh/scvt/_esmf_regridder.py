from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp
import esmpy
from numba import njit


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.planet import PlanetParameterSet
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    Float = typing.Float
    Array = typing.Array


ESMF_OBJECTS = esmpy.Grid | esmpy.Mesh | esmpy.LocStream

ESMF_NORM_OPTIONS = {
    "destarea": esmpy.NormType.DSTAREA,
    "fracarea": esmpy.NormType.FRACAREA
}

ESMF_METHODS = {
    "bilinear": esmpy.RegridMethod.BILINEAR,
    "patch": esmpy.RegridMethod.PATCH,
    "nearest_stod": esmpy.RegridMethod.NEAREST_STOD,
    "nearest_dtos": esmpy.RegridMethod.NEAREST_DTOS,
    "conserve_1st": esmpy.RegridMethod.CONSERVE,
    "conserve_2nd": esmpy.RegridMethod.CONSERVE_2ND,
}

ESMF_EXTRAP_METHODS = {
    "none": esmpy.ExtrapMethod.NONE,
    "nearest_stod": esmpy.ExtrapMethod.NEAREST_STOD,
    "nearest_idavg": esmpy.ExtrapMethod.NEAREST_IDAVG,
    "creep_fill": esmpy.ExtrapMethod.CREEP_FILL
}

ESMF_CONSERVATIVE_METHODS = {"conserve_1st", "conserve_2nd"}

LOC_REMAP_SCVT2ESMF = {
    "cell": "cell",
    "vtx": "node",
    "edge": "edge",
    "lev": "cell",
    "lev_vtx": "node",
    "lev_edge": "edge",
}


def _cell_bounds_from_centers(
    center: Float[Array, "n"],
    lower: float | None = None,
    upper: float | None = None,
) -> Float[Array, "n 2"]:
    if center.shape[0] < 2:
        raise ValueError("center must have at least 2 points")

    edge = jnp.empty((center.shape[0] + 1,), dtype=center.dtype)
    edge = edge.at[1:-1].set(0.5 * (center[:-1] + center[1:]))
    edge = edge.at[0].set(center[0] - 0.5 * (center[1] - center[0]))
    edge = edge.at[-1].set(center[-1] + 0.5 * (center[-1] - center[-2]))
    if lower is not None:
        edge = jnp.maximum(edge, lower)
    if upper is not None:
        edge = jnp.minimum(edge, upper)
    return jnp.stack([edge[:-1], edge[1:]], axis=1)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norm == 0.0, 1.0, norm)


def _sort_conn_counterclockwise_on_sphere(
    elem_conn: np.ndarray,
    node_cart: np.ndarray,
    elem_center_xyz: np.ndarray,
) -> np.ndarray:
    """Sort fixed-width element connectivity counterclockwise around centers."""
    center = _normalize_rows(np.asarray(elem_center_xyz, dtype=np.float64))
    ref = np.zeros_like(center)
    ref[:, 2] = 1.0
    near_pole = np.abs(np.sum(center * ref, axis=1)) > 0.9
    ref[near_pole, :] = np.array([1.0, 0.0, 0.0])

    east = _normalize_rows(np.cross(ref, center))
    north = np.cross(center, east)

    corner = np.asarray(node_cart, dtype=np.float64)[elem_conn]
    tangent = corner - np.sum(
        corner * center[:, None, :],
        axis=2,
        keepdims=True,
    ) * center[:, None, :]
    x = np.sum(tangent * east[:, None, :], axis=2)
    y = np.sum(tangent * north[:, None, :], axis=2)
    order = np.argsort(np.arctan2(y, x), axis=1)
    return np.take_along_axis(elem_conn, order, axis=1)


def _global_size(mesh: SCVTMesh, attr: str) -> int:
    parent = getattr(mesh, "parent", None)
    source = parent if parent is not None else mesh
    return int(getattr(source, attr))


def _positive_esmf_ids(global_ids: np.ndarray) -> np.ndarray:
    global_ids = np.asarray(global_ids, dtype=np.int64)
    if np.any(global_ids < 0):
        raise ValueError("ESMF ids require non-negative global ids.")
    return (global_ids + 1).astype(np.int32)


@njit(cache=True)
def _build_counted_connectivity(
    elem_nodes: np.ndarray,
    elem_counts: np.ndarray,
    elem_start: int,
    elem_end: int,
    node_start: int,
    node_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_lidx = np.empty(node_end + 1, dtype=np.int32)
    node_lidx[:] = -1
    used_nodes = np.empty(node_end - node_start + 1, dtype=np.int64)

    num_node = 0
    num_conn = 0
    for elem in range(elem_start, elem_end + 1):
        count = int(elem_counts[elem])
        num_conn += count
        for inb in range(count):
            node = int(elem_nodes[elem, inb])
            if node < node_start or node > node_end:
                raise ValueError("local node index is outside mesh bounds")
            if node_lidx[node] < 0:
                node_lidx[node] = num_node
                used_nodes[num_node] = node
                num_node += 1

    elem_conn = np.empty(num_conn, dtype=np.int32)
    pos = 0
    for elem in range(elem_start, elem_end + 1):
        count = int(elem_counts[elem])
        for inb in range(count):
            elem_conn[pos] = node_lidx[int(elem_nodes[elem, inb])]
            pos += 1

    return used_nodes[:num_node], elem_conn


@njit(cache=True)
def _build_valid_connectivity(
    elem_nodes: np.ndarray,
    elem_start: int,
    elem_end: int,
    node_start: int,
    node_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_lidx = np.empty(node_end + 1, dtype=np.int32)
    node_lidx[:] = -1
    used_nodes = np.empty(node_end - node_start + 1, dtype=np.int64)
    elem_types = np.empty(elem_end - elem_start + 1, dtype=np.int32)

    num_node = 0
    num_conn = 0
    width = elem_nodes.shape[1]
    for elem in range(elem_start, elem_end + 1):
        count = 0
        for inb in range(width):
            node = int(elem_nodes[elem, inb])
            if node < 0:
                continue
            if node < node_start or node > node_end:
                raise ValueError("local node index is outside mesh bounds")
            if node_lidx[node] < 0:
                node_lidx[node] = num_node
                used_nodes[num_node] = node
                num_node += 1
            count += 1
        elem_types[elem - elem_start] = count
        num_conn += count

    elem_conn = np.empty(num_conn, dtype=np.int32)
    pos = 0
    for elem in range(elem_start, elem_end + 1):
        for inb in range(width):
            node = int(elem_nodes[elem, inb])
            if node < 0:
                continue
            elem_conn[pos] = node_lidx[node]
            pos += 1

    return used_nodes[:num_node], elem_types, elem_conn


@njit(cache=True)
def _build_edge_connectivity(
    vtx_of_edge: np.ndarray,
    cell_of_edge: np.ndarray,
    elem_start: int,
    elem_end: int,
    vtx_start: int,
    vtx_end: int,
    cell_start: int,
    cell_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vtx_lidx = np.empty(vtx_end + 1, dtype=np.int32)
    cell_lidx = np.empty(cell_end + 1, dtype=np.int32)
    vtx_lidx[:] = -1
    cell_lidx[:] = -1

    max_node = 4 * (elem_end - elem_start + 1)
    node_kind = np.empty(max_node, dtype=np.int32)
    node_index = np.empty(max_node, dtype=np.int64)
    elem_conn = np.empty((elem_end - elem_start + 1, 4), dtype=np.int32)

    num_node = 0
    for elem in range(elem_start, elem_end + 1):
        v0 = int(vtx_of_edge[elem, 0])
        v1 = int(vtx_of_edge[elem, 1])
        c0 = int(cell_of_edge[elem, 0])
        c1 = int(cell_of_edge[elem, 1])
        if v0 < vtx_start or v0 > vtx_end:
            raise ValueError("local edge vertex index is outside mesh bounds")
        if v1 < vtx_start or v1 > vtx_end:
            raise ValueError("local edge vertex index is outside mesh bounds")
        if c0 < cell_start or c0 > cell_end:
            raise ValueError("local edge cell index is outside mesh bounds")
        if c1 < cell_start or c1 > cell_end:
            raise ValueError("local edge cell index is outside mesh bounds")

        if vtx_lidx[v0] < 0:
            vtx_lidx[v0] = num_node
            node_kind[num_node] = 0
            node_index[num_node] = v0
            num_node += 1
        if vtx_lidx[v1] < 0:
            vtx_lidx[v1] = num_node
            node_kind[num_node] = 0
            node_index[num_node] = v1
            num_node += 1
        if cell_lidx[c0] < 0:
            cell_lidx[c0] = num_node
            node_kind[num_node] = 1
            node_index[num_node] = c0
            num_node += 1
        if cell_lidx[c1] < 0:
            cell_lidx[c1] = num_node
            node_kind[num_node] = 1
            node_index[num_node] = c1
            num_node += 1

        elem_pos = elem - elem_start
        elem_conn[elem_pos, 0] = vtx_lidx[v0]
        elem_conn[elem_pos, 1] = cell_lidx[c0]
        elem_conn[elem_pos, 2] = vtx_lidx[v1]
        elem_conn[elem_pos, 3] = cell_lidx[c1]

    return node_kind[:num_node], node_index[:num_node], elem_conn


@njit(cache=True)
def _cell_owner_array(
    global_cell_id: np.ndarray,
    cell_proc_idmap: np.ndarray,
    cell_indices: np.ndarray,
    default_owner: int,
) -> np.ndarray:
    owners = np.empty(cell_indices.size, dtype=np.int32)
    if cell_proc_idmap.size == 0:
        owners[:] = default_owner
        return owners

    for i in range(cell_indices.size):
        global_id = int(global_cell_id[int(cell_indices[i])])
        if global_id < 0 or global_id >= cell_proc_idmap.size:
            raise ValueError("invalid global cell id")
        owners[i] = int(cell_proc_idmap[global_id])
    return owners


@njit(cache=True)
def _vtx_owner_array(
    cell_of_vtx: np.ndarray,
    global_cell_id: np.ndarray,
    cell_proc_idmap: np.ndarray,
    vtx_indices: np.ndarray,
    cell_start: int,
    cell_end: int,
    default_owner: int,
) -> np.ndarray:
    owners = np.empty(vtx_indices.size, dtype=np.int32)
    if cell_proc_idmap.size == 0:
        owners[:] = default_owner
        return owners

    width = cell_of_vtx.shape[1]
    for i in range(vtx_indices.size):
        vtx = int(vtx_indices[i])
        owner = 2147483647
        for inb in range(width):
            cell = int(cell_of_vtx[vtx, inb])
            if cell < 0:
                continue
            if cell < cell_start or cell > cell_end:
                raise ValueError("local cell_of_vtx index is outside mesh bounds")
            global_id = int(global_cell_id[cell])
            if global_id < 0 or global_id >= cell_proc_idmap.size:
                raise ValueError("invalid global cell id")
            cell_owner = int(cell_proc_idmap[global_id])
            if cell_owner < owner:
                owner = cell_owner
        if owner == 2147483647:
            owner = default_owner
        owners[i] = owner
    return owners


def _default_owner(mesh: SCVTMesh) -> int:
    return 0 if getattr(mesh, "parent", None) is None else int(mesh.id)


def _cell_proc_idmap_for_esmf(mesh: SCVTMesh) -> np.ndarray:
    if getattr(mesh, "parent", None) is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(mesh.cell_proc_idmap, dtype=np.int64)


def _check_owner_array(mesh: SCVTMesh, owners: np.ndarray, where: str) -> None:
    proc_map = _cell_proc_idmap_for_esmf(mesh)
    if proc_map.size > 0:
        np_model = int(np.max(proc_map)) + 1
    else:
        np_model = max(int(mesh.id) + 1, 1)
    if owners.size > 0 and (int(np.min(owners)) < 0 or int(np.max(owners)) >= np_model):
        raise ValueError(
            f"{where}: ESMF node owners are outside [0, {np_model})."
        )


@njit(cache=True)
def _norm3(value: np.ndarray) -> float:
    return np.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


@njit(cache=True)
def _unit_arc_length(p1: np.ndarray, p2: np.ndarray) -> float:
    n1 = _norm3(p1)
    n2 = _norm3(p2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    x = p1[0] / n1 + p2[0] / n2
    y = p1[1] / n1 + p2[1] / n2
    z = p1[2] / n1 + p2[2] / n2
    dot_product = x * x + y * y + z * z
    if dot_product == 0.0:
        return np.pi
    if dot_product > 4.0:
        return 0.0
    return 2.0 * np.arctan(np.sqrt((4.0 - dot_product) / dot_product))


@njit(cache=True)
def _unit_spherical_triangle_area(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    a = _unit_arc_length(p1, p2)
    b = _unit_arc_length(p2, p3)
    c = _unit_arc_length(p3, p1)
    s = (a + b + c) * 0.5
    tmp = (
        np.tan(s * 0.5)
        * np.tan((s - a) * 0.5)
        * np.tan((s - b) * 0.5)
        * np.tan((s - c) * 0.5)
    )
    if tmp < 0.0:
        return 0.0
    return 4.0 * np.arctan(np.sqrt(tmp))


@njit(cache=True)
def _unit_spherical_quad_area(
    node_xyz: np.ndarray,
    elem_conn: np.ndarray,
) -> np.ndarray:
    area = np.empty(elem_conn.shape[0], dtype=np.float64)
    for elem in range(elem_conn.shape[0]):
        p0 = node_xyz[int(elem_conn[elem, 0]), :]
        p1 = node_xyz[int(elem_conn[elem, 1]), :]
        p2 = node_xyz[int(elem_conn[elem, 2]), :]
        p3 = node_xyz[int(elem_conn[elem, 3]), :]
        area[elem] = abs(_unit_spherical_triangle_area(p0, p1, p2)) + abs(
            _unit_spherical_triangle_area(p0, p2, p3)
        )
    return area


def scvt_local_bounds(
    mesh: SCVTMesh,
    loc: str,
    with_halo: bool = False,
) -> tuple[int, int]:
    esmf_loc = LOC_REMAP_SCVT2ESMF[loc]
    if esmf_loc == "cell":
        ims, ime = int(mesh.cell_ims), int(mesh.cell_ime)
        ids, ide = int(mesh.cell_ids), int(mesh.cell_ide)
    elif esmf_loc == "node":
        ims, ime = int(mesh.vtx_ims), int(mesh.vtx_ime)
        ids, ide = int(mesh.vtx_ids), int(mesh.vtx_ide)
    elif esmf_loc == "edge":
        ims, ime = int(mesh.edge_ims), int(mesh.edge_ime)
        ids, ide = int(mesh.edge_ids), int(mesh.edge_ide)
    else:
        raise ValueError(f"Unsupported SCVT location {loc!r}.")

    return (ims, ime) if with_halo else (ids, ide)


def _validate_mesh_payload(
    *,
    num_node: int,
    num_elem: int,
    node_ids: np.ndarray,
    node_owners: np.ndarray,
    node_coords: np.ndarray,
    elem_ids: np.ndarray,
    elem_types: np.ndarray,
    elem_conn: np.ndarray,
    elem_area: np.ndarray,
    elem_coords: np.ndarray,
) -> None:
    if node_ids.shape != (num_node,):
        raise ValueError(f"Invalid node_ids shape: {node_ids.shape}.")
    if node_owners.shape != (num_node,):
        raise ValueError(f"Invalid node_owners shape: {node_owners.shape}.")
    if node_coords.shape != (2 * num_node,):
        raise ValueError(f"Invalid node_coords shape: {node_coords.shape}.")
    if elem_ids.shape != (num_elem,):
        raise ValueError(f"Invalid elem_ids shape: {elem_ids.shape}.")
    if elem_types.shape != (num_elem,):
        raise ValueError(f"Invalid elem_types shape: {elem_types.shape}.")
    if elem_area.shape != (num_elem,):
        raise ValueError(f"Invalid elem_area shape: {elem_area.shape}.")
    if elem_coords.shape != (2 * num_elem,):
        raise ValueError(f"Invalid elem_coords shape: {elem_coords.shape}.")
    if elem_conn.size != int(np.sum(elem_types)):
        raise ValueError(
            f"Invalid elem_conn size: {elem_conn.size}, "
            f"expected {int(np.sum(elem_types))}."
        )
    if elem_conn.size > 0:
        if int(np.min(elem_conn)) < 0 or int(np.max(elem_conn)) >= num_node:
            raise ValueError(
                f"Invalid elem_conn range [{int(np.min(elem_conn))}, "
                f"{int(np.max(elem_conn))}] for {num_node} nodes."
            )
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("node_ids must be unique on each PET.")
    if np.unique(elem_ids).size != elem_ids.size:
        raise ValueError("elem_ids must be unique on each PET.")


def _create_scvt_cell_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_start, node_end = int(mesh.vtx_ims), int(mesh.vtx_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "cell", with_halo)

    vtx_of_cell = np.asarray(mesh.vtx_of_cell, dtype=np.int64)
    cell_nnb    = np.asarray(mesh.cell_nnb, dtype=np.int64)

    num_elem = elem_end - elem_start + 1
    used_nodes, elem_conn = _build_counted_connectivity(
        vtx_of_cell,
        cell_nnb,
        elem_start,
        elem_end,
        node_start,
        node_end,
    )
    num_node = used_nodes.size

    global_vtx_id   = np.asarray(mesh.global_vtx_id, dtype=np.int64)
    vtx_lon_deg     = np.asarray(mesh.vtx_lon_deg, dtype=np.float64)
    vtx_lat_deg     = np.asarray(mesh.vtx_lat_deg, dtype=np.float64)
    global_cell_id  = np.asarray(mesh.global_cell_id, dtype=np.int64)
    cell_proc_idmap = _cell_proc_idmap_for_esmf(mesh)

    node_ids = _positive_esmf_ids(global_vtx_id[used_nodes])
    node_owners = _vtx_owner_array(
        np.asarray(mesh.cell_of_vtx, dtype=np.int64),
        global_cell_id,
        cell_proc_idmap,
        used_nodes,
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        _default_owner(mesh),
    )
    _check_owner_array(mesh, node_owners, "cell mesh")
    node_coords = np.empty(2 * num_node, dtype=np.float64)
    node_coords[0::2] = vtx_lon_deg[used_nodes]
    node_coords[1::2] = vtx_lat_deg[used_nodes]

    elem_types  = np.asarray(cell_nnb[elem_start:elem_end + 1], dtype=np.int32)
    elem_area   = np.asarray(mesh.area_cell[elem_start:elem_end + 1], dtype=np.float64)
    elem_area   = elem_area / float(para.radius.unwrap())**2
    elem_coords = np.stack(
        [
            np.asarray(mesh.cell_lon_deg[elem_start:elem_end + 1]),
            np.asarray(mesh.cell_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).astype(np.float64).ravel()

    elem_ids = _positive_esmf_ids(global_cell_id[elem_start:elem_end + 1])

    return (
        num_node,
        num_elem,
        node_ids,
        node_owners,
        node_coords,
        elem_ids,
        elem_types,
        elem_conn,
        elem_area,
        elem_coords,
    )


def _create_scvt_vtx_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_start, node_end = int(mesh.cell_ims), int(mesh.cell_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "vtx", with_halo)

    cell_of_vtx = np.asarray(mesh.cell_of_vtx, dtype=np.int64)

    num_elem = elem_end - elem_start + 1
    used_nodes, elem_types, elem_conn = _build_valid_connectivity(
        cell_of_vtx,
        elem_start,
        elem_end,
        node_start,
        node_end,
    )
    num_node = used_nodes.size

    global_cell_id  = np.asarray(mesh.global_cell_id, dtype=np.int64)
    cell_proc_idmap = _cell_proc_idmap_for_esmf(mesh)
    cell_lon_deg    = np.asarray(mesh.cell_lon_deg, dtype=np.float64)
    cell_lat_deg    = np.asarray(mesh.cell_lat_deg, dtype=np.float64)
    node_ids        = _positive_esmf_ids(global_cell_id[used_nodes])
    node_owners     = _cell_owner_array(
        global_cell_id,
        cell_proc_idmap,
        used_nodes,
        _default_owner(mesh),
    )
    _check_owner_array(mesh, node_owners, "vtx mesh")
    node_coords = np.empty(2 * num_node, dtype=np.float64)
    node_coords[0::2] = cell_lon_deg[used_nodes]
    node_coords[1::2] = cell_lat_deg[used_nodes]

    elem_area   = np.asarray(mesh.area_vtx[elem_start:elem_end + 1], dtype=np.float64)
    elem_area   = elem_area / float(para.radius.unwrap())**2
    elem_coords = np.stack(
        [
            np.asarray(mesh.vtx_lon_deg[elem_start:elem_end + 1]),
            np.asarray(mesh.vtx_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).astype(np.float64).ravel()

    global_vtx_id = np.asarray(mesh.global_vtx_id, dtype=np.int64)
    elem_ids = _positive_esmf_ids(global_vtx_id[elem_start:elem_end + 1])

    return (
        num_node,
        num_elem,
        node_ids,
        node_owners,
        node_coords,
        elem_ids,
        elem_types,
        elem_conn,
        elem_area,
        elem_coords,
    )


def _create_scvt_edge_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vtx_start , vtx_end  = int(mesh.vtx_ims), int(mesh.vtx_ime)
    cell_start, cell_end = int(mesh.cell_ims), int(mesh.cell_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "edge", with_halo)

    vtx_of_edge  = np.asarray(mesh.vtx_of_edge, dtype=np.int64)
    cell_of_edge = np.asarray(mesh.cell_of_edge, dtype=np.int64)

    num_elem = elem_end - elem_start + 1
    node_kind, node_index, elem_conn_2d = _build_edge_connectivity(
        vtx_of_edge,
        cell_of_edge,
        elem_start,
        elem_end,
        vtx_start,
        vtx_end,
        cell_start,
        cell_end,
    )
    num_node = node_index.size

    node_ids    = np.empty(num_node, dtype=np.int32)
    node_owners = np.empty(num_node, dtype=np.int32)
    node_coords = np.empty(2 * num_node, dtype=np.float64)
    node_xyz    = np.empty((num_node, 3), dtype=np.float64)

    global_vtx_id   = np.asarray(mesh.global_vtx_id, dtype=np.int64)
    global_cell_id  = np.asarray(mesh.global_cell_id, dtype=np.int64)
    cell_proc_idmap = _cell_proc_idmap_for_esmf(mesh)
    vtx_lon_deg     = np.asarray(mesh.vtx_lon_deg, dtype=np.float64)
    vtx_lat_deg     = np.asarray(mesh.vtx_lat_deg, dtype=np.float64)
    cell_lon_deg    = np.asarray(mesh.cell_lon_deg, dtype=np.float64)
    cell_lat_deg    = np.asarray(mesh.cell_lat_deg, dtype=np.float64)
    vtx_xyz         = np.asarray(mesh.vtx_xyz, dtype=np.float64)
    cell_xyz        = np.asarray(mesh.cell_xyz, dtype=np.float64)
    id_offset       = _global_size(mesh, "n_vtx")

    is_vtx_node = node_kind == 0
    is_cell_node = ~is_vtx_node
    vtx_nodes = node_index[is_vtx_node]
    cell_nodes = node_index[is_cell_node]
    node_lon = node_coords[0::2]
    node_lat = node_coords[1::2]

    node_ids[is_vtx_node] = _positive_esmf_ids(global_vtx_id[vtx_nodes])
    node_owners[is_vtx_node] = _vtx_owner_array(
        np.asarray(mesh.cell_of_vtx, dtype=np.int64),
        global_cell_id,
        cell_proc_idmap,
        vtx_nodes,
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        _default_owner(mesh),
    )
    node_lon[is_vtx_node] = vtx_lon_deg[vtx_nodes]
    node_lat[is_vtx_node] = vtx_lat_deg[vtx_nodes]
    node_xyz[is_vtx_node, :] = vtx_xyz[vtx_nodes, :]

    node_ids[is_cell_node] = id_offset + _positive_esmf_ids(global_cell_id[cell_nodes])
    node_owners[is_cell_node] = _cell_owner_array(
        global_cell_id,
        cell_proc_idmap,
        cell_nodes,
        _default_owner(mesh),
    )
    node_lon[is_cell_node] = cell_lon_deg[cell_nodes]
    node_lat[is_cell_node] = cell_lat_deg[cell_nodes]
    node_xyz[is_cell_node, :] = cell_xyz[cell_nodes, :]
    _check_owner_array(mesh, node_owners, "edge mesh")

    elem_types  = np.full (num_elem, esmpy.MeshElemType.QUAD, dtype=np.int32)
    elem_coords = np.stack(
        [
            np.asarray(mesh.dual_edge_lon_deg[elem_start:elem_end + 1]),
            np.asarray(mesh.dual_edge_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).astype(np.float64).ravel()

    global_edge_id = np.asarray(mesh.global_edge_id, dtype=np.int64)
    center_xyz     = np.asarray(mesh.dual_edge_xyz[elem_start:elem_end + 1, :], dtype=np.float64)
    elem_conn_2d = _sort_conn_counterclockwise_on_sphere(
        elem_conn=elem_conn_2d,
        node_cart=node_xyz,
        elem_center_xyz=center_xyz,
    )
    elem_conn = elem_conn_2d.astype(np.int32, copy=False).ravel()
    elem_ids = _positive_esmf_ids(global_edge_id[elem_start:elem_end + 1])
    elem_area = _unit_spherical_quad_area(node_xyz, elem_conn_2d)

    return (
        num_node,
        num_elem,
        node_ids,
        node_owners,
        node_coords,
        elem_ids,
        elem_types,
        elem_conn,
        elem_area,
        elem_coords,
    )


def create_latlon_grid(lon: np.ndarray, lat: np.ndarray) -> esmpy.Grid:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    nlon = lon.size
    nlat = lat.size

    lon_bnds = np.asarray(_cell_bounds_from_centers(jnp.asarray(lon)))
    lat_bnds = np.asarray(
        _cell_bounds_from_centers(jnp.asarray(lat), lower=-90.0, upper=90.0)
    )
    lon_edge = np.concatenate([lon_bnds[:, 0], lon_bnds[-1:, 1]])
    lat_edge = np.concatenate([lat_bnds[:, 0], lat_bnds[-1:, 1]])

    grid = esmpy.Grid(
        max_index=np.array([nlon, nlat], dtype=np.int32),
        staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
        coord_sys=esmpy.CoordSys.SPH_DEG,
    )

    lon_c = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
    lat_c = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
    if lon_c is None or lat_c is None:
        raise RuntimeError(
            "Failed to access source grid center coordinates in ESMPy"
        )
    lon_m, lat_m = np.meshgrid(lon, lat, indexing="ij")
    lon_c[...] = lon_m
    lat_c[...] = lat_m

    lon_v = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
    lat_v = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)
    if lon_v is None or lat_v is None:
        raise RuntimeError(
            "Failed to access source grid corner coordinates in ESMPy"
        )
    lon_vm, lat_vm = np.meshgrid(lon_edge, lat_edge, indexing="ij")
    lon_v[...] = lon_vm
    lat_v[...] = lat_vm

    return grid


def create_scvt_mesh(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    loc: str,
    layout: str,
    with_halo: bool = False,
) -> esmpy.Mesh:
    if layout != "triskc":
        raise ValueError(f"Invalid layout: {layout} for scvt mesh")

    esmf_loc = LOC_REMAP_SCVT2ESMF[loc]
    if esmf_loc == "cell":
        (
            num_node,
            num_elem,
            nodeId,
            nodeOwner,
            nodeCoord,
            elemId,
            elemType,
            elemConn,
            elemArea,
            elemCoord,
        ) = _create_scvt_cell_mesh_payload(para, mesh, with_halo=with_halo)
    elif esmf_loc == "node":
        (
            num_node,
            num_elem,
            nodeId,
            nodeOwner,
            nodeCoord,
            elemId,
            elemType,
            elemConn,
            elemArea,
            elemCoord,
        ) = _create_scvt_vtx_mesh_payload(para, mesh, with_halo=with_halo)
    elif esmf_loc == "edge":
        (
            num_node,
            num_elem,
            nodeId,
            nodeOwner,
            nodeCoord,
            elemId,
            elemType,
            elemConn,
            elemArea,
            elemCoord,
        ) = _create_scvt_edge_mesh_payload(para, mesh, with_halo=with_halo)
    else:
        raise ValueError(f"Invalid loc: {loc} for {layout} layout on scvt mesh")

    _validate_mesh_payload(
        num_node=num_node,
        num_elem=num_elem,
        node_ids=nodeId,
        node_owners=nodeOwner,
        node_coords=nodeCoord,
        elem_ids=elemId,
        elem_types=elemType,
        elem_conn=elemConn,
        elem_area=elemArea,
        elem_coords=elemCoord,
    )

    esmf_mesh = esmpy.Mesh(
        parametric_dim=2,
        spatial_dim=2,
        coord_sys=esmpy.CoordSys.SPH_DEG,
    )

    esmf_mesh.add_nodes(
        node_count=num_node,
        node_ids=nodeId,
        node_coords=nodeCoord,
        node_owners=nodeOwner,
    )
    esmf_mesh.add_elements(
        element_count=num_elem,
        element_ids=elemId,
        element_types=elemType,
        element_conn=elemConn,
        element_area=elemArea,
        element_coords=elemCoord,
    )
    return esmf_mesh


def create_locstream(mesh: SCVTMesh, loc: str, layout: str) -> esmpy.LocStream:
    if layout == "triskc":
        if LOC_REMAP_SCVT2ESMF[loc] == "edge":
            lon = np.asarray(mesh.dual_edge_lon_deg)
            lat = np.asarray(mesh.dual_edge_lat_deg)
        else:
            raise ValueError(
                f"Invalid loc: {loc} for {layout} layout on scvt mesh"
            )
    else:
        raise ValueError(f"Invalid layout: {layout} for scvt mesh")

    locstream = esmpy.LocStream(
        location_count=lon.shape[0],
        coord_sys=esmpy.CoordSys.SPH_DEG
    )
    locstream["ESMF:Lon"] = lon
    locstream["ESMF:Lat"] = lat
    return locstream


class ESMFRegridder:
    """ESMF regridder supporting all major ESMF methods."""

    def __init__(
        self,
        src: ESMF_OBJECTS,
        dst: ESMF_OBJECTS,
        *,
        method: str = "bilinear",
        ndbounds: tuple[int, ...] | None = None,
        extrap: bool = True,
        extrap_method: str = "none",
        conserve_opt: dict[str, Any] = {"norm_opt": "fracarea"},
        precompute_weights: bool = True
    ):
        self.src = src
        self.dst = dst
        if method not in ESMF_METHODS:
            raise ValueError(
                f"Unsupported ESMF method: {method}. "
                f"Supported: {sorted(ESMF_METHODS.keys())}"
            )
        self.method: str = method
        self.ndbounds: tuple[int, ...] | None = ndbounds
        self.extrap = extrap
        self.extrap_method = extrap_method
        self.conserve_opt = conserve_opt
        self._state: dict[str, Any] | None = None

        if precompute_weights:
            self._setup_state()

    def __call__(self, src_data: Float[Array, "nlev nlat nlon"]):
        return self.regrid(src_data)

    def _setup_state(self) -> dict[str, Any]:
        # =======================================================
        # src
        if isinstance(self.src, esmpy.Grid):
            src_field = esmpy.Field(
                self.src,
                name="src",
                staggerloc=esmpy.StaggerLoc.CENTER,
                ndbounds=self.ndbounds,
            )
        elif isinstance(self.src, esmpy.Mesh):
            src_field = esmpy.Field(
                self.src,
                name="src",
                meshloc=esmpy.MeshLoc.ELEMENT,
                ndbounds=self.ndbounds,
            )
        elif isinstance(self.src, esmpy.LocStream):
            src_field = esmpy.Field(
                self.src,
                name="src",
                ndbounds=self.ndbounds,
            )
        else:
            raise ValueError(f"Unsupported ESMF object type: {type(self.src)}")
        # =======================================================
        # dst
        if isinstance(self.dst, esmpy.Grid):
            dst_field = esmpy.Field(
                self.dst,
                name="dst",
                staggerloc=esmpy.StaggerLoc.CENTER,
                ndbounds=self.ndbounds,
            )
        elif isinstance(self.dst, esmpy.Mesh):
            dst_field = esmpy.Field(
                self.dst,
                name="dst",
                meshloc=esmpy.MeshLoc.ELEMENT,
                ndbounds=self.ndbounds,
            )
        elif isinstance(self.dst, esmpy.LocStream):
            dst_field = esmpy.Field(
                self.dst,
                name="dst",
                ndbounds=self.ndbounds,
            )
        else:
            raise ValueError(f"Unsupported ESMF object type: {type(self.dst)}")
        # =======================================================
        if self.method in ESMF_CONSERVATIVE_METHODS:
            regrid = esmpy.Regrid(
                src_field,
                dst_field,
                regrid_method=ESMF_METHODS[self.method],
                norm_type=ESMF_NORM_OPTIONS[self.conserve_opt["norm_opt"]],
                extrap_method=ESMF_EXTRAP_METHODS[self.extrap_method],
                unmapped_action=esmpy.UnmappedAction.IGNORE,
                ignore_degenerate=True,
            )
        else:
            regrid = esmpy.Regrid(
                src_field,
                dst_field,
                regrid_method=ESMF_METHODS[self.method],
                extrap_method=ESMF_EXTRAP_METHODS[self.extrap_method],
                unmapped_action=esmpy.UnmappedAction.IGNORE,
                ignore_degenerate=True,
            )
        self._state = {"src_field": src_field, "dst_field": dst_field, "regrid": regrid}
        return self._state

    def _get_spatial_rank(self, obj: ESMF_OBJECTS) -> int:
        return 2 if isinstance(obj, esmpy.Grid) else 1

    def regrid(self, src_data: Float[Array, "nlev nlat nlon"] | Float[Array, "nlev npoints"]):
        # ========================================================
        src_data_np = np.asarray(src_data)
        src_spatial_rank = self._get_spatial_rank(self.src)
        # Because ESMF following Fortran indexing conventions,
        # we need to transpose the data to match the expected shape.
        # The expected shape is (nlon, nlat) for 2D spatial data and (npoints,) for 1D spatial data.
        # If the input data has an extra leading dimension for vertical levels,
        # we also need to move that dimension to the end.
        if src_data_np.ndim == src_spatial_rank:
            ndbounds: tuple[int, ...] | None = None
            if src_spatial_rank == 2:
                # (nlat, nlon) -> (nlon, nlat)
                src_data_esmf = np.transpose(src_data_np, (1, 0))
            else:
                # (npoint,) -> (npoint,)
                src_data_esmf = src_data_np
        elif src_data_np.ndim == src_spatial_rank + 1:
            ndbounds = (int(src_data_np.shape[0]),)
            if src_spatial_rank == 2:
                # (nlev, nlat, nlon) -> (nlon, nlat, nlev)
                src_data_esmf = np.transpose(src_data_np, (2, 1, 0))
            else:
                # (nlev, npoint) -> (npoint, nlev)
                src_data_esmf = np.transpose(src_data_np, (1, 0))
        else:
            raise ValueError(
                f"Invalid src_data shape: {src_data_np.shape}, expected rank "
                f"{src_spatial_rank} or {src_spatial_rank + 1}"
            )
        if ndbounds is not None and self.ndbounds is not None and ndbounds != self.ndbounds:
            raise ValueError(
                f"Source data ndbounds {ndbounds} does not match regridder "
                f"ndbounds {self.ndbounds}"
            )
        # ========================================================
        if self._state is not None:
            state = self._state
        else:
            state = self._setup_state()
        src_field = state["src_field"]
        dst_field = state["dst_field"]
        regrid = state["regrid"]
        # ========================================================
        src_field.data[...] = src_data_esmf
        dst_field = regrid(src_field, dst_field)
        out = np.asarray(dst_field.data)
        if self.method in ESMF_CONSERVATIVE_METHODS and not self.extrap:
            out = np.where(np.isfinite(out), out, np.nan)

        # Transpose back to TerraFlow's spatial convention:
        # Grid data is (nlat, nlon), mesh/locstream data is (npoints,).
        dst_spatial_rank = self._get_spatial_rank(self.dst)
        if ndbounds is None:
            if dst_spatial_rank == 2:
                out = np.transpose(out, (1, 0))
            else:
                out = out.reshape(-1)
        else:
            nlev = ndbounds[0]
            if dst_spatial_rank == 2:
                out = np.transpose(out, (2, 1, 0))
            else:
                out = out.reshape(-1, nlev).T
        # ========================================================
        return jnp.asarray(out)

    @staticmethod
    def create_latlon_grid(lon: np.ndarray, lat: np.ndarray) -> esmpy.Grid:
        return create_latlon_grid(lon=lon, lat=lat)

    @staticmethod
    def create_scvt_mesh(
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        loc: str,
        layout: str,
        with_halo: bool = False,
    ) -> esmpy.Mesh:
        return create_scvt_mesh(
            para=para,
            mesh=mesh,
            loc=loc,
            layout=layout,
            with_halo=with_halo,
        )

    @staticmethod
    def create_locstream(mesh: SCVTMesh, loc: str, layout: str) -> esmpy.LocStream:
        return create_locstream(mesh=mesh, loc=loc, layout=layout)
