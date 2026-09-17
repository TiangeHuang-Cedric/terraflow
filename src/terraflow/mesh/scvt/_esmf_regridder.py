from __future__ import annotations
from typing import Any, cast, TYPE_CHECKING
import numpy as np
import esmpy
from numba import njit


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.backend.abstract import AbstractBackend
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.mesh.scvt.mesh_type import SCVTMesh

    Float = typing.Float
    Array = typing.Array


ESMF_OBJECTS = esmpy.Grid | esmpy.Mesh | esmpy.LocStream

ESMF_NORM_OPTIONS = {
    "destarea": esmpy.NormType.DSTAREA ,
    "fracarea": esmpy.NormType.FRACAREA
}

ESMF_METHODS = {
    "bilinear"    : esmpy.RegridMethod.BILINEAR    ,
    "patch"       : esmpy.RegridMethod.PATCH       ,
    "nearest_stod": esmpy.RegridMethod.NEAREST_STOD,
    "nearest_dtos": esmpy.RegridMethod.NEAREST_DTOS,
    "conserve_1st": esmpy.RegridMethod.CONSERVE    ,
    "conserve_2nd": esmpy.RegridMethod.CONSERVE_2ND,
}

ESMF_EXTRAP_METHODS = {
    "none"         : esmpy.ExtrapMethod.NONE         ,
    "nearest_stod" : esmpy.ExtrapMethod.NEAREST_STOD ,
    "nearest_idavg": esmpy.ExtrapMethod.NEAREST_IDAVG,
    "creep_fill"   : esmpy.ExtrapMethod.CREEP_FILL
}

ESMF_CONSERVATIVE_METHODS = {"conserve_1st", "conserve_2nd"}
ESMF_MISSING_LIMIT = 1.0e20
ESMF_MESH_ELEMENT_ORDER_KEY = "terraflow_elem_order"

LOC_REMAP_SCVT2ESMF = {
    "cell"    : "cell",
    "vtx"     : "node",
    "edge"    : "edge",
    "lev"     : "cell",
    "lev_vtx" : "node",
    "lev_edge": "edge",
}


def cell_bounds_from_centers(
    center: Float[Array, "n"],
    lower: float | None = None,
    upper: float | None = None,
) -> Float[Array, "n 2"]:
    center = np.asarray(center)
    if center.shape[0] < 2:
        raise ValueError("center must have at least 2 points")

    edge = np.empty((center.shape[0] + 1,), dtype=center.dtype)
    edge[1:-1] = 0.5 * (center[:-1] + center[1:])
    edge[0] = center[0] - 0.5 * (center[1] - center[0])
    edge[-1] = center[-1] + 0.5 * (center[-1] - center[-2])
    if lower is not None:
        edge = np.maximum(edge, lower)
    if upper is not None:
        edge = np.minimum(edge, upper)
    return np.stack([edge[:-1], edge[1:]], axis=1)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norm == 0.0, 1.0, norm)


def sort_conn_counterclockwise_on_sphere(
    elem_conn: np.ndarray,
    node_cart: np.ndarray,
    elem_center_xyz: np.ndarray,
) -> np.ndarray:
    """Sort fixed-width element connectivity counterclockwise around centers."""
    center = normalize_rows(np.asarray(elem_center_xyz))
    ref = np.zeros_like(center)
    ref[:, 2] = 1.0
    near_pole = np.abs(np.sum(center * ref, axis=1)) > 0.9
    ref[near_pole, :] = np.array([1.0, 0.0, 0.0])

    east = normalize_rows(np.cross(ref, center))
    north = np.cross(center, east)

    corner = np.asarray(node_cart)[elem_conn]
    tangent = corner - np.sum(
        corner * center[:, None, :],
        axis=2,
        keepdims=True,
    ) * center[:, None, :]
    x = np.sum(tangent * east [:, None, :], axis=2)
    y = np.sum(tangent * north[:, None, :], axis=2)
    order = np.argsort(np.arctan2(y, x), axis=1)
    return np.take_along_axis(elem_conn, order, axis=1)


def positive_esmf_ids(global_ids: np.ndarray) -> np.ndarray:
    global_ids = np.asarray(global_ids)
    if np.any(global_ids < 0):
        raise ValueError("ESMF ids require non-negative global ids.")
    return (global_ids + 1)


def scoped_element_ids(
    mesh: SCVTMesh,
    global_ids: np.ndarray,
    offset: int,
) -> np.ndarray:
    if offset <= 0:
        raise ValueError("ESMF scoped element id offset must be positive.")
    return positive_esmf_ids(global_ids) + int(mesh.dev.proc.id_model) * int(offset)


def clean_esmf_output(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return np.where(
        np.isfinite(value) & (np.abs(value) < ESMF_MISSING_LIMIT),
        value,
        0.0,
    )


def elem_coord_distance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    dlon = np.abs(lon1 - lon2)
    dlon = np.minimum(dlon, np.abs(360.0 - dlon))
    dlat = lat1 - lat2
    return dlon * dlon + dlat * dlat


def mesh_element_order(esmf_mesh: esmpy.Mesh) -> np.ndarray | None:
    return cast(dict[str, Any], esmf_mesh._meta).get(ESMF_MESH_ELEMENT_ORDER_KEY)


def set_mesh_element_order(esmf_mesh: esmpy.Mesh, elem_coords: np.ndarray) -> None:
    local_coords = np.asarray(elem_coords, dtype=np.float64).reshape(-1, 2)
    owned_lon = np.asarray(
        esmf_mesh.get_coords(0, meshloc=esmpy.MeshLoc.ELEMENT),
        dtype=np.float64,
    )
    owned_lat = np.asarray(
        esmf_mesh.get_coords(1, meshloc=esmpy.MeshLoc.ELEMENT),
        dtype=np.float64,
    )
    num_owned = owned_lon.size
    if local_coords.shape[0] != num_owned:
        raise ValueError(
            "ESMF owned element count does not match local SCVT element count: "
            f"{num_owned} != {local_coords.shape[0]}."
        )

    order = np.empty(num_owned, dtype=np.int64)
    used = np.zeros(local_coords.shape[0], dtype=bool)
    local_lon = local_coords[:, 0]
    local_lat = local_coords[:, 1]
    for i in range(num_owned):
        dist = elem_coord_distance(
            owned_lon[i],
            owned_lat[i],
            local_lon,
            local_lat,
        )
        dist[used] = np.inf
        best = int(np.argmin(dist))
        if not np.isfinite(dist[best]) or dist[best] > 1.0e-12:
            raise ValueError("Failed to match ESMF owned element order.")
        order[i] = best
        used[best] = True

    cast(dict[str, Any], esmf_mesh._meta)[ESMF_MESH_ELEMENT_ORDER_KEY] = order


@njit(cache=True)
def build_counted_connectivity(
    elem_nodes: np.ndarray,
    elem_counts: np.ndarray,
    elem_start: int,
    elem_end: int,
    node_start: int,
    node_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_lidx = np.empty(node_end + 1, dtype=elem_nodes.dtype)
    node_lidx[:] = -1
    used_nodes = np.empty(node_end - node_start + 1, dtype=elem_nodes.dtype)

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

    elem_conn = np.empty(num_conn, dtype=elem_nodes.dtype)
    pos = 0
    for elem in range(elem_start, elem_end + 1):
        count = int(elem_counts[elem])
        for inb in range(count):
            elem_conn[pos] = node_lidx[int(elem_nodes[elem, inb])]
            pos += 1

    return used_nodes[:num_node], elem_conn


@njit(cache=True)
def build_valid_connectivity(
    elem_nodes: np.ndarray,
    elem_start: int,
    elem_end: int,
    node_start: int,
    node_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_lidx = np.empty(node_end + 1, dtype=elem_nodes.dtype)
    node_lidx[:] = -1
    used_nodes = np.empty(node_end - node_start + 1, dtype=elem_nodes.dtype)
    elem_types = np.empty(elem_end - elem_start + 1, dtype=elem_nodes.dtype)

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

    elem_conn = np.empty(num_conn, dtype=elem_nodes.dtype)
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
def build_edge_connectivity(
    vtx_of_edge: np.ndarray,
    cell_of_edge: np.ndarray,
    elem_start: int,
    elem_end: int,
    vtx_start: int,
    vtx_end: int,
    cell_start: int,
    cell_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vtx_lidx  = np.empty(vtx_end + 1 , dtype=vtx_of_edge.dtype)
    cell_lidx = np.empty(cell_end + 1, dtype=vtx_of_edge.dtype)
    vtx_lidx [:] = -1
    cell_lidx[:] = -1

    max_node = 4 * (elem_end - elem_start + 1)
    node_kind  = np.empty(max_node                      , dtype=vtx_of_edge.dtype)
    node_index = np.empty(max_node                      , dtype=vtx_of_edge.dtype)
    elem_conn  = np.empty((elem_end - elem_start + 1, 4), dtype=vtx_of_edge.dtype)

    num_node = 0
    for elem in range(elem_start, elem_end + 1):
        v0 = int(vtx_of_edge [elem, 0])
        v1 = int(vtx_of_edge [elem, 1])
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
        elem_conn[elem_pos, 0] = vtx_lidx [v0]
        elem_conn[elem_pos, 1] = cell_lidx[c0]
        elem_conn[elem_pos, 2] = vtx_lidx [v1]
        elem_conn[elem_pos, 3] = cell_lidx[c1]

    return node_kind[:num_node], node_index[:num_node], elem_conn


@njit(cache=True)
def cell_owner_array(
    global_cell_id: np.ndarray,
    cell_proc_idmap: np.ndarray,
    cell_indices: np.ndarray,
    default_owner: int,
) -> np.ndarray:
    owners = np.empty(cell_indices.size, dtype=global_cell_id.dtype)
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
def vtx_owner_array(
    cell_of_vtx: np.ndarray,
    global_cell_id: np.ndarray,
    cell_proc_idmap: np.ndarray,
    vtx_indices: np.ndarray,
    cell_start: int,
    cell_end: int,
    default_owner: int,
) -> np.ndarray:
    owners = np.empty(vtx_indices.size, dtype=cell_of_vtx.dtype)
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


def default_owner(mesh: SCVTMesh) -> int:
    return int(mesh.id)


def parent_cell_proc_idmap(mesh: SCVTMesh) -> np.ndarray:
    if getattr(mesh, "parent", None) is None:
        raise ValueError("ESMF SCVT mesh creation requires a local mesh with a parent global mesh.")
    if mesh.parent.cell_proc_idmap is None:
        raise ValueError("ESMF SCVT mesh creation requires parent.cell_proc_idmap.")
    bkd = mesh.dev.backend
    return bkd.to_numpy(mesh.parent.cell_proc_idmap)


def check_owner_array(mesh: SCVTMesh, owners: np.ndarray, where: str) -> None:
    proc_map = parent_cell_proc_idmap(mesh)
    if proc_map.size > 0:
        np_model = int(np.max(proc_map)) + 1
    else:
        np_model = max(int(mesh.id) + 1, 1)
    if owners.size > 0 and (int(np.min(owners)) < 0 or int(np.max(owners)) >= np_model):
        raise ValueError(
            f"{where}: ESMF node owners are outside [0, {np_model})."
        )


@njit(cache=True)
def norm3(value: np.ndarray) -> float:
    return np.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


@njit(cache=True)
def unit_arc_length(p1: np.ndarray, p2: np.ndarray) -> float:
    n1 = norm3(p1)
    n2 = norm3(p2)
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
def unit_spherical_triangle_area(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    a = unit_arc_length(p1, p2)
    b = unit_arc_length(p2, p3)
    c = unit_arc_length(p3, p1)
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
def unit_spherical_quad_area(
    node_xyz: np.ndarray,
    elem_conn: np.ndarray,
) -> np.ndarray:
    area = np.empty(elem_conn.shape[0], dtype=node_xyz.dtype)
    for elem in range(elem_conn.shape[0]):
        p0 = node_xyz[int(elem_conn[elem, 0]), :]
        p1 = node_xyz[int(elem_conn[elem, 1]), :]
        p2 = node_xyz[int(elem_conn[elem, 2]), :]
        p3 = node_xyz[int(elem_conn[elem, 3]), :]
        area[elem] = (
            abs(unit_spherical_triangle_area(p0, p1, p2)) 
            + abs(unit_spherical_triangle_area(p0, p2, p3))
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
        ims, ime = int(mesh.vtx_ims ), int(mesh.vtx_ime )
        ids, ide = int(mesh.vtx_ids ), int(mesh.vtx_ide )
    elif esmf_loc == "edge":
        ims, ime = int(mesh.edge_ims), int(mesh.edge_ime)
        ids, ide = int(mesh.edge_ids), int(mesh.edge_ide)
    else:
        raise ValueError(f"Unsupported SCVT location {loc!r}.")

    return (ims, ime) if with_halo else (ids, ide)


def validate_mesh_payload(
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


def create_scvt_cell_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bkd = mesh.dev.backend
    node_start, node_end = int(mesh.vtx_ims), int(mesh.vtx_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "cell", with_halo)

    vtx_of_cell = bkd.to_numpy(mesh.vtx_of_cell)
    cell_nnb    = bkd.to_numpy(mesh.cell_nnb   )

    num_elem = elem_end - elem_start + 1
    used_nodes, elem_conn = build_counted_connectivity(
        vtx_of_cell,
        cell_nnb,
        elem_start,
        elem_end,
        node_start,
        node_end,
    )
    num_node = used_nodes.size

    global_vtx_id   = bkd.to_numpy(mesh.global_vtx_id         )
    vtx_lon_deg     = bkd.to_numpy(mesh.vtx_lon_deg           )
    vtx_lat_deg     = bkd.to_numpy(mesh.vtx_lat_deg           )
    global_cell_id  = bkd.to_numpy(mesh.global_cell_id        )
    cell_proc_idmap = parent_cell_proc_idmap(mesh)

    node_ids = positive_esmf_ids(global_vtx_id[used_nodes])
    node_owners = vtx_owner_array(
        bkd.to_numpy(mesh.cell_of_vtx),
        global_cell_id,
        cell_proc_idmap,
        used_nodes,
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        default_owner(mesh),
    )
    check_owner_array(mesh, node_owners, "cell mesh")
    node_coords = np.empty(2 * num_node, dtype=vtx_lon_deg.dtype)
    node_coords[0::2] = vtx_lon_deg[used_nodes]
    node_coords[1::2] = vtx_lat_deg[used_nodes]

    elem_types  = bkd.to_numpy(cell_nnb[elem_start:elem_end + 1])
    elem_area   = bkd.to_numpy(mesh.area_cell[elem_start:elem_end + 1])
    elem_area   = elem_area / float(para.radius.unwrap())**2
    elem_coords = np.stack(
        [
            bkd.to_numpy(mesh.cell_lon_deg[elem_start:elem_end + 1]),
            bkd.to_numpy(mesh.cell_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).ravel()

    elem_ids = scoped_element_ids(
        mesh,
        global_cell_id[elem_start:elem_end + 1],
        int(mesh.parent.n_cell),
    )

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


def create_scvt_vtx_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bkd = mesh.dev.backend
    node_start, node_end = int(mesh.cell_ims), int(mesh.cell_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "vtx", with_halo)

    cell_of_vtx = bkd.to_numpy(mesh.cell_of_vtx)

    num_elem = elem_end - elem_start + 1
    used_nodes, elem_types, elem_conn = build_valid_connectivity(
        cell_of_vtx,
        elem_start,
        elem_end,
        node_start,
        node_end,
    )
    num_node = used_nodes.size

    global_cell_id  = bkd.to_numpy(mesh.global_cell_id        )
    cell_proc_idmap = parent_cell_proc_idmap(mesh)
    cell_lon_deg    = bkd.to_numpy(mesh.cell_lon_deg          )
    cell_lat_deg    = bkd.to_numpy(mesh.cell_lat_deg          )
    node_ids        = positive_esmf_ids(global_cell_id[used_nodes])
    node_owners     = cell_owner_array(
        global_cell_id,
        cell_proc_idmap,
        used_nodes,
        default_owner(mesh),
    )
    check_owner_array(mesh, node_owners, "vtx mesh")
    node_coords = np.empty(2 * num_node, dtype=cell_lon_deg.dtype)
    node_coords[0::2] = cell_lon_deg[used_nodes]
    node_coords[1::2] = cell_lat_deg[used_nodes]

    elem_area   = bkd.to_numpy(mesh.area_vtx[elem_start:elem_end + 1])
    elem_area   = elem_area / float(para.radius.unwrap())**2
    elem_coords = np.stack(
        [
            bkd.to_numpy(mesh.vtx_lon_deg[elem_start:elem_end + 1]),
            bkd.to_numpy(mesh.vtx_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).ravel()

    global_vtx_id = bkd.to_numpy(mesh.global_vtx_id)
    elem_ids = scoped_element_ids(
        mesh,
        global_vtx_id[elem_start:elem_end + 1],
        int(mesh.parent.n_vtx),
    )

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


def create_scvt_edge_mesh_payload(
    para: PlanetParameterSet,
    mesh: SCVTMesh,
    with_halo: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bkd = mesh.dev.backend
    vtx_start , vtx_end  = int(mesh.vtx_ims ), int(mesh.vtx_ime )
    cell_start, cell_end = int(mesh.cell_ims), int(mesh.cell_ime)
    elem_start, elem_end = scvt_local_bounds(mesh, "edge", with_halo)

    vtx_of_edge  = bkd.to_numpy(mesh.vtx_of_edge )
    cell_of_edge = bkd.to_numpy(mesh.cell_of_edge)

    num_elem = elem_end - elem_start + 1
    node_kind, node_index, elem_conn_2d = build_edge_connectivity(
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

    global_vtx_id   = bkd.to_numpy(mesh.global_vtx_id         )
    global_cell_id  = bkd.to_numpy(mesh.global_cell_id        )
    cell_proc_idmap = parent_cell_proc_idmap(mesh)
    vtx_lon_deg     = bkd.to_numpy(mesh.vtx_lon_deg           )
    vtx_lat_deg     = bkd.to_numpy(mesh.vtx_lat_deg           )
    cell_lon_deg    = bkd.to_numpy(mesh.cell_lon_deg          )
    cell_lat_deg    = bkd.to_numpy(mesh.cell_lat_deg          )
    vtx_xyz         = bkd.to_numpy(mesh.vtx_xyz               )
    cell_xyz        = bkd.to_numpy(mesh.cell_xyz              )
    id_offset       = mesh.parent.n_vtx
    
    node_ids    = np.empty(num_node     , dtype=vtx_of_edge .dtype)
    node_owners = np.empty(num_node     , dtype=cell_of_edge.dtype)
    node_coords = np.empty(2 * num_node , dtype=cell_xyz    .dtype)
    node_xyz    = np.empty((num_node, 3), dtype=cell_xyz    .dtype)    

    is_vtx_node = node_kind == 0
    is_cell_node = ~is_vtx_node
    vtx_nodes = node_index[is_vtx_node]
    cell_nodes = node_index[is_cell_node]
    node_lon = node_coords[0::2]
    node_lat = node_coords[1::2]

    node_ids[is_vtx_node] = positive_esmf_ids(global_vtx_id[vtx_nodes])
    node_owners[is_vtx_node] = vtx_owner_array(
        bkd.to_numpy(mesh.cell_of_vtx),
        global_cell_id,
        cell_proc_idmap,
        vtx_nodes,
        int(mesh.cell_ims),
        int(mesh.cell_ime),
        default_owner(mesh),
    )
    node_lon[is_vtx_node] = vtx_lon_deg[vtx_nodes]
    node_lat[is_vtx_node] = vtx_lat_deg[vtx_nodes]
    radius = float(para.radius.unwrap())
    node_xyz[is_vtx_node, :] = vtx_xyz[vtx_nodes, :] / radius

    node_ids[is_cell_node] = id_offset + positive_esmf_ids(global_cell_id[cell_nodes])
    node_owners[is_cell_node] = cell_owner_array(
        global_cell_id,
        cell_proc_idmap,
        cell_nodes,
        default_owner(mesh),
    )
    node_lon[is_cell_node] = cell_lon_deg[cell_nodes]
    node_lat[is_cell_node] = cell_lat_deg[cell_nodes]
    node_xyz[is_cell_node, :] = cell_xyz[cell_nodes, :] / radius
    check_owner_array(mesh, node_owners, "edge mesh")

    elem_types  = np.full (num_elem, esmpy.MeshElemType.QUAD, dtype=node_ids.dtype)
    elem_coords = np.stack(
        [
            bkd.to_numpy(mesh.dual_edge_lon_deg[elem_start:elem_end + 1]),
            bkd.to_numpy(mesh.dual_edge_lat_deg[elem_start:elem_end + 1]),
        ],
        axis=1,
    ).ravel()

    global_edge_id = bkd.to_numpy(mesh.global_edge_id)
    center_xyz     = bkd.to_numpy(mesh.dual_edge_xyz[elem_start:elem_end + 1, :] / radius)
    elem_conn_2d = sort_conn_counterclockwise_on_sphere(
        elem_conn=elem_conn_2d,
        node_cart=node_xyz,
        elem_center_xyz=center_xyz,
    )
    elem_conn = elem_conn_2d.ravel()
    elem_ids = scoped_element_ids(
        mesh,
        global_edge_id[elem_start:elem_end + 1],
        int(mesh.parent.n_edge),
    )
    elem_area = unit_spherical_quad_area(node_xyz, elem_conn_2d)

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


def create_latlon_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    periodic_lon: bool = False,
) -> esmpy.Grid:
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    nlon = lon.size
    nlat = lat.size

    lon_bnds = np.asarray(cell_bounds_from_centers(lon))
    lat_bnds = np.asarray(
        cell_bounds_from_centers(lat, lower=-90.0, upper=90.0)
    )
    lon_edge = np.concatenate([lon_bnds[:, 0], lon_bnds[-1:, 1]])
    lat_edge = np.concatenate([lat_bnds[:, 0], lat_bnds[-1:, 1]])

    grid_kwargs: dict[str, Any] = {}
    if periodic_lon:
        grid_kwargs["num_peri_dims"] = 1
        grid_kwargs["periodic_dim"] = 0
    grid = esmpy.Grid(
        max_index=np.array([nlon, nlat], dtype=np.int32),
        staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
        coord_sys=esmpy.CoordSys.SPH_DEG,
        **grid_kwargs,
    )

    lon_c = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
    lat_c = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
    if lon_c is None or lat_c is None:
        raise RuntimeError(
            "Failed to access source grid center coordinates in ESMPy"
        )
    center_lb = grid.lower_bounds[esmpy.StaggerLoc.CENTER]
    center_ub = grid.upper_bounds[esmpy.StaggerLoc.CENTER]
    if center_lb is None or center_ub is None:
        raise RuntimeError("Failed to access ESMF center local bounds.")
    lon_par = lon[center_lb[0]:center_ub[0]]
    lat_par = lat[center_lb[1]:center_ub[1]]
    lon_m, lat_m = np.meshgrid(lon_par, lat_par, indexing="ij")
    lon_c[...] = lon_m
    lat_c[...] = lat_m

    lon_v = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
    lat_v = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)
    if lon_v is None or lat_v is None:
        raise RuntimeError(
            "Failed to access source grid corner coordinates in ESMPy"
        )
    corner_lb = grid.lower_bounds[cast(Any, esmpy.StaggerLoc.CORNER)]
    corner_ub = grid.upper_bounds[cast(Any, esmpy.StaggerLoc.CORNER)]
    if corner_lb is None or corner_ub is None:
        raise RuntimeError("Failed to access ESMF corner local bounds.")
    lon_edge_par = lon_edge[corner_lb[0]:corner_ub[0]]
    lat_edge_par = lat_edge[corner_lb[1]:corner_ub[1]]
    lon_vm, lat_vm = np.meshgrid(lon_edge_par, lat_edge_par, indexing="ij")
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
        ) = create_scvt_cell_mesh_payload(para, mesh, with_halo=with_halo)
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
        ) = create_scvt_vtx_mesh_payload(para, mesh, with_halo=with_halo)
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
        ) = create_scvt_edge_mesh_payload(para, mesh, with_halo=with_halo)
    else:
        raise ValueError(f"Invalid loc: {loc} for {layout} layout on scvt mesh")

    validate_mesh_payload(
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
    set_mesh_element_order(esmf_mesh, elemCoord)
    return esmf_mesh


def create_locstream(
    mesh: SCVTMesh,
    loc: str,
    layout: str,
    with_halo: bool = False,
) -> esmpy.LocStream:
    bkd = mesh.dev.backend
    if layout == "triskc":
        esmf_loc = LOC_REMAP_SCVT2ESMF[loc]
        if esmf_loc == "cell":
            start, end = scvt_local_bounds(mesh, "cell", with_halo)
            lon = bkd.to_numpy(mesh.cell_lon_deg[start:end + 1])
            lat = bkd.to_numpy(mesh.cell_lat_deg[start:end + 1])
        elif esmf_loc == "node":
            start, end = scvt_local_bounds(mesh, "vtx", with_halo)
            lon = bkd.to_numpy(mesh.vtx_lon_deg[start:end + 1])
            lat = bkd.to_numpy(mesh.vtx_lat_deg[start:end + 1])
        elif esmf_loc == "edge":
            start, end = scvt_local_bounds(mesh, "edge", with_halo)
            lon = bkd.to_numpy(mesh.dual_edge_lon_deg[start:end + 1])
            lat = bkd.to_numpy(mesh.dual_edge_lat_deg[start:end + 1])
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
        precompute_weights: bool = True,
        backend: AbstractBackend,
    ):
        self.src = src
        self.dst = dst
        self.backend = backend
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
        self._states: dict[tuple[int, ...] | None, dict[str, Any]] = {}

        if precompute_weights:
            self.setup_state(self.ndbounds)

    def __call__(self, src_data: Float[Array, "nlev nlat nlon"]):
        return self.regrid(src_data)

    def setup_state(
        self,
        ndbounds: tuple[int, ...] | None,
    ) -> dict[str, Any]:
        typekind = {
            self.backend.float32: esmpy.TypeKind.R4,
            self.backend.float64: esmpy.TypeKind.R8,
            self.backend.int32: esmpy.TypeKind.I4,
            self.backend.int64: esmpy.TypeKind.I8
        }
        # =======================================================
        # src
        if isinstance(self.src, esmpy.Grid):
            src_field = esmpy.Field(
                self.src,
                name="src",
                staggerloc=esmpy.StaggerLoc.CENTER,
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type]
            )
        elif isinstance(self.src, esmpy.Mesh):
            src_field = esmpy.Field(
                self.src,
                name="src",
                meshloc=esmpy.MeshLoc.ELEMENT,
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type] 
            )
        elif isinstance(self.src, esmpy.LocStream):
            src_field = esmpy.Field(
                self.src,
                name="src",
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type]
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
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type]
            )
        elif isinstance(self.dst, esmpy.Mesh):
            dst_field = esmpy.Field(
                self.dst,
                name="dst",
                meshloc=esmpy.MeshLoc.ELEMENT,
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type]
            )
        elif isinstance(self.dst, esmpy.LocStream):
            dst_field = esmpy.Field(
                self.dst,
                name="dst",
                ndbounds=ndbounds,
                typekind=typekind[self.backend.float_type]
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
        state = {"src_field": src_field, "dst_field": dst_field, "regrid": regrid}
        self._states[ndbounds] = state
        return state

    def _get_spatial_rank(self, obj: ESMF_OBJECTS) -> int:
        return 2 if isinstance(obj, esmpy.Grid) else 1

    def _grid_center_slices(self, grid: esmpy.Grid) -> tuple[slice, slice]:
        lower = grid.lower_bounds[esmpy.StaggerLoc.CENTER]
        upper = grid.upper_bounds[esmpy.StaggerLoc.CENTER]
        if lower is None or upper is None:
            raise RuntimeError("Failed to access ESMF center local bounds.")
        lon_slice = slice(int(lower[0]), int(upper[0]))
        lat_slice = slice(int(lower[1]), int(upper[1]))
        return lon_slice, lat_slice

    def _local_src_grid_data(
        self,
        src_data_esmf: np.ndarray,
        src_field_data: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(self.src, esmpy.Grid):
            return src_data_esmf
        if src_data_esmf.shape == src_field_data.shape:
            return src_data_esmf

        lon_slice, lat_slice = self._grid_center_slices(self.src)
        if src_data_esmf.ndim == 2:
            src_data_local = src_data_esmf[lon_slice, lat_slice]
        elif src_data_esmf.ndim == 3:
            src_data_local = src_data_esmf[lon_slice, lat_slice, :]
        else:
            src_data_local = src_data_esmf

        if src_data_local.shape != src_field_data.shape:
            raise ValueError(
                f"Source data shape {src_data_esmf.shape} cannot be assigned "
                f"to ESMF local source field shape {src_field_data.shape}."
            )
        return src_data_local

    def _local_src_mesh_data(
        self,
        src_data_esmf: np.ndarray,
        src_field_data: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(self.src, esmpy.Mesh):
            return src_data_esmf

        order = mesh_element_order(self.src)
        if order is None:
            return src_data_esmf
        if order.size != src_field_data.shape[0]:
            raise ValueError(
                "ESMF source mesh element order size does not match source "
                f"field storage: {order.size} != {src_field_data.shape[0]}."
            )
        if src_data_esmf.ndim == 1:
            src_data_esmf = src_data_esmf[order]
        elif src_data_esmf.ndim == 2:
            src_data_esmf = src_data_esmf[order, :]

        if src_data_esmf.shape != src_field_data.shape:
            raise ValueError(
                f"Source data shape {src_data_esmf.shape} cannot be assigned "
                f"to ESMF local source field shape {src_field_data.shape}."
            )
        return src_data_esmf

    def _ordered_dst_mesh_data(self, out: np.ndarray) -> np.ndarray:
        if not isinstance(self.dst, esmpy.Mesh):
            return out

        order = mesh_element_order(self.dst)
        if order is None:
            return out
        if order.size != out.shape[0]:
            raise ValueError(
                "ESMF destination mesh element order size does not match "
                f"destination field storage: {order.size} != {out.shape[0]}."
            )

        ordered = np.zeros_like(out)
        if out.ndim == 1:
            ordered[order] = out
        else:
            ordered[order, ...] = out
        return ordered

    def dst_grid_slices(self) -> tuple[slice, slice]:
        if not isinstance(self.dst, esmpy.Grid):
            raise ValueError("Destination object is not an ESMF Grid.")
        lon_slice, lat_slice = self._grid_center_slices(self.dst)
        return lat_slice, lon_slice

    def regrid(self, src_data: Float[Array, "nlev nlat nlon"] | Float[Array, "nlev npoints"]):
        # ========================================================
        src_data_np = self.backend.to_numpy(src_data)
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
        if self.ndbounds is not None and ndbounds != self.ndbounds:
            raise ValueError(
                f"Source data ndbounds {ndbounds} does not match regridder "
                f"ndbounds {self.ndbounds}"
            )
        state_ndbounds = self.ndbounds if self.ndbounds is not None else ndbounds
        # ========================================================
        state = self._states.get(state_ndbounds)
        if state is None:
            state = self.setup_state(state_ndbounds)
        src_field = state["src_field"]
        dst_field = state["dst_field"]
        regrid = state["regrid"]
        # ========================================================
        src_data_esmf = self._local_src_grid_data(src_data_esmf, src_field.data)
        src_data_esmf = self._local_src_mesh_data(src_data_esmf, src_field.data)
        src_field.data[...] = src_data_esmf
        dst_field.data[...] = 0.0
        dst_field = regrid(src_field, dst_field)
        out = clean_esmf_output(self.backend.to_numpy(dst_field.data))
        out = self._ordered_dst_mesh_data(out)

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
        return self.backend.asarray(out, device=self.backend())

    @staticmethod
    def create_latlon_grid(
        lon: np.ndarray,
        lat: np.ndarray,
        periodic_lon: bool = False,
    ) -> esmpy.Grid:
        return create_latlon_grid(lon=lon, lat=lat, periodic_lon=periodic_lon)

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
    def create_locstream(
        mesh: SCVTMesh,
        loc: str,
        layout: str,
        with_halo: bool = False,
    ) -> esmpy.LocStream:
        return create_locstream(
            mesh=mesh,
            loc=loc,
            layout=layout,
            with_halo=with_halo,
        )
