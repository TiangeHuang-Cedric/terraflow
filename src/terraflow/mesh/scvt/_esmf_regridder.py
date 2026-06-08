from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp
import esmpy

if TYPE_CHECKING:
    from terraflow.util import typing
    from .construct import SphericalCentroidVoronoiTessellations as SCVT_MESH

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


def _cell_bounds_from_centers(center: Float[Array, "n"]) -> Float[Array, "n 2"]:
    if center.shape[0] < 2:
        raise ValueError("center must have at least 2 points")

    edge = jnp.empty((center.shape[0] + 1,), dtype=center.dtype)
    edge = edge.at[1:-1].set(0.5 * (center[:-1] + center[1:]))
    edge = edge.at[0].set(center[0] - 0.5 * (center[1] - center[0]))
    edge = edge.at[-1].set(center[-1] + 0.5 * (center[-1] - center[-2]))
    return jnp.stack([edge[:-1], edge[1:]], axis=1)


def create_latlon_grid(lon: np.ndarray, lat: np.ndarray) -> esmpy.Grid:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    nlon = lon.size
    nlat = lat.size

    lon_bnds = np.asarray(_cell_bounds_from_centers(jnp.asarray(lon)))
    lat_bnds = np.asarray(_cell_bounds_from_centers(jnp.asarray(lat)))
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


def create_scvt_mesh(mesh: SCVT_MESH, loc: str, entity: str) -> esmpy.Mesh:
    # ===================================================
    if entity == "primal":
        if loc == "cell":
            elements = mesh.primal.cell
            nodes = mesh.primal.node
        elif loc == "node":
            elements = mesh.dual.cell
            nodes = mesh.dual.node
        else:
            raise ValueError(
                f"Invalid loc: {loc} for {entity} entity on scvt mesh"
            )
    elif entity == "dual":
        if loc == "cell":
            elements = mesh.dual.cell
            nodes = mesh.dual.node
        elif loc == "node":
            elements = mesh.primal.cell
            nodes = mesh.primal.node
        else:
            raise ValueError(
                f"Invalid loc: {loc} for {entity} entity on scvt mesh"
            )
    else:
        raise ValueError(f"Invalid entity: {entity} for scvt mesh")

    num_node = nodes.shape[0]
    num_elem = elements.shape[0]

    # ESMF uses 1-based indexing
    nodeId = np.arange(0, num_node, dtype=np.int32) + 1
    nodeCoord = np.stack(
        [
            np.rad2deg(nodes.coordinate.lon),
            np.rad2deg(nodes.coordinate.lat)
        ],
        axis=1
    ).ravel()
    nodeOwner = np.zeros(num_node, dtype=np.int32)
    # ESMF uses 1-based indexing
    elemId = np.arange(0, num_elem, dtype=np.int32) + 1
    elemType = np.array(elements.n_adjacency, dtype=np.int32)
    # ESMF uses local indexing for element connectivity, not nodeId
    elemConn = np.array(
        elements.nodes[elements.nodes >= 0],
        dtype=np.int32
    )
    elemArea = np.array(elements.area)
    elemCoord = np.stack(
        [
            np.rad2deg(elements.centroid.lon),
            np.rad2deg(elements.centroid.lat)
        ],
        axis=1
    ).ravel()

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


def create_locstream(mesh: SCVT_MESH, loc: str, entity: str) -> esmpy.LocStream:
    if entity == "primal":
        if loc == "edge":
            lon = mesh.primal.edge.midpoint.lon
            lat = mesh.primal.edge.midpoint.lat
        else:
            raise ValueError(
                f"Invalid loc: {loc} for {entity} entity on scvt mesh"
            )
    elif entity == "dual":
        if loc == "edge":
            lon = mesh.dual.edge.midpoint.lon
            lat = mesh.dual.edge.midpoint.lat
        else:
            raise ValueError(
                f"Invalid loc: {loc} for {entity} entity on scvt mesh"
            )
    else:
        raise ValueError(f"Invalid entity: {entity} for scvt mesh")

    locstream = esmpy.LocStream(
        location_count=lon.shape[0],
        coord_sys=esmpy.CoordSys.SPH_DEG
    )
    locstream["ESMF:Lon"] = np.asarray(
        np.rad2deg(lon),
        dtype=np.float64
    )
    locstream["ESMF:Lat"] = np.asarray(
        np.rad2deg(lat),
        dtype=np.float64
    )
    return locstream


def regridder_setup_point_method(
    src: ESMF_OBJECTS,
    dst: ESMF_OBJECTS,
    method: str,
    extrap_method: str = "none",
    ndbounds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    # ===============================================
    src_field = esmpy.Field(
        src,
        name="src",
        staggerloc=esmpy.StaggerLoc.CENTER,
        ndbounds=ndbounds
    )
    dst_field = esmpy.Field(
        dst,
        name="dst",
        meshloc=esmpy.MeshLoc.ELEMENT,
        ndbounds=ndbounds
    )
    # ===============================================
    regrid = esmpy.Regrid(
        src_field,
        dst_field,
        regrid_method=ESMF_METHODS[method],
        extrap_method=ESMF_EXTRAP_METHODS[extrap_method],
        unmapped_action=esmpy.UnmappedAction.IGNORE,
    )
    # ===============================================
    return {"src_field": src_field, "dst_field": dst_field, "regrid": regrid}


def regridder_setup_conservative_method(
    src: ESMF_OBJECTS,
    dst: ESMF_OBJECTS,
    method: str,
    norm_opt: str,
    extrap_method: str = "none",
    ndbounds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    # ===============================================
    src_field = esmpy.Field(
        src,
        name="src",
        staggerloc=esmpy.StaggerLoc.CENTER,
        ndbounds=ndbounds
    )
    dst_field = esmpy.Field(
        dst,
        name="dst",
        meshloc=esmpy.MeshLoc.ELEMENT,
        ndbounds=ndbounds,
    )
    # ===============================================
    regrid = esmpy.Regrid(
        src_field,
        dst_field,
        regrid_method=ESMF_METHODS[method],
        norm_type=ESMF_NORM_OPTIONS[norm_opt],
        extrap_method=ESMF_EXTRAP_METHODS[extrap_method],
        unmapped_action=esmpy.UnmappedAction.IGNORE,
        ignore_degenerate=True,
    )
    # ===============================================
    return {"src_field": src_field, "dst_field": dst_field, "regrid": regrid}


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
        conserve_opt: dict[str, Any] = {"norm_opt": "destarea"},
        precompute_weights: bool = True,
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
        if self.method in ESMF_CONSERVATIVE_METHODS:
            self._state = regridder_setup_conservative_method(
                src=self.src,
                dst=self.dst,
                method=self.method,
                norm_opt=self.conserve_opt["norm_opt"],
                extrap_method=self.extrap_method,
                ndbounds=self.ndbounds,
            )
            return self._state

        self._state = regridder_setup_point_method(
            src=self.src,
            dst=self.dst,
            method=self.method,
            extrap_method=self.extrap_method,
            ndbounds=self.ndbounds,
        )
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

        # Transpose back to JAX expected shape
        if ndbounds is None:
            out = out.reshape(-1)
        else:
            nlev = ndbounds[0]
            out = out.reshape(-1, nlev).T
        # ========================================================
        return jnp.asarray(out)

    @staticmethod
    def create_latlon_grid(lon: np.ndarray, lat: np.ndarray) -> esmpy.Grid:
        return create_latlon_grid(lon=lon, lat=lat)

    @staticmethod
    def create_scvt_mesh(mesh: SCVT_MESH, loc: str, entity: str) -> esmpy.Mesh:
        return create_scvt_mesh(mesh=mesh, loc=loc, entity=entity)

    @staticmethod
    def create_locstream(mesh: SCVT_MESH, loc: str, entity: str) -> esmpy.LocStream:
        return create_locstream(mesh=mesh, loc=loc, entity=entity)
