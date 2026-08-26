from __future__ import annotations
from typing import Any, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp
import esmpy

from terraflow.util import sphere_geometry

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


def _cell_bounds_from_centers(center: Float[Array, "n"]) -> Float[Array, "n 2"]:
    if center.shape[0] < 2:
        raise ValueError("center must have at least 2 points")

    edge = jnp.empty((center.shape[0] + 1,), dtype=center.dtype)
    edge = edge.at[1:-1].set(0.5 * (center[:-1] + center[1:]))
    edge = edge.at[0].set(center[0] - 0.5 * (center[1] - center[0]))
    edge = edge.at[-1].set(center[-1] + 0.5 * (center[-1] - center[-2]))
    return jnp.stack([edge[:-1], edge[1:]], axis=1)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norm == 0.0, 1.0, norm)


def _sort_conn_counterclockwise_on_sphere(
    elem_conn: np.ndarray,
    node_cart: np.ndarray,
    elem_center_cart: np.ndarray,
) -> np.ndarray:
    """Sort fixed-width element connectivity counterclockwise around centers."""
    center = _normalize_rows(np.asarray(elem_center_cart, dtype=np.float64))
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


def create_scvt_mesh(para: PlanetParameterSet, mesh: SCVTMesh, loc: str, layout: str) -> esmpy.Mesh:
    # ===================================================
    if LOC_REMAP_SCVT2ESMF[loc] == "edge":
        if layout == "triskc":
            node_lon = np.asarray(mesh.vtx_lon )
            node_lat = np.asarray(mesh.vtx_lat )
            cell_lon = np.asarray(mesh.cell_lon)
            cell_lat = np.asarray(mesh.cell_lat)
            node_xyz = np.asarray(mesh.vtx_xyz)
            cell_xyz = np.asarray(mesh.cell_xyz)
            vtx_of_edge  = np.asarray(mesh.vtx_of_edge)
            cell_of_edge = np.asarray(mesh.cell_of_edge)
            if np.any(vtx_of_edge < 0) or np.any(cell_of_edge < 0):
                raise ValueError(
                    "Cannot build an ESMF edge auxiliary mesh with missing edge "
                    "endpoints or adjacent cells."
                )

            aux_lon = np.concatenate([node_lon, cell_lon])
            aux_lat = np.concatenate([node_lat, cell_lat])
            aux_xyz = np.concatenate([node_xyz, cell_xyz], axis=0)
            num_node = aux_lon.shape[0]
            num_elem = vtx_of_edge.shape[0]
            cell_offset = node_lon.shape[0]
            # ===========================================
            elemConn = np.column_stack(
                [
                    vtx_of_edge[:, 0],
                    cell_of_edge[:, 0] + cell_offset,
                    vtx_of_edge[:, 1],
                    cell_of_edge[:, 1] + cell_offset,
                ]
            ).astype(np.int32)
            elemConn = _sort_conn_counterclockwise_on_sphere(
                elem_conn=elemConn,
                node_cart=aux_xyz,
                elem_center_cart=np.asarray(mesh.dual_edge_xyz),
            )
            elemType = np.full(num_elem, 4, dtype=np.int32)

            p0 = aux_xyz[elemConn[:, 0], :]
            p1 = aux_xyz[elemConn[:, 1], :]
            p2 = aux_xyz[elemConn[:, 2], :]
            p3 = aux_xyz[elemConn[:, 3], :]
            elemArea = np.asarray(
                np.abs(
                    sphere_geometry.spherical_triangle_area(
                        para.radius.unwrap(),
                        jnp.array(p0),
                        jnp.array(p1),
                        jnp.array(p2)
                    )
                )
                + np.abs(
                    sphere_geometry.spherical_triangle_area(
                        para.radius.unwrap(),
                        jnp.array(p0),
                        jnp.array(p2),
                        jnp.array(p3)
                    )
                ),
                dtype=np.float64,
            ) / float(para.radius.unwrap())**2
            nodeCoord = np.stack(
                [np.rad2deg(aux_lon), np.rad2deg(aux_lat)],
                axis=1,
            ).ravel()
            elemCoord = np.stack(
                [
                    np.rad2deg(mesh.dual_edge_lon),
                    np.rad2deg(mesh.dual_edge_lat),
                ],
                axis=1,
            ).ravel()            
        else:
            raise ValueError(f"Invalid layout: {layout} for scvt mesh")
    else:
        if layout == "triskc":
            if LOC_REMAP_SCVT2ESMF[loc] == "cell":
                num_node   = mesh.vtx_ime  - mesh.vtx_ims  + 1
                num_elem   = mesh.cell_ime - mesh.cell_ims + 1
                node_lon   = mesh.vtx_lon
                node_lat   = mesh.vtx_lat
                elem_lon   = mesh.cell_lon
                elem_lat   = mesh.cell_lat
                elem_nnb   = mesh.cell_nnb
                elem_nodes = mesh.vtx_of_cell
                elem_area  = mesh.area_cell
            elif LOC_REMAP_SCVT2ESMF[loc] == "node":
                num_node   = mesh.cell_ime - mesh.cell_ims + 1
                num_elem   = mesh.vtx_ime  - mesh.vtx_ims  + 1
                node_lon   = mesh.cell_lon
                node_lat   = mesh.cell_lat
                elem_lon   = mesh.vtx_lon
                elem_lat   = mesh.vtx_lat
                elem_nnb   = mesh.vtx_nnb
                elem_nodes = mesh.cell_of_vtx
                elem_area  = mesh.area_vtx
            else:
                raise ValueError(
                    f"Invalid loc: {loc} for {layout} layout on scvt mesh"
                )
        else:
            raise ValueError(f"Invalid layout: {layout} for scvt mesh")

        nodeCoord = np.stack(
            [
                np.rad2deg(node_lon),
                np.rad2deg(node_lat)
            ],
            axis=1
        ).ravel()
        elemType = np.array(elem_nnb, dtype=np.int32)
        # ESMF uses local indexing for element connectivity, not nodeId
        elemConn = np.array(
            elem_nodes[elem_nodes >= 0],
            dtype=np.int32
        )
        elemArea = np.array(elem_area) / float(para.radius.unwrap())**2
        elemCoord = np.stack(
            [
                np.rad2deg(elem_lon),
                np.rad2deg(elem_lat)
            ],
            axis=1
        ).ravel()

    # ESMF uses 1-based indexing
    nodeId = np.arange(0, num_node, dtype=np.int32) + 1
    nodeOwner = np.zeros(num_node, dtype=np.int32)
    # ESMF uses 1-based indexing
    elemId = np.arange(0, num_elem, dtype=np.int32) + 1

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
            lon = mesh.dual_edge_lon
            lat = mesh.dual_edge_lat
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
    locstream["ESMF:Lon"] = np.asarray(
        np.rad2deg(lon),
        dtype=np.float64
    )
    locstream["ESMF:Lat"] = np.asarray(
        np.rad2deg(lat),
        dtype=np.float64
    )
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
    def create_scvt_mesh(para: PlanetParameterSet, mesh: SCVTMesh, loc: str, layout: str) -> esmpy.Mesh:
        return create_scvt_mesh(para=para, mesh=mesh, loc=loc, layout=layout)

    @staticmethod
    def create_locstream(mesh: SCVTMesh, loc: str, layout: str) -> esmpy.LocStream:
        return create_locstream(mesh=mesh, loc=loc, layout=layout)
