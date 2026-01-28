from __future__ import annotations
from typing import TYPE_CHECKING
import netCDF4 as nc4
import jax.numpy as jnp
import numpy as np
import uxarray as ux

from terraflow.util.history.setup.base import AbstractHistorySetup

if TYPE_CHECKING:
    from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh


class AbstractMeshHistorySetup(AbstractHistorySetup):
    pass


class OrthogonalDualMeshHistorySetup(AbstractMeshHistorySetup):
    file_path: str
    mesh: OrthogonalDualMesh

    def __init__(self, file_path: str, mesh: OrthogonalDualMesh, output_type: str):
        if output_type not in ["ugrid", "unstructured", "structured"]:
            raise ValueError(
                f"Unsupported output_type value: {output_type}. "
                "Supported values are 'ugrid', 'unstructured', and 'structured'."
            )
        super().__init__(file_path=file_path)
        self.mesh = mesh
        self.output_type = output_type

    def setup(self, **kwargs):
        if self.has_built():
            return
        if self.output_type == "ugrid":
            use_dual = kwargs.get("use_dual", False)
            self.ugrid(use_dual=use_dual)
        elif self.output_type == "unstructured":
            self.unstructured()
        elif self.output_type == "structured":
            self.structured(**kwargs)
        else:
            raise ValueError(
                f"Unsupported output_type value: {self.output_type}. "
                "Supported values are 'ugrid_primal', 'ugrid_dual', "
                "'unstructured', and 'structured'."
            )
        if not self.has_built():
            self._built = True

    def ugrid(self, use_dual: bool):
        if self.has_built():
            return
        if use_dual:
            entity = self.mesh.dual
            mesh_name = "dual"
        else:
            entity = self.mesh.primal
            mesh_name = "primal"
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("n_face", int(entity.cell.shape[0]))
        ds.createDimension("n_node", int(entity.node.shape[0]))
        ds.createDimension("n_edge", int(entity.edge.shape[0]))
        ds.createDimension(
            "n_max_face_nodes",
            int(entity.cell.adjacency.shape[1])
        )
        ds.createDimension(
            "n_max_face_edges",
            int(entity.cell.adjacency.shape[1])
        )
        ds.createDimension(
            "n_max_face_faces",
            int(entity.cell.adjacency.shape[1])
        )
        ds.createDimension("two", int(2))
        # Create variables
        mesh = ds.createVariable(mesh_name, "i4", ())
        face_lon = ds.createVariable("face_lon", "f8", ("n_face",))
        face_lat = ds.createVariable("face_lat", "f8", ("n_face",))
        node_lon = ds.createVariable("node_lon", "f8", ("n_node",))
        node_lat = ds.createVariable("node_lat", "f8", ("n_node",))
        edge_lon = ds.createVariable("edge_lon", "f8", ("n_edge",))
        edge_lat = ds.createVariable("edge_lat", "f8", ("n_edge",))
        face_x = ds.createVariable("face_x", "f8", ("n_face",))
        face_y = ds.createVariable("face_y", "f8", ("n_face",))
        face_z = ds.createVariable("face_z", "f8", ("n_face",))
        node_x = ds.createVariable("node_x", "f8", ("n_node",))
        node_y = ds.createVariable("node_y", "f8", ("n_node",))
        node_z = ds.createVariable("node_z", "f8", ("n_node",))
        edge_x = ds.createVariable("edge_x", "f8", ("n_edge",))
        edge_y = ds.createVariable("edge_y", "f8", ("n_edge",))
        edge_z = ds.createVariable("edge_z", "f8", ("n_edge",))
        n_nodes_per_face = ds.createVariable(
            "n_nodes_per_face",
            "i4",
            ("n_face",)
        )
        face_node_connectivity = ds.createVariable(
            "face_node_connectivity",
            "i4",
            ("n_face", "n_max_face_nodes"),
            fill_value=-1
        )
        face_edge_connectivity = ds.createVariable(
            "face_edge_connectivity",
            "i4",
            ("n_face", "n_max_face_edges"),
            fill_value=-1
        )
        face_face_connectivity = ds.createVariable(
            "face_face_connectivity",
            "i4",
            ("n_face", "n_max_face_faces"),
            fill_value=-1
        )
        edge_node_connectivity = ds.createVariable(
            "edge_node_connectivity",
            "i4",
            ("n_edge", "two"),
            fill_value=-1
        )
        edge_face_connectivity = ds.createVariable(
            "edge_face_connectivity",
            "i4",
            ("n_edge", "two"),
            fill_value=-1
        )
        # Fill in coordinate variables
        mesh[:] = 0
        mesh.cf_role = "mesh_topology"
        mesh.topology_dimension = 2
        mesh.edge_dimension = "n_edge"
        mesh.face_dimension = "n_face"
        mesh.node_dimension = "n_node"
        mesh.edge_coordinates = "edge_lon edge_lat"
        mesh.node_coordinates = "node_lon node_lat"
        mesh.face_coordinates = "face_lon face_lat"
        mesh.edge_node_connectivity = "edge_node_connectivity"
        mesh.edge_face_connectivity = "edge_face_connectivity"
        mesh.face_node_connectivity = "face_node_connectivity"
        mesh.face_edge_connectivity = "face_edge_connectivity"
        mesh.face_face_connectivity = "face_face_connectivity"
        face_lon[:] = entity.cell.centroid.lon
        face_lon.long_name = "Face Longitude"
        face_lon.units = "radians_east"
        face_lat[:] = entity.cell.centroid.lat
        face_lat.long_name = "Face Latitude"
        face_lat.units = "radians_north"
        node_lon[:] = entity.node.coordinate.lon
        node_lon.long_name = "Node Longitude"
        node_lon.units = "radians_east"
        node_lat[:] = entity.node.coordinate.lat
        node_lat.long_name = "Node Latitude"
        node_lat.units = "radians_north"
        edge_lon[:] = entity.edge.midpoint.lon
        edge_lon.long_name = "Edge Longitude"
        edge_lon.units = "radians_east"
        edge_lat[:] = entity.edge.midpoint.lat
        edge_lat.long_name = "Edge Latitude"
        edge_lat.units = "radians_north"
        face_x[:] = entity.cell.centroid.x
        face_x.long_name = "Face X Coordinate"
        face_x.units = "m"
        face_y[:] = entity.cell.centroid.y
        face_y.long_name = "Face Y Coordinate"
        face_y.units = "m"
        face_z[:] = entity.cell.centroid.z
        face_z.long_name = "Face Z Coordinate"
        face_z.units = "m"
        node_x[:] = entity.node.coordinate.x
        node_x.long_name = "Node X Coordinate"
        node_x.units = "m"
        node_y[:] = entity.node.coordinate.y
        node_y.long_name = "Node Y Coordinate"
        node_y.units = "m"
        node_z[:] = entity.node.coordinate.z
        node_z.long_name = "Node Z Coordinate"
        node_z.units = "m"
        edge_x[:] = entity.edge.midpoint.x
        edge_x.long_name = "Edge X Coordinate"
        edge_x.units = "m"
        edge_y[:] = entity.edge.midpoint.y
        edge_y.long_name = "Edge Y Coordinate"
        edge_y.units = "m"
        edge_z[:] = entity.edge.midpoint.z
        edge_z.long_name = "Edge Z Coordinate"
        edge_z.units = "m"
        n_nodes_per_face[:] = entity.cell.n_adjacency
        n_nodes_per_face.long_name = "Number of Nodes per Face"
        face_node_connectivity[:, :] = entity.cell.nodes
        face_node_connectivity.long_name = "Face-Node Connectivity"
        face_node_connectivity.start_index = 0
        face_edge_connectivity[:, :] = entity.cell.edges
        face_edge_connectivity.long_name = "Face-Edge Connectivity"
        face_edge_connectivity.start_index = 0
        face_face_connectivity[:, :] = entity.cell.adjacency
        face_face_connectivity.long_name = "Face-Face Connectivity"
        face_face_connectivity.start_index = 0
        edge_node_connectivity[:, :] = entity.edge.endpoints
        edge_node_connectivity.long_name = "Edge-Node Connectivity"
        edge_node_connectivity.start_index = 0
        edge_face_connectivity[:, :] = entity.edge.dual_cells
        edge_face_connectivity.long_name = "Edge-Face Connectivity"
        edge_face_connectivity.start_index = 0
        ds.close()
        self._built = True

    def unstructured(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("ncell", int(self.mesh.primal.cell.shape[0]))
        ds.createDimension("nvtx", int(self.mesh.dual.cell.shape[0]))
        ds.createDimension("nedge", int(self.mesh.primal.edge.shape[0]))
        # Create variables
        ds.createVariable("lon_cell", "f8", ("ncell",))
        ds.createVariable("lat_cell", "f8", ("ncell",))
        ds.createVariable("lon_vtx", "f8", ("nvtx",))
        ds.createVariable("lat_vtx", "f8", ("nvtx",))
        ds.createVariable("lon_edge", "f8", ("nedge",))
        ds.createVariable("lat_edge", "f8", ("nedge",))
        # Fill in coordinate variables
        lon_cell = ds.variables["lon_cell"]
        lon_cell[:] = jnp.rad2deg(self.mesh.primal.cell.centroid.lon)
        lon_cell.long_name = "Longitude at cells"
        lon_cell.units = "degrees_east"
        lat_cell = ds.variables["lat_cell"]
        lat_cell[:] = jnp.rad2deg(self.mesh.primal.cell.centroid.lat)
        lat_cell.long_name = "Latitude at cells"
        lat_cell.units = "degrees_north"
        lon_vtx = ds.variables["lon_vtx"]
        lon_vtx[:] = jnp.rad2deg(self.mesh.dual.cell.centroid.lon)
        lon_vtx.long_name = "Longitude at vertices"
        lon_vtx.units = "degrees_east"
        lat_vtx = ds.variables["lat_vtx"]
        lat_vtx[:] = jnp.rad2deg(self.mesh.dual.cell.centroid.lat)
        lat_vtx.long_name = "Latitude at vertices"
        lat_vtx.units = "degrees_north"
        lon_edge = ds.variables["lon_edge"]
        lon_edge[:] = jnp.rad2deg(self.mesh.primal.edge.midpoint.lon)
        lon_edge.long_name = "Longitude at edges"
        lon_edge.units = "degrees_east"
        lat_edge = ds.variables["lat_edge"]
        lat_edge[:] = jnp.rad2deg(self.mesh.primal.edge.midpoint.lat)
        lat_edge.long_name = "Latitude at edges"
        lat_edge.units = "degrees_north"
        ds.close()
        self._built = True

    def structured(
        self,
        remap_to: str = "faces",
        lon_west: float = -180.0,
        lon_east: float = 180.0,
        lat_south: float = -90.0,
        lat_north: float = 90.0,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
    ):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        to_lat = np.arange(lat_south, lat_north, south_north_resolution)
        to_lon = np.arange(lon_west, lon_east, east_west_resolution)
        dst_grid = ux.Grid.from_structured(lon=to_lon, lat=to_lat)
        # Create dimensions
        if remap_to == "faces":
            lon_name, lat_name = "face_lon", "face_lat"
        elif remap_to == "nodes":
            lon_name, lat_name = "node_lon", "node_lat"
        elif remap_to == "edges":
            lon_name, lat_name = "edge_lon", "edge_lat"
        else:
            raise ValueError(
                f"Unsupported remap_to value: {remap_to}. "
                "Supported values are 'faces', 'nodes', and 'edges'."
            )
        lon_data = np.asarray(
            dst_grid[f"{lon_name}"].data
        ).reshape(len(to_lat), len(to_lon))
        lat_data = np.asarray(
            dst_grid[f"{lat_name}"].data
        ).reshape(len(to_lat), len(to_lon))
        ds.createDimension("lon", lon_data.shape[1])
        ds.createDimension("lat", lat_data.shape[0])
        # Create variables
        lon = ds.createVariable("lon", "f8", ("lon",))
        lat = ds.createVariable("lat", "f8", ("lat",))
        lon[:] = lon_data[1, :]
        lon.long_name = "Longitude"
        lon.units = "degrees_east"
        lat[:] = lat_data[:, 0]
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        ds.close()
        self._built = True
