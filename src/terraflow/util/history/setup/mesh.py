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
        ds.createVariable(mesh_name, "i4", ())
        ds.createVariable("face_lon", "f8", ("n_face",))
        ds.createVariable("face_lat", "f8", ("n_face",))
        ds.createVariable("node_lon", "f8", ("n_node",))
        ds.createVariable("node_lat", "f8", ("n_node",))
        ds.createVariable("edge_lon", "f8", ("n_edge",))
        ds.createVariable("edge_lat", "f8", ("n_edge",))
        ds.createVariable("face_x", "f8", ("n_face",))
        ds.createVariable("face_y", "f8", ("n_face",))
        ds.createVariable("face_z", "f8", ("n_face",))
        ds.createVariable("node_x", "f8", ("n_node",))
        ds.createVariable("node_y", "f8", ("n_node",))
        ds.createVariable("node_z", "f8", ("n_node",))
        ds.createVariable("edge_x", "f8", ("n_edge",))
        ds.createVariable("edge_y", "f8", ("n_edge",))
        ds.createVariable("edge_z", "f8", ("n_edge",))
        ds.createVariable("n_nodes_per_face", "i4", ("n_face",))
        ds.createVariable(
            "face_node_connectivity",
            "i4",
            ("n_face", "n_max_face_nodes")
        )
        ds.createVariable(
            "face_edge_connectivity",
            "i4",
            ("n_face", "n_max_face_edges")
        )
        ds.createVariable(
            "face_face_connectivity",
            "i4",
            ("n_face", "n_max_face_faces")
        )
        ds.createVariable("edge_node_connectivity", "i4", ("n_edge", "two"))
        ds.createVariable("edge_face_connectivity", "i4", ("n_edge", "two"))
        # Fill in coordinate variables
        ds.variables[mesh_name][:] = 0
        ds.variables[mesh_name].cf_role = "mesh_topology"
        ds.variables[mesh_name].topology_dimension = 2
        ds.variables[mesh_name].edge_dimension = "n_edge"
        ds.variables[mesh_name].face_dimension = "n_face"
        ds.variables[mesh_name].node_dimension = "n_node"
        ds.variables[mesh_name].edge_coordinates = "edge_lon edge_lat"
        ds.variables[mesh_name].node_coordinates = "node_lon node_lat"
        ds.variables[mesh_name].face_coordinates = "face_lon face_lat"
        ds.variables[mesh_name].edge_node_connectivity = "edge_node_connectivity"
        ds.variables[mesh_name].edge_face_connectivity = "edge_face_connectivity"
        ds.variables[mesh_name].face_node_connectivity = "face_node_connectivity"
        ds.variables[mesh_name].face_edge_connectivity = "face_edge_connectivity"
        ds.variables[mesh_name].face_face_connectivity = "face_face_connectivity"
        ds.variables["face_lon"][:] = entity.cell.centroid.lon
        ds.variables["face_lon"].long_name = "Face Longitude"
        ds.variables["face_lon"].units = "radians_east"
        ds.variables["face_lat"][:] = entity.cell.centroid.lat
        ds.variables["face_lat"].long_name = "Face Latitude"
        ds.variables["face_lat"].units = "radians_north"
        ds.variables["node_lon"][:] = entity.node.coordinate.lon
        ds.variables["node_lon"].long_name = "Node Longitude"
        ds.variables["node_lon"].units = "radians_east"
        ds.variables["node_lat"][:] = entity.node.coordinate.lat
        ds.variables["node_lat"].long_name = "Node Latitude"
        ds.variables["node_lat"].units = "radians_north"
        ds.variables["edge_lon"][:] = entity.edge.midpoint.lon
        ds.variables["edge_lon"].long_name = "Edge Longitude"
        ds.variables["edge_lon"].units = "radians_east"
        ds.variables["edge_lat"][:] = entity.edge.midpoint.lat
        ds.variables["edge_lat"].long_name = "Edge Latitude"
        ds.variables["edge_lat"].units = "radians_north"
        ds.variables["face_x"][:] = entity.cell.centroid.x
        ds.variables["face_x"].long_name = "Face X Coordinate"
        ds.variables["face_x"].units = "m"
        ds.variables["face_y"][:] = entity.cell.centroid.y
        ds.variables["face_y"].long_name = "Face Y Coordinate"
        ds.variables["face_y"].units = "m"
        ds.variables["face_z"][:] = entity.cell.centroid.z
        ds.variables["face_z"].long_name = "Face Z Coordinate"
        ds.variables["face_z"].units = "m"
        ds.variables["node_x"][:] = entity.node.coordinate.x
        ds.variables["node_x"].long_name = "Node X Coordinate"
        ds.variables["node_x"].units = "m"
        ds.variables["node_y"][:] = entity.node.coordinate.y
        ds.variables["node_y"].long_name = "Node Y Coordinate"
        ds.variables["node_y"].units = "m"
        ds.variables["node_z"][:] = entity.node.coordinate.z
        ds.variables["node_z"].long_name = "Node Z Coordinate"
        ds.variables["node_z"].units = "m"
        ds.variables["edge_x"][:] = entity.edge.midpoint.x
        ds.variables["edge_x"].long_name = "Edge X Coordinate"
        ds.variables["edge_x"].units = "m"
        ds.variables["edge_y"][:] = entity.edge.midpoint.y
        ds.variables["edge_y"].long_name = "Edge Y Coordinate"
        ds.variables["edge_y"].units = "m"
        ds.variables["edge_z"][:] = entity.edge.midpoint.z
        ds.variables["edge_z"].long_name = "Edge Z Coordinate"
        ds.variables["edge_z"].units = "m"
        ds.variables["n_nodes_per_face"][:] = entity.cell.n_adjacency
        ds.variables["n_nodes_per_face"].long_name = "Number of Nodes per Face"
        ds.variables["face_node_connectivity"][:, :] = entity.cell.nodes
        ds.variables["face_node_connectivity"].long_name = "Face-Node Connectivity"
        ds.variables["face_edge_connectivity"][:, :] = entity.cell.edges
        ds.variables["face_edge_connectivity"].long_name = "Face-Edge Connectivity"
        ds.variables["face_face_connectivity"][:, :] = entity.cell.adjacency
        ds.variables["face_face_connectivity"].long_name = "Face-Face Connectivity"
        ds.variables["edge_node_connectivity"][:, :] = entity.edge.endpoints
        ds.variables["edge_node_connectivity"].long_name = "Edge-Node Connectivity"
        ds.variables["edge_face_connectivity"][:, :] = entity.edge.dual_cells
        ds.variables["edge_face_connectivity"].long_name = "Edge-Face Connectivity"
        ds.close()
        self._built = True

    def unstructured(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("nPrimal_Cell", int(self.mesh.primal.cell.shape[0]))
        ds.createDimension("nDual_Cell", int(self.mesh.dual.cell.shape[0]))
        ds.createDimension("nDual_Edge", int(self.mesh.primal.edge.shape[0]))
        # Create variables
        ds.createVariable("lon_pc", "f8", ("nPrimal_Cell",))
        ds.createVariable("lat_pc", "f8", ("nPrimal_Cell",))
        ds.createVariable("lon_de", "f8", ("nDual_Edge",))
        ds.createVariable("lat_de", "f8", ("nDual_Edge",))
        # Fill in coordinate variables
        lon_pc = ds.variables["lon_pc"]
        lon_pc[:] = jnp.rad2deg(self.mesh.primal.cell.centroid.lon)
        lon_pc.long_name = "Face Longitude"
        lon_pc.units = "degrees_east"
        lat_pc = ds.variables["lat_pc"]
        lat_pc[:] = jnp.rad2deg(self.mesh.primal.cell.centroid.lat)
        lat_pc.long_name = "Face Latitude"
        lat_pc.units = "degrees_north"
        lon_de = ds.variables["lon_de"]
        lon_de[:] = jnp.rad2deg(self.mesh.primal.edge.midpoint.lon)
        lon_de.long_name = "Edge Longitude"
        lon_de.units = "degrees_east"
        lat_de = ds.variables["lat_de"]
        lat_de[:] = jnp.rad2deg(self.mesh.primal.edge.midpoint.lat)
        lat_de.long_name = "Edge Latitude"
        lat_de.units = "degrees_north"
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
        lon = np.asarray(
            dst_grid[f"{lon_name}"].data
        ).reshape(len(to_lat), len(to_lon))
        lat = np.asarray(
            dst_grid[f"{lat_name}"].data
        ).reshape(len(to_lat), len(to_lon))
        ds.createDimension("lon", lon.shape[1])
        ds.createDimension("lat", lat.shape[0])
        # Create variables
        ds.createVariable("lon", "f8", ("lon",))
        ds.createVariable("lat", "f8", ("lat",))
        ds.variables["lon"][:] = lon[1, :]
        ds.variables["lon"].long_name = "Longitude"
        ds.variables["lon"].units = "degrees_east"
        ds.variables["lat"][:] = lat[:, 0]
        ds.variables["lat"].long_name = "Latitude"
        ds.variables["lat"].units = "degrees_north"
        ds.close()
        self._built = True
