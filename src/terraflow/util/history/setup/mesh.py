from __future__ import annotations
from typing import TYPE_CHECKING
import netCDF4 as nc4
import jax.numpy as jnp
import numpy as np

from terraflow.util.history.setup.base import AbstractHistorySetup

if TYPE_CHECKING:
    from terraflow.mesh.scvt.mesh_type import SCVTMesh


class AbstractMeshHistorySetup(AbstractHistorySetup):
    pass


class SCVTHistorySetup(AbstractMeshHistorySetup):
    file_path: str
    mesh: SCVTMesh

    def __init__(
        self,
        file_path: str,
        mesh: SCVTMesh,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        if output_type not in ["ugrid", "unstructured", "structured"]:
            raise ValueError(
                f"Unsupported output_type value: {output_type}. "
                "Supported values are 'ugrid', 'unstructured', and 'structured'."
            )
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.mesh = mesh
        self.output_type = output_type

    def setup(self, **kwargs):
        if self.has_built():
            return
        if self.output_type == "ugrid":
            use_dual = kwargs.get("use_dual", False)
            self.ugrid(use_dual=use_dual)
        elif self.output_type == "unstructured":
            self.unstructured(**kwargs)
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
            mesh_name = "dual"
        else:
            mesh_name = "primal"
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # =======================================
        # Create dimensions
        if use_dual:
            ds.createDimension("n_face", int(self.mesh.n_vtx))
            ds.createDimension("n_node", int(self.mesh.n_cell))
            ds.createDimension("n_edge", int(self.mesh.n_edge))
            ds.createDimension(
                "n_max_face_nodes",
                int(self.mesh.maxnnb_vtx)
            )
            ds.createDimension(
                "n_max_face_edges",
                int(self.mesh.maxnnb_vtx)
            )
            ds.createDimension(
                "n_max_face_faces",
                int(self.mesh.maxnnb_vtx)
            )
        else:
            ds.createDimension("n_face", int(self.mesh.n_cell))
            ds.createDimension("n_node", int(self.mesh.n_vtx))
            ds.createDimension("n_edge", int(self.mesh.n_edge))
            ds.createDimension(
                "n_max_face_nodes",
                int(self.mesh.maxnnb_cell)
            )
            ds.createDimension(
                "n_max_face_edges",
                int(self.mesh.maxnnb_cell)
            )
            ds.createDimension(
                "n_max_face_faces",
                int(self.mesh.maxnnb_cell)
            )            
        ds.createDimension("two", int(2))
        # Create variables
        int_type = "i4"
        float_type = "f8"
        mesh = ds.createVariable(mesh_name, int_type, ())
        face_lon = ds.createVariable("face_lon", float_type, ("n_face",))
        face_lat = ds.createVariable("face_lat", float_type, ("n_face",))
        node_lon = ds.createVariable("node_lon", float_type, ("n_node",))
        node_lat = ds.createVariable("node_lat", float_type, ("n_node",))
        edge_lon = ds.createVariable("edge_lon", float_type, ("n_edge",))
        edge_lat = ds.createVariable("edge_lat", float_type, ("n_edge",))
        face_x = ds.createVariable("face_x", float_type, ("n_face",))
        face_y = ds.createVariable("face_y", float_type, ("n_face",))
        face_z = ds.createVariable("face_z", float_type, ("n_face",))
        node_x = ds.createVariable("node_x", float_type, ("n_node",))
        node_y = ds.createVariable("node_y", float_type, ("n_node",))
        node_z = ds.createVariable("node_z", float_type, ("n_node",))
        edge_x = ds.createVariable("edge_x", float_type, ("n_edge",))
        edge_y = ds.createVariable("edge_y", float_type, ("n_edge",))
        edge_z = ds.createVariable("edge_z", float_type, ("n_edge",))
        n_nodes_per_face = ds.createVariable(
            "n_nodes_per_face",
            int_type,
            ("n_face",)
        )
        face_node_connectivity = ds.createVariable(
            "face_node_connectivity",
            int_type,
            ("n_face", "n_max_face_nodes"),
            fill_value=-1
        )
        face_edge_connectivity = ds.createVariable(
            "face_edge_connectivity",
            int_type,
            ("n_face", "n_max_face_edges"),
            fill_value=-1
        )
        face_face_connectivity = ds.createVariable(
            "face_face_connectivity",
            int_type,
            ("n_face", "n_max_face_faces"),
            fill_value=-1
        )
        edge_node_connectivity = ds.createVariable(
            "edge_node_connectivity",
            int_type,
            ("n_edge", "two"),
            fill_value=-1
        )
        edge_face_connectivity = ds.createVariable(
            "edge_face_connectivity",
            int_type,
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
        if use_dual:
            face_lon[:] = mesh.vtx_lon
            face_lon.long_name = "Face Longitude"
            face_lon.units = "radians_east"
            face_lat[:] = mesh.vtx_lat
            face_lat.long_name = "Face Latitude"
            face_lat.units = "radians_north"
            node_lon[:] = mesh.cell_lon
            node_lon.long_name = "Node Longitude"
            node_lon.units = "radians_east"
            node_lat[:] = mesh.cell_lat 
            node_lat.long_name = "Node Latitude"
            node_lat.units = "radians_north"
            edge_lon[:] = mesh.dual_edge_lon
            edge_lon.long_name = "Edge Longitude"
            edge_lon.units = "radians_east"
            edge_lat[:] = mesh.dual_edge_lat
            edge_lat.long_name = "Edge Latitude"
            edge_lat.units = "radians_north"
            face_x[:] = mesh.vtx_xyz[:,0]
            face_x.long_name = "Face X Coordinate"
            face_x.units = "m"
            face_y[:] = mesh.vtx_xyz[:,1]
            face_y.long_name = "Face Y Coordinate"
            face_y.units = "m"
            face_z[:] = mesh.vtx_xyz[:,2]
            face_z.long_name = "Face Z Coordinate"
            face_z.units = "m"
            node_x[:] = mesh.cell_xyz[:,0]
            node_x.long_name = "Node X Coordinate"
            node_x.units = "m"
            node_y[:] = mesh.cell_xyz[:,1]
            node_y.long_name = "Node Y Coordinate"
            node_y.units = "m"
            node_z[:] = mesh.cell_xyz[:,2]
            node_z.long_name = "Node Z Coordinate"
            node_z.units = "m"
            edge_x[:] = mesh.dual_edge_xyz[:,0]
            edge_x.long_name = "Edge X Coordinate"
            edge_x.units = "m"
            edge_y[:] = mesh.dual_edge_xyz[:,1]
            edge_y.long_name = "Edge Y Coordinate"
            edge_y.units = "m"
            edge_z[:] = mesh.dual_edge_xyz[:,2]
            edge_z.long_name = "Edge Z Coordinate"
            edge_z.units = "m"
            n_nodes_per_face[:] = mesh.vtx_nnb
            n_nodes_per_face.long_name = "Number of Nodes per Face"
            face_node_connectivity[:, :] = mesh.cell_of_vtx
            face_node_connectivity.long_name = "Face-Node Connectivity"
            face_node_connectivity.start_index = 0
            face_edge_connectivity[:, :] = mesh.vtx_ed
            face_edge_connectivity.long_name = "Face-Edge Connectivity"
            face_edge_connectivity.start_index = 0
            face_face_connectivity[:, :] = mesh.vtx_nb
            face_face_connectivity.long_name = "Face-Face Connectivity"
            face_face_connectivity.start_index = 0
            edge_node_connectivity[:, :] = mesh.cell_of_edge
            edge_node_connectivity.long_name = "Edge-Node Connectivity"
            edge_node_connectivity.start_index = 0
            edge_face_connectivity[:, :] = mesh.vtx_of_edge
            edge_face_connectivity.long_name = "Edge-Face Connectivity"
            edge_face_connectivity.start_index = 0
        else:
            face_lon[:] = mesh.cell_lon
            face_lon.long_name = "Face Longitude"
            face_lon.units = "radians_east"
            face_lat[:] = mesh.cell_lat
            face_lat.long_name = "Face Latitude"
            face_lat.units = "radians_north"
            node_lon[:] = mesh.vtx_lon
            node_lon.long_name = "Node Longitude"
            node_lon.units = "radians_east"
            node_lat[:] = mesh.vtx_lat 
            node_lat.long_name = "Node Latitude"
            node_lat.units = "radians_north"
            edge_lon[:] = mesh.dual_edge_lon
            edge_lon.long_name = "Edge Longitude"
            edge_lon.units = "radians_east"
            edge_lat[:] = mesh.dual_edge_lat
            edge_lat.long_name = "Edge Latitude"
            edge_lat.units = "radians_north"
            face_x[:] = mesh.cell_xyz[:,0]
            face_x.long_name = "Face X Coordinate"
            face_x.units = "m"
            face_y[:] = mesh.cell_xyz[:,1]
            face_y.long_name = "Face Y Coordinate"
            face_y.units = "m"
            face_z[:] = mesh.cell_xyz[:,2]
            face_z.long_name = "Face Z Coordinate"
            face_z.units = "m"
            node_x[:] = mesh.vtx_xyz[:,0]
            node_x.long_name = "Node X Coordinate"
            node_x.units = "m"
            node_y[:] = mesh.vtx_xyz[:,1]
            node_y.long_name = "Node Y Coordinate"
            node_y.units = "m"
            node_z[:] = mesh.vtx_xyz[:,2]
            node_z.long_name = "Node Z Coordinate"
            node_z.units = "m"
            edge_x[:] = mesh.dual_edge_xyz[:,0]
            edge_x.long_name = "Edge X Coordinate"
            edge_x.units = "m"
            edge_y[:] = mesh.dual_edge_xyz[:,1]
            edge_y.long_name = "Edge Y Coordinate"
            edge_y.units = "m"
            edge_z[:] = mesh.dual_edge_xyz[:,2]
            edge_z.long_name = "Edge Z Coordinate"
            edge_z.units = "m"
            n_nodes_per_face[:] = mesh.cell_nnb
            n_nodes_per_face.long_name = "Number of Nodes per Face"
            face_node_connectivity[:, :] = mesh.vtx_of_cell
            face_node_connectivity.long_name = "Face-Node Connectivity"
            face_node_connectivity.start_index = 0
            face_edge_connectivity[:, :] = mesh.cell_ed
            face_edge_connectivity.long_name = "Face-Edge Connectivity"
            face_edge_connectivity.start_index = 0
            face_face_connectivity[:, :] = mesh.cell_nb
            face_face_connectivity.long_name = "Face-Face Connectivity"
            face_face_connectivity.start_index = 0
            edge_node_connectivity[:, :] = mesh.vtx_of_edge
            edge_node_connectivity.long_name = "Edge-Node Connectivity"
            edge_node_connectivity.start_index = 0
            edge_face_connectivity[:, :] = mesh.cell_of_edge
            edge_face_connectivity.long_name = "Edge-Face Connectivity"
            edge_face_connectivity.start_index = 0            
        ds.close()
        self._built = True

    def unstructured(self, with_vtx: bool = True, with_edge: bool = True):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("cell", int(self.mesh.n_cell))
        if with_vtx:
            ds.createDimension("vtx", int(self.mesh.n_vtx))
        if with_edge:
            ds.createDimension("edge", int(self.mesh.n_edge))
        # Create variables
        float_type = "f8"
        ds.createVariable("cell_lon", float_type, ("cell",))
        ds.createVariable("cell_lat", float_type, ("cell",))
        if with_vtx:
            ds.createVariable("vtx_lon", float_type, ("vtx",))
            ds.createVariable("vtx_lat", float_type, ("vtx",))
        if with_edge:
            ds.createVariable("edge_lon", float_type, ("edge",))
            ds.createVariable("edge_lat", float_type, ("edge",))
        # Fill in coordinate variables
        cell_lon = ds.variables["cell_lon"]
        cell_lon[:] = self.mesh.cell_lon_deg
        cell_lon.long_name = "Longitude at cells"
        cell_lon.units = "degrees_east"
        cell_lat = ds.variables["cell_lat"]
        cell_lat[:] = self.mesh.cell_lat_deg
        cell_lat.long_name = "Latitude at cells"
        cell_lat.units = "degrees_north"
        if with_vtx:
            vtx_lon = ds.variables["vtx_lon"]
            vtx_lon[:] = self.mesh.vtx_lon_deg
            vtx_lon.long_name = "Longitude at vertices"
            vtx_lon.units = "degrees_east"
            vtx_lat = ds.variables["vtx_lat"]
            vtx_lat[:] = self.mesh.vtx_lat_deg
            vtx_lat.long_name = "Latitude at vertices"
            vtx_lat.units = "degrees_north"
        if with_edge:
            edge_lon = ds.variables["edge_lon"]
            edge_lon[:] = self.mesh.dual_edge_lon_deg
            edge_lon.long_name = "Longitude at edges"
            edge_lon.units = "degrees_east"
            edge_lat = ds.variables["edge_lat"]
            edge_lat[:] = self.mesh.dual_edge_lat_deg
            edge_lat.long_name = "Latitude at edges"
            edge_lat.units = "degrees_north"
        ds.close()
        self._built = True

    def structured(
        self,
        dst_lon: np.ndarray,
        dst_lat: np.ndarray,
    ):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("lon", dst_lon.shape[0])
        ds.createDimension("lat", dst_lat.shape[0])
        # Create variables
        float_type = "f8"
        lon = ds.createVariable("lon", float_type, ("lon",))
        lat = ds.createVariable("lat", float_type, ("lat",))
        lon[:] = dst_lon
        lon.long_name = "Longitude"
        lon.units = "degrees_east"
        lat[:] = dst_lat
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        ds.close()
        self._built = True
