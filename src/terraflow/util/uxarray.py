from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing import Optional, cast, Self, overload, Mapping
from typing import Dict, Union, Tuple, Literal, Any, Hashable
from loguru import logger
import datetime
import uxarray as ux
import xarray as xr
import numpy as np
import unxt

from terraflow.util.typing import Array
from terraflow.util.sphere_geometry import convert_longitude_range_in_degrees

if TYPE_CHECKING:
    from xarray.core.dataset_variables import DataVariables
    from xarray.core.types import ErrorOptionsWithWarn
    from terraflow.mesh.scvt.mesh_type import SCVTMesh


class SphericalCoordinates:
    mesh_header: str
    location: str
    lon: Dict[str, xr.DataArray]
    lat: Dict[str, xr.DataArray]

    def __init__(
        self,
        mesh_header: str,
        location: str,
        lon: Array,
        lat: Array,
    ):
        """
        Initialize spherical coordinates.

        Args:
            location (str): The location of the coordinates ("node", "edge", "face").
            lon (Array): The longitude values (in degrees east).
            lat (Array): The latitude values (in degrees north).
        """
        LOCATION = ["node", "edge", "face"]
        if location not in LOCATION:
            raise ValueError(
                f"location must be one of {LOCATION}, got {location}"
            )
        self.mesh_header = mesh_header
        self.location = location
        lon_name = f"{location}_lon"
        self.lon = {
            lon_name: xr.DataArray(
                data=np.asarray(lon),
                dims=(f"n_{location}",),
                attrs=dict(
                    long_name=f"longitude of {location} in {mesh_header} mesh",
                    units="degrees_east"
                ),
            )
        }
        lat_name = f"{location}_lat"
        self.lat = {
            lat_name: xr.DataArray(
                data=np.asarray(lat),
                dims=(f"n_{location}",),
                attrs=dict(
                    long_name=f"latitude of {location} in {mesh_header} mesh",
                    units="degrees_north"
                ),
            )
        }

    def to_dict(self) -> Dict[str, xr.DataArray]:
        coords = {}
        coords.update(self.lon)
        coords.update(self.lat)
        return coords

    @property
    def coordinates_names(self) -> str:
        lon = self.lon.keys().__iter__().__next__()
        lat = self.lat.keys().__iter__().__next__()
        return f"{lon} {lat}"


class CartesianCoordinates:
    mesh_header: str
    x: Dict[str, xr.DataArray]
    y: Dict[str, xr.DataArray]
    z: Dict[str, xr.DataArray]

    def __init__(
        self,
        mesh_header: str,
        location: str,
        x: Array,
        y: Array,
        z: Array,
    ):
        """
        Initialize cartesian coordinates.

        Args:
            mesh_header (str): The mesh header (e.g., "primal", "dual").
            location (str): The location of the coordinates ("node", "edge", "face").
            x (Array): The x direction values (in meters).
            y (Array): The y direction values (in meters).
            z (Array): The z direction values (in meters).
        """
        LOCATION = ["node", "edge", "face"]
        if location not in LOCATION:
            raise ValueError(
                f"location must be one of {LOCATION}, got {location}"
            )
        self.mesh_header = mesh_header
        self.location = location
        x_name = f"{location}_x"
        self.x = {
            x_name: xr.DataArray(
                data=np.asarray(x),
                dims=(f"n_{location}",),
                attrs=dict(
                    long_name=f"x direction of {location} cartesian coordinate in {mesh_header} mesh",
                    units="m"
                ),
            )
        }
        y_name = f"{location}_y"
        self.y = {
            y_name: xr.DataArray(
                data=np.asarray(y),
                dims=(f"n_{location}",),
                attrs=dict(
                    long_name=f"y direction of {location} cartesian coordinate in {mesh_header} mesh",
                    units="m"
                ),
            )
        }
        z_name = f"{location}_z"
        self.z = {
            z_name: xr.DataArray(
                data=np.asarray(z),
                dims=(f"n_{location}",),
                attrs=dict(
                    long_name=f"z direction of {location} cartesian coordinate in {mesh_header} mesh",
                    units="m"
                ),
            )
        }

    def to_dict(self) -> Dict[str, xr.DataArray]:
        coords = {}
        coords.update(self.x)
        coords.update(self.y)
        coords.update(self.z)
        return coords

    @property
    def coordinates_names(self) -> str:
        x = self.x.keys().__iter__().__next__()
        y = self.y.keys().__iter__().__next__()
        z = self.z.keys().__iter__().__next__()
        return f"{x} {y} {z}"


class AbstractElementConvention(ABC):
    """Abstract class for element conventions."""
    _n_element: str
    header: str
    location: str
    sph_coords: SphericalCoordinates
    cart_coords: CartesianCoordinates

    def __init__(
        self,
        header: str,
        lon: Array,
        lat: Array,
        x: Array,
        y: Array,
        z: Array,
    ):
        """Initialize the element convention."""
        self._n_element = f"n_{self.get_location()}"
        self.header = header
        self.sph_coords = SphericalCoordinates(
            mesh_header=header,
            location=self.get_location(),
            lon=lon,
            lat=lat
        )
        self.cart_coords = CartesianCoordinates(
            mesh_header=header,
            location=self.get_location(),
            x=x,
            y=y,
            z=z
        )

    def to_dict(self) -> Dict[str, xr.DataArray]:
        """Convert the element convention to a dictionary of xarray DataArrays."""
        coords = {}
        coords.update(self.sphere_coordinates)
        coords.update(self.cartesian_coordinates)
        return coords

    @staticmethod
    @abstractmethod
    def get_location() -> str:
        """Get the location of the element (node, edge, face)."""
        raise NotImplementedError

    @property
    def sphere_coordinates(self) -> Dict[str, xr.DataArray]:
        """Get the spherical coordinates as a dictionary of xarray DataArrays."""
        return self.sph_coords.to_dict()

    @property
    def cartesian_coordinates(self) -> Dict[str, xr.DataArray]:
        """Get the cartesian coordinates as a dictionary of xarray DataArrays."""
        return self.cart_coords.to_dict()

    @property
    def spherical_coordinates_names(self) -> str:
        """Get the names of the spherical coordinates."""
        return self.sph_coords.coordinates_names

    @property
    def cartesian_coordinates_names(self) -> str:
        """Get the names of the cartesian coordinates."""
        return self.cart_coords.coordinates_names


class NodeConvention(AbstractElementConvention):
    """Node element convention."""

    def __init__(
        self,
        header: str,
        node_lon: Array,
        node_lat: Array,
        node_x: Array,
        node_y: Array,
        node_z: Array
    ):
        """Initialize the node convention."""
        super().__init__(
            header=header,
            lon=node_lon,
            lat=node_lat,
            x=node_x,
            y=node_y,
            z=node_z
        )

    @property
    def n_node(self) -> str:
        """Get the size of the node dimension."""
        return self._n_element

    @staticmethod
    def get_location() -> str:
        """Get the location of the node entity."""
        return "node"

    @classmethod
    def from_scvt_mesh(
        cls,
        mesh: SCVTMesh,
        use_dual: bool = False
    ) -> "NodeConvention":
        """
        Create a NodeConvention from a SphericalCentroidVoronoiTessellations.

        Args:
            mesh (SphericalCentroidVoronoiTessellations): The SCVT mesh.
            use_dual (bool): Whether to output the dual mesh or primal mesh.

        Returns:
            NodeConvention: The node convention for the specified mesh and use_dual flag.
        """
        header = "dual" if use_dual else "primal"
        if use_dual:
            node_x = mesh.cell_xyz[:,0]
            node_y = mesh.cell_xyz[:,1]
            node_z = mesh.cell_xyz[:,2]
            node_lon = mesh.cell_lon_deg
            node_lat = mesh.cell_lat_deg
            node_lon = convert_longitude_range_in_degrees(
                node_lon, from_range=[0, 360], to_range=[-180, 180]
            )
        else:
            node_x = mesh.vtx_xyz[:,0]
            node_y = mesh.vtx_xyz[:,1]
            node_z = mesh.vtx_xyz[:,2]
            node_lon = mesh.vtx_lon_deg
            node_lat = mesh.vtx_lat_deg
            node_lon = convert_longitude_range_in_degrees(
                node_lon, from_range=[0, 360], to_range=[-180, 180]
            )
        return NodeConvention(
            header=header,
            node_lon=node_lon,
            node_lat=node_lat,
            node_x=node_x,
            node_y=node_y,
            node_z=node_z
        )            


class EdgeConvention(AbstractElementConvention):
    """Edge element convention."""
    edge_dimension: str
    edge_node_connectivity: Dict[str, xr.DataArray]
    edge_face_connectivity: Dict[str, xr.DataArray]

    def __init__(
        self,
        header: str,
        is_edge_dimension_first: bool,
        edge_lon: Array,
        edge_lat: Array,
        edge_x: Array,
        edge_y: Array,
        edge_z: Array,
        edge_node_connectivity: Array,
        edge_face_connectivity: Array,
        start_index: int = 0,
        fill_value: int = -1,
    ):
        """Initialize the edge convention."""
        super().__init__(
            header=header,
            lon=edge_lon,
            lat=edge_lat,
            x=edge_x,
            y=edge_y,
            z=edge_z
        )
        self.edge_dimension = f"n_edge"
        dims = (f"n_edge", "two")
        edge_node_connectivity_name = "edge_node_connectivity"
        self.edge_node_connectivity = {
            edge_node_connectivity_name: xr.DataArray(
                data=np.asarray(edge_node_connectivity),
                dims=dims if is_edge_dimension_first else dims[::-1],
                attrs=dict(
                    long_name=f"Nodes at the ends of {header} edges",
                    start_index=start_index,
                    _FillValue=fill_value,
                )
            )
        }
        dims = (f"n_edge", "two")
        edge_face_connectivity_name = "edge_face_connectivity"
        self.edge_face_connectivity = {
            edge_face_connectivity_name: xr.DataArray(
                data=np.asarray(edge_face_connectivity),
                dims=dims if is_edge_dimension_first else dims[::-1],
                attrs=dict(
                    long_name=f"Faces on either side of {header} edges",
                    start_index=start_index,
                    _FillValue=fill_value,
                )
            )
        }

    @property
    def n_edge(self) -> str:
        """Get the size of the edge dimension."""
        return self._n_element

    @staticmethod
    def get_location() -> str:
        """Get the location of the edge entity."""
        return "edge"

    def to_dict(self) -> Dict[str, xr.DataArray]:
        """Convert the edge convention to a dictionary of xarray DataArrays."""
        output = super().to_dict()
        output.update(self.edge_node_connectivity)
        output.update(self.edge_face_connectivity)
        return output

    @property
    def edge_node_connectivity_name(self) -> str:
        """Get the name of the edge node connectivity."""
        return self.edge_node_connectivity.keys().__iter__().__next__()

    @property
    def edge_face_connectivity_name(self) -> str:
        """Get the name of the edge face connectivity."""
        return self.edge_face_connectivity.keys().__iter__().__next__()

    @classmethod
    def from_scvt_mesh(
        cls,
        mesh: SCVTMesh,
        use_dual: bool = False
    ) -> "EdgeConvention":
        """
        Create an edge convention from an SCVT mesh.

        Args:
            mesh (SphericalCentroidVoronoiTessellations): The SCVT mesh.
            use_dual (bool): Whether to output the dual mesh or primal mesh.

        Returns:
            EdgeConvention: The edge convention for the specified mesh and use_dual flag.
        """
        header = "dual" if use_dual else "primal"
        is_edge_dimension_first: bool = True
        if use_dual:
            edge_x = mesh.dual_edge_xyz[:,0]
            edge_y = mesh.dual_edge_xyz[:,0]
            edge_z = mesh.dual_edge_xyz[:,0]
            edge_lon = mesh.dual_edge_lon_deg
            edge_lat = mesh.dual_edge_lat_deg
            edge_lon = convert_longitude_range_in_degrees(
                edge_lon, from_range=[0, 360], to_range=[-180, 180]
            )
            edge_node_connectivity = mesh.cell_of_edge
            edge_face_connectivity = mesh.vtx_of_edge
        else:
            edge_x = mesh.edge_xyz[:,0]
            edge_y = mesh.edge_xyz[:,0]
            edge_z = mesh.edge_xyz[:,0]
            edge_lon = mesh.edge_lon_deg
            edge_lat = mesh.edge_lat_deg
            edge_lon = convert_longitude_range_in_degrees(
                edge_lon, from_range=[0, 360], to_range=[-180, 180]
            )
            edge_node_connectivity = mesh.vtx_of_edge
            edge_face_connectivity = mesh.cell_of_edge
        return EdgeConvention(
            header=header,
            is_edge_dimension_first=is_edge_dimension_first,
            edge_lon=edge_lon,
            edge_lat=edge_lat,
            edge_x=edge_x,
            edge_y=edge_y,
            edge_z=edge_z,
            edge_node_connectivity=edge_node_connectivity,
            edge_face_connectivity=edge_face_connectivity
        )

class FaceConvention(AbstractElementConvention):
    """Face element convention."""
    face_dimension: str
    n_nodes_per_face: Dict[str, xr.DataArray]
    face_node_connectivity: Dict[str, xr.DataArray]
    face_edge_connectivity: Dict[str, xr.DataArray]
    face_face_connectivity: Dict[str, xr.DataArray]

    def __init__(
        self,
        header: str,
        is_face_dimension_first: bool,
        face_lon: Array,
        face_lat: Array,
        face_x: Array,
        face_y: Array,
        face_z: Array,
        n_nodes_per_face: Array,
        face_node_connectivity: Array,
        face_edge_connectivity: Array,
        face_face_connectivity: Array,
        start_index: int = 0,
        fill_value: int = -1,
    ):
        """Initialize the face convention."""
        super().__init__(
            header=header,
            lon=face_lon,
            lat=face_lat,
            x=face_x,
            y=face_y,
            z=face_z
        )
        self.face_dimension = f"n_face"
        dims = (f"n_face",)
        self.n_nodes_per_face = {
            f"n_nodes_per_face": xr.DataArray(
                data=np.asarray(n_nodes_per_face),
                dims=dims,
                attrs=dict(
                    long_name=f"Maximum number of nodes for each face in {header} mesh",
                )
            )
        }
        dims = (f"n_face", f"n_max_face_nodes")
        face_node_connectivity_name = "face_node_connectivity"
        self.face_node_connectivity = {
            face_node_connectivity_name: xr.DataArray(
                data=np.asarray(face_node_connectivity),
                dims=dims if is_face_dimension_first else dims[::-1],
                attrs=dict(
                    long_name=f"Nodes defining {header} faces in {header} mesh",
                    start_index=start_index,
                    _FillValue=fill_value,
                )
            )
        }
        dims = (f"n_face", f"n_max_face_edges")
        face_edge_connectivity_name = "face_edge_connectivity"
        self.face_edge_connectivity = {
            face_edge_connectivity_name: xr.DataArray(
                data=np.asarray(face_edge_connectivity),
                dims=dims if is_face_dimension_first else dims[::-1],
                attrs=dict(
                    long_name=f"Edges defining {header} faces in {header} mesh",
                    start_index=start_index,
                    _FillValue=fill_value,
                )
            )
        }
        dims = (f"n_face", f"n_max_face_faces")
        face_face_connectivity_name = "face_face_connectivity"
        self.face_face_connectivity = {
            face_face_connectivity_name: xr.DataArray(
                data=np.asarray(face_face_connectivity),
                dims=dims if is_face_dimension_first else dims[::-1],
                attrs=dict(
                    long_name=f"Adjacent faces to {header} faces in {header} mesh",
                    start_index=start_index,
                    _FillValue=fill_value,
                )
            )
        }

    @property
    def n_face(self) -> str:
        """Get the size of the face dimension."""
        return self._n_element

    @staticmethod
    def get_location() -> str:
        """Get the location of the face entity."""
        return "face"

    def to_dict(self) -> Dict[str, xr.DataArray]:
        """Convert the face convention to a dictionary of xarray DataArrays."""
        output = super().to_dict()
        output.update(self.n_nodes_per_face)
        output.update(self.face_node_connectivity)
        output.update(self.face_edge_connectivity)
        output.update(self.face_face_connectivity)
        return output

    @property
    def face_node_connectivity_name(self) -> str:
        """Get the name of the face node connectivity."""
        return self.face_node_connectivity.keys().__iter__().__next__()

    @property
    def face_edge_connectivity_name(self) -> str:
        """Get the name of the face edge connectivity."""
        return self.face_edge_connectivity.keys().__iter__().__next__()

    @property
    def face_face_connectivity_name(self) -> str:
        """Get the name of the face face connectivity."""
        return self.face_face_connectivity.keys().__iter__().__next__()

    @classmethod
    def from_scvt_mesh(
        cls,
        mesh: SCVTMesh,
        use_dual: bool = False
    ) -> "FaceConvention":
        """
        Create a FaceConvention from a SphericalCentroidVoronoiTessellations mesh.

        Args:
            mesh (SphericalCentroidVoronoiTessellations): The SCVT mesh.
            use_dual (bool): Whether to output the dual mesh or primal mesh.

        Returns:
            FaceConvention: The face convention for the specified mesh and use_dual flag.
        """
        header = "dual" if use_dual else "primal"
        is_face_dimension_first: bool = True
        if use_dual:
            face_lon = mesh.vtx_lon_deg
            face_lat = mesh.vtx_lat_deg
            face_lon = convert_longitude_range_in_degrees(
                face_lon, from_range=[0, 360], to_range=[-180, 180]
            )
            face_x = mesh.vtx_xyz[:,0]
            face_y = mesh.vtx_xyz[:,1]
            face_z = mesh.vtx_xyz[:,2]
            n_nodes_per_face = mesh.vtx_nnb
            face_node_connectivity = mesh.cell_of_vtx
            face_edge_connectivity = mesh.vtx_ed
            face_face_connectivity = mesh.vtx_nb
        else:
            face_lon = mesh.cell_lon_deg
            face_lat = mesh.cell_lat_deg
            face_lon = convert_longitude_range_in_degrees(
                face_lon, from_range=[0, 360], to_range=[-180, 180]
            )
            face_x = mesh.cell_xyz[:,0]
            face_y = mesh.cell_xyz[:,1]
            face_z = mesh.cell_xyz[:,2]
            n_nodes_per_face = mesh.cell_nnb
            face_node_connectivity = mesh.vtx_of_cell
            face_edge_connectivity = mesh.cell_ed
            face_face_connectivity = mesh.cell_nb
        return FaceConvention(
            header=header,
            is_face_dimension_first=is_face_dimension_first,
            face_lon=face_lon,
            face_lat=face_lat,
            face_x=face_x,
            face_y=face_y,
            face_z=face_z,
            n_nodes_per_face=n_nodes_per_face,
            face_node_connectivity=face_node_connectivity,
            face_edge_connectivity=face_edge_connectivity,
            face_face_connectivity=face_face_connectivity
        )


class MeshConvention:
    """Mesh convention for UGRID representation."""
    header: str
    mesh_representation: int
    cf_role: str
    topology_dimension: int
    node: NodeConvention
    edge: EdgeConvention
    face: FaceConvention

    def __init__(
        self,
        header: str,
        mesh_representation: int,
        topology_dimension: int,
        node: NodeConvention,
        edge: EdgeConvention,
        face: FaceConvention,
        cf_role: str = "mesh_topology"
    ):
        """Initialize the mesh convention."""
        self.header = header
        self.mesh_representation = mesh_representation
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node = node
        self.edge = edge
        self.face = face

    @property
    def mesh_name(self) -> str:
        """Get the name of the mesh."""
        if self.header == "":
            return "mesh"
        return f"{self.header.lower()}_mesh"

    @property
    def edge_dimension(self) -> str:
        """Get the edge dimension name."""
        return self.edge.edge_dimension

    @property
    def face_dimension(self) -> str:
        """Get the face dimension name."""
        return self.face.face_dimension

    @property
    def node_coordinates(self) -> str:
        """Get the node coordinates."""
        return self.node.spherical_coordinates_names

    @property
    def edge_coordinates(self) -> str:
        """Get the edge coordinates."""
        return self.edge.spherical_coordinates_names

    @property
    def face_coordinates(self) -> str:
        """Get the face coordinates."""
        return self.face.spherical_coordinates_names

    @property
    def edge_node_connectivity(self) -> str:
        """Get the name of the edge node connectivity."""
        return self.edge.edge_node_connectivity_name

    @property
    def edge_face_connectivity(self) -> str:
        return self.edge.edge_face_connectivity_name

    @property
    def face_node_connectivity(self) -> str:
        """Get the name of the face node connectivity."""
        return self.face.face_node_connectivity_name

    @property
    def face_edge_connectivity(self) -> str:
        """Get the name of the face edge connectivity."""
        return self.face.face_edge_connectivity_name

    @property
    def face_face_connectivity(self) -> str:
        """Get the name of the face face connectivity."""
        return self.face.face_face_connectivity_name

    def to_dict(self) -> Dict[str, xr.DataArray]:
        """Convert the mesh convention and its topology to an xarray dict."""
        mesh = xr.DataArray(
            data=self.mesh_representation,
            attrs=dict(
                cf_role=self.cf_role,
                topology_dimension=self.topology_dimension,
                edge_dimension=self.edge_dimension,
                face_dimension=self.face_dimension,
                node_coordinates=self.node_coordinates,
                edge_coordinates=self.edge_coordinates,
                face_coordinates=self.face_coordinates,
                edge_node_connectivity=self.edge_node_connectivity,
                edge_face_connectivity=self.edge_face_connectivity,
                face_node_connectivity=self.face_node_connectivity,
                face_edge_connectivity=self.face_edge_connectivity,
                face_face_connectivity=self.face_face_connectivity,
            )
        )
        topology: Dict[str, xr.DataArray] = {
            self.mesh_name: mesh
        }
        topology.update(self.node.to_dict())
        topology.update(self.edge.to_dict())
        topology.update(self.face.to_dict())
        return topology

    @classmethod
    def from_scvt_mesh(
        cls,
        mesh: SCVTMesh,
        use_dual: bool = False
    ) -> "MeshConvention":
        """
        Create a MeshConvention from a SphericalCentroidVoronoiTessellations mesh.

        Args:
            mesh (SphericalCentroidVoronoiTessellations): The SCVT mesh.
            use_dual (bool): Whether to output the dual mesh or primal mesh.

        Returns:
            MeshConvention: The mesh convention for the specified mesh and use_dual flag.
        """
        header: str = "dual" if use_dual else "primal"
        mesh_representation: int = 1 if use_dual else 0
        # TODO: In the Current, we only process meshes in 2D and 3D layered mesh;
        # For fully 3d unstructured mesh, need to change topology_dimension to 3.
        topology_dimension: int = 2
        cf_role: str = "mesh_topology"
        node = NodeConvention.from_scvt_mesh(
            mesh, use_dual=use_dual
        )
        edge = EdgeConvention.from_scvt_mesh(
            mesh, use_dual=use_dual
        )
        face = FaceConvention.from_scvt_mesh(
            mesh, use_dual=use_dual
        )
        return MeshConvention(
            header=header,
            mesh_representation=mesh_representation,
            topology_dimension=topology_dimension,
            node=node,
            edge=edge,
            face=face,
            cf_role=cf_role
        )


class AttributeConvention:
    """Attribute convention for UGRID dataset."""
    conventions: str = "CF-1.11 UGRID-1.0"
    title: str
    institution: str
    source: str
    source_grid_category: str
    history: str

    def __init__(
        self,
        title: str = "UGRID Dataset",
        history: str = f"Created by TerraFlow.util.uxarray module in {datetime.datetime.now()}",
        institution: str = "TerraFlow Project",
        source_grid_category: str = "Spherical Centroid Voronoi Tessellations",
        source: str = "TerraFlow",
        conventions: str = "CF-1.11 UGRID-1.0"
    ):
        """Initialize the attribute convention."""
        self.conventions = conventions
        self.title = title
        self.institution = institution
        self.source_grid_category = source_grid_category
        self.source = source
        self.history = history

    def to_dict(self) -> Dict:
        """Convert the attribute convention to a dictionary."""
        return dict(
            Conventions=self.conventions,
            title=self.title,
            institution=self.institution,
            source=self.source,
            source_grid_category=self.source_grid_category,
            history=self.history,
        )


class UGRIDGrid:
    """UGRID Grid Dataset Builder."""

    def __init__(
        self,
        attributes: AttributeConvention,
        **mesh_conventions: MeshConvention
    ):
        """Initialize the UGRID Grid Dataset."""
        self.attributes = attributes
        for mesh_name, mesh in mesh_conventions.items():
            setattr(self, mesh_name, mesh)

    def copy(self) -> Self:
        """Create a copy of the UGRID Grid Dataset."""
        instance = type(self)(
            attributes=AttributeConvention(
                title=self.attributes.title,
                history=self.attributes.history,
                institution=self.attributes.institution,
                source_grid_category=self.attributes.source_grid_category,
                source=self.attributes.source,
                conventions=self.attributes.conventions
            )
        )
        for mesh_name in [attr for attr in dir(self) if isinstance(getattr(self, attr), MeshConvention)]:
            mesh: MeshConvention = getattr(self, mesh_name)
            setattr(instance, mesh_name, mesh)
        return instance

    @classmethod
    def from_scvt_mesh(
        cls,
        mesh: SCVTMesh,
        attributes: Optional[AttributeConvention] = None,
        use_dual: bool = False,
        **kwargs
    ) -> Self:
        """Create a UGRID dataset from an SCVT mesh."""
        if attributes is None:
            default_attributes = AttributeConvention()
            attributes = AttributeConvention(
                title=kwargs.get("title", default_attributes.title),
                history=kwargs.get("history", default_attributes.history),
                institution=kwargs.get(
                    "institution", default_attributes.institution),
                source_grid_category=kwargs.get(
                    "source_grid_category", default_attributes.source_grid_category
                ),
                source=kwargs.get("source", default_attributes.source),
                conventions=kwargs.get(
                    "conventions", default_attributes.conventions
                )
            )
        if use_dual:
            dual = MeshConvention.from_scvt_mesh(
                mesh=mesh,
                use_dual=True
            )
            return cls(
                attributes=attributes,
                dual=dual
            )
        else:
            primal = MeshConvention.from_scvt_mesh(
                mesh=mesh,
                use_dual=False
            )
            return cls(
                attributes=attributes,
                primal=primal
            )

    def to_xarray(self) -> xr.Dataset:
        """Create a UGRID dataset from the component in SCVT mesh."""
        attributes = self.attributes.to_dict()
        data_vars: Dict[str, xr.DataArray] = {}
        for mesh_name in [attr for attr in dir(self) if isinstance(getattr(self, attr), MeshConvention)]:
            mesh: MeshConvention = getattr(self, mesh_name)
            data_vars.update(mesh.to_dict())
        return xr.Dataset(
            data_vars=data_vars,
            attrs=attributes
        )

    def to_uxgrid(self) -> ux.Grid:
        """Convert to uxarray UxDataset."""
        ds = self.to_xarray()
        return ux.Grid.from_dataset(ds)


class UGRIDDataArray:
    variables: xr.DataArray
    uxgrid: ux.Grid

    def __init__(
        self,
        variables: xr.DataArray,
        uxgrid: ux.Grid
    ):
        """Initialize the UGRIDDataset."""
        self.variables = variables
        self.uxgrid = uxgrid

    @property
    def uxds(self) -> ux.UxDataArray:
        """Get the uxarray UxDataset."""
        return self.to_uxarray()

    @property
    def data(self) -> Any:
        """Get the data of the variables."""
        return self.variables.data

    def _variables_with_spatial_coords(self) -> xr.DataArray:
        """Attach UGRID spatial coordinates to variables for UXarray remapping."""
        da = self.variables
        coords: Dict[str, xr.DataArray] = {}
        for dim in da.dims:
            if dim == "n_face":
                location = "face"
            elif dim == "n_node":
                location = "node"
            elif dim == "n_edge":
                location = "edge"
            else:
                continue
            for coord_name in (f"{location}_lon", f"{location}_lat"):
                coord = getattr(self.uxgrid, coord_name)
                coord_data = np.asarray(coord)
                if coord_data.shape[0] != da.sizes[dim]:
                    continue
                attrs = dict(getattr(coord, "attrs", {}))
                if coord_name.endswith("_lon"):
                    attrs.setdefault("standard_name", "longitude")
                    attrs.setdefault("units", "degrees_east")
                else:
                    attrs.setdefault("standard_name", "latitude")
                    attrs.setdefault("units", "degrees_north")
                coords[coord_name] = xr.DataArray(
                    data=coord_data,
                    dims=(dim,),
                    attrs=attrs,
                )
        if coords:
            return da.assign_coords(coords)
        return da

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> "UGRIDDataArray":
        """Index the UGRIDDataArray."""
        new_vars = self.variables.isel(
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kwargs
        )
        return UGRIDDataArray(
            variables=new_vars,
            uxgrid=self.uxgrid
        )

    @classmethod
    def from_xarray(
        cls,
        da: xr.DataArray,
        uxgrid: ux.Grid,
        attrs: Optional[Dict[str, Any]] = None
    ) -> "UGRIDDataArray":
        """Create a UGRIDDataset from an xarray and UGRIDGrid."""
        data = da.data
        dims = da.dims
        var_name = da.name if da.name is not None else "variable"
        long_name = da.attrs.get("long_name", "")
        unit = da.attrs.get("units", "")
        location = da.attrs.get("location", "")
        if unit == "" or long_name == "":
            raise ValueError(
                f"Variable {var_name} must have 'long_name/units' attribute to be added to UGRIDDataset."
            )
        if location == "" or data is None or dims == ():
            raise ValueError(
                f"Variable {var_name} must have 'location/data/dims' attribute to be added to UGRIDDataset."
            )
        da_attrs = da.attrs
        if attrs is not None:
            da_attrs.update(attrs)
        new_ds = xr.DataArray(
            data=np.asarray(data),
            dims=dims,
            attrs=da_attrs
        )
        return cls(variables=new_ds, uxgrid=uxgrid)

    def to_xarray(self) -> xr.DataArray:
        """Convert to xarray Dataset."""
        xrgrid = self.uxgrid.to_xarray()
        da = self.variables
        da = da.merge(xrgrid)
        return da

    def to_uxarray(self) -> ux.UxDataArray:
        """Convert to uxarray UxDataset."""
        return ux.UxDataArray.from_xarray(
            da=self._variables_with_spatial_coords(),
            uxgrid=self.uxgrid
        )

    @overload
    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180.0,
        lon_east: float = 180.0,
        lat_south: float = -90.0,
        lat_north: float = 90.0,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_da: Literal[False]
    ) -> ux.UxDataArray:
        ...

    @overload
    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_da: Literal[True]
    ) -> Tuple[ux.UxDataArray, xr.DataArray]:
        ...

    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_da: bool = False
    ) -> Union[ux.UxDataArray, Tuple[ux.UxDataArray, xr.DataArray]]:
        """
        Convert to structured grid using specified remapping method.

        Args:
            remap_method (str): The remapping method to use. Options are
                "nearest_neighbor", "bilinear", "inverse_distance_weighted".
            remap_to (str): The location to remap to. Options are "faces",
                "edges", "nodes".
            lon_west (float): The western longitude boundary of the structured grid.
            lon_east (float): The eastern longitude boundary of the structured grid.
            lat_south (float): The southern latitude boundary of the structured grid.
            lat_north (float): The northern latitude boundary of the structured grid.
            east_west_resolution (float): The resolution in the east-west direction.
            south_north_resolution (float): The resolution in the south-north direction.
            with_reshaped_da (bool): Whether to return the reshaped xarray DataArray
                along with the uxarray UxDataArray.

        Returns:
            Union[ux.UxDataArray, Tuple[ux.UxDataArray, xr.DataArray]]: The remapped uxarray UxDataArray,
            and optionally the reshaped xarray DataArray.

            if returns xarray DataArray, its structure will be:
                dims: (lat_{location}, lon_{location})
                coords: {
                    lon_{location}: (lat_{location}, lon_{location}),
                    lat_{location}: (lat_{location}, lon_{location})
                }
        """
        to_lat = np.arange(lat_south, lat_north, south_north_resolution)
        to_lon = np.arange(lon_west, lon_east, east_west_resolution)
        dst_grid = ux.Grid.from_structured(lon=to_lon, lat=to_lat)
        daux = self.to_uxarray()
        if remap_method not in ["nearest_neighbor", "bilinear", "inverse_distance_weighted"]:
            raise ValueError(
                f"remap_method must be one of ['nearest_neighbor', 'bilinear', 'inverse_distance_weighted'], got {remap_method}"
            )
        if remap_method == "nearest_neighbor":
            remap = daux.remap.nearest_neighbor
        elif remap_method == "bilinear":
            # Because remap bilinear method only support variables on cell centers
            # So we need to check if variables are not located on faces first
            if daux.attrs.get("location", "") != "face":
                raise ValueError(
                    "The UxDataArray location attribute must be 'face' for bilinear remapping."
                )
            remap = daux.remap.bilinear
        else:  # inverse_distance_weighted
            remap = daux.remap.inverse_distance_weighted
        uxda = cast(
            ux.UxDataArray,
            remap(destination_grid=dst_grid, remap_to=remap_to)
        )
        if not with_reshaped_da:
            return uxda
        if remap_to == "faces":
            location = "face"
        elif remap_to == "edges":
            location = "edge"
        elif remap_to == "nodes":
            location = "node"
        else:
            raise ValueError(
                f"remap_to must be one of ['faces', 'edges', 'nodes'], got {remap_to}"
            )
        lon_name = f"{location}_lon"
        lat_name = f"{location}_lat"
        lat_remapped = np.asarray(
            getattr(uxda.uxgrid, lat_name)
        ).reshape(len(to_lat), len(to_lon))
        lon_remapped = np.asarray(
            getattr(uxda.uxgrid, lon_name)
        ).reshape(len(to_lat), len(to_lon))
        dims = (f"{lat_name}", f"{lon_name}")
        if len(uxda.data.shape) == 1:
            var_data = np.asarray(uxda.data).reshape(
                len(to_lat), len(to_lon)
            )
        elif len(uxda.data.shape) == 2:
            var_data = np.asarray(uxda.data).reshape(
                uxda.data.shape[0], len(to_lat), len(to_lon)
            )
            dims = (uxda.dims[0], f"{lat_name}", f"{lon_name}")
        elif len(uxda.data.shape) == 3:
            var_data = np.asarray(uxda.data).reshape(
                uxda.data.shape[0],
                uxda.data.shape[1],
                len(to_lat), len(to_lon)
            )
            dims = (
                uxda.dims[0], uxda.dims[1],
                f"{lat_name}", f"{lon_name}"
            )
        else:
            raise ValueError(
                "Only support remapping UxDataArray with up to "
                "3 leading dimensions before lat/lon dimensions."
            )
        da = xr.DataArray(
            data=var_data,
            dims=dims,
            coords={
                lon_name: (
                    (f"{lat_name}", f"{lon_name}"),
                    lon_remapped
                ),
                lat_name: (
                    (f"{lat_name}", f"{lon_name}"),
                    lat_remapped
                )
            },
            attrs=daux.attrs
        )
        return cast(Tuple[ux.UxDataArray, xr.DataArray], (uxda, da))


class UGRIDDataset:
    variables: xr.Dataset
    uxgrid: ux.Grid

    def __init__(
        self,
        variables: xr.Dataset,
        uxgrid: ux.Grid
    ):
        """Initialize the UGRIDDataset."""
        self.variables = variables
        self.uxgrid = uxgrid

    @property
    def uxds(self) -> ux.UxDataset:
        """Get the uxarray UxDataset."""
        return self.to_uxarray()

    @property
    def data_vars(self) -> DataVariables:
        """Get the data variables of the dataset."""
        return self.variables.data_vars

    def _variables_with_spatial_coords(self) -> xr.Dataset:
        """Attach UGRID spatial coordinates to variables for UXarray remapping."""
        ds = self.variables
        coords: Dict[str, xr.DataArray] = {}
        for dim in ds.dims:
            if dim == "n_face":
                location = "face"
            elif dim == "n_node":
                location = "node"
            elif dim == "n_edge":
                location = "edge"
            else:
                continue
            for coord_name in (f"{location}_lon", f"{location}_lat"):
                coord = getattr(self.uxgrid, coord_name)
                coord_data = np.asarray(coord)
                if coord_data.shape[0] != ds.sizes[dim]:
                    continue
                attrs = dict(getattr(coord, "attrs", {}))
                if coord_name.endswith("_lon"):
                    attrs.setdefault("standard_name", "longitude")
                    attrs.setdefault("units", "degrees_east")
                else:
                    attrs.setdefault("standard_name", "latitude")
                    attrs.setdefault("units", "degrees_north")
                coords[coord_name] = xr.DataArray(
                    data=coord_data,
                    dims=(dim,),
                    attrs=attrs,
                )
        if coords:
            return ds.assign_coords(coords)
        return ds

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> "UGRIDDataset":
        """Index the UGRIDDataset."""
        new_vars = self.variables.isel(
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kwargs
        )
        return UGRIDDataset(
            variables=new_vars,
            uxgrid=self.uxgrid
        )

    def add(
        self,
        data: Array,
        dims: Tuple[Hashable, ...],
        long_name: str,
        units: str,
        location: str,
        **kwargs
    ) -> None:
        """Add variable to the UGRIDDataset."""
        new_var = xr.DataArray(
            data=np.asarray(data),
            dims=dims,
            attrs=dict(
                long_name=long_name,
                units=units,
                location=location,
                **kwargs
            )
        )
        self.variables = self.variables.assign({new_var.name: new_var})

    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        uxgrid: ux.Grid,
        attrs: Optional[Dict[str, Any]] = None
    ) -> "UGRIDDataset":
        """Create a UGRIDDataset from an xarray and UGRIDGrid."""
        variables: Dict[Hashable, xr.DataArray] = {}
        for var_name, var_data in ds.items():
            data = var_data.data
            dims = var_data.dims
            location = var_data.attrs.get("location", "")
            if location == "" or data is None or dims == ():
                raise ValueError(
                    f"Variable {var_name} must have 'location/data/dims' attribute to be added to UGRIDDataset."
                )
            variables[var_name] = xr.DataArray(
                data=np.asarray(data),
                dims=dims,
                attrs=var_data.attrs
            )
        return cls(variables=xr.Dataset(data_vars=variables, attrs=attrs), uxgrid=uxgrid)

    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset."""
        xrgrid = self.uxgrid.to_xarray()
        ds = self.variables
        ds = ds.merge(xrgrid)
        return ds

    def to_uxarray(self) -> ux.UxDataset:
        """Convert to uxarray UxDataset."""
        return ux.UxDataset.from_xarray(
            ds=self._variables_with_spatial_coords(),
            uxgrid=self.uxgrid
        )

    @overload
    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180.0,
        lon_east: float = 180.0,
        lat_south: float = -90.0,
        lat_north: float = 90.0,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_ds: Literal[False]
    ) -> ux.UxDataset:
        ...

    @overload
    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_ds: Literal[True]
    ) -> Tuple[ux.UxDataset, xr.Dataset]:
        ...

    def to_structured_grid(
        self,
        *,
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        with_reshaped_ds: bool = False
    ) -> Union[ux.UxDataset, Tuple[ux.UxDataset, xr.Dataset]]:
        """
        Convert to structured grid using specified remapping method.

        Args:
            remap_method (str): The remapping method to use. Options are
                "nearest_neighbor", "bilinear", "inverse_distance_weighted".
            remap_to (str): The location to remap to. Options are "faces",
                "edges", "nodes".
            lon_west (float): The western longitude boundary of the structured grid.
            lon_east (float): The eastern longitude boundary of the structured grid.
            lat_south (float): The southern latitude boundary of the structured grid.
            lat_north (float): The northern latitude boundary of the structured grid.
            east_west_resolution (float): The resolution in the east-west direction.
            south_north_resolution (float): The resolution in the south-north direction.
            with_reshaped_ds (bool): Whether to return the reshaped xarray Dataset
                along with the uxarray UxDataset.

        Returns:
            Union[ux.UxDataset, Tuple[ux.UxDataset, xr.Dataset]]: The remapped uxarray UxDataset,
            and optionally the reshaped xarray Dataset.
        """
        to_lat = np.arange(lat_south, lat_north, south_north_resolution)
        to_lon = np.arange(lon_west, lon_east, east_west_resolution)
        dst_grid = ux.Grid.from_structured(lon=to_lon, lat=to_lat)
        dsux = self.to_uxarray()
        if remap_method not in ["nearest_neighbor", "bilinear", "inverse_distance_weighted"]:
            raise ValueError(
                f"remap_method must be one of ['nearest_neighbor', 'bilinear', 'inverse_distance_weighted'], got {remap_method}"
            )
        if remap_method == "nearest_neighbor":
            remap = dsux.remap.nearest_neighbor
        elif remap_method == "bilinear":
            # Because remap bilinear method only support variables on cell centers
            # So we need to delete variables which are not located on faces first
            vars_dropped: list[Hashable] = []
            for var in dsux.data_vars:
                if dsux[var].attrs.get("location", "") != "face":
                    logger.warning(
                        f"Dropping variable {var} for bilinear remapping because its location is not 'face'."
                    )
                    vars_dropped.append(var)
            dsux = dsux.drop_vars(vars_dropped)
            remap = dsux.remap.bilinear
        else:  # inverse_distance_weighted
            remap = dsux.remap.inverse_distance_weighted
        uxds = cast(
            ux.UxDataset,
            remap(destination_grid=dst_grid, remap_to=remap_to)
        )
        if not with_reshaped_ds:
            return uxds
        vars_remapped: Dict[Hashable, xr.DataArray] = {}
        lat_remapped: Dict[str, np.ndarray] = {}
        lon_remapped: Dict[str, np.ndarray] = {}
        for var_name in uxds.data_vars:
            if remap_to == "faces":
                location = "face"
            elif remap_to == "edges":
                location = "edge"
            elif remap_to == "nodes":
                location = "node"
            else:
                raise ValueError(
                    f"remap_to must be one of ['faces', 'edges', 'nodes'], got {remap_to}"
                )
            lon_name = f"{location}_lon"
            lat_name = f"{location}_lat"
            if lat_name not in lat_remapped:
                lat = np.asarray(getattr(uxds.uxgrid, lat_name)).reshape(
                    len(to_lat), len(to_lon)
                )
                lat_remapped[lat_name] = lat
            if lon_name not in lon_remapped:
                lon = np.asarray(getattr(uxds.uxgrid, lon_name)).reshape(
                    len(to_lat), len(to_lon)
                )
                lon_remapped[lon_name] = lon
            dims = (f"{lat_name}", f"{lon_name}")
            # use the data_vars mapping to get the variable (avoid subscripting uxds directly)
            var = uxds.data_vars[var_name]
            if len(var.dims) == 1:
                var_data = np.asarray(var).reshape(
                    len(to_lat), len(to_lon)
                )
            elif len(var.dims) == 2:
                var_data = np.asarray(var).reshape(
                    var.shape[0], len(to_lat), len(to_lon)
                )
                dims = (var.dims[0], f"{lat_name}", f"{lon_name}")
            elif len(var.dims) == 3:
                var_data = np.asarray(var).reshape(
                    var.shape[0],
                    var.shape[1],
                    len(to_lat), len(to_lon)
                )
                dims = (
                    var.dims[0], var.dims[1],
                    f"{lat_name}", f"{lon_name}"
                )
            else:
                raise ValueError(
                    "Only support remapping UxDataset variables with up to "
                    "3 leading dimensions before lat/lon dimensions."
                )
            vars_remapped[var_name] = xr.DataArray(
                data=var_data,
                dims=dims,
                coords={
                    lon_name: (
                        (f"{lat_name}", f"{lon_name}"),
                        lon_remapped[lon_name]
                    ),
                    lat_name: (
                        (f"{lat_name}", f"{lon_name}"),
                        lat_remapped[lat_name]
                    )
                },
                attrs=dsux[var_name].attrs
            )
        ds = xr.Dataset(
            data_vars=vars_remapped,
            attrs=uxds.attrs
        )
        return cast(Tuple[ux.UxDataset, xr.Dataset], (uxds, ds))
