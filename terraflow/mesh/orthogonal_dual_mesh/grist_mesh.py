from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import xarray as xr

from terraflow.mesh.orthogonal_dual_mesh.base import Point, RefMesh
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimeCell, RefDualCell
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimeEdge, RefDualEdge
from terraflow.mesh.orthogonal_dual_mesh import geometry
from terraflow.util import typing, device
from terraflow.util.io import get_array, get_scalar, get_string, load_netcdf

Int = typing.Int
Float = typing.Float
Str = typing.Str
Array = typing.Array


class GristElementMeta:
    """
    Metadata for a Grist mesh element, including min, max, and mean values.
    """
    min: Float
    max: Float
    mean: Float
    PROPERTY: str

    def _get_meta(self, ds: xr.Dataset, kind: str, var_name: str,
                  prop: str, float_type: type = np.float64) -> float:
        """
        Helper function to get metadata from the dataset.
        Args:
            ds: xarray.Dataset, the dataset loaded from a NetCDF file.
            kind: str, the kind of statistic, e.g., "min", "max", "mean".
            var_name: str, the variable name, e.g., "edt", "edp", "tri", "plg".
            prop: str, the prop name, e.g., "dist", "area", "angle".
            float_type: type, the float type to use, default is np.float64.
        Returns:
            The requested metadata value as a float.
        """
        if kind != "mean" and kind != "min" and kind != "max":
            raise ValueError("kind must be 'mean', 'min' or 'max'")
        string = "mesh_" + kind + "_" + var_name + "_" + prop
        return float_type(get_scalar(ds, string))

    @property
    def property(self) -> str:
        """Property name, to be defined in subclasses."""
        return self.PROPERTY

    def _from_file(self, ds: xr.Dataset, var_name: str, prop: str,
                   float_type: type = np.float64):
        """
        Read the metadata of an element from a NetCDF file.
        Args:
            ds: xarray.Dataset, the dataset loaded from a NetCDF file.
            var_name: str, the variable name, e.g., "edt", "edp", "tri", "plg".
            property: str, the property name, e.g., "dist", "area", "angle".
            float_type: type, the float type to use, default is np.float64.
        Returns:
            A tuple of 3 numbers, the metadata of the element.,
            e.g. (min, max, mean).
        """
        self.min = self._get_meta(
            ds, "min", var_name, prop, float_type=float_type)
        self.max = self._get_meta(
            ds, "max", var_name, prop, float_type=float_type)
        self.mean = self._get_meta(
            ds, "mean", var_name, prop, float_type=float_type)

    def from_file(self, ds: xr.Dataset, var_name: str,
                  float_type: type = np.float64):
        """Read metadata from a NetCDF file."""
        return self._from_file(ds, var_name, self.property, float_type)


class GristEdgeDist(GristElementMeta):
    """Metadata for edge distance in a Grist mesh."""
    PROPERTY = "dist"


class GristCellArea(GristElementMeta):
    """Metadata for cell area in a Grist mesh."""
    PROPERTY = "area"


class GristCellAngle(GristElementMeta):
    """Metadata for cell angle in a Grist mesh."""
    PROPERTY = "angle"


class GristMeta:
    """Metadata for a Grist mesh."""
    grist_0d_file: str
    grist_2d_file: str
    nt: Int
    ne: Int
    nv: Int
    glevel: Int
    maxvnb: Int
    n_dual: Int
    kind: str
    optm: str
    pos: str
    name: str
    edp_dist: GristEdgeDist
    edt_dist: GristEdgeDist
    plg_area: GristCellArea
    tri_area: GristCellArea
    tri_angle: GristCellAngle

    def __init__(self, grist_0d_file: str, grist_2d_file: str,
                 nt: Int, ne: Int, nv: Int, glevel: Int, maxvnb: Int, n_dual: Int,
                 kind: str, optm: str, pos: str, name: str,
                 edp_dist: GristEdgeDist, edt_dist: GristEdgeDist,
                 plg_area: GristCellArea, tri_area: GristCellArea,
                 tri_angle: GristCellAngle):
        """Initialize the metadata for a Grist mesh."""
        self.grist_0d_file = grist_0d_file
        self.grist_2d_file = grist_2d_file
        self.nt = nt
        self.ne = ne
        self.nv = nv
        self.glevel = glevel
        self.maxvnb = maxvnb
        self.n_dual = n_dual
        self.kind = kind
        self.optm = optm
        self.pos = pos
        self.name = name
        self.edp_dist = edp_dist
        self.edt_dist = edt_dist
        self.plg_area = plg_area
        self.tri_area = tri_area
        self.tri_angle = tri_angle

    @property
    def shape(self) -> dict:
        """Return the shape of the mesh as a dictionary."""
        return dict(nv=self.nv, nt=self.nt, ne=self.ne)

    @classmethod
    def from_file(cls, grist_0d_file: Str, grist_2d_file: Str, n_dual: Int = 3,
                  dev: device.Device = device.CPU()):
        """Read the metadata of a Grist mesh from NetCDF files."""
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(grist_0d_file)
        grist_0d_file = grist_0d_file
        grist_2d_file = grist_2d_file
        nt = int_type(get_scalar(ds, "mesh_nt"))
        ne = int_type(get_scalar(ds, "mesh_ne"))
        nv = int_type(get_scalar(ds, "mesh_nv"))
        glevel = int_type(get_scalar(ds, "mesh_glevel"))
        maxvnb = int_type(get_scalar(ds, "mesh_maxvnb"))
        n_dual = int_type(n_dual)
        kind = get_string(ds, "mesh_kind")
        optm = get_string(ds, "mesh_optm")
        pos = get_string(ds, "mesh_pos")
        name = get_string(ds, "mesh_name")
        edt_dist = GristEdgeDist()
        edp_dist = GristEdgeDist()
        plg_area = GristCellArea()
        tri_area = GristCellArea()
        tri_angle = GristCellAngle()
        edp_dist.from_file(ds, "edp", float_type=float_type)
        edt_dist.from_file(ds, "edt", float_type=float_type)
        plg_area.from_file(ds, "plg", float_type=float_type)
        tri_area.from_file(ds, "tri", float_type=float_type)
        tri_angle.from_file(ds, "tri", float_type=float_type)
        return cls(grist_0d_file, grist_2d_file, nt, ne, nv, glevel, maxvnb, n_dual,
                   kind, optm, pos, name, edp_dist, edt_dist, plg_area, tri_area, tri_angle)


class GristPrimeCell(RefPrimeCell):
    """
    Primal cell in the Grist mesh.
    Attributes:
        nnb (Array): Number of neighboring cells, shape (nCell,).
        maxnnb (Int): Maximum number of neighboring cells, scalar.
    """
    nnb: Array
    maxnnb: Int

    def __init__(self, n_cell: Int, area: Array, kite_area: Array,
                 nb: Array, ed: Array, nr: Array, v: Array, cc: Point,
                 nnb: Array, maxnnb: Int):
        """
        Initialize a PrimeCell with the given parameters.
        Args:
            n_cell (Int): Number of cells.
            area (Array): Area of the cells.
            kite_area (Array): Kite areas of the cells.
            nb (Array): Neighbors of the cells.
            ed (Array): Edges of the cells.
            nr (Array): Normal vectors of the edges.
            v (Array): Vertices of the cells.
            cc (Point): Centroid of the cells.
            nnb (Array): Number of neighbors for each cell.
            maxnnb (Int): Maximum number of neighbors for any cell.
        """
        RefPrimeCell.__init__(self, n_cell, area, kite_area, nb, ed, nr, v, cc)
        self.nnb = nnb
        self.maxnnb = maxnnb

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.Device = device.CPU()):
        """Initialize the primal cell from metadata."""
        arch = dev()
        float_type = dev.float_type
        int_type = dev.int_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(get_array(ds, "vtx_tr").T - 1,
                      dtype=int_type, device=arch)
        cc = Point(
            cart=jnp.array(get_array(ds, "vtx_p").T,
                           dtype=float_type, device=arch),
            ltln=jnp.array(get_array(ds, "vtx_ltln_nnb")[0:2, :].T,
                           dtype=float_type, device=arch))
        nb = jnp.array(get_array(ds, "vtx_nb").T - 1,
                       dtype=int_type, device=arch)
        nnb = jnp.array(get_array(ds, "vtx_ltln_nnb")[2, :].T,
                        dtype=int_type, device=arch)
        maxnnb = int_type(meta.maxvnb)
        ed = jnp.array(get_array(ds, "vtx_ed").T - 1,
                       dtype=int_type, device=arch)
        area = jnp.array(get_array(ds, "plg_area")[0, :].T,
                         dtype=float_type, device=arch)
        kite_area = jnp.array(0.0, dtype=float_type, device=arch)
        nr = jnp.array(0, dtype=int_type, device=arch)
        return cls(meta.nv, area, kite_area, nb, ed, nr, v, cc, nnb, maxnnb)


class GristDualCell(RefDualCell):
    """
    Dual cell in the Grist mesh.
    Attributes:
        nnb (Int): Number of dual cells, usually 3 for triangles.
    """
    nnb: Int

    def __init__(self, n_cell: Int, area: Array, kite_area: Array,
                 nb: Array, ed: Array, nr: Array, v: Array, cc: Point,
                 nnb: Int):
        """
        Initialize a DualCell with the given parameters.
        Args:
            n_cell (Int): Number of cells.
            area (Array): Area of the cells.
            kite_area (Array): Kite areas of the cells.
            nb (Array): Neighbors of the cells.
            ed (Array): Edges of the cells.
            nr (Array): Normal vectors of the edges.
            v (Array): Vertices of the cells.
            cc (Point): Centroid of the cells.
            nnb (Int): Number of neighbors for each cell.
        """
        RefDualCell.__init__(self, n_cell, area, kite_area, nb, ed, nr, v, cc)
        self.nnb = nnb

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.Device = device.CPU()):
        """Initialize the dual cell from metadata."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(get_array(ds, "tri_v").T - 1,
                      dtype=int_type, device=arch)
        cc = Point(cart=jnp.array(get_array(ds, "tri_cc_p").T, dtype=float_type, device=arch),
                   ltln=jnp.array(get_array(ds, "tri_cc_ltln").T, dtype=float_type, device=arch))
        nb = jnp.array(get_array(ds, "tri_nb").T - 1,
                       dtype=int_type, device=arch)
        nnb = int_type(meta.n_dual)
        ed = jnp.array(get_array(ds, "tri_ed").T - 1,
                       dtype=int_type, device=arch)
        area = jnp.array(get_array(ds, "tri_area")[0, :].T,
                         dtype=float_type, device=arch)
        kite_area = jnp.array(0.0, dtype=float_type, device=arch)
        nr = jnp.array(0, dtype=int_type, device=arch)
        return cls(meta.nt, area, kite_area, nb, ed, nr, v, cc, nnb)


class GristPrimeEdge(RefPrimeEdge):
    """Primal edge in the Grist mesh."""

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.Device = device.CPU()):
        """Initialize primal edge data from Grist Meta."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(get_array(ds, "edp_v").T - 1,
                      dtype=int_type, device=arch)
        cc = Point(cart=jnp.array(get_array(ds, "edp_cc_p").T, dtype=float_type, device=arch),
                   ltln=jnp.array(get_array(ds, "edp_cc_ltln").T, dtype=float_type, device=arch))
        nr = jnp.array(get_array(ds, "edp_nr").T,
                       dtype=float_type, device=arch)
        tg = jnp.array(get_array(ds, "edp_tg").T,
                       dtype=float_type, device=arch)
        length = jnp.array(get_array(ds, "edp_len")[
                           0, :].T, dtype=float_type, device=arch)
        return cls(meta.ne, length, nr, tg, v, cc)


class GristDualEdge(RefDualEdge):
    """Dual edge in the Grist mesh."""

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.Device = device.CPU()):
        """Initialize dual edge data from Grist Meta."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(get_array(ds, "edt_v").T - 1,
                      dtype=int_type, device=arch)
        cc = Point(cart=jnp.array(get_array(ds, "edt_cc_p").T, dtype=float_type, device=arch),
                   ltln=jnp.array(get_array(ds, "edt_cc_ltln").T, dtype=float_type, device=arch))
        nr = jnp.array(get_array(ds, "edt_nr").T,
                       dtype=float_type, device=arch)
        tg = jnp.array(get_array(ds, "edt_tg").T,
                       dtype=float_type, device=arch)
        length = jnp.array(get_array(ds, "edt_len")[0, :].T,
                           dtype=float_type, device=arch)
        return cls(meta.ne, length, nr, tg, v, cc)


@dataclass(frozen=True)
class GristRefMesh(RefMesh):
    """
    Grist reference mesh class, containing metadata, primal and dual cells and
    edges.
    Attributes:
        meta (GristMeta): Metadata of the Grist mesh.
        plg (GristPrimeCell): Primal cells of the Grist mesh.
        dua (GristDualCell): Dual cells of the Grist mesh.
        edp (GristPrimeEdge): Primal edges of the Grist mesh.
        edd (GristDualEdge): Dual edges of the Grist mesh.
    Methods:
        from_file(grist_0d_file, grist_2d_file, nDual, dev):
            Load the Grist mesh from the specified files.
    """
    meta: GristMeta
    plg: GristPrimeCell
    dua: GristDualCell
    edp: GristPrimeEdge
    edd: GristDualEdge

    @property
    def shape(self) -> dict:
        """Return the shape of the mesh as a dictionary."""
        return self.meta.shape

    @classmethod
    def from_netcdf(cls, grist_0d_file: Str, grist_2d_file: Str, n_dual: Int = 3,
                    dev: device.Device = device.CPU()):
        """Load the Grist mesh from the specified NetCDF files."""
        meta = GristMeta.from_file(
            grist_0d_file, grist_2d_file, n_dual=n_dual, dev=dev)
        plg = GristPrimeCell.from_meta(meta, dev=dev)
        dua = GristDualCell.from_meta(meta, dev=dev)
        edp = GristPrimeEdge.from_meta(meta, dev=dev)
        edd = GristDualEdge.from_meta(meta, dev=dev)
        plg.kite_area = geometry.spherical_plg_kite_areas(plg, edp, dua)
        dua.kite_area = geometry.spherical_tri_kite_areas(dua, edd, plg)
        plg.nr = geometry.correct_normal_vectors(plg, edp)
        dua.nr = geometry.correct_normal_vectors(dua, edd)
        return cls(meta, plg, dua, edp, edd)
