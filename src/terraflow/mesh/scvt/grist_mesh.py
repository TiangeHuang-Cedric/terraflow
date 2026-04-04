from typing import Dict, Callable, Optional
import numpy as np
import jax.numpy as jnp
import xarray as xr
import equinox as eqx
from pathlib import Path

from terraflow.util import typing, device, log
from terraflow.util.io import get_array, get_scalar, get_string, load_netcdf
import terraflow.util.math as math
from terraflow.mesh.scvt.base import (
    Point,
    AbstractRefMesh,
    RefPrimalCell,
    RefDualCell,
    RefPrimalEdge,
    RefDualEdge
)
from terraflow.mesh.scvt import geometry


IntType = typing.IntType
FloatType = typing.FloatType
Str = typing.Str
Array = typing.Array


class GristElementMeta:
    """
    Metadata for a Grist mesh element, including min, max, and mean values.
    """
    MIN: FloatType
    MAX: FloatType
    MEAN: FloatType
    PROPERTY: str

    def _get_meta(
        self,
        ds: xr.Dataset,
        kind: str,
        var_name: str,
        prop: str,
        float_type: type = np.float64
    ) -> float:
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

    def _from_file(
        self,
        ds: xr.Dataset,
        var_name: str,
        prop: str,
        float_type: type = np.float64
    ):
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
        self.MIN = self._get_meta(
            ds, "min", var_name, prop, float_type=float_type
        )
        self.MAX = self._get_meta(
            ds, "max", var_name, prop, float_type=float_type
        )
        self.MEAN = self._get_meta(
            ds, "mean", var_name, prop, float_type=float_type
        )

    def from_file(
        self, ds: xr.Dataset, var_name: str, float_type: type = np.float64
    ):
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
    nt: IntType
    ne: IntType
    nv: IntType
    glevel: IntType
    maxvnb: IntType
    n_dual: IntType
    kind: str
    optm: str
    pos: str
    name: str
    edp_dist: GristEdgeDist
    edt_dist: GristEdgeDist
    plg_area: GristCellArea
    tri_area: GristCellArea
    tri_angle: GristCellAngle

    def __init__(
        self,
        grist_0d_file: str,
        grist_2d_file: str,
        nt: IntType,
        ne: IntType,
        nv: IntType,
        glevel: IntType,
        maxvnb: IntType,
        n_dual: IntType,
        kind: str,
        optm: str,
        pos: str,
        name: str,
        edp_dist: GristEdgeDist,
        edt_dist: GristEdgeDist,
        plg_area: GristCellArea,
        tri_area: GristCellArea,
        tri_angle: GristCellAngle
    ):
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
    def from_file(
        cls,
        grist_0d_file: Str,
        grist_2d_file: Str,
        n_dual: IntType = 3,
        dev: device.AbstractDevice = device.CPU()
    ) -> "GristMeta":
        """Read the metadata of a Grist mesh from NetCDF files."""
        float_type = dev.float_type
        ds = load_netcdf(grist_0d_file)
        grist_0d_file = grist_0d_file
        grist_2d_file = grist_2d_file
        nt = get_scalar(ds, "mesh_nt")
        ne = get_scalar(ds, "mesh_ne")
        nv = get_scalar(ds, "mesh_nv")
        glevel = get_scalar(ds, "mesh_glevel")
        maxvnb = get_scalar(ds, "mesh_maxvnb")
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
        return cls(
            grist_0d_file, grist_2d_file, nt, ne, nv, glevel, maxvnb, n_dual,
            kind, optm, pos, name, edp_dist, edt_dist, plg_area, tri_area, tri_angle
        )


def padding_with_value(
    input: Array,
    condition_fn: Callable[[Array], Array] = lambda x: x >= 0,
    pad_value: FloatType = -1.0,
    dtype: Optional[type] = None
) -> Array:
    """Pad the input array with a specified value based on a condition."""
    mask = condition_fn(input)
    if dtype is not None:
        pad_val_dtype = jnp.array(pad_value, dtype=dtype)
    else:
        pad_val_dtype = jnp.array(pad_value, dtype=input.dtype)
    return jnp.where(mask, input, pad_val_dtype)


class GristPrimalCell(RefPrimalCell):
    """
    Primal cell in the Grist mesh.
    Attributes:
        nnb (Array): Number of neighboring cells, shape (nCell,).
        maxnnb (Int): Maximum number of neighboring cells, scalar.
    """
    nnb: Array
    maxnnb: IntType

    def __init__(
        self,
        n_cell: IntType,
        ids: IntType,
        ide: IntType,
        ims: IntType,
        ime: IntType,
        v: Array,
        area: Array,
        kite_area: Array,
        nb: Array,
        ed: Array,
        nr: Array,
        cc: Point,
        nnb: Array,
        maxnnb: IntType
    ):
        """Initialize the primal cell."""
        super().__init__(
            n_cell=n_cell,
            ids=ids,
            ide=ide,
            ims=ims,
            ime=ime,
            v=v,
            area=area,
            kite_area=kite_area,
            nb=nb,
            ed=ed,
            nr=nr,
            cc=cc
        )
        self.nnb = nnb
        self.maxnnb = maxnnb

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.AbstractDevice = device.CPU()) -> "GristPrimalCell":
        """Initialize the primal cell from metadata."""
        arch = dev()
        float_type = dev.float_type
        int_type = dev.int_type
        ds = load_netcdf(meta.grist_2d_file)
        v = padding_with_value(
            jnp.array(
                get_array(ds, "vtx_tr").T - 1,
                dtype=int_type,
                device=arch
            )
        )
        cart: Array = jnp.array(
            get_array(ds, "vtx_p").T,
            dtype=float_type,
            device=arch
        )
        ltln: Dict[str, Array] = {
            "lat": jnp.array(
                get_array(ds, "vtx_ltln_nnb")[0, :].T,
                dtype=float_type,
                device=arch
            ),
            "lon": math.convert_longitude_range_in_radians(
                jnp.array(
                    get_array(ds, "vtx_ltln_nnb")[1, :].T,
                    dtype=float_type,
                    device=arch
                ),
                from_range=["-pi", "pi"],
                to_range=["0", "2*pi"]
            )
        }
        cc = Point(cart=cart, ltln=ltln)
        nb = padding_with_value(
            jnp.array(
                get_array(ds, "vtx_nb").T - 1,
                dtype=int_type,
                device=arch
            )
        )
        nnb = jnp.array(
            get_array(ds, "vtx_ltln_nnb")[2, :].T,
            dtype=int_type,
            device=arch
        )
        maxnnb = int_type(meta.maxvnb)
        ed = padding_with_value(
            jnp.array(
                get_array(ds, "vtx_ed").T - 1,
                dtype=int_type,
                device=arch
            )
        )
        area = jnp.array(
            get_array(ds, "plg_area")[0, :].T,
            dtype=float_type,
            device=arch
        )
        kite_area = jnp.array(0.0, dtype=float_type, device=arch)
        nr = jnp.array(0, dtype=int_type, device=arch)
        return cls(
            n_cell=meta.nv,
            ids=0,
            ide=meta.nv-1,
            ims=0,
            ime=meta.nv-1,
            v=v,
            area=area,
            kite_area=kite_area,
            nb=nb,
            ed=ed,
            nr=nr,
            cc=cc,
            nnb=nnb,
            maxnnb=maxnnb
        )


class GristDualCell(RefDualCell):
    """
    Dual cell in the Grist mesh.
    Attributes:
        nnb (Int): Number of dual cells, usually 3 for triangles.
    """
    nnb: IntType

    def __init__(
        self,
        n_cell: IntType,
        ids: IntType,
        ide: IntType,
        ims: IntType,
        ime: IntType,
        v: Array,
        area: Array,
        kite_area: Array,
        nb: Array,
        ed: Array,
        nr: Array,
        cc: Point,
        nnb: IntType
    ):
        """Initialize the dual cell."""
        super().__init__(
            n_cell=n_cell,
            ids=ids,
            ide=ide,
            ims=ims,
            ime=ime,
            v=v,
            area=area,
            kite_area=kite_area,
            nb=nb,
            ed=ed,
            nr=nr,
            cc=cc
        )
        self.nnb = nnb

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.AbstractDevice = device.CPU()) -> "GristDualCell":
        """Initialize the dual cell from metadata."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(
            get_array(ds, "tri_v").T - 1,
            dtype=int_type,
            device=arch
        )
        cart = jnp.array(
            get_array(ds, "tri_cc_p").T,
            dtype=float_type,
            device=arch
        )
        ltln: Dict[str, Array] = {
            "lat": jnp.array(
                get_array(ds, "tri_cc_ltln")[0, :].T,
                dtype=float_type,
                device=arch
            ),
            "lon": math.convert_longitude_range_in_radians(
                jnp.array(
                    get_array(ds, "tri_cc_ltln")[1, :].T,
                    dtype=float_type,
                    device=arch
                ),
                from_range=["-pi", "pi"],
                to_range=["0", "2*pi"]
            )
        }
        cc = Point(cart=cart, ltln=ltln)
        nb = jnp.array(
            get_array(ds, "tri_nb").T - 1,
            dtype=int_type,
            device=arch
        )
        nnb = int_type(meta.n_dual)
        ed = padding_with_value(
            jnp.array(
                get_array(ds, "tri_ed").T - 1,
                dtype=int_type,
                device=arch
            )
        )
        area = jnp.array(
            get_array(ds, "tri_area")[0, :].T,
            dtype=float_type,
            device=arch
        )
        kite_area = jnp.array(0.0, dtype=float_type, device=arch)
        nr = jnp.array(0, dtype=int_type, device=arch)
        return cls(
            n_cell=meta.nt,
            ids=0,
            ide=meta.nt-1,
            ims=0,
            ime=meta.nt-1,
            v=v,
            area=area,
            kite_area=kite_area,
            nb=nb,
            ed=ed,
            nr=nr,
            cc=cc,
            nnb=nnb
        )


class GristPrimalEdge(RefPrimalEdge):
    """Primal edge in the Grist mesh."""

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.AbstractDevice = device.CPU()) -> "GristPrimalEdge":
        """Initialize primal edge data from Grist Meta."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(
            get_array(ds, "edp_v").T - 1,
            dtype=int_type,
            device=arch
        )
        cart: Array = jnp.array(
            get_array(ds, "edp_cc_p").T,
            dtype=float_type,
            device=arch
        )
        ltln: Dict[str, Array] = {
            "lat": jnp.array(
                get_array(ds, "edp_cc_ltln")[0, :].T,
                dtype=float_type,
                device=arch
            ),
            "lon": math.convert_longitude_range_in_radians(
                jnp.array(
                    get_array(ds, "edp_cc_ltln")[1, :].T,
                    dtype=float_type,
                    device=arch
                ),
                from_range=["-pi", "pi"],
                to_range=["0", "2*pi"]
            )
        }
        mp = Point(cart=cart, ltln=ltln)
        nr = jnp.array(
            get_array(ds, "edp_nr").T,
            dtype=float_type,
            device=arch
        )
        tg = jnp.array(
            get_array(ds, "edp_tg").T,
            dtype=float_type,
            device=arch
        )
        length = jnp.array(
            get_array(ds, "edp_len")[0, :].T,
            dtype=float_type,
            device=arch
        )
        return cls(
            n_edge=meta.ne,
            ids=0,
            ide=meta.ne-1,
            ims=0,
            ime=meta.ne-1,
            v=v,
            length=length,
            nr=nr,
            tg=tg,
            mp=mp
        )


class GristDualEdge(RefDualEdge):
    """Dual edge in the Grist mesh."""

    @classmethod
    def from_meta(cls, meta: GristMeta, dev: device.AbstractDevice = device.CPU()) -> "GristDualEdge":
        """Initialize dual edge data from Grist Meta."""
        arch = dev()
        int_type = dev.int_type
        float_type = dev.float_type
        ds = load_netcdf(meta.grist_2d_file)
        v = jnp.array(
            get_array(ds, "edt_v").T - 1,
            dtype=int_type,
            device=arch
        )
        cart: Array = jnp.array(
            get_array(ds, "edt_cc_p").T,
            dtype=float_type,
            device=arch
        )
        ltln: Dict[str, Array] = {
            "lat": jnp.array(
                get_array(ds, "edt_cc_ltln")[0, :].T,
                dtype=float_type,
                device=arch
            ),
            "lon": math.convert_longitude_range_in_radians(
                jnp.array(
                    get_array(ds, "edt_cc_ltln")[1, :].T,
                    dtype=float_type,
                    device=arch
                ),
                from_range=["-pi", "pi"],
                to_range=["0", "2*pi"]
            )
        }
        mp = Point(cart=cart, ltln=ltln)
        nr = jnp.array(
            get_array(ds, "edt_nr").T,
            dtype=float_type,
            device=arch
        )
        tg = jnp.array(
            get_array(ds, "edt_tg").T,
            dtype=float_type,
            device=arch
        )
        length = jnp.array(
            get_array(ds, "edt_len")[0, :].T,
            dtype=float_type,
            device=arch
        )
        return cls(
            n_edge=meta.ne,
            ids=0,
            ide=meta.ne-1,
            ims=0,
            ime=meta.ne-1,
            v=v,
            length=length,
            nr=nr,
            tg=tg,
            mp=mp
        )


class GristRefMesh(eqx.Module, AbstractRefMesh):
    """
    Grist reference mesh class, containing metadata, primal and dual cells and
    edges.
    Attributes:
        meta (GristMeta): Metadata of the Grist mesh.
        plg (GristPrimalCell): Primal cells of the Grist mesh.
        dua (GristDualCell): Dual cells of the Grist mesh.
        edp (GristPrimalEdge): Primal edges of the Grist mesh.
        edd (GristDualEdge): Dual edges of the Grist mesh.
    Methods:
        from_file(grist_0d_file, grist_2d_file, nDual, dev):
            Load the Grist mesh from the specified files.
    """
    meta: GristMeta
    plg: GristPrimalCell
    dua: GristDualCell
    edp: GristPrimalEdge
    edd: GristDualEdge

    @property
    def shape(self) -> dict:
        """Return the shape of the mesh as a dictionary."""
        return self.meta.shape

    @classmethod
    def from_netcdf(
        cls,
        grist_0d_file: Str,
        grist_2d_file: Str,
        n_dual: IntType = 3,
        dev: device.AbstractDevice = device.CPU()
    ) -> "GristRefMesh":
        """Load the Grist mesh from the specified NetCDF files."""
        meta = GristMeta.from_file(
            grist_0d_file, grist_2d_file, n_dual=n_dual, dev=dev
        )
        plg = GristPrimalCell.from_meta(meta, dev=dev)
        dua = GristDualCell.from_meta(meta, dev=dev)
        edp = GristPrimalEdge.from_meta(meta, dev=dev)
        edd = GristDualEdge.from_meta(meta, dev=dev)
        # Use eqx.tree_at to update immutable fields
        plg = eqx.tree_at(
            lambda x: (x.kite_area, x.nr),
            plg,
            (geometry.spherical_plg_kite_areas(plg, edp, dua),
             geometry.correct_normal_vectors(plg, edp))
        )
        dua = eqx.tree_at(
            lambda x: (x.kite_area, x.nr),
            dua,
            (geometry.spherical_tri_kite_areas(dua, edd, plg),
             geometry.correct_normal_vectors(dua, edd))
        )
        log.logger.info(
            "Loading GRIST mesh from: "
        )
        log.logger.info(
            f"    └─grist 0d file: {Path(grist_0d_file).absolute()}"
        )
        log.logger.info(
            f"    └─grist 2d file: {Path(grist_2d_file).absolute()}."
        )
        log.logger.info(
            "GRIST Mesh info: "
        )
        log.logger.info(
            f"    └─nv={meta.nv} nt={meta.nt} ne={meta.ne}"
        )
        log.logger.info(
            f"    └─maxvnb={meta.maxvnb} n_dual={meta.n_dual} glevel={meta.glevel}"
        )
        return cls(meta, plg, dua, edp, edd)
