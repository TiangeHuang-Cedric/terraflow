from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

from terraflow.util import typing, device
from terraflow.mesh.orthogonal_dual_mesh.base import Point
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimeCell, RefDualCell
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimeEdge, RefDualEdge
from terraflow.mesh.orthogonal_dual_mesh.grist_mesh import GristRefMesh

Array = typing.Array
Int = typing.Int
Float = typing.Float


class PrimeCell(RefPrimeCell):
    """A cell in the primal mesh."""
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
    def from_grist(cls, grist: GristRefMesh):
        """Initialize a PrimeCell from a GristRefMesh object."""
        n_cell = grist.shape["nv"]
        v = grist.plg.vertex()
        cc = Point(cart=grist.plg.centroid().cartesian(),
                   ltln=grist.plg.centroid().latlon())
        nb = grist.plg.nb
        ed = grist.plg.ed
        area = grist.plg.area
        kite_area = grist.plg.kite_area
        nr = grist.plg.nr
        nnb = grist.plg.nnb
        maxnnb = grist.meta.maxvnb
        return cls(n_cell, area, kite_area, nb, ed, nr, v, cc, nnb, maxnnb)


class DualCell(RefDualCell):
    """A cell in the dual mesh."""
    nnb: Int

    def __init__(self, n_cell: Int, area: Array, kite_area: Array,
                 nb: Array, ed: Array, nr: Array, v: Array, cc: Point, nnb: Int):
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
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the DualCell from a GristRefMesh object."""
        n_cell = grist.shape["nt"]
        v = grist.dua.vertex()
        cc = Point(cart=grist.dua.centroid().cartesian(),
                   ltln=grist.dua.centroid().latlon())
        nnb = grist.meta.n_dual
        nb = grist.dua.nb
        ed = grist.dua.ed
        area = grist.dua.area
        kite_area = grist.dua.kite_area
        nr = grist.dua.nr
        return cls(n_cell, area, kite_area, nb, ed, nr, v, cc, nnb)


class PrimeEdge(RefPrimeEdge):
    """An edge in the primal mesh."""

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the PrimeEdge from a GristRefMesh object."""
        n_edge = grist.shape["ne"]
        v = grist.edp.vertex()
        cc = Point(cart=grist.edp.centroid().cartesian(),
                   ltln=grist.edp.centroid().latlon())
        length = grist.edp.length
        nr = grist.edp.nr
        tg = grist.edp.tg
        return cls(n_edge, length, nr, tg, v, cc)


class DualEdge(RefDualEdge):
    """An edge in the dual mesh."""

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the DualEdge from a GristRefMesh object."""
        n_edge = grist.shape["ne"]
        v = grist.edd.vertex()
        cc = Point(cart=grist.edd.centroid().cartesian(),
                   ltln=grist.edd.centroid().latlon())
        length = grist.edd.length
        nr = grist.edd.nr
        tg = grist.edd.tg
        return cls(n_edge, length, nr, tg, v, cc)


@partial(jax.jit, static_argnames=("edp", "edd"))
def calc_edptg_eddnr(edp: PrimeEdge, edd: DualEdge) -> Array:
    """
    Return the sign between edp tangent and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    return jnp.sign(jnp.sum(edp.tg * edd.nr, axis=-1)).astype(edp.v.dtype)


@partial(jax.jit, static_argnames=("edp", "edd"))
def calc_edpnr_eddtg(edp: PrimeEdge, edd: DualEdge) -> Array:
    """
    Return the sign between edp normal and edd tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    return jnp.sign(jnp.sum(edp.nr * edd.tg, axis=-1)).astype(edp.v.dtype)


@partial(jax.jit, static_argnames=("edp", "dua"))
def calc_edp12_edptg(edp: PrimeEdge, dua: DualCell) -> Array:
    """
    Return the sign between edp1 to edp2 and edp tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    edp1 = edp.vertex()[:, 0]  # (n_edge, )
    edp2 = edp.vertex()[:, 1]  # (n_edge, )
    dua_cc = dua.centroid().cartesian()  # (n_dual, 3)
    edp12 = dua_cc[edp2] - dua_cc[edp1]  # (n_edge, 3)
    return jnp.sign(jnp.sum(edp12 * edp.tg, axis=-1)).astype(edp.v.dtype)


@partial(jax.jit, static_argnames=("edp", "edd", "dua"))
def calc_edp12_eddnr(edp: PrimeEdge, edd: DualEdge, dua: DualCell) -> Array:
    """
    Return the sign between edp1 to edp2 and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    edp1 = edp.vertex()[:, 0]  # (n_edge, )
    edp2 = edp.vertex()[:, 1]  # (n_edge, )
    dua_cc = dua.centroid().cartesian()  # (n_polg, 3)
    edp12 = dua_cc[edp2] - dua_cc[edp1]  # (n_edge, 3)
    return jnp.sign(jnp.sum(edp12 * edd.nr, axis=-1)).astype(edp.v.dtype)


class AuxiliaryData:
    """Auxiliary data for the orthogonal dual mesh."""
    edptg_eddnr: Array
    edpnr_eddtg: Array
    edp12_edptg: Array
    edp12_eddnr: Array

    def __init__(self, edptg_eddnr: Array, edpnr_eddtg: Array,
                 edp12_edptg: Array, edp12_eddnr: Array):
        self.edptg_eddnr = edptg_eddnr
        self.edpnr_eddtg = edpnr_eddtg
        self.edp12_edptg = edp12_edptg
        self.edp12_eddnr = edp12_eddnr

    @classmethod
    def from_elements(cls, plg: PrimeCell, dua: DualCell,
                      edp: PrimeEdge, edd: DualEdge) -> "AuxiliaryData":
        """Return auxiliary data for the mesh."""

        def check_auxiliary_data(aux: "AuxiliaryData"):
            """Check the validity of the auxiliary data."""
            if not jnp.all(jnp.abs(edptg_eddnr) == 1):
                raise ValueError("Invalid edptg_eddnr values; must be ±1.")
            if not jnp.all(jnp.abs(edpnr_eddtg) == 1):
                raise ValueError("Invalid edpnr_eddtg values; must be ±1.")
            if not jnp.all(jnp.abs(edp12_edptg) == 1):
                raise ValueError("Invalid edp12_edptg values; must be ±1.")
            if not jnp.all(jnp.abs(edp12_eddnr) == 1):
                raise ValueError("Invalid edp12_eddnr values; must be ±1.")

        edptg_eddnr = calc_edptg_eddnr(edp, edd)
        edpnr_eddtg = calc_edpnr_eddtg(edp, edd)
        edp12_edptg = calc_edp12_edptg(edp, dua)
        edp12_eddnr = calc_edp12_eddnr(edp, edd, dua)
        aux = cls(edptg_eddnr, edpnr_eddtg, edp12_edptg, edp12_eddnr)
        check_auxiliary_data(aux)
        return aux


@dataclass(frozen=True)
class OrthogonalDualMesh:
    """An orthogonal dual mesh consisting of primal and dual cells and edges."""
    n_polg: Int
    n_dual: Int
    n_edge: Int
    plg: PrimeCell
    dua: DualCell
    edp: PrimeEdge
    edd: DualEdge
    aux: AuxiliaryData

    @property
    def shape(self) -> dict:
        """Return the shape of the mesh as (n_polg, n_dual, n_edge)."""
        return dict(n_polg=self.n_polg, n_dual=self.n_dual, n_edge=self.n_edge)

    @classmethod
    def from_grist(
        cls, grisd_0d_file: str,
        grist_2d_file: str,
        n_dual: Int = 3,
        dev: device.Device = device.CPU()
    ) -> "OrthogonalDualMesh":
        """Initialize the OrthogonalDualMesh from Grist NetCDF Files."""
        grist = GristRefMesh.from_netcdf(
            grisd_0d_file, grist_2d_file, n_dual=n_dual, dev=dev)
        n_polg = grist.shape["nv"]
        n_dual = grist.shape["nt"]
        n_edge = grist.shape["ne"]
        plg = PrimeCell.from_grist(grist)
        dua = DualCell.from_grist(grist)
        edp = PrimeEdge.from_grist(grist)
        edd = DualEdge.from_grist(grist)
        aux = AuxiliaryData.from_elements(plg, dua, edp, edd)
        return cls(n_polg, n_dual, n_edge, plg, dua, edp, edd, aux)
