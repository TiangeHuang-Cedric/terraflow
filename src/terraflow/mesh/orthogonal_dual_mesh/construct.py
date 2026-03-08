from __future__ import annotations
from re import I
from typing import (
    TYPE_CHECKING,
    Optional,
    Self,
    Mapping,
    List,
    Literal,
    Union,
    overload
)
import jax
import jax.numpy as jnp
import equinox as eqx
from loguru import logger

from terraflow.util import typing, device
from terraflow.util import math, history
from terraflow.util.uxarray import UGRIDGrid, AttributeConvention
from terraflow.mesh import AbstractUnstructuredMesh
from terraflow.mesh.orthogonal_dual_mesh.base import Point
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimalCell, RefDualCell
from terraflow.mesh.orthogonal_dual_mesh.base import RefPrimalEdge, RefDualEdge
from terraflow.mesh.orthogonal_dual_mesh.grist_mesh import GristRefMesh


if TYPE_CHECKING:
    from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
    import numpy as np


Float = typing.Float
Array = typing.Array
IntType = typing.IntType
FloatType = typing.FloatType


class PrimalCell(RefPrimalCell):
    """A cell in the primal mesh."""
    nnb: Array
    maxnnb: IntType

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize a PrimalCell from a GristRefMesh object."""
        n_cell = int(grist.shape["nv"])
        v = grist.plg.vertex
        cc = Point(
            cart=grist.plg.centroid.cartesian,
            ltln=grist.plg.centroid.latlon
        )
        nb = grist.plg.nb
        ed = grist.plg.ed
        area = grist.plg.area
        kite_area = grist.plg.kite_area
        nr = grist.plg.nr
        nnb = grist.plg.nnb
        maxnnb = int(grist.meta.maxvnb)
        return cls(v, n_cell, area, kite_area, nb, ed, nr, cc, nnb, maxnnb)


class DualCell(RefDualCell):
    """A cell in the dual mesh."""
    nnb: IntType

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the DualCell from a GristRefMesh object."""
        n_cell = int(grist.shape["nt"])
        v = grist.dua.vertex
        cc = Point(
            cart=grist.dua.centroid.cartesian,
            ltln=grist.dua.centroid.latlon
        )
        nnb = int(grist.meta.n_dual)
        nb = grist.dua.nb
        ed = grist.dua.ed
        area = grist.dua.area
        kite_area = grist.dua.kite_area
        nr = grist.dua.nr
        return cls(v, n_cell, area, kite_area, nb, ed, nr, cc, nnb)


class PrimalEdge(RefPrimalEdge):
    """An edge in the primal mesh."""

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the PrimalEdge from a GristRefMesh object."""
        n_edge = int(grist.shape["ne"])
        v = grist.edp.vertex
        cc = Point(
            cart=grist.edp.midpoint.cartesian,
            ltln=grist.edp.midpoint.latlon
        )
        length = grist.edp.length
        nr = grist.edp.nr
        tg = grist.edp.tg
        return cls(v, n_edge, length, nr, tg, cc)


class DualEdge(RefDualEdge):
    """An edge in the dual mesh."""

    @classmethod
    def from_grist(cls, grist: GristRefMesh):
        """Initialize the DualEdge from a GristRefMesh object."""
        n_edge = int(grist.shape["ne"])
        v = grist.edd.vertex
        cc = Point(
            cart=grist.edd.midpoint.cartesian,
            ltln=grist.edd.midpoint.latlon
        )
        length = grist.edd.length
        nr = grist.edd.nr
        tg = grist.edd.tg
        return cls(v, n_edge, length, nr, tg, cc)


@jax.jit
def calc_edptg_eddnr(edp: PrimalEdge, edd: DualEdge) -> Array:
    """
    Return the sign between edp tangent and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    return jnp.sign(jnp.sum(edp.tg * edd.nr, axis=-1)).astype(edp.v.dtype)


@jax.jit
def calc_edpnr_eddtg(edp: PrimalEdge, edd: DualEdge) -> Array:
    """
    Return the sign between edp normal and edd tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    return jnp.sign(jnp.sum(edp.nr * edd.tg, axis=-1)).astype(edp.v.dtype)


@jax.jit
def calc_edp12_edptg(edp: PrimalEdge, dua: DualCell) -> Array:
    """
    Return the sign between edp1 to edp2 and edp tangent vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    edp1 = edp.vertex[:, 0]  # (n_edge, )
    edp2 = edp.vertex[:, 1]  # (n_edge, )
    dua_cc = dua.centroid.cartesian  # (n_dual, 3)
    edp12 = dua_cc[edp2] - dua_cc[edp1]  # (n_edge, 3)
    return jnp.sign(jnp.sum(edp12 * edp.tg, axis=-1)).astype(edp.v.dtype)


@jax.jit
def calc_edp12_eddnr(edp: PrimalEdge, edd: DualEdge, dua: DualCell) -> Array:
    """
    Return the sign between edp1 to edp2 and edd normal vectors. If they are
    the same direction, return 1; if opposite, return -1.
    """
    edp1 = edp.vertex[:, 0]  # (n_edge, )
    edp2 = edp.vertex[:, 1]  # (n_edge, )
    dua_cc = dua.centroid.cartesian  # (n_polg, 3)
    edp12 = dua_cc[edp2] - dua_cc[edp1]  # (n_edge, 3)
    return jnp.sign(jnp.sum(edp12 * edd.nr, axis=-1)).astype(edp.v.dtype)


class AuxiliaryData(eqx.Module):
    """Auxiliary data for the orthogonal dual mesh."""
    edptg_eddnr: Array
    edpnr_eddtg: Array
    edp12_edptg: Array
    edp12_eddnr: Array

    @classmethod
    def from_elements(
        cls,
        plg: PrimalCell,
        dua: DualCell,
        edp: PrimalEdge,
        edd: DualEdge
    ) -> "AuxiliaryData":
        """Return auxiliary data for the mesh."""

        def check_auxiliary_data(aux: "AuxiliaryData"):
            """Check the validity of the auxiliary data."""
            if not jnp.all(jnp.abs(aux.edptg_eddnr) == 1):
                raise ValueError("Invalid edptg_eddnr values; must be ±1.")
            if not jnp.all(jnp.abs(aux.edpnr_eddtg) == 1):
                raise ValueError("Invalid edpnr_eddtg values; must be ±1.")
            if not jnp.all(jnp.abs(aux.edp12_edptg) == 1):
                raise ValueError("Invalid edp12_edptg values; must be ±1.")
            if not jnp.all(jnp.abs(aux.edp12_eddnr) == 1):
                raise ValueError("Invalid edp12_eddnr values; must be ±1.")

        edptg_eddnr = calc_edptg_eddnr(edp, edd)
        edpnr_eddtg = calc_edpnr_eddtg(edp, edd)
        edp12_edptg = calc_edp12_edptg(edp, dua)
        edp12_eddnr = calc_edp12_eddnr(edp, edd, dua)
        aux = cls(edptg_eddnr, edpnr_eddtg, edp12_edptg, edp12_eddnr)
        check_auxiliary_data(aux)
        return aux


class SingleCellEntityView:
    """
    Lightweight view for accessing a single cell without data copying.

    This class provides indexed access to cell properties while maintaining
    a reference to the original cell data. Uses __slots__ for memory efficiency.

    Attributes:
        _cell: Reference to the parent cell (PrimalCell or DualCell)
        _index: Index of this cell in the parent cell array
        _nnb: Number of valid neighbors for this cell
    """
    __slots__ = ('_cell', '_index', '_nnb')

    def __init__(self, cell: Union["PrimalCell", "DualCell"], index: int):
        """
        Initialize a view for a single cell.

        Args:
            cell: Parent cell object (PrimalCell or DualCell)
            index: Index of the cell to view
        """
        self._cell = cell
        self._index = index

        # Pre-compute the number of valid neighbors for this cell
        if isinstance(cell, PrimalCell):
            self._nnb = int(cell.nnb[index])
        else:
            self._nnb = int(cell.nnb)  # DualCell has fixed nnb

    def __repr__(self) -> str:
        """Return string representation of this cell view."""
        cell_type = "PrimalCell" if isinstance(
            self._cell, PrimalCell) else "DualCell"
        return f"SingleCellEntityView({cell_type}, index={self._index}, nnb={self._nnb})"

    @property
    def index(self) -> IntType:
        """Return the global index of this cell."""
        return self._index

    @property
    def adjacency(self) -> Array:
        """
        Return the neighbor cell indices (valid neighbors only).

        Returns:
            Array of shape (nnb,) containing neighboring cell indices
        """
        return self._cell.nb[self._index, :self._nnb]

    @property
    def n_adjacency(self) -> IntType:
        """
        Return the number of adjacent cells for this cell.

        Returns:
            Number of adjacent cells (nnb)
        """
        return self._nnb

    @property
    def centroid(self) -> Point:
        """
        Return the centroid coordinates of this cell.

        Returns:
            Point object containing Cartesian and lat-lon coordinates
        """
        return self._cell.centroid[self._index]

    @property
    def nodes(self) -> Array:
        """
        Return the node indices of this cell (valid neighbors only).

        Returns:
            Array of shape (nnb,) containing node indices
        """
        return self._cell.v[self._index, :self._nnb]

    @property
    def edges(self) -> Array:
        """
        Return the edge indices of this cell (valid neighbors only).

        Returns:
            Array of shape (nnb,) containing edge indices
        """
        return self._cell.ed[self._index, :self._nnb]

    @property
    def max_adjacency(self) -> IntType:
        """
        Return the maximum possible number of neighbors for any cell.

        Returns:
            Maximum neighbor count (PrimalCell: maxnnb, DualCell: fixed nnb)
        """
        if isinstance(self._cell, PrimalCell):
            return self._cell.maxnnb
        else:
            return self._cell.nnb

    @property
    def num_neighbors(self) -> int:
        """Return the number of valid neighbors for this cell."""
        return self._nnb

    @property
    def area(self) -> FloatType:
        """
        Return the area of this cell.

        Returns:
            Scalar area value
        """
        return float(self._cell.area[self._index])

    @property
    def kite_area(self) -> Array:
        """
        Return the kite sub-areas of this cell (valid neighbors only).

        Returns:
            Array of shape (nnb,) containing kite area values
        """
        return self._cell.kite_area[self._index, :self._nnb]

    @property
    def orient(self) -> Array:
        """
        Return the orientation signs for edges of this cell (valid neighbors only).

        Returns:
            Array of shape (nnb,) containing orientation signs (±1)
        """
        return self._cell.nr[self._index, :self._nnb]


class CellEntityView:
    """
    Lightweight view for accessing multiple cells without data copying.

    This class stores only the indices of selected cells and provides
    batch operations on the original cell data (zero-copy). Supports
    chained indexing and iteration.

    Attributes:
        _cell: Reference to the parent cell (PrimalCell or DualCell)
        _indices: JAX array of cell indices to view
    """
    __slots__ = ('_cell', '_indices')

    def __init__(self, cell: Union["PrimalCell", "DualCell"], indices: Array):
        """
        Initialize a view for multiple cells.

        Args:
            cell: Parent cell object (reference, not copied)
            indices: JAX array of cell indices to include in this view
        """
        self._cell = cell
        self._indices = indices

    @overload
    def __getitem__(self, index: int) -> SingleCellEntityView:
        """Access a single cell by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "CellEntityView":
        """Access multiple cells by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> "CellEntityView":
        """Access multiple cells by array of indices."""
        ...

    def __getitem__(self, index: Union[int, slice, Array]) -> Union["SingleCellEntityView", "CellEntityView"]:
        """
        Support chained indexing on the view.

        Args:
            index: Integer, slice, or array index

        Returns:
            SingleCellEntityView for int index, CellEntityView for slice/array

        Examples:
            >>> view = mesh.primal.cell[10:20]  # CellEntityView with 10 cells
            >>> cell = view[5]  # SingleCellEntityView for original cell 15
            >>> sub_view = view[2:5]  # CellEntityView for original cells 12-14
        """
        if isinstance(index, int):
            # Map to original cell index
            original_idx = int(self._indices[index])
            return SingleCellEntityView(self._cell, original_idx)
        elif isinstance(index, slice):
            # Slice the index array
            sub_indices = self._indices[index]
            return CellEntityView(self._cell, sub_indices)
        elif isinstance(index, (list, jnp.ndarray)):
            # Advanced indexing on the index array
            sub_indices = self._indices[jnp.asarray(index)]
            return CellEntityView(self._cell, sub_indices)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        """Return the number of cells in this view."""
        return len(self._indices)

    def __iter__(self):
        """
        Iterate over individual cell views.

        Yields:
            SingleCellEntityView for each cell in this view
        """
        # Convert to Python list once for efficiency
        for idx in self._indices.tolist():
            yield SingleCellEntityView(self._cell, idx)

    def __repr__(self) -> str:
        """Return string representation of this view."""
        cell_type = "PrimalCell" if isinstance(
            self._cell, PrimalCell) else "DualCell"
        return f"CellEntityView({cell_type}, n_cells={len(self)})"

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of this cell view."""
        return (len(self),)

    @property
    def indices(self) -> Array:
        """
        Return the indices of cells in this view.

        Returns:
            JAX array of shape (n_cells,) containing cell indices
        """
        return self._indices

    @overload
    def get_adjacency(self, with_padding: Literal[True] = True) -> Array:
        """Return adjacency matrix as fixed-size array with padding."""
        ...

    @overload
    def get_adjacency(self, with_padding: Literal[False]) -> List[Array]:
        """Return adjacency matrix as list of variable-length arrays."""
        ...

    def get_adjacency(self, with_padding: bool = True) -> Union[Array, List[Array]]:
        """
        Return adjacency matrix for cells in this view.

        Args:
            with_padding: If True (default), return fixed-size array with padding (-1).
                         If False, return list of variable-length arrays (one per cell).

        Returns:
            If with_padding=True: Array of shape (n_cells, max_neighbors) with padding
            If with_padding=False: List of n_cells arrays, each with shape (actual_neighbors,)
        """
        if with_padding:
            return self._cell.nb[self._indices]

        # Return list of variable-length arrays
        # Note: Due to variable output shapes, Python loop is unavoidable but optimized
        # JAX's array slicing (nb_data[i, :nnb]) is highly optimized at C/XLA level
        # Shape: (n_cells, max_neighbors)
        nb_data = self._cell.nb[self._indices]

        if isinstance(self._cell, PrimalCell):
            # Variable neighbor counts
            nnb_data = self._cell.nnb[self._indices]
            # Simple slicing is fastest: JAX optimizes this at the XLA level
            return [nb_data[i, :int(nnb_data[i])] for i in range(len(self._indices))]
        else:
            # DualCell has fixed nnb for all cells
            nnb = int(self._cell.nnb)
            return [nb_data[i, :nnb] for i in range(len(self._indices))]

    @property
    def adjacency(self) -> Array:
        """
        Return the adjacency matrix for cells in this view with padding (-1).

        Returns:
            Array of shape (n_cells, max_neighbors) with padding (-1)
        """
        return self.get_adjacency(with_padding=True)

    @property
    def n_adjacency(self) -> Array:
        """
        Return the number of adjacent cells for each cell in this view.

        Returns:
            Array of shape (n_cells,) containing the number of adjacent cells for each cell
        """
        if isinstance(self._cell, PrimalCell):
            return self._cell.nnb[self._indices]
        else:
            return jnp.ones(len(self), dtype=jnp.int32) * self._cell.nnb

    @property
    def max_adjacency(self) -> IntType:
        """
        Return the maximum possible number of neighbors.

        Returns:
            Maximum neighbor count
        """
        if isinstance(self._cell, PrimalCell):
            return self._cell.maxnnb
        else:
            return self._cell.nnb

    @property
    def centroid(self) -> Point:
        """
        Return centroids of all cells in this view.

        Returns:
            Point object with arrays of shape (n_cells, 3) for cart and (n_cells, 2) for ltln
        """
        return self._cell.centroid[self._indices]

    @property
    def area(self) -> Array:
        """
        Return areas of all cells in this view.

        Returns:
            Array of shape (n_cells,)
        """
        return self._cell.area[self._indices]

    @property
    def nodes(self) -> Array:
        """
        Return node arrays for cells in this view.

        Returns:
            Array of shape (n_cells, max_neighbors)
        """
        return self._cell.v[self._indices]

    @property
    def edges(self) -> Array:
        """
        Return edge arrays for cells in this view.

        Returns:
            Array of shape (n_cells, max_neighbors)
        """
        return self._cell.ed[self._indices]

    @property
    def kite_area(self) -> Array:
        """
        Return kite areas for cells in this view.

        Returns:
            Array of shape (n_cells, max_neighbors)
        """
        return self._cell.kite_area[self._indices]

    @property
    def orient(self) -> Array:
        """
        Return orientation signs for cells in this view.

        Returns:
            Array of shape (n_cells, max_neighbors)
        """
        return self._cell.nr[self._indices]


class CellEntity:
    """
    Entity accessor for cell collections supporting efficient indexing.

    This class provides the main interface for accessing cells in a mesh.
    It supports integer indexing (returns SingleCellEntityView), slice/array
    indexing (returns CellEntityView), and batch operations on all cells.

    Attributes:
        _cell: The underlying cell data (PrimalCell or DualCell)
    """

    def __init__(self, cell: Union["PrimalCell", "DualCell"]):
        """
        Initialize the cell entity accessor.

        Args:
            cell: PrimalCell or DualCell instance containing the data
        """
        self._cell = cell

    def __repr__(self) -> str:
        return f"CellEntity({self._cell})"

    @overload
    def __getitem__(self, index: int) -> SingleCellEntityView:
        """Access a single cell by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> CellEntityView:
        """Access multiple cells by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> CellEntityView:
        """Access multiple cells by array of indices."""
        ...

    def __getitem__(
        self,
        index: Union[int, slice, Array]
    ) -> Union[SingleCellEntityView, CellEntityView]:
        """
        Access cells by index with zero-copy views.

        Args:
            index: Integer (single cell), slice, or array of indices

        Returns:
            SingleCellEntityView for integer index (single cell access)
            CellEntityView for slice/array index (multi-cell access)

        Raises:
            IndexError: If integer index is out of bounds
            TypeError: If index type is not supported

        Examples:
            >>> cell = mesh.primal.cell[5]  # Single cell view
            >>> cells = mesh.primal.cell[10:20]  # View of 10 cells
            >>> cells = mesh.primal.cell[[1, 5, 9]]  # View of specific cells
        """
        if isinstance(index, int):
            # Single cell access: return lightweight view
            if index < 0 or index >= self._cell.n_cell:
                raise IndexError(
                    f"Cell index {index} out of range [0, {self._cell.n_cell})")
            return SingleCellEntityView(self._cell, index)

        elif isinstance(index, slice):
            # Slice access: return view with index array
            start, stop, step = index.indices(self._cell.n_cell)
            indices = jnp.arange(start, stop, step)
            return CellEntityView(self._cell, indices)

        elif isinstance(index, (list, jnp.ndarray)):
            # Array indexing: return view with index array
            indices = jnp.asarray(index)
            if indices.ndim != 1:
                raise ValueError("Index array must be 1-dimensional")
            return CellEntityView(self._cell, indices)

        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> IntType:
        """Return the number of cells in the cell entity."""
        return self._cell.n_cell

    def __iter__(self):
        """Iterate over the single cell entities."""
        for i in range(self._cell.n_cell):
            yield self[i]

    @property
    def shape(self) -> tuple[IntType]:
        """Return the shape of the cell entity."""
        return (self._cell.n_cell,)

    @overload
    def get_adjacency(self, with_padding: Literal[True] = True) -> Array:
        """Return adjacency matrix as fixed-size array with padding."""
        ...

    @overload
    def get_adjacency(self, with_padding: Literal[False]) -> List[Array]:
        """Return adjacency matrix as list of variable-length arrays."""
        ...

    def get_adjacency(self, with_padding: bool = True) -> Union[Array, List[Array]]:
        """
        Return the adjacency matrix of the cell entity.

        Args:
            with_padding: If True (default), return fixed-size array with padding (-1000001).
                         If False, return list of variable-length arrays (one per cell).

        Returns:
            If with_padding=True: Array of shape (n_cells, max_neighbors) with padding
            If with_padding=False: List of n_cells arrays, each with shape (actual_neighbors,)
        """
        if with_padding:
            return self._cell.nb

        # Return list of variable-length arrays using fully JIT-accelerated approach
        nb_data = self._cell.nb  # Shape: (n_cells, max_neighbors)

        if isinstance(self._cell, PrimalCell):
            nnb_data = self._cell.nnb  # Variable neighbor counts
            return [nb_data[i, :int(nnb_data[i])] for i in range(self._cell.n_cell)]
        else:
            # DualCell has fixed nnb for all cells
            nnb = int(self._cell.nnb)
            return [nb_data[i, :nnb] for i in range(self._cell.n_cell)]

    @property
    def adjacency(self) -> Array:
        """Return the adjacency matrix of the cell entity with padding (-1)."""
        return self.get_adjacency(with_padding=True)

    @property
    def n_adjacency(self) -> Array:
        """Return the number of adjacent cells for each cell in the entity."""
        if isinstance(self._cell, PrimalCell):
            return self._cell.nnb
        else:
            return jnp.ones(self._cell.n_cell, dtype=jnp.int32) * self._cell.nnb

    @property
    def max_adjacency(self) -> IntType:
        """Return the maximum adjacency number of the cell entity."""
        if isinstance(self._cell, PrimalCell):
            return self._cell.maxnnb
        else:
            return self._cell.nnb  # DualCell has fixed nnb

    @property
    def centroid(self) -> Point:
        """Return the centroid of the cell entity."""
        return self._cell.centroid

    @property
    def nodes(self) -> Array:
        """Return the nodes of the cell entity."""
        return self._cell.v

    @property
    def edges(self) -> Array:
        """Return the edges of the cell entity."""
        return self._cell.ed

    @property
    def area(self) -> Array:
        """Return the area of the cell entity."""
        return self._cell.area

    @property
    def kite_area(self) -> Array:
        """Return the kite area of the cell entity."""
        return self._cell.kite_area

    @property
    def orient(self) -> Array:
        """Return the orientation sign of the cell entity."""
        return self._cell.nr


class SingleEdgeEntityView:
    """
    Lightweight view for accessing a single edge without data copying.

    This class provides indexed access to edge properties while maintaining
    a reference to the original edge data. Uses __slots__ for memory efficiency.

    Attributes:
        _edge: Reference to the parent edge (PrimalEdge or DualEdge)
        _dual_edge: Reference to the dual edge (PrimalEdge or DualEdge)
        _index: Index of this edge in the parent edge array
    """
    __slots__ = ("_edge", "_dual_edge", "_index")

    def __init__(
        self,
        edge: Union["PrimalEdge", "DualEdge"],
        dual_edge: Union["PrimalEdge", "DualEdge"],
        index: int
    ):
        """
        Initialize a view for a single edge.

        Args:
            edge: Parent edge object (PrimalEdge or DualEdge)
            index: Index of the edge to view
        """
        self._edge = edge
        self._dual_edge = dual_edge
        self._index = index

    def __repr__(self) -> str:
        """Return string representation of this edge view."""
        edge_type = "PrimalEdge" if isinstance(
            self._edge, PrimalEdge
        ) else "DualEdge"
        return f"SingleEdgeEntityView({edge_type}, index={self._index})"

    @property
    def index(self) -> IntType:
        """Return the global index of this edge."""
        return self._index

    @property
    def midpoint(self) -> Point:
        """
        Return the midpoint coordinates of this edge.

        Returns:
            Point object containing Cartesian and lat-lon coordinates
        """
        return self._edge.midpoint[self._index]

    @property
    def endpoints(self) -> Array:
        """
        Return the node indices at the two endpoints of this edge.

        Returns:
            Array of shape (2,) containing endpoint node indices
        """
        return self._edge.v[self._index]

    @property
    def dual_cells(self) -> Array:
        """Return the dual cell adjacent to this edge.

        Returns:
            Array of shape (2,) containing adjacent face indices for this edge
        """
        return self._dual_edge.vertex[self._index]

    @property
    def length(self) -> FloatType:
        """
        Return the length of this edge.

        Returns:
            Scalar length value
        """
        return float(self._edge.length[self._index])

    @property
    def normal(self) -> Array:
        """
        Return the normal vector of this edge.

        Returns:
            Array of shape (3,) for 3D normal vector
        """
        return self._edge.nr[self._index]

    @property
    def tangent(self) -> Array:
        """
        Return the tangent vector of this edge.

        Returns:
            Array of shape (3,) for 3D tangent vector
        """
        return self._edge.tg[self._index]


class EdgeEntityView:
    """
    Lightweight view for accessing multiple edges without data copying.

    This class stores only the indices of selected edges and provides
    batch operations on the original edge data (zero-copy). Supports
    chained indexing and iteration.

    Attributes:
        _edge: Reference to the parent edge (PrimalEdge or DualEdge)
        _indices: JAX array of edge indices to view
    """
    __slots__ = ("_edge", "_dual_edge", "_indices")

    def __init__(
        self,
        edge: Union["PrimalEdge", "DualEdge"],
        dual_edge: Union["PrimalEdge", "DualEdge"],
        indices: Array
    ):
        """
        Initialize a view for multiple edges.

        Args:
            edge: Parent edge object (reference, not copied)
            indices: JAX array of edge indices to include in this view
        """
        self._edge = edge
        self._dual_edge = dual_edge
        self._indices = indices

    def __repr__(self) -> str:
        """Return string representation of this view."""
        edge_type = "PrimalEdge" if isinstance(
            self._edge, PrimalEdge) else "DualEdge"
        return f"EdgeEntityView({edge_type}, n_edges={len(self)})"

    @overload
    def __getitem__(self, index: int) -> SingleEdgeEntityView:
        """Access a single edge by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "EdgeEntityView":
        """Access multiple edges by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> "EdgeEntityView":
        """Access multiple edges by array of indices."""
        ...

    def __getitem__(self, index: Union[int, slice, Array]) -> Union["SingleEdgeEntityView", "EdgeEntityView"]:
        """
        Support chained indexing on the view.

        Args:
            index: Integer, slice, or array index

        Returns:
            SingleEdgeEntityView for int index, EdgeEntityView for slice/array

        Examples:
            >>> view = mesh.primal.edge[10:20]  # EdgeEntityView with 10 edges
            >>> edge = view[5]  # SingleEdgeEntityView for original edge 15
            >>> sub_view = view[2:5]  # EdgeEntityView for original edges 12-14
        """
        if isinstance(index, int):
            # Map to original edge index
            original_idx = int(self._indices[index])
            return SingleEdgeEntityView(self._edge, self._dual_edge, original_idx)
        elif isinstance(index, slice):
            # Slice the index array
            sub_indices = self._indices[index]
            return EdgeEntityView(self._edge, self._dual_edge, sub_indices)
        elif isinstance(index, (list, jnp.ndarray)):
            # Advanced indexing on the index array
            sub_indices = self._indices[jnp.asarray(index)]
            return EdgeEntityView(self._edge, self._dual_edge, sub_indices)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        """Return the number of edges in this view."""
        return len(self._indices)

    def __iter__(self):
        """
        Iterate over individual edge views.

        Yields:
            SingleEdgeEntityView for each edge in this view
        """
        # Convert to Python list once for efficiency
        for idx in self._indices.tolist():
            yield SingleEdgeEntityView(self._edge, self._dual_edge, idx)

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the edge view."""
        return (len(self),)

    @property
    def indices(self) -> Array:
        """
        Return the indices of edges in this view.

        Returns:
            JAX array of shape (n_edges,) containing edge indices
        """
        return self._indices

    @property
    def midpoint(self) -> Point:
        """
        Return midpoints of all edges in this view.

        Returns:
            Point object with arrays of shape (n_edges, 3) for cart and (n_edges, 2) for ltln
        """
        return self._edge.midpoint[self._indices]

    @property
    def length(self) -> Array:
        """
        Return lengths of all edges in this view.

        Returns:
            Array of shape (n_edges,)
        """
        return self._edge.length[self._indices]

    @property
    def endpoints(self) -> Array:
        """
        Return endpoint node indices for edges in this view.

        Returns:
            Array of shape (n_edges, 2)
        """
        return self._edge.v[self._indices]

    @property
    def dual_cells(self) -> Array:
        """Return the dual cells adjacent to the edge entity in this view.

        Returns:
            Array of shape (n_edges, 2)
        """
        return self._dual_edge.vertex[self._indices]

    @property
    def normal(self) -> Array:
        """
        Return normal vectors for edges in this view.

        Returns:
            Array of shape (n_edges, 3)
        """
        return self._edge.nr[self._indices]

    @property
    def tangent(self) -> Array:
        """
        Return tangent vectors for edges in this view.

        Returns:
            Array of shape (n_edges, 3)
        """
        return self._edge.tg[self._indices]


class EdgeEntity:
    """
    Entity accessor for edge collections supporting efficient indexing.

    This class provides the main interface for accessing edges in a mesh.
    It supports integer indexing (returns SingleEdgeEntityView), slice/array
    indexing (returns EdgeEntityView), and batch operations on all edges.

    Attributes:
        _edge: The underlying edge data (PrimalEdge or DualEdge)
    """

    def __init__(self, edge: Union["PrimalEdge", "DualEdge"], dual_edge: Union["DualEdge", "PrimalEdge"]):
        """
        Initialize the edge entity accessor.

        Args:
            edge: PrimalEdge or DualEdge instance containing the data
            dual_edge: The dual edge object corresponding to the given edge
        """
        self._edge = edge
        self._dual_edge = dual_edge

    def __repr__(self) -> str:
        return f"EdgeEntity({self._edge})"

    @overload
    def __getitem__(self, index: int) -> SingleEdgeEntityView:
        """Access a single edge by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> EdgeEntityView:
        """Access multiple edges by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> EdgeEntityView:
        """Access multiple edges by array of indices."""
        ...

    def __getitem__(
        self,
        index: Union[int, slice, Array]
    ) -> Union[SingleEdgeEntityView, EdgeEntityView]:
        """
        Access edges by index with zero-copy views.

        Args:
            index: Integer (single edge), slice, or array of indices

        Returns:
            SingleEdgeEntityView for integer index (single edge access)
            EdgeEntityView for slice/array index (multi-edge access)

        Raises:
            IndexError: If integer index is out of bounds
            TypeError: If index type is not supported

        Examples:
            >>> edge = mesh.primal.edge[5]  # Single edge view
            >>> edges = mesh.primal.edge[10:20]  # View of 10 edges
            >>> edges = mesh.primal.edge[[1, 5, 9]]  # View of specific edges
        """
        if isinstance(index, int):
            # Single edge access: return lightweight view
            if index < 0 or index >= self._edge.n_edge:
                raise IndexError(
                    f"Edge index {index} out of range [0, {self._edge.n_edge})")
            return SingleEdgeEntityView(self._edge, self._dual_edge, index)

        elif isinstance(index, slice):
            # Slice access: return view with index array
            start, stop, step = index.indices(self._edge.n_edge)
            indices = jnp.arange(start, stop, step)
            return EdgeEntityView(self._edge, self._dual_edge, indices)

        elif isinstance(index, (list, jnp.ndarray)):
            # Array indexing: return view with index array
            indices = jnp.asarray(index)
            if indices.ndim != 1:
                raise ValueError("Index array must be 1-dimensional")
            return EdgeEntityView(self._edge, self._dual_edge, indices)

        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        """Return the number of edges in the edge entity."""
        return int(self._edge.n_edge)

    def __iter__(self):
        """Iterate over the single edge entities."""
        for i in range(len(self)):
            yield self[i]

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the edge entity."""
        return (len(self),)

    @property
    def midpoint(self) -> Point:
        """Return the midpoint of the edge entity."""
        return self._edge.midpoint

    @property
    def endpoints(self) -> Array:
        """Return the endpoints of the edge entity."""
        return self._edge.vertex

    @property
    def dual_cells(self) -> Array:
        """Return the dual cells adjacent to the edge entity."""
        return self._dual_edge.vertex

    @property
    def length(self) -> Array:
        """Return the length of the edge entity."""
        return self._edge.length

    @property
    def normal(self) -> Array:
        """Return the normal vector of the edge entity."""
        return self._edge.normal_vector

    @property
    def tangent(self) -> Array:
        """Return the tangent vector of the edge entity."""
        return self._edge.tangent_vector


class SingleNodeEntityView:
    """
    Lightweight view for accessing a single node without data copying.

    Nodes are represented by dual cells (for primal mesh) or primal cells
    (for dual mesh). This class provides indexed access to node properties
    while maintaining a reference to the original data.

    Attributes:
        _dual: Reference to the parent dual/primal cell
        _index: Index of this node in the parent cell array
    """
    __slots__ = ('_dual', '_index', '_nnb')

    def __init__(self, dual: Union["DualCell", "PrimalCell"], index: int):
        """
        Initialize a view for a single node.

        Args:
            dual: Parent cell object representing nodes (DualCell or PrimalCell)
            index: Index of the node to view
        """
        self._dual = dual
        self._index = index

        # Pre-compute the number of valid neighbors for this node
        if isinstance(dual, PrimalCell):
            self._nnb = int(dual.nnb[index])
        else:
            self._nnb = int(dual.nnb)  # DualCell has fixed nnb

    def __repr__(self) -> str:
        """Return string representation of this node view."""
        dual_type = "DualCell" if isinstance(
            self._dual, DualCell) else "PrimalCell"
        return f"SingleNodeEntityView({dual_type}, index={self._index})"

    @property
    def index(self) -> IntType:
        """Return the global index of this node."""
        return self._index

    @property
    def adjacency(self) -> Array:
        """
        Return the adjacent node indices for this node.

        Returns:
            Array of adjacent indices
        """
        return self._dual.nb[self._index, :self._nnb]

    @property
    def n_adjacency(self) -> IntType:
        """
        Return the number of adjacent nodes for this node.

        Returns:
            Number of adjacent nodes
        """
        return self._nnb

    @property
    def max_adjacency(self) -> IntType:
        """
        Return the maximum possible number of neighbors for this node.

        Returns:
            Maximum neighbor count
        """
        if isinstance(self._dual, PrimalCell):
            return self._dual.maxnnb
        else:
            return self._dual.nnb

    @property
    def adjacent_cells(self) -> Array:
        """
        Return the adjacent cells for this node.

        Returns:
            Array of adjacent cell indices
        """
        return self.adjacency

    @property
    def coordinate(self) -> Point:
        """
        Return the coordinates for nodes in this view.

        Returns:
            Point Instance
        """
        return self._dual.centroid[self._index]


class NodeEntityView:
    """
    Lightweight view for accessing multiple nodes without data copying.

    This class stores only the indices of selected nodes and provides
    batch operations on the original node data (zero-copy). Supports
    chained indexing and iteration.

    Attributes:
        _dual: Reference to the parent dual/primal cell
        _indices: JAX array of node indices to view
    """
    __slots__ = ('_dual', '_indices')

    def __init__(self, dual: Union["DualCell", "PrimalCell"], indices: Array):
        """
        Initialize a view for multiple nodes.

        Args:
            dual: Parent cell object representing nodes (reference, not copied)
            indices: JAX array of node indices to include in this view
        """
        self._dual = dual
        self._indices = indices

    def __repr__(self) -> str:
        """Return string representation of this view."""
        dual_type = "DualCell" if isinstance(
            self._dual, DualCell) else "PrimalCell"
        return f"NodeEntityView({dual_type}, n_nodes={len(self)})"

    @overload
    def __getitem__(self, index: int) -> SingleNodeEntityView:
        """Access a single node by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "NodeEntityView":
        """Access multiple nodes by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> "NodeEntityView":
        """Access multiple nodes by array of indices."""
        ...

    def __getitem__(self, index: Union[int, slice, Array]) -> Union["SingleNodeEntityView", "NodeEntityView"]:
        """
        Support chained indexing on the view.

        Args:
            index: Integer, slice, or array index

        Returns:
            SingleNodeEntityView for int index, NodeEntityView for slice/array

        Examples:
            >>> view = mesh.primal.node[10:20]  # NodeEntityView with 10 nodes
            >>> node = view[5]  # SingleNodeEntityView for original node 15
            >>> sub_view = view[2:5]  # NodeEntityView for original nodes 12-14
        """
        if isinstance(index, int):
            # Map to original node index
            original_idx = int(self._indices[index])
            return SingleNodeEntityView(self._dual, original_idx)
        elif isinstance(index, slice):
            # Slice the index array
            sub_indices = self._indices[index]
            return NodeEntityView(self._dual, sub_indices)
        elif isinstance(index, (list, jnp.ndarray)):
            # Advanced indexing on the index array
            sub_indices = self._indices[jnp.asarray(index)]
            return NodeEntityView(self._dual, sub_indices)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        """Return the number of nodes in this view."""
        return len(self._indices)

    def __iter__(self):
        """
        Iterate over individual node views.

        Yields:
            SingleNodeEntityView for each node in this view
        """
        # Convert to Python list once for efficiency
        for idx in self._indices.tolist():
            yield SingleNodeEntityView(self._dual, idx)

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of this node view."""
        return (len(self._indices),)

    @property
    def indices(self) -> Array:
        """
        Return the indices of nodes in this view.

        Returns:
            JAX array of shape (n_nodes,) containing node indices
        """
        return self._indices

    @overload
    def get_adjacency(self, with_padding: Literal[True] = True) -> Array:
        """Return adjacency matrix as fixed-size array with padding."""
        ...

    @overload
    def get_adjacency(self, with_padding: Literal[False]) -> List[Array]:
        """Return adjacency matrix as list of variable-length arrays."""
        ...

    def get_adjacency(self, with_padding: bool = True) -> Union[Array, List[Array]]:
        """
        Return adjacency matrix for nodes in this view.

        Args:
            with_padding: If True (default), return fixed-size array with padding (-1).
                         If False, return list of variable-length arrays (one per node).

        Returns:
            If with_padding=True: Array of shape (n_nodes, max_neighbors) with padding
            If with_padding=False: List of n_nodes arrays, each with shape (actual_neighbors,)
        """
        if with_padding:
            return self._dual.nb[self._indices]

        # Return list of variable-length arrays using fully JIT-accelerated approach
        # Shape: (n_nodes, max_neighbors)
        nb_data = self._dual.nb[self._indices]

        if isinstance(self._dual, PrimalCell):
            # Variable neighbor counts
            nnb_data = self._dual.nnb[self._indices]
            return [nb_data[i, :int(nnb_data[i])] for i in range(len(self._indices))]
        else:
            # DualCell has fixed nnb for all nodes
            nnb = int(self._dual.nnb)
            return [nb_data[i, :nnb] for i in range(len(self._indices))]

    @property
    def adjacency(self):
        """Return the adjacency matrix for nodes in this view with padding (-1)."""
        return self.get_adjacency(with_padding=True)

    @property
    def n_adjacency(self) -> Array:
        """
        Return the number of adjacent nodes for each node in this view.

        Returns:
            Array of shape (n_nodes,) containing the number of adjacent nodes for each node
        """
        if isinstance(self._dual, PrimalCell):
            return self._dual.nnb[self._indices]
        else:
            return jnp.ones(len(self), dtype=jnp.int32) * self._dual.nnb

    @property
    def max_adjacency(self) -> IntType:
        """
        Return the maximum possible number of neighbors for each nodes in this view.

        Returns:
            Maximum neighbor count
        """
        if isinstance(self._dual, PrimalCell):
            return self._dual.maxnnb
        else:
            return self._dual.nnb

    @property
    def adjacent_cells(self) -> Array:
        """
        Return the adjacent cells for nodes in this view.

        Returns:
            Array of shape (n_nodes, max_neighbors)
        """
        return self._dual.vertex[self._indices]

    @property
    def coordinate(self) -> Point:
        """
        Return the centroid point for nodes in this view.

        Returns:
            Point Instance
        """
        return self._dual.centroid[self._indices]


class NodeEntity:
    """
    Entity accessor for node collections supporting efficient indexing.

    This class provides the main interface for accessing nodes (vertices) in a mesh.
    Nodes are represented by the dual of the mesh (dual cells for primal nodes,
    primal cells for dual nodes). It supports integer indexing (returns
    SingleNodeEntityView), slice/array indexing (returns NodeEntityView), and
    batch operations on all nodes.

    Attributes:
        _dual: The underlying dual cell data (DualCell or PrimalCell)
    """

    def __init__(self, dual: Union["DualCell", "PrimalCell"]):
        """
        Initialize the node entity accessor.

        Args:
            dual: DualCell or PrimalCell instance representing nodes
        """
        self._dual = dual

    @overload
    def __getitem__(self, index: int) -> SingleNodeEntityView:
        """Access a single node by integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> NodeEntityView:
        """Access multiple nodes by slice."""
        ...

    @overload
    def __getitem__(self, index: Array) -> NodeEntityView:
        """Access multiple nodes by array of indices."""
        ...

    def __getitem__(
        self,
        index: Union[int, slice, Array]
    ) -> Union[SingleNodeEntityView, NodeEntityView]:
        """
        Access nodes by index with zero-copy views.

        Args:
            index: Integer (single node), slice, or array of indices

        Returns:
            SingleNodeEntityView for integer index (single node access)
            NodeEntityView for slice/array index (multi-node access)

        Raises:
            IndexError: If integer index is out of bounds
            TypeError: If index type is not supported

        Examples:
            >>> node = mesh.primal.node[5]  # Single node view
            >>> nodes = mesh.primal.node[10:20]  # View of 10 nodes
            >>> nodes = mesh.primal.node[[1, 5, 9]]  # View of specific nodes
        """
        if isinstance(index, int):
            # Single node access: return lightweight view
            if index < 0 or index >= self._dual.n_cell:
                raise IndexError(
                    f"Node index {index} out of range [0, {self._dual.n_cell})")
            return SingleNodeEntityView(self._dual, index)

        elif isinstance(index, slice):
            # Slice access: return view with index array
            start, stop, step = index.indices(self._dual.n_cell)
            indices = jnp.arange(start, stop, step)
            return NodeEntityView(self._dual, indices)

        elif isinstance(index, (list, jnp.ndarray)):
            # Array indexing: return view with index array
            indices = jnp.asarray(index)
            if indices.ndim != 1:
                raise ValueError("Index array must be 1-dimensional")
            return NodeEntityView(self._dual, indices)

        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        """Return the number of nodes in the node entity."""
        return int(self._dual.n_cell)

    def __iter__(self):
        """Iterate over the single node entities."""
        for i in range(len(self)):
            yield self[i]

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the node entity as (n_nodes,)."""
        return (len(self),)

    @overload
    def get_adjacency(self, with_padding: Literal[True] = True) -> Array:
        """Return adjacency matrix as fixed-size array with padding."""
        ...

    @overload
    def get_adjacency(self, with_padding: Literal[False]) -> List[Array]:
        """Return adjacency matrix as list of variable-length arrays."""
        ...

    def get_adjacency(self, with_padding: bool = True) -> Union[Array, List[Array]]:
        """Return the adjacency matrix of the node entity.

        Args:
            with_padding: If True (default), return fixed-size array with padding (-1).
                         If False, return list of variable-length arrays with actual neighbors.

        Returns:
            Array of shape (n_nodes, max_neighbors) when with_padding=True, or
            List[Array] with variable-length arrays when with_padding=False.
        """
        if with_padding:
            return self._dual.nb

        # Return list of variable-length arrays using fully JIT-accelerated approach
        nb_data = self._dual.nb  # Shape: (n_nodes, max_neighbors)

        if isinstance(self._dual, PrimalCell):
            nnb_data = self._dual.nnb  # Variable neighbor counts
            return [nb_data[i, :int(nnb_data[i])] for i in range(self._dual.n_cell)]
        else:
            # DualCell has fixed nnb for all nodes
            nnb = int(self._dual.nnb)
            return [nb_data[i, :nnb] for i in range(self._dual.n_cell)]

    @property
    def adjacency(self) -> Array:
        """Return the adjacency matrix of the node entity with padding (-1)."""
        return self.get_adjacency(with_padding=True)

    @property
    def n_adjacency(self) -> Array:
        """Return the number of adjacent nodes for each node in the entity."""
        if isinstance(self._dual, PrimalCell):
            return self._dual.nnb
        else:
            return jnp.ones(self._dual.n_cell, dtype=jnp.int32) * self._dual.nnb

    @property
    def max_adjacency(self) -> IntType:
        """Return the maximum adjacency nodes number of the node entity."""
        if isinstance(self._dual, PrimalCell):
            return self._dual.maxnnb
        else:
            return self._dual.nnb  # DualCell has fixed nnb

    @property
    def adjacent_cells(self) -> Array:
        """Return the adjacent cells for each node in the entity."""
        return self._dual.vertex

    @property
    def coordinate(self) -> Point:
        """Return the centroid point of the node entity"""
        return self._dual.centroid


class AbstractMeshEntity(eqx.Module):
    _mesh: "OrthogonalDualMesh"

    def __init__(self, mesh: "OrthogonalDualMesh"):
        self._mesh = mesh

    @property
    def cell(self) -> CellEntity:
        """Return the cell entity."""
        raise NotImplementedError

    @property
    def edge(self) -> EdgeEntity:
        """Return the edge entity."""
        raise NotImplementedError

    @property
    def node(self) -> NodeEntity:
        """Return the node entity."""
        raise NotImplementedError


class PrimalMeshEntity(AbstractMeshEntity):

    @property
    def cell(self) -> CellEntity:
        """Return the primal cell entity."""
        return CellEntity(self._mesh._plg)

    @property
    def edge(self) -> EdgeEntity:
        """Return the primal edge entity."""
        return EdgeEntity(self._mesh._edp, self._mesh._edd)

    @property
    def node(self) -> NodeEntity:
        """Return the primal node entity."""
        return NodeEntity(self._mesh._dua)


class DualMeshEntity(AbstractMeshEntity):

    @property
    def cell(self) -> CellEntity:
        """Return the dual cell entity."""
        return CellEntity(self._mesh._dua)

    @property
    def edge(self) -> EdgeEntity:
        """Return the dual edge entity."""
        return EdgeEntity(self._mesh._edd, self._mesh._edp)

    @property
    def node(self) -> NodeEntity:
        """Return the dual node entity."""
        return NodeEntity(self._mesh._plg)


class OrthogonalDualMesh(AbstractUnstructuredMesh):
    """An orthogonal dual mesh consisting of primal and dual cells and edges.

    This mesh type is registered in the catalog with alias 'odm' through
    the registration system in terraflow.manager.plugins.mesh.registrations.
    """
    n_polg: IntType
    n_dual: IntType
    n_edge: IntType
    _plg: PrimalCell
    _dua: DualCell
    _edp: PrimalEdge
    _edd: DualEdge
    _aux: AuxiliaryData

    @staticmethod
    def alias_name() -> str:
        """Retrun the alias name for this mesh type."""
        return "odm"

    def get_layout(
        self,
        layout: str = "trisk"
    ) -> TriSKBasedCGridStaggeringLayout:
        """Return the layout object for this mesh type."""
        import terraflow.layout.orthogonal_dual_mesh as odm_layout
        if layout == "trisk":
            return odm_layout.TriSKBasedCGridStaggeringLayout()
        else:
            raise ValueError(f"Unsupported layout: {layout}")

    @property
    def shape(self) -> dict:
        """Return the shape of the mesh as (n_polg, n_dual, n_edge)."""
        return dict(n_polg=self.n_polg, n_dual=self.n_dual, n_edge=self.n_edge)

    @property
    def primal(self) -> PrimalMeshEntity:
        """Return the primal mesh entity."""
        # TODO: For now, just provide access to primal and dual mesh entities.
        # In the future, we may want to use other ways to access entities instead.
        return PrimalMeshEntity(self)

    @property
    def dual(self) -> DualMeshEntity:
        """Return the dual mesh entity."""
        # TODO: For now, just provide access to primal and dual mesh entities.
        # In the future, we may want to use other ways to access entities instead.
        return DualMeshEntity(self)

    @property
    def aux(self) -> AuxiliaryData:
        """Return the auxiliary data of the mesh."""
        return self._aux

    @classmethod
    def from_grist(
        cls,
        grist_0d_file: str,
        grist_2d_file: str,
        n_dual: IntType = 3,
        dev: device.AbstractDevice = device.CPU()
    ) -> Self:
        """Initialize the OrthogonalDualMesh from Grist NetCDF Files."""
        grist = GristRefMesh.from_netcdf(
            grist_0d_file, grist_2d_file, n_dual=n_dual, dev=dev
        )
        n_polg = int(grist.shape["nv"])
        n_dual = int(grist.shape["nt"])
        n_edge = int(grist.shape["ne"])
        plg = PrimalCell.from_grist(grist)
        dua = DualCell.from_grist(grist)
        edp = PrimalEdge.from_grist(grist)
        edd = DualEdge.from_grist(grist)
        aux = AuxiliaryData.from_elements(plg, dua, edp, edd)
        logger.success("Initialized OrthogonalDualMesh from Grist mesh.")
        return cls(dev, n_polg, n_dual, n_edge, plg, dua, edp, edd, aux)

    def reconstruct_vector(
        self,
        normal_u_at_edge: Float[Array, "n_edge"],
        from_loc: Literal["dual_edge"] = "dual_edge",
        to_loc: Literal["primal_cell", "dual_cell"] = "primal_cell",
    ) -> Mapping[str, Float[Array, "n_cell"]]:
        """Reconstruct vector components at specified location from edge normal values.

        Currently supports only 'primal_cell_from_dual_edge' location.

        Arguments:
            normal_u_at_edge: (n_edge,) array of normal velocity values at edges.
            from_loc: Location of input normal values ('dual_edge' supported).
            to_loc: Location to reconstruct vector components ('primal_cell' supported).

        Returns:
            Dict[str, jnp.ndarray]: Dictionary with keys 'u' and 'v' mapping to arrays each shape (n_cell,).

        Raises:
            ValueError: If the specified location is not supported.
        """
        if from_loc == "dual_edge" and to_loc == "primal_cell":
            return reconstruct_vector_at_primal_cell_from_dual_edge(
                self, normal_u_at_edge
            )
        else:
            raise ValueError(
                f"Unsupported reconstruction location: {from_loc} to {to_loc}"
            )

    def to_ugrid(
        self,
        attributes: Optional[AttributeConvention] = None,
        use_dual: bool = False,
        **kwargs
    ) -> UGRIDGrid:
        """Convert the OrthogonalDualMesh to a UGRIDGrid representation."""
        return UGRIDGrid.from_orthogonal_dual_mesh(
            self, attributes=attributes, use_dual=use_dual, **kwargs
        )

    def history_builder(
        self,
        file_path: str,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
    ) -> history.setup.mesh.OrthogonalDualMeshHistorySetup:
        """Get history builder for the mesh."""
        return history.setup.mesh.OrthogonalDualMeshHistorySetup(
            file_path=file_path,
            mesh=self,
            output_type=output_type,
            float_type=float_type,
            int_type=int_type
        )


def reconstruct_vector_at_primal_cell_from_dual_edge(
    mesh: OrthogonalDualMesh,
    normal_u_at_edge: Float[Array, "nlev n_edge"]
) -> Mapping[str, Float[Array, "nlev n_cell"]]:
    """Reconstruct prime-cell vector components (u, v) from edge normal values.

    Arguments:
        mesh: OrthogonalDualMesh instance.
        normal_u_at_edge: (nlev, n_edge) array of normal velocity values at edges.

    Returns:
        Dict[str, jnp.ndarray]: Dictionary with keys 'u' and 'v' mapping to arrays each shape (nlev, n_cell).

    Notes:
        - The implementation handles variable neighbour counts using masking.
        - All arrays are expected to be jax-compatible.
    """
    cell = mesh.primal.cell
    cell_centroid = cell.centroid.cartesian  # (ncell, 3)
    cell_edges = cell.edges  # (ncell, maxnnb)
    edp = mesh.primal.edge
    edd = mesh.dual.edge
    cell_nnb = cell.n_adjacency  # (ncell,)
    cell_area = cell.area  # (ncell,)
    edge_midpoint = edd.midpoint.cartesian  # (nedge, 3)
    edge_nr = edp.normal  # (nedge, 3)
    edge_length = edp.length  # (nedge,)

    eps = jnp.finfo(cell_centroid.dtype).eps

    def reconstruct_single_level(
        normal_u_at_edge: Float[Array, "n_edge"]
    ) -> Mapping[str, Float[Array, "n_cell"]]:

        # Vectorized computation over all cells
        # Gather edge data for all cells at once: (ncell, maxnnb, ...)
        OPi = edge_midpoint[cell_edges]  # (ncell, maxnnb, 3)
        nei = edge_nr[cell_edges]  # (ncell, maxnnb, 3)
        wind = normal_u_at_edge[cell_edges]  # (ncell, maxnnb)
        leng = edge_length[cell_edges]  # (ncell, maxnnb)

        # Create mask for valid neighbors: (ncell, maxnnb)
        # Use shape inference to get maxnnb instead of accessing it as a field
        maxnnb = cell_edges.shape[1]
        pos = jnp.arange(maxnnb)[None, :]  # (1, maxnnb)
        mask = pos < cell_nnb[:, None]  # (ncell, maxnnb)

        # Project edge midpoints onto tangent plane at cell centroids
        # OPc: (ncell, 3) -> (ncell, 1, 3) for broadcasting
        OPc = cell_centroid[:, None, :]  # (ncell, 1, 3)
        rei = jax.vmap(
            lambda opc, opi: math.project_sphere2plane(opc[0], opi),
            in_axes=(0, 0)
        )(OPc, OPi)  # (ncell, maxnnb, 3)

        # Compute norms and normalize
        rei_norm = jnp.linalg.norm(rei, axis=-1)  # (ncell, maxnnb)
        rei_norm = jnp.maximum(rei_norm, eps * 100)
        rei_unit = rei / rei_norm[:, :, None]  # (ncell, maxnnb, 3)

        # Dot product: nei . rei_unit
        dot_val = jnp.sum(nei * rei_unit, axis=-1)  # (ncell, maxnnb)

        # Contribution per neighbor: rei * wind * dot_val * leng
        # (ncell, maxnnb, 3)
        contrib = rei * (wind * dot_val * leng)[:, :, None]

        # Apply mask to zero out invalid neighbors
        contrib = contrib * mask[:, :, None]  # (ncell, maxnnb, 3)

        # Sum contributions across neighbors
        vector_cart = jnp.sum(contrib, axis=1)  # (ncell, 3)

        # Divide by cell area
        area = jnp.maximum(cell_area, eps * 100)  # (ncell,)
        vector_cart = vector_cart / area[:, None]  # (ncell, 3)

        # Convert to lat-lon coordinates
        result = math.convert_vector_cart2ltln(
            vector_cart, cell_centroid
        )  # returns dict with 'u', 'v'
        return result

    return jax.vmap(
        reconstruct_single_level,
        in_axes=0,
        out_axes=0
    )(normal_u_at_edge)
