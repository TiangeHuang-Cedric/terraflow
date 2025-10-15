import jax.numpy as jnp
from jax import lax, debug
from dataclasses import dataclass

from terraflow.util import typing

Int = typing.Int
Float = typing.Float
Array = typing.Array


@dataclass
class Point:
    """
    A point in 3D space, represented in both Cartesian and
    Latitude/Longitude formats.
    Attributes:
        cartesian (Array): 3D Cartesian coordinates, shape (3,).
        ltln (Array): Latitude and Longitude in radians,
                      shape (2,) as [lat, lon].
    """
    cart: Array
    ltln: Array

    def cartesian(self) -> Array:
        """Return the Cartesian coordinates of the point."""
        return self.cart

    def latlon(self) -> Array:
        """Return the latitude and longitude of the point."""
        return self.ltln


class Element:
    """
    Base class for mesh elements such as Cell and Edge.
    Attributes:
        v (Array): Generating vertex of element, shape (nCell, nnb).
        cc (Point): Centroid center point of the element.
    """
    v: Array
    cc: Point

    def __init__(self, v: Array, cc: Point):
        """
        Initialize an Element with vertices and centroid.
        Args:
            v (Array): Vertices of the element.
            cc (Point): Centroid of the element.
        """
        self.v = v
        self.cc = cc

    def vertex(self) -> Array:
        """Return the vertices of the element."""
        return self.v

    def centroid(self) -> Point:
        """Return the centroid of the element."""
        return self.cc


class Cell(Element):
    """A Cell element in the mesh."""
    n_cell: Int
    area: Array
    kite_area: Array
    nb: Array
    ed: Array
    nr: Array

    def __init__(self, n_cell: Int, area: Array, kite_area: Array,
                 nb: Array, ed: Array, nr: Array, v: Array, cc: Point):
        """
        Initialize a Cell with the given parameters.
        Args:
            n_cell (Int): Number of cells.
            area (Array): Area of the cells.
            kite_area (Array): Kite areas of the cells.
            nb (Array): Neighbors of the cells.
            ed (Array): Edges of the cells.
            nr (Array): Normal vectors of the edges.
            v (Array): Vertices of the cells.
            cc (Point): Centroid of the cells.
        """
        Element.__init__(self, v, cc)
        self.n_cell = n_cell
        self.area = area
        self.kite_area = kite_area
        self.nb = nb
        self.ed = ed
        self.nr = nr

    @property
    def shape(self) -> tuple:
        """Return the shape of the cell element."""
        return self.neighbors().shape

    @property
    def length(self) -> Int:
        """Return the number of neighboring cells (length of nnb)."""
        return self.shape[0]

    @property
    def max_neighbors(self) -> Int:
        """Return the maximum number of neighbors (maxnnb)."""
        return self.shape[1]

    def edge(self) -> Array:
        """Return the edges of the cell element."""
        return self.ed

    def neighbors(self) -> Array:
        """Return the neighboring cells of the cell element."""
        return self.nb

    def valid_neighbors(self) -> Array:
        """Return the valid neighboring cells of the cell element."""
        return self.neighbors() >= 0

    def check_kite_areas(self, kite_area: Array):
        """Check if the sum of kite areas matches the cell area."""
        def raise_error(_):
            debug.print("Sum of kite areas does not match cell area.")

        def do_nothing(_):
            return None

        lax.cond(
            jnp.allclose(
                jnp.sum(kite_area, axis=-1), self.area, rtol=1e-5, atol=1e-5),
            do_nothing,
            raise_error,
            None
        )


class Edge(Element):
    """Edge element in the mesh."""
    n_edge: Int
    length: Array
    nr: Array
    tg: Array

    def __init__(self, n_edge: Int, length: Array,
                 nr: Array, tg: Array, v: Array, cc: Point):
        Element.__init__(self, v, cc)
        self.n_edge = n_edge
        self.length = length
        self.nr = nr
        self.tg = tg

    def normal_vector(self):
        """Return the normal vectors of the edge element."""
        return self.nr

    def tangent_vector(self):
        """Return the tangent vectors of the edge element."""
        return self.tg


class RefPrimeCell(Cell):
    """Prime Cell element in the dual mesh."""
    pass


class RefDualCell(Cell):
    """Dual Cell element in the dual mesh."""
    pass


class RefPrimeEdge(Edge):
    """Prime Edge element in the dual mesh."""
    pass


class RefDualEdge(Edge):
    """Dual Edge element in the dual mesh."""
    pass


class RefMesh:
    """Reference mesh class."""

    def __init__(self):
        raise NotImplementedError("RefMesh is an abstract base class.")
