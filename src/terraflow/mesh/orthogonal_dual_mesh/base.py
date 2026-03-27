from typing import Self, Union, Dict, Sequence
from typing import overload
import jax.numpy as jnp
from jax import lax, debug
import equinox as eqx

from terraflow.util import typing

Float = typing.Float
Int = typing.Int
IntType = typing.IntType
FloatType = typing.FloatType
Array = typing.Array


class Point(eqx.Module):
    """
    A point in 3D space, represented in both Cartesian and
    Latitude/Longitude formats.
    Attributes:
        cartesian (Float[Array, "npoint 3"])         : 3D Cartesian coordinates, shape (3,).
        ltln      (Dict[str, Float[Array, "npoint"]]): Latitude and Longitude in radians.
    """
    cart: Float[Array, "npoint 3"]
    ltln: Dict[str, Float[Array, "npoint"]]

    @overload
    def __getitem__(self, indices: int) -> Self: ...

    @overload
    def __getitem__(self, indices: Sequence[int]) -> Self: ...

    @overload
    def __getitem__(self, indices: Array) -> Self: ...

    def __getitem__(
        self,
        indices: Union[int, Sequence[int], Array]
    ) -> Self:
        cart = self.cart[indices, :]
        ltln = dict(
            lon=self.lon[indices],
            lat=self.lat[indices]
        )
        return type(self)(
            cart=cart, ltln=ltln
        )

    @property
    def cartesian(self) -> Array:
        """Return the Cartesian coordinates of the point."""
        return self.cart

    @property
    def x(self) -> Array:
        """Return the x direction of cartesian coordinates of the point"""
        return self.cart[:, 0]

    @property
    def y(self) -> Array:
        """Return the y direction of cartesian coordinates of the point"""
        return self.cart[:, 1]

    @property
    def z(self) -> Array:
        """Return the z direction of cartesian coordinates of the point"""
        return self.cart[:, 2]

    @property
    def latlon(self) -> Dict[str, Array]:
        """Return the latitude and longitude of the point."""
        return self.ltln

    @property
    def lat(self) -> Array:
        """Return the latitude of the point."""
        if "lat" in self.ltln:
            return self.ltln["lat"]
        raise ValueError("Latitude not found.")

    @property
    def lon(self) -> Array:
        """Return the longitude of the point."""
        if "lon" in self.ltln:
            return self.ltln["lon"]
        raise ValueError("Longitude not found.")


class Element(eqx.Module):
    """
    Base class for mesh elements such as Cell and Edge.
    Attributes:
        v (Array): Generating vertex of element, shape (nCell, nnb).
    """
    v: Array

    @property
    def vertex(self) -> Array:
        """Return the vertices of the element."""
        return self.v


class Cell(Element):
    """A Cell element in the mesh."""
    n_cell: IntType
    area: Float[Array, "ncell"]
    kite_area: Float[Array, "ncell maxnnb"]
    nb: Int[Array, "ncell maxnnb"]
    ed: Int[Array, "ncell maxnnb"]
    nr: Int[Array, "ncell maxnnb"]
    cc: Point

    @property
    def shape(self) -> tuple:
        """Return the shape of the cell element."""
        return self.neighbors.shape

    @property
    def length(self) -> IntType:
        """Return the number of neighboring cells (length of nnb)."""
        return self.shape[0]

    @property
    def max_neighbors(self) -> IntType:
        """Return the maximum number of neighbors (maxnnb)."""
        return self.shape[1]

    @property
    def centroid(self) -> Point:
        """Return the centroid of the element."""
        return self.cc

    @property
    def edge(self) -> Array:
        """Return the edges of the cell element."""
        return self.ed

    @property
    def neighbors(self) -> Array:
        """Return the neighboring cells of the cell element."""
        return self.nb

    def valid_neighbors(self) -> Array:
        """Return the valid neighboring cells of the cell element."""
        return self.neighbors >= 0

    def check_kite_areas(self, kite_area: Array):
        """Check if the sum of kite areas matches the cell area."""
        def raise_error(_):
            debug.print("Sum of kite areas does not match cell area.")

        def do_nothing(_):
            return None

        lax.cond(
            jnp.allclose(
                jnp.sum(kite_area, dtype=jnp.float64, axis=-1),
                self.area,
                rtol=1e-5,
                atol=1e-5
            ),
            do_nothing,
            raise_error,
            None
        )


class Edge(Element):
    """Edge element in the mesh."""
    n_edge: IntType
    length: Float[Array, "nedge"]
    nr: Float[Array, "nedge 3"]
    tg: Float[Array, "nedge 3"]
    mp: Point

    @property
    def midpoint(self) -> Point:
        """Return the midpoint of the edge element."""
        return self.mp

    @property
    def normal_vector(self):
        """Return the normal vectors of the edge element."""
        return self.nr

    @property
    def tangent_vector(self):
        """Return the tangent vectors of the edge element."""
        return self.tg


class RefPrimalCell(Cell):
    """primal Cell element in the dual mesh."""
    pass


class RefDualCell(Cell):
    """Dual Cell element in the dual mesh."""
    pass


class RefPrimalEdge(Edge):
    """primal Edge element in the dual mesh."""
    pass


class RefDualEdge(Edge):
    """Dual Edge element in the dual mesh."""
    pass


class AbstractRefMesh:
    """Reference mesh class."""

    def __init__(self):
        raise NotImplementedError("RefMesh is an abstract base class.")
