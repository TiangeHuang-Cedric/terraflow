from jax import numpy as jnp

from terraflow.util import typing
from terraflow.mesh.orthogonal_dual_mesh import PrimeCell, DualCell
from terraflow.mesh.orthogonal_dual_mesh import PrimeEdge, DualEdge
from terraflow.operator.fvm import FluxReconstruction


Int = typing.Int
Float = typing.Float
Array = typing.Array


class FluxReconstructionAtEdge(FluxReconstruction):
    pass


class CentralDifference2ndOrder(FluxReconstructionAtEdge):
    def __call__(self, x_at_cell: Array, edge: PrimeEdge | DualEdge) -> Array:
        vertex = edge.vertex()
        v1 = vertex[:, 0]
        v2 = vertex[:, 1]
        x1 = x_at_cell[v1]
        x2 = x_at_cell[v2]
        return 0.5 * (x1 + x2)


class LeastSquareBasedFluxReconstruction(FluxReconstructionAtEdge):
    pass


class LeastSquare3rd4thOrder(LeaastSquareBasedFluxReconstruction):
    stencil_num: Array
    stencil_idx: Array
    edd_v_fcoe: Array
    def __init__(self, stencil_num: Int, stencil_idx: Array, v_fcoe: Array):
        self.stencil_num = stencil_num
        self.stencil_idx = stencil_idx
        self.v_fcoe = v_fcoe
        
        
class LeastSquare5th6thOrder(LeaastSquareBasedFluxReconstruction):
    stencil_2nd_num: Array
    stencil_2nd_idx: Array
    edd_2nd_v_fcoe: Array
    stencil_4th_num: Array
    stencil_4th_idx: Array
    edd_4th_v_fcoe: Array
    
    def __init__(self, stencil_num: Int, stencil_idx: Array, v_fcoe: Array):
        self.stencil_num = stencil_num
        self.stencil_idx = stencil_idx
        self.v_fcoe = v_fcoe