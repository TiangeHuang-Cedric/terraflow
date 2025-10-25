from terraflow import bind
from terraflow.util import typing
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import DualEdge
from terraflow.operator.fvm.stagc4odm import interp
import terraflow.equation.atmosphere.shallow_water_equation\
    .vector_invariant_form.equ as equ


Array = typing.Array

prefix = "equation.atmosphere.shallow_water_equation.vector_invariant_form"
momentum = f"{prefix}.momentum"


@bind(f"{momentum}.fvm.stagc4odm", tag="nonlinear_coriolis_term")
class TRiSKSchemeBasedNonlinearCoriolisTerm(equ.NonlinearCoriolisTerm):
    edp_numb_on_edge: Array
    edp_edge_on_edge: Array
    edp_trsk_on_edge: Array
    edp_edpl_on_edge: Array

    def __init__(self, mesh: OrthogonalDualMesh):
        collect = interp.trisk_weighted_interpolation_operator(mesh)
        numb, edge, trsk, edpl = collect
        self.edp_numb_on_edge = numb
        self.edp_edge_on_edge = edge
        self.edp_trsk_on_edge = trsk
        self.edp_edpl_on_edge = edpl

    def __call__(self, x_at_edge: Array, edd: DualEdge) -> Array:
        return interp.interp_weighted_add_pair_meeting_edges2edge(
            x_at_edge,
            self.edp_edge_on_edge,
            self.edp_trsk_on_edge,
            self.edp_edpl_on_edge,
            edd
        )
