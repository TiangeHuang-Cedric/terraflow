from terraflow.util import typing
from terraflow.planet import PlanetParameterSet
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.discretization.abstract import DiscretizationInfo
from terraflow.discretization.abstract import AbstractEquationSetAuxiliary
from terraflow.discretization.fvm.odm.trisk import coriolis, base
from terraflow.discretization.fvm.odm.trisk.single_lev.dynamics_types import Block

Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
]


class AuxiliaryTerms(AbstractEquationSetAuxiliary):

    def __init__(
        self,
        disc: DiscretizationInfo,
        coord: CoordSys,
        para: PlanetParameterSet,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc, coord, para, method)
        self.ut = coriolis.CoriolisForceAtDualEdge(
            coord, para
        )
        self.rel_vor = base.RelativeVorticityAtDualCell(
            disc, coord, para, method
        )
        self.abs_vor = base.AbsoluteVorticityAtDualCell(
            disc, coord, para, method
        )
        self.z_e = base.GeopotentialHeightAtPrimalEdge(
            disc, coord, para, method
        )
        self.mfn_e = base.MassFluxAtPrimalEdge(
            disc, coord, para, method
        )
        self.z_vtx = base.GeopotentialHeightAtDualCell(
            disc, coord, para, method
        )
        self.pv_vtx = base.PotentialVorticityAtDualCell(
            disc, coord, para, method
        )
        self.pv_e = base.PotentialVorticityAtDualEdge(
            disc, coord, para, method
        )
        self.ke = base.KineticEnergyAtPrimalCell(
            disc, coord, para, method
        )

    def build_default(self) -> "AuxiliaryTerms":
        self.rel_vor.build_default()
        self.abs_vor.build_default()
        self.z_e.build_default()
        self.mfn_e.build_default()
        self.z_vtx.build_default()
        self.pv_vtx.build_default()
        self.pv_e.build_default()
        self.ke.build_default()
        return self

    def __call__(
        self,
        block: Block,
        pass_tag: str = "all_pass"
    ) -> Block:

        def forward_call(block: Block) -> Block:
            block = self.z_e(block)
            return block

        def backward_call(block: Block) -> Block:
            block = self.ut.recon_tangent_velocity(block)
            block = self.rel_vor(block)
            block = self.abs_vor(block)
            block = self.mfn_e(block)
            block = self.z_vtx(block)
            block = self.pv_vtx(block)
            block = self.pv_e(block)
            block = self.ke(block)
            return block

        if pass_tag == "all_pass":
            block = forward_call(block)
            block = backward_call(block)
        elif pass_tag == "forward_pass":
            block = forward_call(block)
        elif pass_tag == "backward_pass":
            block = backward_call(block)

        return block
