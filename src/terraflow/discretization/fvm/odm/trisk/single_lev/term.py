from __future__ import annotations
from typing import Self
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import Field
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractPrepareSpaceOperators,
    AbstractEquationOperator
)
from terraflow.discretization.fvm.odm.trisk import base, adv, coriolis, pgf
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block
from terraflow.dynamics.shallow_water import types

Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]
BLOCK = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]


class FluidThickness(AbstractEquationOperator[FiniteVolumeMethod]):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=False,
            hydrostatic=False,
            nonhydrostatic=False,
            ideal_dry_core=False,
            advection=False
        )
        self.recon_at_edge = adv.flux_reconstruct.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: BLOCK) -> BLOCK:
        coord = block.coord
        g = block.para.gravity.unwrap()
        un = block.state.un.unwrap()
        gz = block.state.gz.unwrap()
        gzs = block.static.gzs.unwrap()
        dmg = (gz - gzs) / g
        dmg_e = self.recon_at_edge(
            x_at_primal_cell=dmg,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        dmg_vtx = op.interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: (b.aux.dmg.data, b.aux.dmg_e.data, b.aux.dmg_vtx.data,),
            block,
            replace=(dmg, dmg_e, dmg_vtx)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_at_edge.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.recon_at_edge.prepare_block(block)


class TendencyOfDivergenceOfMassFlux(
    types.DivergenceOfMassFluxProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: Block) -> Block:
        g = block.para.gravity.unwrap()
        dmf = block.aux.dmf.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dgzdt_dmf.data,
            block,
            replace=-dmf * g
        )

    def build_default(self) -> "TendencyOfDivergenceOfMassFlux":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dgzdt_dmf


class NonlinearCoriolisTermAtDualEdge(
    types.NonlinearCoriolisTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: Block) -> Block:
        coriolis = block.aux.coriolis.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dundt_coriolis.data,
            block,
            replace=coriolis
        )

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_coriolis


class PressureGradientForceAtDualEdge(
    types.PressureGradientForceTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.scheme = pgf.swm.PressureGradientForce(disc=disc, method=method)

    def __call__(self, block: Block) -> Block:
        return self.scheme(block)

    def build_default(self) -> "PressureGradientForceAtDualEdge":
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_pgf


class GradientOfKineticEnergyAtDualEdge(
    types.GradientOfKineticEnergyProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: Block) -> Block:
        ke = block.aux.ke.unwrap()
        grad_ke = - op.grad_at_dual_edge(
            x_at_primal_cell=ke,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_grad_ke.data,
            block,
            replace=grad_ke
        )

    def build_default(self) -> "GradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_grad_ke


class ExPrepareSpaceOperators(AbstractPrepareSpaceOperators[FiniteVolumeMethod]):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        baroclinic: bool,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            baroclinic=baroclinic,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.dmg = FluidThickness(disc=disc, method=method)
        self.mf = base.MassFlux(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.dmf = base.DivergenceOfMassFluxAtCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.coriolis = coriolis.schemes.CoriolisForceAtEdge(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.vor = base.RelativeVorticityAtDualCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.pv = base.PotentialVorticity(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.ke = base.KineticEnergyAtPrimalCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def build_default(self) -> "ExPrepareSpaceOperators":
        if not self.has_built():
            self.dmg.build_default()
            self.mf.build_default()
            self.dmf.build_default()
            self.vor.build_default()
            self.coriolis.build_default()
            self.pv.build_default()
            self.ke.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmg(block)
        block = self.mf.prepare_block(block)
        block = self.dmf.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.coriolis.prepare_block(block)
        block = self.pv.prepare_block(block)
        block = self.ke.prepare_block(block)
        return block

    def __call__(
        self,
        block: BLOCK,
        pass_tag: str = "all_pass",
        explicit: bool = True
    ) -> BLOCK:
        block = self.dmg(block)
        if pass_tag == "all_pass":
            block = self.mf(block)
            block = self.dmf(block)
            block = self.coriolis.recon_ut(block)
            block = self.vor(block)
            block = self.pv(block)
            block = self.coriolis(block)
            block = self.ke(block)
        elif pass_tag == "forward_pass":
            block = self.mf(block)
            block = self.dmf(block)
        elif pass_tag == "backward_pass":
            block = self.coriolis.recon_ut(block)
            block = self.vor(block)
            block = self.pv(block)
            block = self.coriolis(block)
            block = self.ke(block)
        return block


class ExSpaceOperators(
    types.ExVifShallowWaterEquationSet4UGRIDSpaceOperators[
        TendencyOfDivergenceOfMassFlux,
        NonlinearCoriolisTermAtDualEdge,
        GradientOfKineticEnergyAtDualEdge,
        PressureGradientForceAtDualEdge,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(disc=disc, method=method)

    def __call__(self, block: BLOCK, pass_tag: str = "all_pass", explicit: bool = True) -> BLOCK:
        block = self.prepare(
            block=block,
            pass_tag=pass_tag,
            explicit=explicit
        )
        if pass_tag == "all_pass":
            block = self.dgzdt(block)
            block = self.dundt(block)
        elif pass_tag == "forward_pass":
            block = self.dgzdt(block)
        elif pass_tag == "backward_pass":
            block = self.dundt(block)
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return block
