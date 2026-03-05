from __future__ import annotations
from typing import TYPE_CHECKING
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord import OneLayerVertCoordinate, HybridVertCoordinate
from terraflow.layout import CoordinateSystem, Field
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.discretization.abstract import DiscretizationInfo
from terraflow.discretization.fvm.odm.trisk import base, coriolis, pgf
from terraflow.discretization.fvm.odm.trisk.single_lev.aux import (
    DivergenceOfMassFluxAtCell,
)
from terraflow.dynamics.shallow_water.types import (
    DivergenceOfMassFluxProtocol,
    NonlinearCoriolisTermProtocol,
    PressureGradientForceTermProtocol,
    GradientOfKineticEnergyProtocol,
)

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block

Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate | HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]


class TendencyOfDivergenceOfMassFlux(
    DivergenceOfMassFluxProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.mf = base.MassFlux(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.dmf = DivergenceOfMassFluxAtCell(
            disc=disc,
            method=method
        )

    def __call__(self, block: Block) -> Block:
        g = block.para.gravity.unwrap()
        block = self.mf(block)
        block = self.dmf(block)
        dmf = block.aux.dmf.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dgzdt_dmf.data,
            block,
            replace=-dmf * g
        )

    def build_default(self) -> "TendencyOfDivergenceOfMassFlux":
        if not self.has_built():
            self.mf.build_default()
            self.dmf.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.mf.prepare(block)
        block = self.dmf.prepare(block)
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dgzdt_dmf


class NonlinearCoriolisTermAtDualEdge(
    NonlinearCoriolisTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.vor = base.RelativeVorticityAtDualCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.coriolis = coriolis.CoriolisForceAtEdge(
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

    def __call__(self, block: Block) -> Block:
        block = self.vor(block)
        block = self.coriolis.recon_tangent_velocity(block)
        block = self.pv(block)
        return self.coriolis(block)

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self.vor.build_default()
            self.coriolis.build_default()
            self.pv.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.vor.prepare(block)
        block = self.coriolis.prepare(block)
        block = self.pv.prepare(block)
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_coriolis


class PressureGradientForceAtDualEdge(
    PressureGradientForceTermProtocol[FiniteVolumeMethod]
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

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_pgf


class GradientOfKineticEnergyAtDualEdge(
    GradientOfKineticEnergyProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
    ):
        super().__init__(disc=disc, method=method)
        self.ke = base.KineticEnergyAtPrimalCell(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def __call__(self, block: Block) -> Block:
        block = self.ke(block)
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
            self.ke.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.ke.prepare(block)
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_grad_ke
