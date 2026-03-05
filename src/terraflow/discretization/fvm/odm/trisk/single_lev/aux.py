from __future__ import annotations
from typing import TYPE_CHECKING, Self
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as op
from terraflow.vert_coord import OneLayerVertCoordinate
from terraflow.layout import CoordinateSystem
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationSetAuxiliaryTerms
)
from terraflow.discretization.fvm.odm.trisk.adv import flux_reconstruct as fr


if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


Float = typing.Float
Array = typing.Array
CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    OneLayerVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    OneLayerVertCoordLayout
]


class FluidThickness(AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]):

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
        self.recon_at_edge = fr.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: Block) -> Block:
        coord = block.coord
        g = block.para.gravity.unwrap()
        un = block.state.un.unwrap()
        gz = block.state.gz.unwrap()
        gzs = block.static.gzs.unwrap()
        dmg = (gz - gzs) / g
        block = eqx.tree_at(
            lambda b: b.aux.dmg.data,
            block,
            replace=dmg
        )
        dmg_e = self.recon_at_edge(
            x_at_primal_cell=dmg,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        block = eqx.tree_at(
            lambda b: b.aux.dmg_e.data,
            block,
            replace=dmg_e
        )
        dmg_vtx = op.interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: b.aux.dmg_vtx.data,
            block,
            replace=dmg_vtx
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_at_edge.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return self.recon_at_edge.prepare(block)


class DivergenceOfMassFluxAtCell(AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod
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

    def __call__(self, block: Block) -> Block:
        mfn = block.aux.mfn.unwrap()
        dmf = op.div_at_primal_cell(
            x_at_edge=mfn,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.aux.dmf.data,
            block,
            replace=dmf
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class AuxiliaryTerms(AbstractEquationSetAuxiliaryTerms[FiniteVolumeMethod]):

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

    def build_default(self) -> "AuxiliaryTerms":
        if not self.has_built():
            self.dmg.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.dmg(block)
        return block

    def __call__(self, block: Block) -> Block:
        return self.dmg(block)
