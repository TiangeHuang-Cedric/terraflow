from typing import Self
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh.operator import (
    div_at_primal_cell,
    grad_at_dual_edge
)
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import Field, ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.dynamics.baroclinic import types
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationTerm,
)
from terraflow.discretization.fvm.odm.trisk import pgf
from terraflow.discretization.fvm.odm.trisk.hybrid_lev import operator as hybrid_op
from terraflow.discretization.fvm.odm.trisk.hybrid_lev.aux import (
    PotentialTemperatureHorizontalAdvection,
    PotentialTemperatureVerticalAdvection
)
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block


Term = AbstractEquationTerm[FiniteVolumeMethod]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]


Float = typing.Float
Array = typing.Array


class TendencyOfDryAirMassAtSurface(
    types.DivergenceOfMassFluxAtSurfaceProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        mfn = block.aux.mfn.unwrap()
        dmgsdt = - jnp.sum(
            div_at_primal_cell(
                x_at_edge=mfn,
                coord=block.coord,
                para=block.para
            ),
            axis=0
        )
        return eqx.tree_at(
            lambda b: b.tend.dmgsdt.data,
            block,
            dmgsdt
        )

    def unwrap(self, block: Block) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dmgsdt

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class TendencyOfPotentialTemperatureDueToDivergenceOfFlux(
    types.DivergenceOfThermodynamicMassFluxProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.pthf = PotentialTemperatureHorizontalAdvection(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(
        self,
        block: Block
    ) -> Block:
        block = self.pthf(block)
        ptfn = block.adv_batch_pt.qmfn.unwrap()
        dmfn = div_at_primal_cell(
            x_at_edge=ptfn,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dptdt_dptfn,
            block,
            dmfn
        )

    def unwrap(self, block: Block) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dptdt_dptfn

    def build_default(self) -> Self:
        if not self.has_built():
            self.pthf.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.pthf.prepare(block)
        return block


class TendencyOfPotentialTemperatureDueToVerticalAdvection(
    types.VerticalAdvectionOfThermodynamicTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.ptvf = PotentialTemperatureVerticalAdvection(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(
        self,
        block: Block
    ) -> Block:
        block = self.ptvf(block)
        batch = block.adv_batch_pt
        adv_vert = hybrid_op.delta_at_full_lev(batch.qmfz.unwrap())
        return eqx.tree_at(
            lambda b: b.tend.dptdt_ptvf,
            block,
            adv_vert
        )

    def unwrap(self, block: Block) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dptdt_ptvf

    def build_default(self) -> Self:
        if not self.has_built():
            self.ptvf.build_default()
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        block = self.ptvf.prepare(block)
        return block


class TendencyOfNormalVelocityDueToVerticalAdvection(
    types.VerticalAdvectionOfMomentumTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.interp = hybrid_op.VerticalInterpMethod()

    def __call__(self, block: Block) -> Block:
        un = block.state.un.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        dmg = block.aux.dmg.unwrap()
        un_lev = self.interp.cell2lev(
            coord=block.coord,
            x=un,
            extrap_opt=True
        )
        mfz_e_lev = self.interp.cell2lev(
            coord=block.coord,
            x=mfz_lev,
            extrap_opt=True
        )
        dundt_wedundeta = - hybrid_op.adv_vert_at_full_lev(
            x=un,
            x_lev=un_lev,
            mfz_lev=mfz_e_lev,
            dmg=dmg
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_wedundeta,
            block,
            dundt_wedundeta
        )

    def unwrap(self, block: Block) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_wedundeta

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block


class TendencyOfNormalVelocityDueToPressureGradientForce(
    types.PressureGradientForceTermProtocol[FiniteVolumeMethod]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.pgf_scheme = pgf.schemes.PressureGradientForce(
            disc=disc,
            method=method,
            baroclinic=self.baroclinic,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def __call__(self, block: Block) -> Block:
        return self.pgf_scheme(block)

    def unwrap(self, block: Block) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_pgf

    def prepare(self, block: Block) -> Block:
        return self.pgf_scheme.prepare(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.pgf_scheme.build_default()
            self._built = True
        return self


class NonlinearCoriolisTermAtDualEdge(
    types.NonlinearCoriolisTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        coriolis = block.aux.coriolis.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dundt_coriolis.data,
            block,
            coriolis
        )

    def build_default(self) -> "NonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_coriolis


class GradientOfKineticEnergyAtDualEdge(
    types.GradientOfKineticEnergyProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        hydrostatic: bool,
        nonhydrostatic: bool,
        ideal_dry_core: bool,
        advection: bool
    ):
        super().__init__(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: Block) -> Block:
        ke = block.aux.ke.unwrap()
        grad_ke = - grad_at_dual_edge(
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

    def prepare(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block) -> Field:
        return block.tend.dundt_grad_ke
