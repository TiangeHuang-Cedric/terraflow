from typing import Self
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh.operator import (
    div_at_primal_cell,
    grad_at_dual_edge,
    interp_weighted_add_plg2dua
)
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import Field, ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.dynamics.baroclinic import types
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationOperator,
    AbstractPrepareSpaceOperators,
)
from terraflow.discretization.fvm.odm.trisk import base, pgf, coriolis, adv
from terraflow.discretization.fvm.odm.trisk.hybrid_lev import operator as hybrid_op
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block
from terraflow.dynamics.baroclinic.types import ExVifBaroclinicEquationSet4UGRIDSpaceOperators

FDT = ArrayField[StaggeredFieldMetaData]

Operator = AbstractEquationOperator[FiniteVolumeMethod]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

Float = typing.Float
Array = typing.Array

BLOCK = Block[HybridVertCoordinate, HybridVertCoordLayout]


class Temperature(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        pt = block.state.pt.unwrap()
        mg = block.aux.mg.unwrap()
        exner = hybrid_op.exner_at_full_lev(
            coord=block.coord,
            para=block.para,
            mg=mg
        )
        t = pt * exner
        return eqx.tree_at(
            lambda b: b.aux.t.data,
            block,
            t
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class VerticalMassFluxAtHalfLev(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmgdt_lev = block.coord.vert.calc_dmgdt_lev(dmgsdt)
        dmf = block.aux.dmf.unwrap()
        mfz_lev = - dmgdt_lev[1:-1, :] - jnp.cumsum(dmf[:-1, :], axis=0)
        mfz_lev = jnp.pad(mfz_lev, ((1, 1), (0, 0)))
        return eqx.tree_at(
            lambda b: b.aux.mfz_lev.data,
            block,
            mfz_lev
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class DryAirWeight(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        mgs = block.state.mgs.unwrap()
        mg_lev = block.coord.vert.calc_mg_lev(mgs)
        mg = 0.5 * (mg_lev[:-1, :] + mg_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.aux.mg_lev.data, b.aux.mg.data,),
            block,
            (mg_lev, mg)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class DeltaDryAirWeight(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.recon_at_edge = adv.flux_reconstruct.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: BLOCK) -> BLOCK:
        coord = block.coord
        un = block.state.un.unwrap()
        mg = block.aux.mg.unwrap()
        mg_lev = block.aux.mg_lev.unwrap()
        dmg = hybrid_op.delta_at_full_lev(mg_lev)
        dmg_lev = hybrid_op.delta_at_half_lev(x=mg, x_lev=mg_lev)
        dmg_e = self.recon_at_edge(
            x_at_primal_cell=dmg,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        dmg_vtx = interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            coord=coord,
        )
        return eqx.tree_at(
            lambda b: (
                b.aux.dmg.data, b.aux.dmg_lev.data,
                b.aux.dmg_e.data, b.aux.dmg_vtx.data
            ),
            block,
            replace=(dmg, dmg_lev, dmg_e, dmg_vtx)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_at_edge.build_default()
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.recon_at_edge.prepare_block(block)


class DensityOfDryAir(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        gz_lev = block.state.gz_lev.unwrap()
        dmg = block.aux.dmg.unwrap()
        dgz = hybrid_op.delta_at_full_lev(gz_lev)
        rhod = - dmg / dgz
        return eqx.tree_at(
            lambda b: b.aux.rhod.data,
            block,
            rhod
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class Geopotential(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        para = block.para
        gzs = block.static.gzs.unwrap()
        ph_lev = block.state.ph_lev.unwrap()
        t = block.aux.t.unwrap()
        rd = para.rd.unwrap()
        tmp = rd * t * jnp.log(ph_lev[1:, :] / ph_lev[:-1, :])
        gz_lev = gzs + jnp.cumsum(
            jnp.flip(tmp, axis=0),
            axis=0
        )
        gz_lev = jnp.concatenate(
            [gzs.reshape(1, -1), gz_lev],
            axis=0
        )
        gz_lev = jnp.flip(gz_lev, axis=0)
        gz = 0.5 * (gz_lev[:-1, :] + gz_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.state.gz_lev.data, b.state.gz.data),
            block,
            (gz_lev, gz)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class HydrostaticPressure(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        mg_lev = block.aux.mg_lev.unwrap()
        mgs = block.state.mgs.unwrap()
        dmg = block.aux.dmg.unwrap()
        phs = mgs
        ph_lev = mg_lev[0, :] + jnp.cumsum(dmg, axis=0)
        ph_lev = jnp.concatenate(
            [mg_lev[0, :].reshape(1, -1), ph_lev],
            axis=0
        )
        ph = 0.5 * (ph_lev[:-1, :] + ph_lev[1:, :])
        return eqx.tree_at(
            lambda b: (b.state.phs.data, b.state.ph.data, b.state.ph_lev.data),
            block,
            (phs, ph, ph_lev)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class Pressure(Operator):
    # nh
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        if self.hydrostatic:
            return eqx.tree_at(
                lambda b: (b.state.p.data, b.state.p_lev.data),
                block,
                (block.state.ph.data, block.state.ph_lev.data)
            )
        coord = block.coord
        para = block.para
        gz = block.state.gz.unwrap()
        pt = block.state.pt.unwrap()
        dmg = block.aux.dmg.unwrap()
        rd = para.rd.unwrap()
        cp = para.cp.unwrap()
        cv = para.cv.unwrap()
        gamma = cp / cv
        p0 = coord.vert.p0
        dgz = hybrid_op.delta_at_full_lev(gz)
        p = - rd * dmg * pt / (p0 * dgz)
        p = p ** gamma * p0
        return eqx.tree_at(
            lambda b: b.state.p.data,
            block,
            p
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class PotentialTemperatureHorizontalAdvection(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.adv_scheme = adv.schemes.HorizontalAdvectionScheme()

    def __call__(self, block: BLOCK) -> BLOCK:
        pt = block.state.pt.unwrap()
        batch = block.adv_batch_pt
        coord = block.coord
        aux = block.equaux.adv
        batch = self.adv_scheme(
            q=pt,
            batch=batch,
            coord=coord,
            aux=aux
        )
        return eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )

    def upwind(self, order: int = 3) -> adv.upwind.UpwindHorizontalAdvection:
        self.adv_scheme.upwind(order=order)
        return self.adv_scheme.scheme

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def unwrap(self, block: BLOCK) -> tuple[FDT, FDT]:
        return (block.adv_batch_pt.qmfn, block.adv_batch_pt.qmfz)


class PotentialTemperatureVerticalAdvection(Operator):
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
            baroclinic=True,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.adv_scheme = adv.schemes.VerticalAdvectionScheme()

    def __call__(self, block: BLOCK) -> BLOCK:
        pt = block.state.pt.unwrap()
        batch = block.adv_batch_pt
        batch = self.adv_scheme(q=pt, batch=batch)
        return eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )

    def upwind(self, order: int = 3) -> adv.upwind.UpwindVerticalAdvection:
        self.adv_scheme.upwind(order=order)
        return self.adv_scheme.scheme

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def unwrap(self, block: BLOCK) -> tuple[FDT, FDT]:
        return (block.adv_batch_pt.qmfn, block.adv_batch_pt.qmfz)


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

    def __call__(self, block: BLOCK) -> BLOCK:
        dmf = block.aux.dmf.unwrap()
        dmgsdt_dmf = - jnp.sum(dmf, axis=0)
        return eqx.tree_at(
            lambda b: b.tend.dmgsdt_dmf.data,
            block,
            dmgsdt_dmf
        )

    def unwrap(self, block: BLOCK) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dmgsdt_dmf

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
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
        self.mfz_lev = VerticalMassFluxAtHalfLev(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )
        self.adv_scheme = PotentialTemperatureHorizontalAdvection(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.mfz_lev(block)
        un = block.state.un.unwrap()
        ut = block.aux.ut.unwrap()
        mfn = block.aux.mfn.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        dmg = block.aux.dmg.unwrap()
        batch = block.adv_batch_pt
        batch = batch.set_wind(
            un=un,
            ut=ut,
            mfn=mfn,
            mfz=mfz_lev,
            m=dmg
        )
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        block = self.adv_scheme(block)
        ptfn = block.adv_batch_pt.qmfn.unwrap()
        dmfn = div_at_primal_cell(
            x_at_edge=ptfn,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dptdt_dptfn.data,
            block,
            dmfn
        )

    def unwrap(self, block: BLOCK) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dptdt_dptfn

    def build_default(self) -> Self:
        if not self.has_built():
            self.mfz_lev.build_default()
            self.adv_scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.mfz_lev.prepare_block(block)
        block = self.adv_scheme.prepare_block(block)
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
        self.adv_scheme = PotentialTemperatureVerticalAdvection(
            disc=disc,
            method=method,
            hydrostatic=hydrostatic,
            nonhydrostatic=nonhydrostatic,
            ideal_dry_core=ideal_dry_core,
            advection=advection
        )

    def __call__(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme(block)
        adv_vert = hybrid_op.delta_at_full_lev(
            x_lev=block.adv_batch_pt.qmfz.unwrap()
        )
        return eqx.tree_at(
            lambda b: b.tend.dptdt_ptvf.data,
            block,
            adv_vert
        )

    def unwrap(self, block: BLOCK) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dptdt_ptvf

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
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
        self.recon_at_edge = adv.flux_reconstruct.FluxReconstructAtEdgeOnPrimalMesh()
        self.vert_interp = hybrid_op.VerticalInterpMethod()

    def __call__(self, block: BLOCK) -> BLOCK:
        un = block.state.un.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        dmg_e = block.aux.dmg_e.unwrap()
        un_lev = self.vert_interp.cell2lev(
            coord=block.coord,
            x=un,
            extrap_opt=False
        )
        mfz_e_lev = self.recon_at_edge(
            x_at_primal_cell=mfz_lev,
            normal_u_at_edge=un_lev,
            coord=block.coord,
            aux=block.equaux.adv
        )
        dundt_wedundeta = - hybrid_op.adv_vert_at_full_lev(
            x=un,
            x_lev=un_lev,
            mfz_lev=mfz_e_lev,
            dmg=dmg_e
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_wedundeta.data,
            block,
            dundt_wedundeta
        )

    def unwrap(self, block: BLOCK) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_wedundeta

    def build_default(self) -> Self:
        if not self.has_built():
            self.recon_at_edge.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
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

    def __call__(self, block: BLOCK) -> BLOCK:
        return self.pgf_scheme(block)

    def unwrap(self, block: BLOCK) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_pgf

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.pgf_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.pgf_scheme.build_default()
            self._built = True
        return self


class TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge(
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

    def __call__(self, block: BLOCK) -> BLOCK:
        coriolis = block.aux.coriolis.unwrap()
        return eqx.tree_at(
            lambda b: b.tend.dundt_coriolis.data,
            block,
            coriolis
        )

    def build_default(self) -> "TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block

    def unwrap(self, block: BLOCK) -> Field:
        return block.tend.dundt_coriolis


class TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge(
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

    def __call__(self, block: BLOCK) -> BLOCK:
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

    def build_default(self) -> "TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block

    def unwrap(self, block: BLOCK) -> Field:
        return block.tend.dundt_grad_ke


class ExPrepareSpaceOperators(AbstractPrepareSpaceOperators):

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
        self.mg = DryAirWeight(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.dmg = DeltaDryAirWeight(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
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
        self.ph = HydrostaticPressure(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.p = Pressure(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.t = Temperature(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )
        self.gz = Geopotential(
            disc=disc,
            method=method,
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
        self.rhod = DensityOfDryAir(
            disc=disc,
            method=method,
            hydrostatic=self.hydrostatic,
            nonhydrostatic=self.nonhydrostatic,
            ideal_dry_core=self.ideal_dry_core,
            advection=self.advection
        )

    def build_default(self) -> "ExPrepareSpaceOperators":
        if not self.has_built():
            self.mg.build_default()
            self.dmg.build_default()
            self.mf.build_default()
            self.dmf.build_default()
            self.ph.build_default()
            self.p.build_default()
            self.t.build_default()
            self.gz.build_default()
            self.vor.build_default()
            self.coriolis.build_default()
            self.pv.build_default()
            self.ke.build_default()
            self.rhod.build_default()
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        block = self.mg.prepare_block(block)
        block = self.dmg.prepare_block(block)
        block = self.mf.prepare_block(block)
        block = self.dmf.prepare_block(block)
        block = self.ph.prepare_block(block)
        block = self.p.prepare_block(block)
        block = self.t.prepare_block(block)
        block = self.gz.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.coriolis.prepare_block(block)
        block = self.pv.prepare_block(block)
        block = self.ke.prepare_block(block)
        block = self.rhod.prepare_block(block)
        return block

    def __call__(
        self,
        block: BLOCK,
        pass_tag: str = "all_pass",
        explicit: bool = True
    ) -> BLOCK:
        # TODO: optimize by using forward-backward scheme
        block = self.mg(block)
        block = self.dmg(block)
        block = self.mf(block)
        block = self.dmf(block)
        block = self.ph(block)
        block = self.p(block)
        block = self.t(block)
        block = self.gz(block)
        block = self.coriolis.recon_ut(block)
        block = self.vor(block)
        block = self.pv(block)
        block = self.coriolis(block)
        block = self.ke(block)
        block = self.rhod(block)
        return block


class ExSpaceOperators(
    ExVifBaroclinicEquationSet4UGRIDSpaceOperators[
        TendencyOfDryAirMassAtSurface,
        TendencyOfPotentialTemperatureDueToDivergenceOfFlux,
        TendencyOfPotentialTemperatureDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge,
        TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge,
        TendencyOfNormalVelocityDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToPressureGradientForce,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators,
    ]
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

    def __call__(self, block: BLOCK, pass_tag: str, explicit: bool) -> BLOCK:
        block = self.prepare(
            block=block,
            pass_tag=pass_tag,
            explicit=explicit
        )
        if pass_tag == "all_pass":
            block = self.dmgsdt(block)
            block = self.dptdt(block)
            block = self.dundt(block)
        else:
            raise NotImplementedError
        return block
