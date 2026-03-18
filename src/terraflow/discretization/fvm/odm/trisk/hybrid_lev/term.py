from typing import Self, TypeVar
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
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractEquationOperator,
    AbstractPrepareSpaceOperators,
)
from terraflow.discretization.fvm.odm.trisk.common import (
    coriolis,
    pgf,
    adv,
    operators as common_op,
)
from terraflow.discretization.fvm.odm.trisk.hybrid_lev import operator as hybrid_op
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block, State
from terraflow.dynamics.baroclinic import types

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

Float = typing.Float
Array = typing.Array

BLOCK = Block[HybridVertCoordinate, HybridVertCoordLayout]

BEM = TypeVar("BEM", bound=types.AbstractBaroclinicEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, BEM]


class Temperature(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        pt = star_state.pt.unwrap()
        mg = star_state.mg.unwrap()
        exner = hybrid_op.exner_at_full_lev(
            coord=block.coord,
            para=block.para,
            mg=mg
        )
        t = pt * exner
        block = eqx.tree_at(
            lambda b: b.aux.t.data,
            block,
            t
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TangentMassFluxAtEdge(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        ut = block.aux.ut.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        mft = dmg_e * ut
        block = eqx.tree_at(
            lambda b: b.aux.mft.data,
            block,
            mft
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmgdt_lev = block.coord.vert.calc_dmgdt_lev(dmgsdt)
        dmf = block.aux.dmf.unwrap()
        mfz_lev = - dmgdt_lev[1:-1, :] - jnp.cumsum(dmf[:-1, :], axis=0)
        mfz_lev = jnp.pad(mfz_lev, ((1, 1), (0, 0)))
        block = eqx.tree_at(
            lambda b: b.aux.mfz_lev.data,
            block,
            mfz_lev
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mgs = star_state.mgs.unwrap()
        mg_lev = block.coord.vert.calc_mg_lev(mgs)
        mg = 0.5 * (mg_lev[:-1, :] + mg_lev[1:, :])
        star_state = eqx.tree_at(
            lambda s: (s.mg.data, s.mg_lev.data),
            star_state,
            (mg, mg_lev)
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.interp = hybrid_op.InterpMethod()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        coord = block.coord
        mg = star_state.mg.unwrap()
        mg_lev = star_state.mg_lev.unwrap()
        dmg = hybrid_op.delta_at_full_lev(mg_lev)
        dmg_lev = hybrid_op.delta_at_half_lev(x=mg, x_lev=mg_lev)
        dmg_e = self.interp.cell2edge(
            x=dmg,
            coord=coord
        )
        dmg_vtx = interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            coord=coord,
        )
        star_state = eqx.tree_at(
            lambda s: (
                s.dmg.data,
                s.dmg_lev.data,
                s.dmg_e.data,
                s.dmg_vtx.data
            ),
            star_state,
            (dmg, dmg_lev, dmg_e, dmg_vtx)
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class DensityOfDryAir(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        gz_lev = star_state.gz_lev.unwrap()
        dmg = star_state.dmg.unwrap()
        dgz = hybrid_op.delta_at_full_lev(gz_lev)
        rhod = - dmg / dgz
        block = eqx.tree_at(
            lambda b: b.aux.rhod.data,
            block,
            rhod
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        para = block.para
        gzs = block.static.gzs.unwrap()
        ph_lev = star_state.ph_lev.unwrap()
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
        star_state = eqx.tree_at(
            lambda s: (s.gz.data, s.gz_lev.data),
            star_state,
            (gz, gz_lev)
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mg_lev = star_state.mg_lev.unwrap()
        dmg = star_state.dmg.unwrap()
        # TODO: For ideal dry core, need to add mixing ratio effect
        ph_lev = mg_lev[0, :] + jnp.cumsum(dmg, axis=0)
        ph_lev = jnp.concatenate(
            [mg_lev[0, :].reshape(1, -1), ph_lev],
            axis=0
        )
        ph = 0.5 * (ph_lev[:-1, :] + ph_lev[1:, :])
        phs = ph_lev[-1, :]
        star_state = eqx.tree_at(
            lambda s: (s.phs.data, s.ph.data, s.ph_lev.data),
            star_state,
            (phs, ph, ph_lev)
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        if self.mode.hydrostatic:
            star_state = eqx.tree_at(
                lambda s: (s.p.data, s.p_lev.data),
                star_state,
                (star_state.ph.data, star_state.ph_lev.data)
            )
            return block, star_state
        coord = block.coord
        para = block.para
        gz = star_state.gz.unwrap()
        pt = star_state.pt.unwrap()
        dmg = star_state.dmg.unwrap()
        rd = para.rd.unwrap()
        cp = para.cp.unwrap()
        cv = para.cv.unwrap()
        gamma = cp / cv
        p0 = coord.vert.p0
        dgz = hybrid_op.delta_at_full_lev(gz)
        p = - rd * dmg * pt / (p0 * dgz)
        p = p ** gamma * p0
        # TODO: Need to add p_lev calculation for nonhydrostatic case
        star_state = eqx.tree_at(
            lambda s: (s.p.data,),
            star_state,
            (p,)
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = adv.schemes.HorizontalAdvectionScheme()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        pt = star_state.pt.unwrap()
        batch = block.adv_batch_pt
        coord = block.coord
        aux = block.equaux.adv
        batch = self.adv_scheme(
            q=pt,
            batch=batch,
            coord=coord,
            aux=aux
        )
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        return block, star_state

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
        mode: types.AbstractBaroclinicEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = adv.schemes.VerticalAdvectionScheme()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        pt = star_state.pt.unwrap()
        batch = block.adv_batch_pt
        batch = self.adv_scheme(q=pt, batch=batch)
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        return block, star_state

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
    types.DivergenceOfMassFluxAtSurfaceProtocol[FiniteVolumeMethod, BEM]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        dmf = block.aux.dmf.unwrap()
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmgsdt_dmf = - jnp.sum(dmf, axis=0)
        dmgsdt = dmgsdt + dmgsdt_dmf
        block = eqx.tree_at(
            lambda b: (b.tend.dmgsdt.data, b.tend.dmgsdt_dmf.data),
            block,
            (dmgsdt, dmgsdt_dmf)
        )
        return block, star_state

    def unwrap(self, block: BLOCK, itime: int) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dmgsdt_dmf

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfPotentialTemperatureDueToDivergenceOfFlux(
    types.DivergenceOfThermodynamicMassFluxProtocol[FiniteVolumeMethod, BEM]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = PotentialTemperatureHorizontalAdvection(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        un = star_state.un.unwrap()
        dmg = star_state.dmg.unwrap()
        ut = block.aux.ut.unwrap()
        mfn = block.aux.mfn.unwrap()
        mft = block.aux.mft.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        batch = block.adv_batch_pt
        batch = batch.set_wind(
            un=un,
            ut=ut,
            mfn=mfn,
            mft=mft,
            mfz=mfz_lev,
            m=dmg
        )
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        block, star_state = self.adv_scheme(block, star_state)
        ptfn = block.adv_batch_pt.qmfn.unwrap()
        dptdt_dptfn = - div_at_primal_cell(
            x_at_edge=ptfn,
            coord=block.coord,
            para=block.para
        )
        dptdt = dptdt + dptdt_dptfn
        block = eqx.tree_at(
            lambda b: (b.tend.dptdt.data, b.tend.dptdt_dptfn.data),
            block,
            (dptdt, dptdt_dptfn)
        )
        return block, star_state

    def unwrap(self, block: BLOCK, itime: int) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dptdt_dptfn

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
        return block


class TendencyOfPotentialTemperatureDueToVerticalAdvection(
    types.VerticalAdvectionOfThermodynamicTermProtocol[FiniteVolumeMethod, BEM]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = PotentialTemperatureVerticalAdvection(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        dptdt = block.tend.dptdt.unwrap()
        block, star_state = self.adv_scheme(block, star_state)
        dptdt_ptvf = - hybrid_op.delta_at_full_lev(
            x_lev=block.adv_batch_pt.qmfz.unwrap()
        )
        dptdt = dptdt + dptdt_ptvf
        block = eqx.tree_at(
            lambda b: (b.tend.dptdt.data, b.tend.dptdt_ptvf.data),
            block,
            (dptdt, dptdt_ptvf)
        )
        return block, star_state

    def unwrap(self, block: BLOCK, itime: int) -> ArrayField[StaggeredFieldMetaData]:
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
    types.VerticalAdvectionOfMomentumTermProtocol[FiniteVolumeMethod, BEM]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.interp = hybrid_op.InterpMethod()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        un = star_state.un.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        dundt = block.tend.dundt.unwrap()
        un_lev = self.interp.edge2lev(
            coord=block.coord,
            x=un,
            extrap_opt=False
        )
        mfz_e_lev = self.interp.lev2edge(
            x=mfz_lev,
            coord=block.coord,
        )
        dundt_wedundeta = - hybrid_op.adv_vert_at_full_lev(
            x=un,
            x_lev=un_lev,
            mfz_lev=mfz_e_lev,
            dmg=dmg_e
        )
        dundt = dundt + dundt_wedundeta
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_wedundeta.data),
            block,
            (dundt, dundt_wedundeta)
        )
        return block, star_state

    def unwrap(self, block: BLOCK, itime: int) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_wedundeta

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfNormalVelocityDueToPressureGradientForce(
    types.PressureGradientForceTermProtocol[FiniteVolumeMethod, BEM]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.pgf_scheme = pgf.schemes.PressureGradientForce(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        block, star_state = self.pgf_scheme(block, star_state=star_state)
        dundt = block.tend.dundt.unwrap()
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt + dundt_pgf
        block = eqx.tree_at(
            lambda b: b.tend.dundt.data,
            block,
            dundt
        )
        return block, star_state

    def unwrap(self, block: BLOCK, itime: int) -> ArrayField[StaggeredFieldMetaData]:
        return block.tend.dundt_pgf

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.pgf_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.pgf_scheme.build_default()
            self._built = True
        return self


class TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge(
    types.NonlinearCoriolisTermProtocol[FiniteVolumeMethod, BEM]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        coriolis = block.aux.coriolis.unwrap()
        dundt = block.tend.dundt.unwrap()
        dundt_coriolis = coriolis
        dundt = dundt + dundt_coriolis
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_coriolis.data),
            block,
            (dundt, dundt_coriolis)
        )
        return block, star_state

    def build_default(self) -> "TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block

    def unwrap(self, block: BLOCK, itime: int) -> Field:
        return block.tend.dundt_coriolis


class TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge(
    types.GradientOfKineticEnergyProtocol[FiniteVolumeMethod, BEM]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        ke = block.aux.ke.unwrap()
        dundt = block.tend.dundt.unwrap()
        dundt_grad_ke = - grad_at_dual_edge(
            x_at_primal_cell=ke,
            coord=block.coord,
            para=block.para
        )
        dundt = dundt + dundt_grad_ke
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_grad_ke.data),
            block,
            (dundt, dundt_grad_ke)
        )
        return block, star_state

    def build_default(self) -> "TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block

    def unwrap(self, block: BLOCK, itime: int) -> Field:
        return block.tend.dundt_grad_ke


class ExPrepareSpaceOperators(
    AbstractPrepareSpaceOperators[
        FiniteVolumeMethod,
        BEM
    ]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.mg = DryAirWeight(
            disc=disc, method=method, mode=mode
        )
        self.dmg = DeltaDryAirWeight(
            disc=disc, method=method, mode=mode
        )
        self.mfn = common_op.NormalMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.mft = TangentMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.dmf = common_op.DivergenceOfMassFluxAtCell(
            disc=disc, method=method, mode=mode
        )
        self.ph = HydrostaticPressure(
            disc=disc, method=method, mode=mode
        )
        self.p = Pressure(
            disc=disc, method=method, mode=mode
        )
        self.t = Temperature(
            disc=disc, method=method, mode=mode
        )
        self.gz = Geopotential(
            disc=disc, method=method, mode=mode
        )
        self.coriolis = coriolis.schemes.CoriolisForceAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.vor = common_op.RelativeVorticityAtDualCell(
            disc=disc, method=method, mode=mode
        )
        self.pv = common_op.PotentialVorticity(
            disc=disc, method=method, mode=mode
        )
        self.ke = common_op.KineticEnergyAtPrimalCell(
            disc=disc, method=method, mode=mode
        )
        self.rhod = DensityOfDryAir(
            disc=disc, method=method, mode=mode
        )

    def first_call(self, block: BLOCK) -> BLOCK:
        old_state = block.get_state(itime=0)
        block, old_state = self.mg(block, old_state)
        block, old_state = self.dmg(block, old_state)
        block, old_state = self.mfn(block, old_state)
        block, old_state = self.dmf(block, old_state)
        block, old_state = self.ph(block, old_state)
        block, old_state = self.p(block, old_state)
        block, old_state = self.t(block, old_state)
        block, old_state = self.gz(block, old_state)
        block, old_state = self.coriolis.recon_ut(block, old_state)
        block, old_state = self.mft(block, old_state)
        block, old_state = self.vor(block, old_state)
        block, old_state = self.pv(block, old_state)
        block, old_state = self.coriolis(block, old_state)
        block, old_state = self.ke(block, old_state)
        block, old_state = self.rhod(block, old_state)
        block = block.set_state(itime=0, state=old_state)
        return block

    def __call__(
        self,
        block: BLOCK,
        star_state: State,
        pass_tag: str,
        explicit: bool
    ) -> tuple[BLOCK, State]:
        # TODO: optimize by using forward-backward scheme
        if pass_tag == "all_pass":
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.ph(block, star_state)
            block, star_state = self.p(block, star_state)
            block, star_state = self.t(block, star_state)
            block, star_state = self.gz(block, star_state)
            block, star_state = self.coriolis.recon_ut(block, star_state)
            block, star_state = self.mft(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.coriolis(block, star_state)
            block, star_state = self.ke(block, star_state)
            block, star_state = self.rhod(block, star_state)
        elif pass_tag == "forward_pass":
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.mft(block, star_state)
            block, star_state = self.coriolis.recon_ut(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.coriolis(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "backward_pass":
            block, star_state = self.ph(block, star_state)
            block, star_state = self.p(block, star_state)
            block, star_state = self.t(block, star_state)
            block, star_state = self.gz(block, star_state)
            block, star_state = self.rhod(block, star_state)
        return block, star_state

    def build_default(self) -> "ExPrepareSpaceOperators":
        if not self.has_built():
            self.mg.build_default()
            self.dmg.build_default()
            self.mfn.build_default()
            self.mft.build_default()
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
        block = self.mfn.prepare_block(block)
        block = self.mft.prepare_block(block)
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


class ExSpaceOperators(
    types.ExVifBaroclinicEquationSet4UGRIDSpaceOperators[
        TendencyOfDryAirMassAtSurface,
        TendencyOfPotentialTemperatureDueToDivergenceOfFlux,
        TendencyOfPotentialTemperatureDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge,
        TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge,
        TendencyOfNormalVelocityDueToVerticalAdvection,
        TendencyOfNormalVelocityDueToPressureGradientForce,
        FiniteVolumeMethod,
        BEM,
        ExPrepareSpaceOperators,
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: BEM,
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.mfz_lev = VerticalMassFluxAtHalfLev(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        old_idx: int,
        star_idx: int,
        new_idx: int,
        pass_tag: str,
        explicit: bool
    ) -> BLOCK:
        block = block.reset(pass_tag=pass_tag)
        if pass_tag == "all_pass":
            # ==================================================
            # Get state at star idx
            star_state = block.get_state(itime=star_idx)
            # ==================================================
            # Prepare state and block for tendency calculation
            block, star_state = self.prepare(
                block=block,
                star_state=star_state,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block, star_state = self.dmgsdt(block, star_state)
            block, star_state = self.mfz_lev(block, star_state)
            block, star_state = self.dptdt(block, star_state)
            block, star_state = self.dundt(block, star_state)
            # ==================================================
            # Update state at star idx
            block = block.set_state(itime=star_idx, state=star_state)
            # ==================================================
        elif pass_tag == "forward_pass":
            # =================================================
            # Get state at star idx
            star_state = block.get_state(itime=star_idx)
            # ==================================================
            # Prepare state and block for tendency calculation
            block, star_state = self.prepare(
                block=block,
                star_state=star_state,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block, star_state = self.dmgsdt(block, star_state)
            block, star_state = self.mfz_lev(block, star_state)
            block, star_state = self.dptdt(block, star_state)
            # ==================================================
            # Update state at star idx
            block = block.set_state(itime=star_idx, state=star_state)
            # ==================================================
        elif pass_tag == "backward_pass":
            # =================================================
            # Get state at star idx and new idx
            star_state = block.get_state(itime=star_idx)
            new_state = block.get_state(itime=new_idx)
            # ==================================================
            # Prepare state and block for tendency calculation
            block, new_state = self.prepare(
                block=block,
                star_state=new_state,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block, star_state = self.dundt.grad_ke(block, star_state)
            block, new_state = self.dundt.pgf(block, new_state)
            block, star_state = self.dundt.coriolis(block, star_state)
            block, star_state = self.dundt.adv_vert(block, star_state)
            # ==================================================
            # Update state at star idx and new idx
            block = block.set_state(itime=star_idx, state=star_state)
            block = block.set_state(itime=new_idx, state=new_state)
            # ==================================================
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )
        return block

    def build_default(self) -> Self:
        if not self.has_built():
            self.mfz_lev.build_default()
            super().build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.mfz_lev.prepare_block(block)
        block = super().prepare_block(block)
        return block

    def _update_mgs(
        self,
        block: Block,
        old_state: State,
        new_state: State,
        dt: float,
        pass_tag: str
    ) -> State:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            old_mgs = old_state.mgs.unwrap()
            dmgsdt = block.tend.dmgsdt.unwrap()
            # Update mgs
            new_mgs = old_mgs + dmgsdt * dt
            # Wrap to new state and update dmg
            new_state = eqx.tree_at(
                lambda s: s.mgs.data,
                new_state,
                new_mgs
            )
            # Update dmg
            block, new_state = self.prepare.mg(block, new_state)
            block, new_state = self.prepare.dmg(block, new_state)
            return new_state
        elif pass_tag == "backward_pass":
            return new_state
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def _update_pt(
        self,
        block: Block,
        old_state: State,
        new_state: State,
        dt: float,
        pass_tag: str
    ) -> State:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            old_pt = old_state.pt.unwrap()
            old_dmg = old_state.dmg.unwrap()
            new_dmg = new_state.dmg.unwrap()
            dptdt = block.tend.dptdt.unwrap()
            new_pt = (old_pt * old_dmg + dptdt * dt)/new_dmg
            new_state = eqx.tree_at(
                lambda s: s.pt.data,
                new_state,
                new_pt
            )
            return new_state
        elif pass_tag == "backward_pass":
            return new_state
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def _update_un(
        self,
        block: Block,
        old_state: State,
        new_state: State,
        dt: float,
        pass_tag: str
    ) -> State:
        if pass_tag == "all_pass" or pass_tag == "backward_pass":
            old_un = old_state.un.unwrap()
            dundt = block.tend.dundt.unwrap()
            new_un = old_un + dundt * dt
            new_state = eqx.tree_at(
                lambda s: s.un.data,
                new_state,
                new_un
            )
            return new_state
        elif pass_tag == "forward_pass":
            return new_state
        else:
            raise ValueError(
                f"Unsupported pass tag {pass_tag}!"
            )

    def update_state(
        self,
        block: BLOCK,
        old_idx: int,
        new_idx: int,
        dt: float,
        pass_tag: str,
    ) -> State:
        old_state = block.get_state(old_idx)
        new_state = block.get_state(new_idx)
        # ================================================
        # Update mgs
        new_state = self._update_mgs(
            block=block,
            old_state=old_state,
            new_state=new_state,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        # Update pt
        new_state = self._update_pt(
            block=block,
            old_state=old_state,
            new_state=new_state,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        # Update un
        new_state = self._update_un(
            block=block,
            old_state=old_state,
            new_state=new_state,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        return new_state
