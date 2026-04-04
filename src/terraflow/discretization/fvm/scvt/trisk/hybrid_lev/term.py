from __future__ import annotations
from typing import TYPE_CHECKING, Self, TypeVar
import equinox as eqx
import jax.numpy as jnp

from terraflow.util import typing, formula
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt.operators import (
    div_at_primal_cell,
    grad_at_dual_edge,
    interp_weighted_add_plg2dua
)
from terraflow.discretization.abstract import (
    AbstractEquationOperator,
    AbstractPrepareSpaceOperators,
)
from terraflow.dynamics.baroclinic import types
from . import operators as hybrid_op
from ..parallel import tree_at

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.common.coriolis.schemes import CoriolisForceAtEdge
    from terraflow.discretization.fvm.scvt.trisk.common.adv.upwind import (
        UpwindHorizontalAdvectionAtCell,
        UpwindVerticalAdvection
    )


Float = typing.Float
Array = typing.Array


BEM = TypeVar("BEM", bound=types.AbstractBaroclinicEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, BEM]

if TYPE_CHECKING:
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo
    from ..dynamics_types import Block, State
    BLOCK = Block[HybridVertCoordinate, HybridVertCoordLayout]


class Temperature(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AbstractBaroclinicEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        pt = star_state.pt.unwrap()
        p = star_state.p.unwrap()
        t = block.aux.t.unwrap()
        tv = block.aux.tv.unwrap()
        # ==========================================================
        t = t.at[full_kds:full_kde+1, ids:ihe+1].set(
            formula.temperature(
                pt=pt[full_kds:full_kde+1, ids:ihe+1],
                p=p[full_kds:full_kde+1, ids:ihe+1],
                qv=jnp.asarray(0.0, dtype=pt.dtype),
                para=block.para
            )
        )
        # TODO: For ideal dry core, need to add mixing ratio effect
        tv = tv.at[full_kds:full_kde+1, ids:ihe+1].set(
            t[full_kds:full_kde+1, ids:ihe+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.aux.t, b.aux.tv),
            pytree=block,
            replace=(t, tv)
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ihs, ihe = mesh.edge_ihs[0], mesh.edge_ihe[0]
        kds, kde = vert.half_kds, vert.half_kde
        ut = block.aux.ut.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        mft = block.aux.mft.unwrap()
        # ==========================================================
        mft = mft.at[kds:kde+1, ihs:ihe+1].set(
            dmg_e[kds:kde+1, ihs:ihe+1] * ut[kds:kde+1, ihs:ihe+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.aux.mft,
            pytree=block,
            replace=mft
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
        half_kds = block.coord.vert.half_kds
        half_kde = block.coord.vert.half_kde
        polg_ids = block.coord.mesh.polg_ids
        polg_ide = block.coord.mesh.polg_ide
        dmgsdt = block.tend.dmgsdt.unwrap()
        dmf = block.aux.dmf.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        # ==========================================================
        dmgdt_lev = block.coord.vert.calc_dmgdt_lev(
            dmgsdt[polg_ids:polg_ide+1],
            ks=half_kds + 1,
            ke=half_kde - 1
        )
        mfz_lev = mfz_lev.at[half_kds+1:half_kde].set(
            - dmgdt_lev - jnp.cumsum(
                dmf[:half_kde-1, polg_ids:polg_ide+1],
                axis=0
            )
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.aux.mfz_lev,
            pytree=block,
            replace=mfz_lev,
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        polg_ids = mesh.polg_ids
        polg_ihe = mesh.polg_ihe[0]
        full_kds = vert.full_kds
        full_kde = vert.full_kde
        half_kds = vert.half_kds
        half_kde = vert.half_kde
        mgs = star_state.mgs.unwrap()
        mg_lev = star_state.mg_lev.unwrap()
        mg = star_state.mg.unwrap()
        # ==========================================================
        mg_lev = mg_lev.at[half_kds:half_kde+1, polg_ids:polg_ihe+1].set(
            block.coord.vert.calc_mg_lev(
                mgs[polg_ids:polg_ihe+1],
                ks=half_kds,
                ke=half_kde
            )
        )
        mg = mg.at[full_kds:full_kde+1, polg_ids:polg_ihe+1].set(
            0.5 * (
                mg_lev[full_kds:full_kde+1, polg_ids:polg_ihe+1]
                + mg_lev[full_kds+1:full_kde+2, polg_ids:polg_ihe+1]
            )
        )
        # ==========================================================
        star_state = tree_at(
            coord=block.coord,
            where=lambda s: (s.mg, s.mg_lev),
            pytree=star_state,
            replace=(mg, mg_lev)
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
        dmg = star_state.dmg.unwrap()
        dmg_lev = star_state.dmg_lev.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        dmg_vtx = star_state.dmg_vtx.unwrap()
        # ==========================================================
        dmg = hybrid_op.delta_at_full_lev(
            coord=coord,
            x_lev=mg_lev,
            dx=dmg,
        )
        dmg_lev = hybrid_op.delta_at_half_lev(
            coord=coord,
            x=mg,
            x_lev=mg_lev,
            dx_lev=dmg_lev
        )
        dmg_e = self.interp.cell2edge(
            x=dmg,
            y=dmg_e,
            coord=coord
        )
        dmg_vtx = interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,
            x_at_dual_cell=dmg_vtx,
            coord=coord,
        )
        # ==========================================================
        star_state = tree_at(
            coord=coord,
            where=lambda s: (s.dmg, s.dmg_lev, s.dmg_e, s.dmg_vtx),
            pytree=star_state,
            replace=(dmg, dmg_lev, dmg_e, dmg_vtx)
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        kds, kde = vert.full_kds, vert.full_kde
        gz_lev = star_state.gz_lev.unwrap()
        dmg = star_state.dmg.unwrap()
        rhod = block.aux.rhod.unwrap()
        # ==========================================================
        dgz = hybrid_op.delta_at_full_lev(
            coord=block.coord,
            x_lev=gz_lev,
            hori_entity="polg",
            hori_end_at="halo0",
        )
        rhod = rhod.at[kds:kde+1, ids:ihe+1].set(
            - dmg[kds:kde+1, ids:ihe+1] / dgz
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.aux.rhod,
            pytree=block,
            replace=rhod
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        para = block.para
        gzs = block.static.gzs.unwrap()
        ph_lev = star_state.ph_lev.unwrap()
        tv = block.aux.tv.unwrap()
        rd = para.rd.unwrap()
        gz_lev = star_state.gz_lev.unwrap()
        gz = star_state.gz.unwrap()
        # ====================================================
        # Calculat gz_lev
        kds, kde = half_kds, half_kde
        # =======================================
        # From half_kde to half_kds
        tmp = rd * tv * jnp.log(
            ph_lev[kds+1:kde+1, ids:ihe+1] / ph_lev[kds:kde, ids:ihe+1]
        )
        _gz_lev = gzs + jnp.cumulative_sum(
            jnp.flip(tmp, axis=0),
            axis=0,
            include_initial=True
        )
        # =======================================
        # Flip back, from half_kds to half_kde
        _gz_lev = jnp.flip(_gz_lev, axis=0)
        # =======================================
        # Wrap Back
        gz_lev = gz_lev.at[kds:kde+1, ids:ihe+1].set(_gz_lev)
        # ===================================================
        kds, kde = full_kds, full_kde
        # Calculate gz
        gz = gz.at[kds:kde+1, ids:ihe+1].set(
            0.5 * (
                gz_lev[kds:kde+1, ids:ihe+1]
                + gz_lev[kds+1:kde+2, ids:ihe+1]
            )
        )
        # ===================================================
        star_state = tree_at(
            coord=block.coord,
            where=lambda s: (s.gz, s.gz_lev),
            pytree=star_state,
            replace=(gz, gz_lev)
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        mg_lev = star_state.mg_lev.unwrap()
        dmg = star_state.dmg.unwrap()
        ph_lev = star_state.ph_lev.unwrap()
        ph = star_state.ph.unwrap()
        phs = star_state.phs.unwrap()
        # TODO: For ideal dry core, need to add mixing ratio effect
        # =======================================================
        # Calculate ph at half level
        kds, kde = half_kds, half_kde
        ph_lev = ph_lev.at[kds:kde+1, ids:ihe+1].set(
            mg_lev[kds, :] + jnp.cumulative_sum(
                dmg[kds:kde+1, ids:ihe+1],
                axis=0,
                include_initial=True
            )
        )
        # =======================================================
        # Calculate ph at full level
        kds, kde = full_kds, full_kde
        ph = ph.at[kds:kde+1, ids:ihe+1].set(
            0.5 * (
                ph_lev[kds:kde+1, ids:ihe+1]
                + ph_lev[kds+1:kde+2, ids:ihe+1]
            )
        )
        # =======================================================
        # Calculate phs
        kde = half_kde
        phs = phs.at[ids:ihe+1].set(ph_lev[kde, ids:ihe+1])
        # =======================================================
        star_state = tree_at(
            coord=block.coord,
            where=lambda s: (s.phs, s.ph, s.ph_lev),
            pytree=star_state,
            replace=(phs, ph, ph_lev)
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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ihe = mesh.polg_ids, mesh.polg_ihe[0]
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        ph = star_state.ph.unwrap()
        ph_lev = star_state.ph_lev.unwrap()
        p = star_state.p.unwrap()
        p_lev = star_state.p_lev.unwrap()
        # =======================================================
        if self.mode.hydrostatic:
            p_lev = p_lev.at[half_kds:half_kde+1, ids:ihe+1].set(
                ph_lev[half_kds:half_kde+1, ids:ihe+1]
            )
            p = p.at[full_kds:full_kde+1, ids:ihe+1].set(
                ph[full_kds:full_kde+1, ids:ihe+1]
            )
        else:
            raise NotImplementedError
        # =======================================================
        star_state = tree_at(
            coord=block.coord,
            where=lambda s: (s.p, s.p_lev),
            pytree=star_state,
            replace=(p, p_lev)
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
        from ..common.adv.schemes import HorizontalAdvectionSchemeAtCell
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        pt = star_state.pt.unwrap()
        batch = block.adv_batch_pt
        coord = block.coord
        aux = block.equaux.adv
        # ==========================================================
        batch = self.adv_scheme(
            q=pt,
            batch=batch,
            coord=coord,
            aux=aux
        )
        # ==========================================================
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        return block, star_state

    def upwind(self, order: int = 3, optimized: bool = True) -> UpwindHorizontalAdvectionAtCell:
        self.adv_scheme.upwind(order=order, optimized=optimized)
        return self.adv_scheme.scheme

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self


class PotentialTemperatureVerticalAdvection(Operator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AbstractBaroclinicEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.adv.schemes import VerticalAdvectionScheme
        self.adv_scheme = VerticalAdvectionScheme()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        pt = star_state.pt.unwrap()
        batch = block.adv_batch_pt
        # ==========================================================
        batch = self.adv_scheme(q=pt, batch=batch, coord=block.coord)
        # ==========================================================
        block = eqx.tree_at(
            lambda b: b.adv_batch_pt,
            block,
            batch
        )
        return block, star_state

    def upwind(self, order: int = 3) -> UpwindVerticalAdvection:
        self.adv_scheme.upwind(order=order)
        return self.adv_scheme.scheme

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self


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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        dmf = block.aux.dmf.unwrap()
        dmgsdt_dmf = block.tend.dmgsdt_dmf.unwrap()
        dmgsdt = block.tend.dmgsdt.unwrap()
        # ==========================================================
        dmgsdt_dmf = dmgsdt_dmf.at[ids:ide+1].set(
            - jnp.sum(dmf[kds:kde+1, ids:ide+1], axis=0)
        )
        dmgsdt = dmgsdt.at[ids:ide+1].set(
            dmgsdt[ids:ide+1] + dmgsdt_dmf[ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dmgsdt, b.tend.dmgsdt_dmf),
            pytree=block,
            replace=(dmgsdt, dmgsdt_dmf)
        )
        return block, star_state

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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = star_state.un.unwrap()
        dmg = star_state.dmg.unwrap()
        ut = block.aux.ut.unwrap()
        mfn = block.aux.mfn.unwrap()
        mft = block.aux.mft.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        batch = block.adv_batch_pt
        dptdt_dptfn = block.tend.dptdt_dptfn.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        # ==========================================================
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
        dptdt_dptfn = dptdt_dptfn.at[kds:kde+1, ids:ide+1].set(
            - div_at_primal_cell(
                x_at_primal_edge=ptfn,
                coord=block.coord,
                para=block.para
            )
        )
        dptdt = dptdt.at[kds:kde+1, ids:ide+1].set(
            dptdt[kds:kde+1, ids:ide+1] + dptdt_dptfn[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dptdt, b.tend.dptdt_dptfn),
            pytree=block,
            replace=(dptdt, dptdt_dptfn)
        )
        return block, star_state

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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        dptdt_ptvf = block.tend.dptdt_ptvf.unwrap()
        dptdt = block.tend.dptdt.unwrap()
        # ==========================================================
        block, star_state = self.adv_scheme(block, star_state)
        dptdt_ptvf = dptdt_ptvf.at[kds:kde+1, ids:ide+1].set(
            - hybrid_op.delta_at_full_lev(
                coord=block.coord,
                x_lev=block.adv_batch_pt.qmfz.unwrap()
            )
        )
        dptdt = dptdt.at[kds:kde+1, ids:ide+1].set(
            dptdt[kds:kde+1, ids:ide+1] + dptdt_ptvf[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dptdt, b.tend.dptdt_ptvf),
            pytree=block,
            replace=(dptdt, dptdt_ptvf)
        )
        return block, star_state

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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        un = star_state.un.unwrap()
        un_lev = block.aux.un_lev.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        mfz_lev = block.aux.mfz_lev.unwrap()
        mfz_e_lev = block.aux.mfz_e_lev.unwrap()
        dundt_wedundeta = block.tend.dundt_wedundeta.unwrap()
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        un_lev = self.interp.edge2lev(
            coord=block.coord,
            x=un,
            y=un_lev,
            extrap_opt=False
        )
        mfz_e_lev = self.interp.lev2edge(
            x=mfz_lev,
            y=mfz_e_lev,
            coord=block.coord,
        )
        dundt_wedundeta = dundt_wedundeta.at[kds:kde+1, ids:ide+1].set(
            - hybrid_op.adv_vert_at_full_lev(
                coord=block.coord,
                x=un,
                x_lev=un_lev,
                mfz_lev=mfz_e_lev,
                dmg=dmg_e,
                hori_entity="edge",
            )
        )
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_wedundeta[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_wedundeta),
            pytree=block,
            replace=(dundt, dundt_wedundeta)
        )
        return block, star_state

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
        from ..common.pgf.schemes import PressureGradientForce
        self.pgf_scheme = PressureGradientForce(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block, star_state = self.pgf_scheme(block, star_state=star_state)
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_pgf[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )
        return block, star_state

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
        from ..common.coriolis.schemes import CoriolisForceAtEdge
        self.scheme = CoriolisForceAtEdge(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block, star_state = self.scheme(block, star_state)
        dundt_coriolis = block.tend.dundt_coriolis.unwrap()
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_coriolis[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_coriolis),
            pytree=block,
            replace=(dundt, dundt_coriolis)
        )
        return block, star_state

    def build_default(self) -> "TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge":
        if not self.has_built():
            self.scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.scheme.prepare_block(block)


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
        mesh = block.coord.mesh
        vert = block.coord.vert
        ids, ide = mesh.edge_ids, mesh.edge_ide
        kds, kde = vert.full_kds, vert.full_kde
        ke = block.aux.ke.unwrap()
        dundt_grad_ke = block.tend.dundt_grad_ke.unwrap()
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        dundt_grad_ke = dundt_grad_ke.at[kds:kde+1, ids:ide+1].set(
            - grad_at_dual_edge(
                x_at_primal_cell=ke,
                coord=block.coord,
                para=block.para
            )
        )
        dundt = dundt.at[kds:kde+1, ids:ide+1].set(
            dundt[kds:kde+1, ids:ide+1] + dundt_grad_ke[kds:kde+1, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_grad_ke),
            pytree=block,
            replace=(dundt, dundt_grad_ke)
        )
        return block, star_state

    def build_default(self) -> "TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge":
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


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
        from ..common import operators as common_op
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
        self.ut = common_op.TangentVelocityAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.div = common_op.DivergenceOfVelocityAtCell(
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
        old_state = block.get_state(itime=block.old)
        block, old_state = self.mg(block, old_state)
        block, old_state = self.dmg(block, old_state)
        block, old_state = self.mfn(block, old_state)
        block, old_state = self.dmf(block, old_state)
        block, old_state = self.ph(block, old_state)
        block, old_state = self.p(block, old_state)
        block, old_state = self.t(block, old_state)
        block, old_state = self.gz(block, old_state)
        block, old_state = self.ut(block, old_state)
        block, old_state = self.mft(block, old_state)
        block, old_state = self.vor(block, old_state)
        block, old_state = self.pv(block, old_state)
        block, old_state = self.ke(block, old_state)
        block, old_state = self.rhod(block, old_state)
        block = block.set_state(itime=block.old, state=old_state)
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
            block, star_state = self.ut(block, star_state)
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.mft(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.ph(block, star_state)
            block, star_state = self.p(block, star_state)
            block, star_state = self.t(block, star_state)
            block, star_state = self.gz(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.ke(block, star_state)
            block, star_state = self.rhod(block, star_state)
        elif pass_tag == "forward_pass":
            block, star_state = self.ut(block, star_state)
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.mft(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "backward_pass":
            block, star_state = self.ph(block, star_state)
            block, star_state = self.p(block, star_state)
            block, star_state = self.t(block, star_state)
            block, star_state = self.gz(block, star_state)
            block, star_state = self.rhod(block, star_state)
        else:
            raise ValueError(f"Unsupported pass tag: {pass_tag}!")
        return block, star_state

    def build_default(self, coriolis_scheme: CoriolisForceAtEdge) -> "ExPrepareSpaceOperators":
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
            self.div.build_default()
            self.vor.build_default()
            self.ut.build_default(coriolis_scheme=coriolis_scheme)
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
        block = self.div.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.ut.prepare_block(block)
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

    def diagnose(
        self,
        block: BLOCK,
        itime: int,
    ) -> BLOCK:
        state = block.get_state(itime)
        block, state = self.prepare.div(block, state)
        block = block.set_state(itime=itime, state=state)
        block = block.diagnose(itime=itime)
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
            new_state = tree_at(
                coord=block.coord,
                where=lambda s: s.mgs,
                pytree=new_state,
                replace=new_mgs,
                fill_halo=True
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
            new_state = tree_at(
                coord=block.coord,
                where=lambda s: s.pt,
                pytree=new_state,
                replace=new_pt,
                fill_halo=True
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
            new_state = tree_at(
                coord=block.coord,
                where=lambda s: s.un,
                pytree=new_state,
                replace=new_un,
                fill_halo=True
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

    def build_default(self) -> Self:
        if not self.has_built():
            self.mfz_lev.build_default()
            self.dmgsdt.build_default()
            self.dptdt.build_default()
            self.dundt.build_default()
            self.prepare.build_default(
                coriolis_scheme=self.dundt.coriolis.scheme
            )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmgsdt.prepare_block(block)
        block = self.dptdt.prepare_block(block)
        block = self.dundt.prepare_block(block)
        block = self.mfz_lev.prepare_block(block)
        block = self.prepare.prepare_block(block)
        return block
