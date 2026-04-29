from __future__ import annotations
from typing import Self, TYPE_CHECKING
import jax
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as op
from terraflow.discretization.abstract import (
    AbstractPrepareSpaceOperators,
    AbstractDynamicalEquationOperator,
    AbstractAdvectionEquationOperator
)
from terraflow.discretization.fvm.scvt.trisk.common.adv.schemes import HorizontalAdvectionSchemeAtCell
from terraflow.discretization.fvm.scvt.trisk.common.adv.flux_limiter import HorizontalLimiterAtCell
from terraflow.dynamics.adv import types
from ..parallel import tree_at


if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo
    from terraflow.discretization.fvm.scvt.trisk.common.adv.batch import AdvBatchType
    from ..dynamics_types import Block

    Float = typing.Float
    Int = typing.Int
    Array = typing.Array
    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    BLOCK = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]


DynOperator = AbstractDynamicalEquationOperator[
    FiniteVolumeMethod,
    types.AdvectionEquation2DMode
]

AdvOperator = AbstractAdvectionEquationOperator[
    FiniteVolumeMethod,
    types.AdvectionEquation2DMode
]


class MassFluxAtEdge(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        batch = block.adv_batch_bg
        m = block.state.dmg.unwrap()[star_idx, ...]
        # ===================================
        # Calculate mass flux at edge
        batch = self.adv_scheme.calc_mass_hflx(
            m=m,
            batch=batch,
            coord=coord,
            aux=block.equaux.adv,
        )
        # ===================================
        return eqx.tree_at(
            where=lambda b: b.adv_batch_bg,
            pytree=block,
            replace=batch,
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.adv_scheme.prepare_block(block)


class SetWindForBackground(DynOperator):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, itime: Int[Array, ""] | int) -> BLOCK:
        un = block.state.un.unwrap()[itime, ...]
        m = block.state.dmg.unwrap()[itime, ...]
        return eqx.tree_at(
            lambda b: (
                b.adv_batch_bg.un.data,
                b.adv_batch_bg.m.data,
            ),
            block,
            (un, m)
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class AccumWind(DynOperator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK) -> BLOCK:
        m = block.adv_batch_bg.m.unwrap()
        mfn = block.adv_batch_bg.mfn.unwrap()
        for batch_name, batch in block.adv_batches.items():
            batch = batch.accum_wind(
                coord=block.coord,
                m=m,
                mfn=mfn,
            )
            block = eqx.tree_at(
                lambda b: b.adv_batches[batch_name],
                block,
                batch
            )
        return block

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class TendencyOfDivergenceOfMassFlux(
    types.DivergenceOfMassFluxProtocol[
        FiniteVolumeMethod,
        types.AdvectionEquation2DMode
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.mfn = MassFluxAtEdge(disc=disc, method=method, mode=mode)

    def __call__(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str,
    ) -> BLOCK:
        coord = block.coord
        para = block.para
        m_old = block.state.dmg.unwrap()[old_idx, :, :]
        # ===================================
        block = self.mfn(block=block, star_idx=star_idx)
        mfn = block.adv_batch_bg.mfn.unwrap()
        dmf = op.div_at_primal_cell(
            x_at_primal_edge=mfn,
            coord=coord,
            para=para,
        )
        m_new = m_old - dt * dmf
        # ===================================
        return tree_at(
            coord=coord,
            where=lambda b: b.state.dmg,
            pytree=block,
            replace=m_new,
            at_dims=0,
            at_idxes=new_idx
        )

    def build_default(self) -> Self:
        if not self.has_built():
            self.mfn.build_default()
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return self.mfn.prepare_block(block)


class TendencyOfDivergenceOfTracerFlux(
    types.DivergenceOfTracerFluxProtocol[
        FiniteVolumeMethod,
        types.AdvectionEquation2DMode
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc, method, mode)
        self.adv_scheme = HorizontalAdvectionSchemeAtCell()
        self.limiter = HorizontalLimiterAtCell()

    def __call__(
        self,
        batch: AdvBatchType,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> tuple[AdvBatchType, BLOCK]:
        coord = block.coord
        idx = batch.idx
        q_old = block.tracers.q.unwrap()[old_idx, idx, ...]
        q_star = block.tracers.q.unwrap()[star_idx, idx, ...]
        q_new = block.tracers.q.unwrap()[new_idx, idx, ...]
        # ==========================================
        # Run tracers
        q_new_, batch = self._calc_at_tracers(
            q=q_star,
            q_old=q_old,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        q_new = q_new.at[idx].set(q_new_)
        block = tree_at(
            coord=coord,
            where=lambda b: b.tracers.q,
            pytree=block,
            replace=q_new,
            at_dims=0,
            at_idxes=new_idx
        )
        # # ==========================================
        # Set new mass for current batch
        batch = batch.set_new_mass(block=block, substep=substep)
        # ==========================================
        return batch, block

    def _calc_at_tracers(
        self,
        q: Float[Array, "ntracers nlev ncell"],
        q_old: Float[Array, "ntracers nlev ncell"],
        batch: AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "ntracers nlev nedges"], AdvBatchType]:

        def body_fn(
            carry_batch: AdvBatchType,
            xs: tuple[Float[Array, "nlev ncell"], Float[Array, "nlev ncell"]]
        ) -> tuple[AdvBatchType, Float[Array, "nlev nedges"]]:
            q_in, q_old_in = xs
            q_new, carry_batch = self._calc_at_single_tracer(
                q=q_in,
                q_old=q_old_in,
                batch=carry_batch,
                block=block,
                dt=dt,
                substep=substep
            )
            return carry_batch, q_new

        batch, q_new = jax.lax.scan(
            f=body_fn,
            init=batch,
            xs=(q, q_old)
        )
        return q_new, batch

    def _calc_at_single_tracer(
        self,
        q: Float[Array, "nlev npoints"],
        q_old: Float[Array, "nlev npoints"],
        batch: AdvBatchType,
        block: BLOCK,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[Float[Array, "nlev nedges"], AdvBatchType]:
        coord = block.coord
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        kds, kde = vert.full_kds, vert.full_kde
        para = block.para
        aux = block.equaux.adv
        q_old_ = q_old[kds:kde+1, ids:ide+1]
        m_old = batch.m.unwrap()[kds:kde+1, ids:ide+1]
        m_new = block.adv_batch_bg.m.unwrap()[kds:kde+1, ids:ide+1]
        # ===================================
        batch = self.adv_scheme.calc_tracer_hflx(
            q=q,
            batch=batch,
            coord=coord,
            aux=aux
        )
        batch = self.limiter(
            q=q_old,
            q_old=q_old,
            batch=batch,
            block=block,
            dt=dt,
            substep=substep
        )
        dqdt = op.div_at_primal_cell(
            x_at_primal_edge=batch.qmfn.unwrap(),
            coord=coord,
            para=para,
        )
        # ===================================
        q_new = (m_old * q_old_ - dt * dqdt) / m_new
        return q_new, batch

    def build_default(self) -> Self:
        if not self.has_built():
            self.adv_scheme.build_default()
            self.limiter.build_default()
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.adv_scheme.prepare_block(block)
        block = self.limiter.prepare_block(block)
        return block


class ExPrepareSpaceOperators(
    AbstractPrepareSpaceOperators[
        FiniteVolumeMethod,
        types.AdvectionEquation2DMode
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def first_call(self, block: BLOCK) -> BLOCK:
        m_old = block.state.dmg.unwrap()[block.old, ...]
        un_old = block.state.un.unwrap()[block.old, ...]
        block = eqx.tree_at(
            lambda b: (
                b.adv_batch_bg.m.data,
                b.adv_batch_bg.un.data
            ),
            block,
            (m_old, un_old),
        )
        for batch_name in block.adv_batches.keys():
            block = eqx.tree_at(
                lambda b: (
                    b.adv_batches[batch_name].m.data,
                    b.adv_batches[batch_name].un.data,
                ),
                block,
                (m_old, un_old),
            )
        return block

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        explicit: bool
    ) -> BLOCK:
        return block

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        return block


class ExSpaceOperators(
    types.ExVif2DAdvectionEquationSet4UGRIDSpaceOperators[
        TendencyOfDivergenceOfMassFlux,
        TendencyOfDivergenceOfTracerFlux,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.AdvectionEquation2DMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        self.set_bg_wind = SetWindForBackground(
            disc=disc, method=method, mode=mode
        )
        self.accum_wind = AccumWind(
            disc=disc, method=method, mode=mode
        )

    def __call__(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str,
        explicit: bool,
    ) -> BLOCK:
        block = self.prepare(
            block=block,
            star_idx=star_idx,
            explicit=explicit
        )
        block = self.dmdt.dmf(
            block=block,
            old_idx=old_idx,
            star_idx=star_idx,
            new_idx=new_idx,
            dt=dt,
            substep=substep,
            pass_tag=pass_tag,
        )
        return block

    def d2a_coupling(self, block: BLOCK) -> BLOCK:
        block = block.testcase_set_un(block.old)
        block = self.set_bg_wind(block, block.old)
        block = self.accum_wind(block)
        return block

    def advection_space_operators(
        self,
        batch: AdvBatchType,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int
    ) -> tuple[AdvBatchType, BLOCK]:
        batch, block = self.dqdt.dtf(
            batch=batch,
            block=block,
            old_idx=old_idx,
            star_idx=star_idx,
            new_idx=new_idx,
            dt=dt,
            substep=substep,
        )
        return batch, block

    def diagnose(
        self,
        block: BLOCK,
        itime_str: str,
    ) -> BLOCK:
        block = block.diagnose(itime_str=itime_str)
        return block

    def update_state(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        substep: int,
        pass_tag: str,
    ) -> BLOCK:
        return block

    def build_default(self) -> Self:
        if not self.has_built():
            self.dmdt.build_default()
            self.dqdt.build_default()
            self.accum_wind.build_default()
            self.prepare.build_default()
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmdt.prepare_block(block)
        block = self.dqdt.prepare_block(block)
        block = self.accum_wind.prepare_block(block)
        block = self.prepare.prepare_block(block)
        return block
