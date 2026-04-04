from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as op

from terraflow.discretization.abstract import (
    AbstractPrepareSpaceOperators,
    AbstractEquationOperator
)
from terraflow.dynamics.shallow_water import types
from ..parallel import tree_at


if TYPE_CHECKING:
    from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.layout.abstract import CoordinateSystem
    from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
    from terraflow.layout.onelayer_vert_coord import OneLayerVertCoordLayout
    from terraflow.discretization.abstract import DiscretizationInfo
    from terraflow.discretization.fvm.scvt.trisk.common.coriolis.schemes import CoriolisForceAtEdge
    from ..dynamics_types import Block, State

    Float = typing.Float
    Array = typing.Array
    CoordSys = CoordinateSystem[
        SphericalCentroidVoronoiTessellations,
        OneLayerVertCoordinate,
        TriSKBasedCGridStaggeringLayout,
        OneLayerVertCoordLayout
    ]

    BLOCK = Block[OneLayerVertCoordinate, OneLayerVertCoordLayout]


class FluidThickness(
    AbstractEquationOperator[
        FiniteVolumeMethod,
        types.ShallowWaterEquationMode
    ]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=types.ShallowWaterEquationMode()
        )
        from ..common.adv.flux_reconstruct import FluxReconstructAtEdgeOnPrimalMesh
        self.recon_at_edge = FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        coord = block.coord
        p_ids = block.coord.mesh.polg_ids
        p_ihe = block.coord.mesh.polg_ihe[0]
        g = block.para.gravity.unwrap()
        un = star_state.un.unwrap()
        gz = star_state.gz.unwrap()
        gzs = block.static.gzs.unwrap()
        dmg = star_state.dmg.unwrap()
        dmg_e = star_state.dmg_e.unwrap()
        dmg_vtx = star_state.dmg_vtx.unwrap()
        # ==========================================================
        dmg = dmg.at[0, p_ids:p_ihe+1].set(
            gz[0, p_ids:p_ihe+1] - gzs[p_ids:p_ihe+1]
        ) / g
        dmg_e = self.recon_at_edge(
            x_at_primal_cell=dmg,
            x_at_primal_edge=dmg_e,
            normal_u_at_edge=un,
            coord=block.coord,
            aux=block.equaux.adv
        )
        dmg_vtx = op.interp_weighted_add_plg2dua(
            x_at_primal_cell=dmg,    # in
            x_at_dual_cell=dmg_vtx,  # inout
            coord=coord,
        )
        # ==========================================================
        star_state = tree_at(
            coord=coord,
            where=lambda s: s.dmg,
            pytree=star_state,
            replace=dmg
        )
        star_state = tree_at(
            coord=coord,
            where=lambda s: (s.dmg_e, s.dmg_vtx),
            pytree=star_state,
            replace=(dmg_e, dmg_vtx),
            fill_halo=True
        )
        return block, star_state

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
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        ids, ide = block.coord.mesh.polg_ids, block.coord.mesh.polg_ide
        g = block.para.gravity.unwrap()
        dmf = block.aux.dmf.unwrap()
        dgzdt = block.tend.dgzdt.unwrap()
        dgzdt_dmf = block.tend.dgzdt_dmf.unwrap()
        # ==========================================================
        dgzdt_dmf = dgzdt_dmf.at[0, ids:ide+1].set(
            -dmf[0, ids:ide+1] * g
        )
        dgzdt = dgzdt.at[0, ids:ide+1].set(
            dgzdt[0, ids:ide+1] + dgzdt_dmf[0, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dgzdt, b.tend.dgzdt_dmf),
            pytree=block,
            replace=(dgzdt, dgzdt_dmf)
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge(
    types.NonlinearCoriolisTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.coriolis.schemes import CoriolisForceAtEdge
        self.scheme = CoriolisForceAtEdge(
            disc=disc, method=method, mode=mode
        )

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        ids, ide = block.coord.mesh.edge_ids, block.coord.mesh.edge_ide
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block, star_state = self.scheme(block, star_state)
        dundt_coriolis = block.tend.dundt_coriolis.unwrap()
        dundt = dundt.at[0, ids:ide+1].set(
            dundt[0, ids:ide+1] + dundt_coriolis[0, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self.scheme.build_default()
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return self.scheme.prepare_block(block)


class TendencyOfNormalVelocityDueToPressureGradientForceAtDualEdge(
    types.PressureGradientForceTermProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)
        from ..common.pgf.swm import PressureGradientForce
        self.scheme = PressureGradientForce(
            disc=disc, method=method)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        ids, ide = block.coord.mesh.edge_ids, block.coord.mesh.edge_ide
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block, star_state = self.scheme(block, star_state)
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt.at[0, ids:ide+1].set(
            dundt[0, ids:ide+1] + dundt_pgf[0, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )
        return block, star_state

    def build_default(self) -> Self:
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge(
    types.GradientOfKineticEnergyProtocol[FiniteVolumeMethod]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        ids, ide = block.coord.mesh.edge_ids, block.coord.mesh.edge_ide
        ke = block.aux.ke.unwrap()
        dundt = block.tend.dundt.unwrap()
        dundt_grad_ke = block.tend.dundt_grad_ke.unwrap()
        # ==========================================================
        dundt_grad_ke = - op.grad_at_dual_edge(
            x_at_primal_cell=ke,
            x_at_dual_edge=dundt_grad_ke,
            coord=block.coord,
            para=block.para
        )
        dundt = dundt.at[0, ids:ide+1].set(
            dundt[0, ids:ide+1] + dundt_grad_ke[0, ids:ide+1]
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_grad_ke),
            pytree=block,
            replace=(dundt, dundt_grad_ke),
            fill_halo=True
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block


class ExPrepareSpaceOperators(
    AbstractPrepareSpaceOperators[
        FiniteVolumeMethod,
        types.ShallowWaterEquationMode
    ]
):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(
            disc=disc,
            method=method,
            mode=mode
        )
        from ..common import operators as common_op
        self.dmg = FluidThickness(disc=disc, method=method)
        self.mfn = common_op.NormalMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.dmf = common_op.DivergenceOfMassFluxAtCell(
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
            disc=disc,
            method=method,
            mode=mode
        )

    def first_call(self, block: BLOCK) -> BLOCK:
        star_state = block.get_state(block.old)
        block, star_state = self.dmg(block, star_state)
        block, star_state = self.mfn(block, star_state)
        block, star_state = self.dmf(block, star_state)
        block, star_state = self.ut(block, star_state)
        block, star_state = self.vor(block, star_state)
        block, star_state = self.pv(block, star_state)
        block, star_state = self.ke(block, star_state)
        block = block.set_state(itime=block.old, state=star_state)
        return block

    def __call__(
        self,
        block: BLOCK,
        star_state: State,
        pass_tag: str,
        explicit: bool
    ) -> tuple[BLOCK, State]:
        if pass_tag == "all_pass":
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.ut(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "forward_pass":
            block, star_state = self.mfn(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.ut(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "backward_pass":
            pass
        return block, star_state

    def build_default(self, coriolis_scheme: CoriolisForceAtEdge) -> "ExPrepareSpaceOperators":
        if not self.has_built():
            self.dmg.build_default()
            self.mfn.build_default()
            self.dmf.build_default()
            self.div.build_default()
            self.vor.build_default()
            self.ut.build_default(coriolis_scheme=coriolis_scheme)
            self.pv.build_default()
            self.ke.build_default()
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dmg.prepare_block(block)
        block = self.mfn.prepare_block(block)
        block = self.dmf.prepare_block(block)
        block = self.div.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.ut.prepare_block(block)
        block = self.pv.prepare_block(block)
        block = self.ke.prepare_block(block)
        return block


class ExSpaceOperators(
    types.ExVifShallowWaterEquationSet4UGRIDSpaceOperators[
        TendencyOfDivergenceOfMassFlux,
        TendencyOfNormalVelocityDueToNonlinearCoriolisTermAtDualEdge,
        TendencyOfNormalVelocityDueToGradientOfKineticEnergyAtDualEdge,
        TendencyOfNormalVelocityDueToPressureGradientForceAtDualEdge,
        FiniteVolumeMethod,
        ExPrepareSpaceOperators
    ]
):
    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

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
            # Get state at star_idx
            star_state = block.get_state(star_idx)
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
            block, star_state = self.dgzdt(block, star_state)
            block, star_state = self.dundt(block, star_state)
            # ==================================================
            # Update state at star idx
            block = block.set_state(itime=star_idx, state=star_state)
            # ==================================================
        elif pass_tag == "forward_pass":
            # ==================================================
            # Get state at star_idx
            star_state = block.get_state(star_idx)
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
            block, star_state = self.dgzdt(block, star_state)
            # ==================================================
            # Update state at star idx
            block = block.set_state(itime=star_idx, state=star_state)
            # ==================================================
        elif pass_tag == "backward_pass":
            # ==================================================
            # Get state at star_idx and new_idx
            star_state = block.get_state(star_idx)
            new_state = block.get_state(new_idx)
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
            # ==================================================
            # Update state at star idx and new idx
            block = block.set_state(itime=star_idx, state=star_state)
            block = block.set_state(itime=new_idx, state=new_state)
            # ==================================================
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
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

    def _update_gz(
        self,
        block: BLOCK,
        old_state: State,
        new_state: State,
        dt: float,
        pass_tag: str
    ) -> State:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            old_gz = old_state.gz.unwrap()
            dgzdt = block.tend.dgzdt.unwrap()
            new_gz = old_gz + dgzdt * dt
            new_state = tree_at(
                coord=block.coord,
                where=lambda s: s.gz,
                pytree=new_state,
                replace=new_gz,
                fill_halo=True
            )
            # Update dmg
            block, new_state = self.prepare.dmg(block, new_state)
            return new_state
        elif pass_tag == "backward_pass":
            return new_state
        else:
            raise ValueError(f"Unknown pass tag: {pass_tag}")

    def _update_un(
        self,
        block: BLOCK,
        old_state: State,
        new_state: State,
        dt: float,
        pass_tag: str
    ) -> State:
        if pass_tag == "all_pass" or pass_tag == "backward_pass":
            old_un = old_state.un.unwrap()
            dundt = block.tend.dundt.unwrap()
            new_un = old_un + dundt * dt
            return tree_at(
                coord=block.coord,
                where=lambda s: s.un,
                pytree=new_state,
                replace=new_un,
                fill_halo=True
            )
        elif pass_tag == "forward_pass":
            return new_state
        else:
            raise ValueError(f"Unknown pass tag: {pass_tag}")

    def update_state(
        self,
        block: BLOCK,
        old_idx: int,
        new_idx: int,
        pass_tag: str,
        dt: float
    ) -> State:
        old_state = block.get_state(old_idx)
        new_state = block.get_state(new_idx)
        # Update gz
        new_state = self._update_gz(
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
            self.dgzdt.build_default()
            self.dundt.build_default()
            self.prepare.build_default(
                coriolis_scheme=self.dundt.coriolis.scheme
            )
            self._built = True
        return self

    def prepare_block(self, block: BLOCK) -> BLOCK:
        block = self.dgzdt.prepare_block(block)
        block = self.dundt.prepare_block(block)
        block = self.prepare.prepare_block(block)
        return block
