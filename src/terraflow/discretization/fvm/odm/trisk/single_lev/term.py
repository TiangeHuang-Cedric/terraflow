from __future__ import annotations
from typing import Self
import equinox as eqx

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operators as op
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
from terraflow.discretization.fvm.odm.trisk.common import (
    adv,
    coriolis,
    pgf,
    operators as common_op
)
from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block, State
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
        self.recon_at_edge = adv.flux_reconstruct.FluxReconstructAtEdgeOnPrimalMesh()

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        coord = block.coord
        g = block.para.gravity.unwrap()
        un = star_state.un.unwrap()
        gz = star_state.gz.unwrap()
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
        star_state = eqx.tree_at(
            lambda s: (s.dmg.data, s.dmg_e.data, s.dmg_vtx.data),
            star_state,
            (dmg, dmg_e, dmg_vtx)
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
        g = block.para.gravity.unwrap()
        dmf = block.aux.dmf.unwrap()
        dgzdt = block.tend.dgzdt.unwrap()
        dgzdt_dmf = -dmf * g
        dgzdt = dgzdt + dgzdt_dmf
        block = eqx.tree_at(
            lambda b: (b.tend.dgzdt.data, b.tend.dgzdt_dmf.data),
            block,
            replace=(dgzdt, dgzdt_dmf)
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block, itime: int) -> Field:
        return block.tend.dgzdt_dmf


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

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        dundt_coriolis = block.aux.coriolis.unwrap()
        dundt = block.tend.dundt.unwrap()
        dundt = dundt + dundt_coriolis
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_coriolis.data),
            block,
            replace=(dundt, dundt_coriolis)
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block, itime: int) -> Field:
        return block.tend.dundt_coriolis


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
        self.scheme = pgf.swm.PressureGradientForce(
            disc=disc, method=method)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        block, star_state = self.scheme(block, star_state)
        dundt = block.tend.dundt.unwrap()
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt + dundt_pgf
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_pgf.data),
            block,
            replace=(dundt, dundt_pgf)
        )
        return block, star_state

    def build_default(self) -> Self:
        self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block, itime: int) -> Field:
        return block.tend.dundt_pgf


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
        ke = block.aux.ke.unwrap()
        dundt = block.tend.dundt.unwrap()
        dundt_grad_ke = - op.grad_at_dual_edge(
            x_at_primal_cell=ke,
            coord=block.coord,
            para=block.para
        )
        dundt = dundt + dundt_grad_ke
        block = eqx.tree_at(
            lambda b: (b.tend.dundt.data, b.tend.dundt_grad_ke.data),
            block,
            replace=(dundt, dundt_grad_ke)
        )
        return block, star_state

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self

    def prepare_block(self, block: Block) -> Block:
        return block

    def unwrap(self, block: Block, itime: int) -> Field:
        return block.tend.dundt_grad_ke


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
        self.dmg = FluidThickness(disc=disc, method=method)
        self.mf = common_op.NormalMassFluxAtEdge(
            disc=disc, method=method, mode=mode
        )
        self.dmf = common_op.DivergenceOfMassFluxAtCell(
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
            disc=disc,
            method=method,
            mode=mode
        )

    def first_call(self, block: BLOCK) -> BLOCK:
        star_state = block.get_state(0)
        block, star_state = self.dmg(block, star_state)
        block, star_state = self.mf(block, star_state)
        block, star_state = self.dmf(block, star_state)
        block, star_state = self.coriolis.recon_ut(block, star_state)
        block, star_state = self.vor(block, star_state)
        block, star_state = self.pv(block, star_state)
        block, star_state = self.coriolis(block, star_state)
        block, star_state = self.ke(block, star_state)
        block = block.set_state(itime=0, state=star_state)
        return block

    def __call__(
        self,
        block: BLOCK,
        star_state: State,
        pass_tag: str,
        explicit: bool
    ) -> tuple[BLOCK, State]:
        if pass_tag == "all_pass":
            block, star_state = self.mf(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.coriolis.recon_ut(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.coriolis(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "forward_pass":
            block, star_state = self.mf(block, star_state)
            block, star_state = self.dmf(block, star_state)
            block, star_state = self.coriolis.recon_ut(block, star_state)
            block, star_state = self.vor(block, star_state)
            block, star_state = self.pv(block, star_state)
            block, star_state = self.coriolis(block, star_state)
            block, star_state = self.ke(block, star_state)
        elif pass_tag == "backward_pass":
            pass
        return block, star_state

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
        block = self.dmg.prepare_block(block)
        block = self.mf.prepare_block(block)
        block = self.dmf.prepare_block(block)
        block = self.vor.prepare_block(block)
        block = self.coriolis.prepare_block(block)
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
            new_state = eqx.tree_at(
                lambda s: s.gz.data,
                new_state,
                new_gz
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
            return eqx.tree_at(
                lambda s: s.un.data,
                new_state,
                new_un
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
