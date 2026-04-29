from __future__ import annotations
from typing import Self, TYPE_CHECKING

from terraflow.util import typing
from terraflow.operator import FiniteVolumeMethod
from terraflow.mesh.scvt import operators as op

from terraflow.discretization.abstract import (
    AbstractPrepareSpaceOperators,
    AbstractDynamicalEquationOperator
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


class FluidThickness(
    AbstractDynamicalEquationOperator[
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

    def __call__(self, block: BLOCK, star_idx: Int[Array, ""] | int) -> BLOCK:
        coord = block.coord
        p_ids = block.coord.mesh.polg_ids
        p_ihe = block.coord.mesh.polg_ihe[0]
        g = block.para.gravity.unwrap()
        un = block.state.un.unwrap()[star_idx, :, :]
        gz = block.state.gz.unwrap()[star_idx, :, :]
        gzs = block.static.gzs.unwrap()
        dmg = block.state.dmg.unwrap()[star_idx, :, :]
        dmg_e = block.state.dmg_e.unwrap()[star_idx, :, :]
        dmg_vtx = block.state.dmg_vtx.unwrap()[star_idx, :, :]
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
        block = tree_at(
            coord=coord,
            where=lambda b: b.state.dmg,
            pytree=block,
            replace=dmg,
            at_dims=0,
            at_idxes=star_idx
        )
        return tree_at(
            coord=coord,
            where=lambda b: (b.state.dmg_e, b.state.dmg_vtx),
            pytree=block,
            replace=(dmg_e, dmg_vtx),
            at_dims=(0, 0),
            at_idxes=(star_idx, star_idx),
            fill_halo=True
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
        method: FiniteVolumeMethod,
        mode: types.ShallowWaterEquationMode
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(
        self,
        block: Block,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> Block:
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
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dgzdt, b.tend.dgzdt_dmf),
            pytree=block,
            replace=(dgzdt, dgzdt_dmf)
        )

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

    def __call__(
        self,
        block: Block,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> Block:
        ids, ide = block.coord.mesh.edge_ids, block.coord.mesh.edge_ide
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block = self.scheme(block, star_idx)
        dundt_coriolis = block.tend.dundt_coriolis.unwrap()
        dundt = dundt.at[0, ids:ide+1].set(
            dundt[0, ids:ide+1] + dundt_coriolis[0, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )

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

    def __call__(
        self,
        block: Block,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> Block:
        ids, ide = block.coord.mesh.edge_ids, block.coord.mesh.edge_ide
        dundt = block.tend.dundt.unwrap()
        # ==========================================================
        block = self.scheme(block, star_idx)
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        dundt = dundt.at[0, ids:ide+1].set(
            dundt[0, ids:ide+1] + dundt_pgf[0, ids:ide+1]
        )
        # ==========================================================
        return tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt,
            pytree=block,
            replace=dundt
        )

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

    def __call__(
        self,
        block: Block,
        star_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
    ) -> Block:
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
        return tree_at(
            coord=block.coord,
            where=lambda b: (b.tend.dundt, b.tend.dundt_grad_ke),
            pytree=block,
            replace=(dundt, dundt_grad_ke),
            fill_halo=True
        )

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
        old = block.old
        block = self.dmg(block, old)
        block = self.mfn(block, old)
        block = self.dmf(block, old)
        block = self.ut(block, old)
        block = self.vor(block, old)
        block = self.pv(block, old)
        block = self.ke(block, old)
        return block

    def __call__(
        self,
        block: BLOCK,
        star_idx: Int[Array, ""] | int,
        pass_tag: str,
        explicit: bool
    ) -> BLOCK:
        if pass_tag == "all_pass":
            block = self.mfn(block, star_idx)
            block = self.dmf(block, star_idx)
            block = self.ut(block, star_idx)
            block = self.vor(block, star_idx)
            block = self.pv(block, star_idx)
            block = self.ke(block, star_idx)
        elif pass_tag == "forward_pass":
            block = self.mfn(block, star_idx)
            block = self.dmf(block, star_idx)
            block = self.ut(block, star_idx)
            block = self.vor(block, star_idx)
            block = self.pv(block, star_idx)
            block = self.ke(block, star_idx)
        elif pass_tag == "backward_pass":
            pass
        return block

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
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        pass_tag: str,
        explicit: bool
    ) -> BLOCK:
        block = block.reset(pass_tag=pass_tag)
        if pass_tag == "all_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=star_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dgzdt(block, star_idx, dt, substep)
            block = self.dundt(block, star_idx, dt, substep)
            # ==================================================
        elif pass_tag == "forward_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=star_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dgzdt(block, star_idx, dt, substep)
            # ==================================================
        elif pass_tag == "backward_pass":
            # ==================================================
            # Prepare state and block for tendency calculation
            block = self.prepare(
                block=block,
                star_idx=new_idx,
                pass_tag=pass_tag,
                explicit=explicit
            )
            # ==================================================
            # Calculate tendencies
            block = self.dundt.grad_ke(block, star_idx, dt, substep)
            block = self.dundt.pgf(block, new_idx, dt, substep)
            block = self.dundt.coriolis(block, star_idx, dt, substep)
            # ==================================================
        else:
            raise ValueError(f"Unknown pass_tag: {pass_tag}")
        return block

    def d2a_coupling(self, block: BLOCK, *args, **kwargs) -> BLOCK:
        return block

    def advection_space_operators(
        self,
        batch: AdvBatchType,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        star_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: Float[Array, ""] | float,
        substep: int,
        explicit: bool
    ) -> tuple[AdvBatchType, BLOCK]:
        return batch, block

    def diagnose(
        self,
        block: BLOCK,
        itime_str: str,
    ) -> BLOCK:
        itime = block.get_time_idx(itime_str=itime_str)
        block = self.prepare.div(block, itime)
        block = block.diagnose(itime_str=itime_str)
        return block

    def _update_gz(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        pass_tag: str
    ) -> BLOCK:
        if pass_tag == "all_pass" or pass_tag == "forward_pass":
            mesh = block.coord.mesh
            vert = block.coord.vert
            ids, ide = mesh.polg_ids, mesh.polg_ide
            kds, kde = vert.full_kds, vert.full_kde
            gz = block.state.gz.unwrap()
            old_gz = gz[old_idx, :, :]
            new_gz = gz[new_idx, :, :]
            dgzdt = block.tend.dgzdt.unwrap()
            # ==================================================
            new_gz = new_gz.at[kds:kde+1, ids:ide+1].set(
                old_gz[kds:kde+1, ids:ide+1] + dgzdt[kds:kde+1, ids:ide+1] * dt
            )
            # ==================================================
            block = tree_at(
                coord=block.coord,
                where=lambda b: b.state.gz,
                pytree=block,
                replace=new_gz,
                at_dims=0,
                at_idxes=new_idx,
                fill_halo=True
            )
            # Update dmg
            return self.prepare.dmg(block, new_idx)
        elif pass_tag == "backward_pass":
            return block
        else:
            raise ValueError(f"Unknown pass tag: {pass_tag}")

    def _update_un(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        pass_tag: str
    ) -> BLOCK:
        if pass_tag == "all_pass" or pass_tag == "backward_pass":
            mesh = block.coord.mesh
            vert = block.coord.vert
            ids, ide = mesh.edge_ids, mesh.edge_ide
            kds, kde = vert.full_kds, vert.full_kde
            un = block.state.un.unwrap()
            old_un = un[old_idx, :, :]
            new_un = un[new_idx, :, :]
            dundt = block.tend.dundt.unwrap()
            # ==================================================
            new_un = new_un.at[kds:kde+1, ids:ide+1].set(
                old_un[kds:kde+1, ids:ide+1] + dundt[kds:kde+1, ids:ide+1] * dt
            )
            # ==================================================
            return tree_at(
                coord=block.coord,
                where=lambda b: b.state.un,
                pytree=block,
                replace=new_un,
                at_dims=0,
                at_idxes=new_idx,
                fill_halo=True
            )
        elif pass_tag == "forward_pass":
            return block
        else:
            raise ValueError(f"Unknown pass tag: {pass_tag}")

    def update_state(
        self,
        block: BLOCK,
        old_idx: Int[Array, ""] | int,
        new_idx: Int[Array, ""] | int,
        dt: float,
        substep: int,
        pass_tag: str,
    ) -> BLOCK:
        # ==================================================
        # Update gz
        block = self._update_gz(
            block=block,
            old_idx=old_idx,
            new_idx=new_idx,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        # Update un
        block = self._update_un(
            block=block,
            old_idx=old_idx,
            new_idx=new_idx,
            dt=dt,
            pass_tag=pass_tag
        )
        # ================================================
        return block

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
