from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Self
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, formula
from terraflow.operator import FiniteVolumeMethod
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operators as mesh_op
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationOperator,
    DiscretizationInfo,
    AbstractEquationTermAuxiliaryData
)
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.dynamics_types import Block, State

Float = typing.Float

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

EM = TypeVar("EM", bound=AbstractEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, EM]


class RefProfileData(AbstractEquationTermAuxiliaryData):
    mg: Array
    gz: Array
    gz_lev: Array
    ad: Array
    mpt: Array
    dmgdn: Array

    def __init__(
        self,
        mg: Array,
        gz: Array,
        gz_lev: Array,
        ad: Array,
        mpt: Array,
        dmgdn: Array
    ):
        self.mg = mg
        self.gz = gz
        self.gz_lev = gz_lev
        self.ad = ad
        self.mpt = mpt
        self.dmgdn = dmgdn
        self._init = False

    @classmethod
    def empty(cls) -> Self:
        return cls(
            mg=jnp.array([]),
            gz=jnp.array([]),
            gz_lev=jnp.array([]),
            ad=jnp.array([]),
            mpt=jnp.array([]),
            dmgdn=jnp.array([]),
        )

    @classmethod
    def from_coord(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        ref_ps: Float[Array, "ncell"]
    ) -> Self:
        a = 0.09923
        b = 247.7874
        c = -1.0385
        ps0 = 1013.0  # hPa
        rd = para.rd.unwrap()
        # =============================
        # For half lev
        p_lev = coord.vert.calc_mg_lev(ref_ps) / 100.0
        gz_lev = -rd * (
            a * (p_lev - ps0)
            + b / (1 + c) * (p_lev**(1 + c) - ps0**(1 + c))
        )
        mg_lev = p_lev * 100.0
        # =============================
        # For full lev
        p = coord.vert.calc_mg(ref_ps) / 100.0
        t = p * (a + b * p**c)
        mg = p * 100.0
        gz = -rd * (
            a * (p - ps0)
            + b / (1 + c) * (p**(1 + c) - ps0**(1 + c))
        )
        ad = rd * t / (p * 100.0)
        mpt = formula.dry_potential_temperature(
            t=t,
            p=mg,
            para=para
        ) * (mg_lev[1:, :] - mg_lev[:-1, :])
        dmgdn = mesh_op.grad_at_dual_edge(
            x_at_primal_cell=mg,
            coord=coord,
            para=para
        )
        # =============================
        return cls(
            mg=mg,
            gz=gz,
            gz_lev=gz_lev,
            ad=ad,
            mpt=mpt,
            dmgdn=dmgdn
        )


class PerturbationPressureGradientForce(Operator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: Block, star_state: State) -> tuple[Block, State]:
        mesh = block.coord.mesh
        edd_v = mesh.dual.edge.endpoints  # (nedge, 2)
        cell_1 = edd_v[:, 0]
        cell_2 = edd_v[:, 1]
        pro = block.equaux.pgf.pro
        dmg = star_state.dmg.unwrap()
        p = star_state.p.unwrap()
        p_lev = star_state.p_lev.unwrap()
        gz = star_state.gz.unwrap()
        rhod = block.aux.rhod.unwrap()

        # perturbation quantities
        p_ptb = p - pro.mg
        gz_ptb = gz - pro.gz
        dp_ptb = (p_lev[1:, :] - p_lev[:-1, :]) - dmg
        ad_ptb = 1.0 / rhod - pro.ad

        # compute pgf tendency
        # TODO: For ideal dry core, need to add mixing ratio effect
        L = 1
        tmp1 = pro.dmgdn * 0.5 * (ad_ptb[:, cell_1] + ad_ptb[:, cell_2])
        tmp2 = 0.5 * (1.0 / rhod[:, cell_1] + 1.0 / rhod[:, cell_2])
        tmp2 = tmp2 * mesh_op.grad_at_dual_edge(
            x_at_primal_cell=p_ptb,
            coord=block.coord,
            para=block.para
        )
        tmp3 = mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz_ptb,
            coord=block.coord,
            para=block.para
        )
        tmp4 = 0.5 * (
            dp_ptb[:, cell_1] / dmg[:, cell_1] +
            dp_ptb[:, cell_2] / dmg[:, cell_2]
        )
        tmp4 = tmp4 * mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=block.coord,
            para=block.para
        )
        block = eqx.tree_at(
            lambda b: b.tend.dundt_pgf.data,
            block,
            - (tmp1 + tmp2 + tmp3 + tmp4) / L
        )
        return block, star_state

    def prepare_block(self, block: Block) -> Block:
        pro = RefProfileData.from_coord(
            coord=block.coord,
            para=block.para,
            ref_ps=block.static.ref_ps.unwrap()
        )
        block = eqx.tree_at(
            lambda b: b.equaux.pgf.pro,
            block,
            pro
        )
        return block

    def build_default(self) -> Self:
        if not self.has_built():
            self._built = True
        return self
