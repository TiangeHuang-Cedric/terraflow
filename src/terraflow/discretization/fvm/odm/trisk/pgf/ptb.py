from __future__ import annotations
from typing import TYPE_CHECKING, Self
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, formula
from terraflow.operator import FiniteVolumeMethod
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.orthogonal_dual_mesh.construct import OrthogonalDualMesh
from terraflow.mesh.orthogonal_dual_mesh import operator as mesh_op
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.discretization.abstract import (
    DiscretizationInfo,
    AbstractPrepareSpaceOperators,
    AbstractEquationTermAuxiliaryData
)
from terraflow.layout.abstract import CoordinateSystem
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.orthogonal_dual_mesh import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout

if TYPE_CHECKING:
    from terraflow.discretization.fvm.odm.trisk.hybrid_lev.dry.hdc.dynamics_types import Block

Float = typing.Float

Array = typing.Array

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    OrthogonalDualMesh,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

Term = AbstractPrepareSpaceOperators[FiniteVolumeMethod]


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
        nlevp = coord.vert.nlev + 1
        ref_ps_lev = jnp.tile(ref_ps, (nlevp, 1))  # in hPa
        p = coord.vert.calc_mg_lev(ref_ps) / 100.0
        gz_lev = -rd * (
            a * (p - ps0)
            + b / (1 + c) * (p**(1 + c) - ps0**(1 + c))
        )
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
        ) * (ref_ps_lev[1:, :] - ref_ps_lev[:-1, :])
        dmgdn = mesh_op.grad_at_dual_edge(
            x_at_primal_cell=mg,
            coord=coord,
            para=para
        )
        return cls(
            mg=mg,
            gz=gz,
            gz_lev=gz_lev,
            ad=ad,
            mpt=mpt,
            dmgdn=dmgdn
        )


class PerturbationPressureGradientForce(Term):

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

    def __call__(self, block: Block) -> Block:
        mesh = block.coord.mesh
        edd_v = mesh.dual.edge.dual_cells  # (nedge, 2)
        pro = block.equaux.pgf.pro
        dmg = block.aux.dmg.unwrap()
        p = block.state.p.unwrap()
        p_lev = block.state.p_lev.unwrap()
        rhod = block.aux.rhod.unwrap()
        gz = block.state.gz.unwrap()

        # perturbation quantities
        p_ptb = p - pro.mg
        gz_ptb = gz - pro.gz
        dp_ptb = (p_lev[1:, :] - p_lev[:-1, :]) - dmg
        ad_ptb = 1.0 / rhod - pro.ad

        # compute pgf tendency
        # TODO: For ideal dry core, need to add mixing ratio effect
        tmp1 = pro.dmgdn * 0.5 * jnp.sum(ad_ptb[:, edd_v], axis=-1)
        tmp2 = 0.5 * jnp.sum(1.0 / rhod[:, edd_v], axis=-1)
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
        tmp4 = 0.5 * jnp.sum(dp_ptb[:, edd_v] / dmg[:, edd_v], axis=-1)
        tmp4 = tmp4 * mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=block.coord,
            para=block.para
        )
        return eqx.tree_at(
            lambda b: b.tend.dundt_pgf.data,
            block,
            - (tmp1 + tmp2 + tmp3 + tmp4)
        )

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
