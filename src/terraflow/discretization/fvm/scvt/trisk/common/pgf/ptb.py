from __future__ import annotations
from typing import TypeVar, Self, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, formula
from terraflow.operator import FiniteVolumeMethod
from terraflow.planet import PlanetParameterSet
from terraflow.mesh.scvt.construct import SphericalCentroidVoronoiTessellations
from terraflow.mesh.scvt import operators as mesh_op
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
from terraflow.discretization.abstract import (
    AbstractEquationMode,
    AbstractEquationOperator,
    DiscretizationInfo,
    AbstractEquationTermAuxiliaryData
)
from terraflow.layout.abstract import CoordinateSystem, MeshEntity, Basis, VertCoordLocation
from terraflow.layout.field import ArrayField, StaggeredFieldMetaData
from terraflow.layout.scvt import TriSKBasedCGridStaggeringLayout
from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
from terraflow.discretization.fvm.scvt.trisk.parallel import tree_at

Float = typing.Float
Array = typing.Array

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"

FDT = ArrayField[StaggeredFieldMetaData]

CoordSys = CoordinateSystem[
    SphericalCentroidVoronoiTessellations,
    HybridVertCoordinate,
    TriSKBasedCGridStaggeringLayout,
    HybridVertCoordLayout
]

EM = TypeVar("EM", bound=AbstractEquationMode)

Operator = AbstractEquationOperator[FiniteVolumeMethod, EM]

if TYPE_CHECKING:
    from terraflow.discretization.fvm.scvt.trisk.dynamics_types import Block, State
    BLOCK = Block[HybridVertCoordinate, HybridVertCoordLayout]


class RefProfileData(AbstractEquationTermAuxiliaryData):
    mg: FDT
    gz: FDT
    gz_lev: FDT
    ad: FDT
    mpt: FDT
    dmgdn: FDT

    def __init__(
        self,
        mg: FDT,
        gz: FDT,
        gz_lev: FDT,
        ad: FDT,
        mpt: FDT,
        dmgdn: FDT
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
            mg=FDT.empty(),
            gz=FDT.empty(),
            gz_lev=FDT.empty(),
            ad=FDT.empty(),
            mpt=FDT.empty(),
            dmgdn=FDT.empty(),
        )

    @classmethod
    def from_coord(
        cls,
        coord: CoordSys,
        para: PlanetParameterSet,
        ref_ps: Float[Array, "ncell"]
    ) -> Self:
        float_type = coord.dev.float_type
        a = jnp.array(0.09923, dtype=float_type)
        b = jnp.array(247.7874, dtype=float_type)
        c = jnp.array(-1.0385, dtype=float_type)
        ps0 = jnp.array(1013.0, dtype=float_type)  # hPa
        mesh = coord.mesh
        vert = coord.vert
        ids, ide = mesh.polg_ids, mesh.polg_ide
        ihe = mesh.polg_ihe[0]
        e_ids, e_ide = mesh.edge_ids, mesh.edge_ide
        full_kds, full_kde = vert.full_kds, vert.full_kde
        half_kds, half_kde = vert.half_kds, vert.half_kde
        rd = para.rd.unwrap()
        # Init profile data with zeros
        pro = cls.zeros_like(coord=coord)
        mg = pro.mg.data
        gz = pro.gz.data
        gz_lev = pro.gz_lev.data
        ad = pro.ad.data
        mpt = pro.mpt.data
        dmgdn = pro.dmgdn.data
        # =============================
        # For half lev
        p_lev = coord.vert.calc_mg_lev(
            mgs=ref_ps[ids:ide+1],
            ks=half_kds,
            ke=half_kde
        ) / 100.0
        mg_lev = p_lev * 100.0
        gz_lev = gz_lev.at[half_kds:half_kde+1, ids:ide+1].set(
            -rd * (
                a * (p_lev - ps0)
                + b / (1.0 + c) * (p_lev**(1.0 + c) - ps0**(1.0 + c))
            )
        )
        # =============================
        # For full lev
        p = coord.vert.calc_mg(
            mgs=ref_ps[ids:ihe+1],
            ks=full_kds,
            ke=full_kde
        ) / 100.0
        t = p * (a + b * p**c)
        mg = mg.at[full_kds:full_kde+1, ids:ihe+1].set(p * 100.0)
        gz = gz.at[full_kds:full_kde+1, ids:ihe+1].set(
            -rd * (
                a * (p - ps0)
                + b / (1.0 + c) * (p**(1.0 + c) - ps0**(1.0 + c))
            )
        )
        ad = ad.at[full_kds:full_kde+1, ids:ihe+1].set(
            rd * (a + b * p**c) / 100.0
        )
        mpt = mpt.at[full_kds:full_kde+1, ids:ihe+1].set(
            formula.dry_potential_temperature(
                t=t,
                p=mg,
                para=para
            ) * (mg_lev[1:, :] - mg_lev[:-1, :])
        )
        dmgdn = dmgdn.at[full_kds:full_kde+1, e_ids:e_ide+1].set(
            mesh_op.grad_at_dual_edge(
                x_at_primal_cell=mg,
                coord=coord,
                para=para
            )
        )
        # =============================
        return eqx.tree_at(
            lambda pro: (
                pro.mg.data, pro.gz.data, pro.gz_lev.data,
                pro.ad.data, pro.mpt.data, pro.dmgdn.data,
                pro._init
            ),
            pro,
            (mg, gz, gz_lev, ad, mpt, dmgdn, True)
        )

    @classmethod
    def zeros_like(cls, coord: CoordSys) -> Self:
        meta = get_profile_meta()
        return cls(
            mg=meta["mg"].zeros_like(coord=coord),
            gz=meta["gz"].zeros_like(coord=coord),
            gz_lev=meta["gz_lev"].zeros_like(coord=coord),
            ad=meta["ad"].zeros_like(coord=coord),
            mpt=meta["mpt"].zeros_like(coord=coord),
            dmgdn=meta["dmgdn"].zeros_like(coord=coord)
        )


def get_profile_meta():
    return {
        "mg": StaggeredFieldMetaData(
            name="mg",
            dims=(NLEV, NCELL),
            long_name="reference dry-air pressure",
            description="Reference dry-air pressure profile",
            units="Pa",
            mesh_name="primal",
            hori_location=MeshEntity.Cell,
            vert_location=VertCoordLocation.Cell,
            basis=Basis.Scalar
        ),
        "gz": StaggeredFieldMetaData(
            name="gz",
            dims=(NLEV, NCELL),
            long_name="reference geopotential",
            description="Reference geopotential",
            units="m2 / s2",
            mesh_name="primal",
            hori_location=MeshEntity.Cell,
            vert_location=VertCoordLocation.Cell,
            basis=Basis.Scalar
        ),
        "gz_lev": StaggeredFieldMetaData(
            name="gz_lev",
            dims=(NLEVP, NCELL),
            long_name="reference geopotential at half level",
            description="Reference geopotential at half level",
            units="m2 / s2",
            mesh_name="primal",
            hori_location=MeshEntity.Cell,
            vert_location=VertCoordLocation.Lev,
            basis=Basis.Scalar
        ),
        "ad": StaggeredFieldMetaData(
            name="ad",
            dims=(NLEV, NCELL),
            long_name="reference inverse dry-air density",
            description="Reference inverse dry-air density",
            units="m3 / kg",
            mesh_name="primal",
            hori_location=MeshEntity.Cell,
            vert_location=VertCoordLocation.Cell,
            basis=Basis.Scalar
        ),
        "mpt": StaggeredFieldMetaData(
            name="mpt",
            dims=(NLEVP, NCELL),
            long_name="reference dry potential temperature at half level",
            description="Reference mass weighted potential temperature",
            units="K",
            mesh_name="primal",
            hori_location=MeshEntity.Cell,
            vert_location=VertCoordLocation.Lev,
            basis=Basis.Scalar
        ),
        "dmgdn": StaggeredFieldMetaData(
            name="dmgdn",
            dims=(NLEV, NEDGE),
            long_name="reference dry-air normal mass flux gradient at edge",
            description="Reference dry-air normal mass flux gradient at edge",
            units="Pa / m",
            mesh_name="primal",
            hori_location=MeshEntity.Edge,
            vert_location=VertCoordLocation.Cell,
            basis=Basis.Normal
        )
    }


class PerturbationPressureGradientForce(Operator):

    def __init__(
        self,
        disc: DiscretizationInfo,
        method: FiniteVolumeMethod,
        mode: AbstractEquationMode,
    ):
        super().__init__(disc=disc, method=method, mode=mode)

    def __call__(self, block: BLOCK, star_state: State) -> tuple[BLOCK, State]:
        mesh = block.coord.mesh
        vert = block.coord.vert
        full_kds = vert.full_kds
        full_kde = vert.full_kde
        e_ids = mesh.edge_ids
        e_ide = mesh.edge_ide
        c_ids = mesh.polg_ids
        c_ihe = mesh.polg_ihe[0]
        edd_v = mesh.dual.edge.endpoints[e_ids:e_ide+1, :]  # (nedge, 2)
        cell_1 = edd_v[e_ids:e_ide+1, 0]
        cell_2 = edd_v[e_ids:e_ide+1, 1]
        # Extract profile data
        pro = block.equaux.pgf.pro
        pro_mg = pro.mg.data[full_kds:full_kde+1, c_ids:c_ihe+1]
        pro_gz = pro.gz.data[full_kds:full_kde+1, c_ids:c_ihe+1]
        pro_ad = pro.ad.data[full_kds:full_kde+1, c_ids:c_ihe+1]
        # Extract block and state data
        dmg = star_state.dmg.unwrap()[full_kds:full_kde+1, c_ids:c_ihe+1]
        p = star_state.p.unwrap()[full_kds:full_kde+1, c_ids:c_ihe+1]
        p_lev = star_state.p_lev.unwrap()[full_kds:full_kde+2, c_ids:c_ihe+1]
        gz = star_state.gz.unwrap()[full_kds:full_kde+1, c_ids:c_ihe+1]
        rhod = block.aux.rhod.unwrap()[full_kds:full_kde+1, c_ids:c_ihe+1]
        ad = 1.0 / rhod
        dundt_pgf = block.tend.dundt_pgf.unwrap()
        # ==========================================================
        # perturbation quantities
        p_ptb = (p - pro_mg)
        gz_ptb = gz - pro_gz
        dp_ptb = (p_lev[1:, :] - p_lev[:-1, :]) - dmg
        ad_ptb = ad - pro_ad

        # compute pgf tendency
        # TODO: For ideal dry core, need to add mixing ratio effect
        L = 1
        tmp1 = pro.dmgdn * 0.5 * (
            jnp.take(ad_ptb, cell_1, axis=1)
            + jnp.take(ad_ptb, cell_2, axis=1)
        )
        tmp2 = 0.5 * (
            jnp.take(ad, cell_1, axis=1)
            + jnp.take(ad, cell_2, axis=1)
        )
        tmp2 = tmp2 * mesh_op.grad_at_dual_edge(
            x_at_primal_cell=p_ptb,
            coord=block.coord,
            para=block.para,
        )
        tmp3 = mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz_ptb,
            coord=block.coord,
            para=block.para
        )
        tmp4 = 0.5 * (
            jnp.take(dp_ptb, cell_1, axis=1) / jnp.take(dmg, cell_1, axis=1)
            + jnp.take(dp_ptb, cell_2, axis=1) / jnp.take(dmg, cell_2, axis=1)
        )
        tmp4 = tmp4 * mesh_op.grad_at_dual_edge(
            x_at_primal_cell=gz,
            coord=block.coord,
            para=block.para
        )

        dundt_pgf = dundt_pgf.at[full_kds:full_kde+1, e_ids:e_ide+1].set(
            -(tmp1 + tmp2 + tmp3 + tmp4) / L
        )
        # ==========================================================
        block = tree_at(
            coord=block.coord,
            where=lambda b: b.tend.dundt_pgf,
            pytree=block,
            replace=dundt_pgf
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
