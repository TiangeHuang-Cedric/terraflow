from __future__ import annotations
from typing import Self, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device, formula
from terraflow.util.pytree import tree_at
from terraflow.mesh.scvt.mesh_type import SCVTMesh
from terraflow.mesh.scvt.field import SCVTField2D, SCVTFieldMeta
from terraflow.dynamics.abstract import (
    AbstractEquationAuxiliaryData
)

NLEVP = "ilev"
NLEV = "lev"
NCELL = "cell"
NEDGE = "edge"
NVTX = "vtx"


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block

    Int = typing.Int
    Float = typing.Float
    Array = typing.Array


class RefProfileData(AbstractEquationAuxiliaryData):
    mg    : SCVTField2D
    gz    : SCVTField2D
    gz_lev: SCVTField2D
    ad    : SCVTField2D
    mpt   : SCVTField2D
    dmgdn : SCVTField2D

    @classmethod
    def empty(cls) -> Self:
        return cls(
            initialized=False,
            mg=SCVTField2D.empty(),
            gz=SCVTField2D.empty(),
            gz_lev=SCVTField2D.empty(),
            ad=SCVTField2D.empty(),
            mpt=SCVTField2D.empty(),
            dmgdn=SCVTField2D.empty(),
        )
        
    @classmethod
    def initialize(
        cls,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        ref_ps: Float[Array, "ncell"],
        dev: device.AbstractDevice
    ) -> Self:
        if cfg.pgf_scheme == "ptb":
            return cls.from_coord(
                para=para,
                mesh=mesh,
                vert_coord=vert_coord,
                ref_ps=ref_ps,
                dev=dev
            )
        else:
            return cls.empty()

    @classmethod
    def from_coord(
        cls,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        ref_ps: Float[Array, "ncell"],
        dev: device.AbstractDevice
    ) -> Self:
        a    = jnp.array(0.09923 , dtype=dev.float_type)
        b    = jnp.array(247.7874, dtype=dev.float_type)
        c    = jnp.array(-1.0385 , dtype=dev.float_type)
        ps0  = jnp.array(1013.0  , dtype=dev.float_type)  # hPa
        ids     , ide     , ihe = mesh.cell_ids, mesh.cell_ide, mesh.cell_ihe[0]
        e_ids   , e_ide         = mesh.edge_ids, mesh.edge_ide
        full_kds, full_kde      = mesh.full_kds, mesh.full_kde
        half_kds, half_kde      = mesh.half_kds, mesh.half_kde
        rd     = para.rd.unwrap()
        # =========================================================
        # Init profile data with zeros
        pro    = cls.zeros_like(mesh=mesh, dev=dev)
        mg     = pro.mg.d
        gz     = pro.gz.d
        gz_lev = pro.gz_lev.d
        ad     = pro.ad.d
        mpt    = pro.mpt.d
        dmgdn  = pro.dmgdn.d
        # =========================================================
        # For half lev
        p_lev = vert_coord.calc_mg_lev(
            mgs=ref_ps[ids:ide+1],
            ks=half_kds,
            ke=half_kde
        ) / 100.0
        gz_lev = gz_lev.at[half_kds:half_kde+1, ids:ide+1].set(
            -rd * (
                a * (p_lev - ps0)
                + b / (1.0 + c) * (p_lev**(1.0 + c) - ps0**(1.0 + c))
            )
        )
        # =========================================================
        # For full lev
        mg_lev = vert_coord.calc_mg_lev(
            mgs=ref_ps[ids:ihe+1],
            ks=half_kds,
            ke=half_kde
        )
        p = vert_coord.calc_mg(
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
                p=p * 100.0,
                para=para
            ) * (mg_lev[1:, :] - mg_lev[:-1, :])
        )
        v1 = mesh.cell_of_edge[e_ids:e_ide+1, 0]
        v2 = mesh.cell_of_edge[e_ids:e_ide+1, 1]
        mg1 = mg[full_kds:full_kde+1, v1]
        mg2 = mg[full_kds:full_kde+1, v2]
        dmgdn = dmgdn.at[full_kds:full_kde+1, e_ids:e_ide+1].set(
            (mg2 - mg1) * mesh.edpnr_eddtg[e_ids:e_ide+1] / mesh.de[e_ids:e_ide+1]
        )
        # ==========================================================
        return eqx.tree_at(
            lambda pro: (
                pro.mg.d, pro.gz.d, pro.gz_lev.d,
                pro.ad.d, pro.mpt.d, pro.dmgdn.d
            ),
            pro,
            (mg, gz, gz_lev, ad, mpt, dmgdn)
        )

    @classmethod
    def zeros_like(cls, mesh: SCVTMesh, dev: device.AbstractDevice) -> Self:
        fields: dict[str, SCVTField2D] = {}
        fields["mg"] = SCVTField2D.from_mesh(
            name="mg",
            long_name="reference dry-air pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev
        )
        fields["gz"] = SCVTField2D.from_mesh(
            name="gz",
            long_name="reference geopotential",
            units="m2 / s2",
            loc="cell",
            mesh=mesh,
            dev=dev
        )
        fields["gz_lev"] = SCVTField2D.from_mesh(
            name="gz_lev",
            long_name="reference geopotential at half level",
            units="m2 / s2",
            loc="lev",
            mesh=mesh,
            dev=dev
        )
        fields["ad"] = SCVTField2D.from_mesh(
            name="ad",
            long_name="reference inverse dry-air density",
            units="m3 / kg",
            loc="cell",
            mesh=mesh,
            dev=dev
        )
        fields["mpt"] = SCVTField2D.from_mesh(
            name="mpt",
            long_name="reference dry potential temperature at half level",
            units="K",
            loc="lev",
            mesh=mesh,
            dev=dev            
        )
        fields["dmgdn"] = SCVTField2D.from_mesh(
            name="dmgdn",
            long_name="reference dry-air normal mass flux gradient at edge",
            units="Pa / m",
            loc="edge",
            mesh=mesh,
            dev=dev
        )
        return cls(initialized=True, **fields)


def get_profile_meta():
    return {
        "mg": SCVTFieldMeta(
            name="mg",
            long_name="reference dry-air pressure",
            units="Pa",
            loc="cell"
        ),
        "gz": SCVTFieldMeta(
            name="gz",
            long_name="reference geopotential",
            units="m2 / s2",
            loc="cell"
        ),
        "gz_lev": SCVTFieldMeta(
            name="gz_lev",
            long_name="reference geopotential at half level",
            units="m2 / s2",
            loc="lev"
        ),
        "ad": SCVTFieldMeta(
            name="ad",
            long_name="reference inverse dry-air density",
            units="m3 / kg",
            loc="cell"
        ),
        "mpt": SCVTFieldMeta(
            name="mpt",
            long_name="reference dry potential temperature at half level",
            units="K",
            loc="lev"
        ),
        "dmgdn": SCVTFieldMeta(
            name="dmgdn",
            long_name="reference dry-air normal mass flux gradient at edge",
            units="Pa / m",
            loc="edge"
        )
    }


def pgf_ptb_run(block: Block, star_idx: Int[Array, ""] | int) -> Block:
    mesh = block.mesh
    kds  , kde   = mesh.full_kds, mesh.full_kde
    e_ids, e_ide = mesh.edge_ids, mesh.edge_ide
    c_ids, c_ihe = mesh.cell_ids, mesh.cell_ihe[0]
    v1 = mesh.cell_of_edge[e_ids:e_ide+1, 0]
    v2 = mesh.cell_of_edge[e_ids:e_ide+1, 1]
    # Extract profile data
    pro = block.equaux.pgf.pro
    pro_mg = pro.mg.d
    pro_gz = pro.gz.d
    pro_ad = pro.ad.d
    # Extract block and state data
    dmg       = block.state  .dmg      .unwrap()[star_idx, :, :]
    p         = block.state  .p        .unwrap()[star_idx, :, :]
    p_lev     = block.state  .p_lev    .unwrap()[star_idx, :, :]
    gz        = block.state  .gz       .unwrap()[star_idx, :, :]
    qm        = block.tracers.qm       .unwrap()
    rhod      = block.aux    .rhod     .unwrap()
    dundt     = block.tend   .dundt    .unwrap()
    dundt_pgf = block.tend   .dundt_pgf.unwrap()
    # ==========================================================
    # perturbation quantities
    ids, ihe = c_ids, c_ihe
    p_ptb  = (p[kds:kde+1, ids:ihe+1] - pro_mg[kds:kde+1, ids:ihe+1])
    gz_ptb = gz[kds:kde+1, ids:ihe+1] - pro_gz[kds:kde+1, ids:ihe+1]
    dp_ptb = (
        p_lev[kds+1:kde+2, ids:ihe+1] - p_lev[kds:kde+1, ids:ihe+1]
    ) - dmg[kds:kde+1, ids:ihe+1]
    ad_ptb = 1 / rhod[kds:kde+1, ids:ihe+1] - pro_ad[kds:kde+1, ids:ihe+1]

    # compute pgf tendency
    ids, ide = e_ids, e_ide
    L = 1 + 0.5 * (qm[kds:kde+1, v1] + qm[kds:kde+1, v2])
    tmp1 = pro.dmgdn.d[kds:kde+1, ids:ide+1] * 0.5 * (
        ad_ptb[kds:kde+1, v1] + ad_ptb[kds:kde+1, v2]
    )
    tmp2 = 0.5 * (
        1 / rhod[kds:kde+1, v1] + 1 / rhod[kds:kde+1, v2]
    )
    tmp2 = (
        tmp2 * (p_ptb[kds:kde+1, v2] - p_ptb[kds:kde+1, v1]) 
        * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
    )
    tmp3 = (
        (gz_ptb[kds:kde+1, v2] - gz_ptb[kds:kde+1, v1]) 
        * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
    )
    tmp4 = 0.5 * (
        dp_ptb[kds:kde+1, v1] / dmg[kds:kde+1, v1] + 
        dp_ptb[kds:kde+1, v2] / dmg[kds:kde+1, v2]
    )
    tmp4 = (
        tmp4 * (gz[kds:kde+1, v2] - gz[kds:kde+1, v1]) 
        * mesh.edpnr_eddtg[ids:ide+1] / mesh.de[ids:ide+1]
    )
    tmp = -(tmp1 + tmp2 + tmp3 + tmp4) / L
    dundt     = dundt    .at[kds:kde+1, ids:ide+1].set(
        dundt[kds:kde+1, ids:ide+1] + tmp
    )
    dundt_pgf = dundt_pgf.at[kds:kde+1, ids:ide+1].set(tmp)
    # ==========================================================
    return tree_at(
        where=lambda b: (b.tend.dundt.d, b.tend.dundt_pgf.d),
        pytree=block,
        replace=(dundt, dundt_pgf)
    )    
