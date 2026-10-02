from __future__ import annotations
from typing import (
    Self,
    TYPE_CHECKING
)


from terraflow.util.device import AbstractDevice
from terraflow.util import equinox as eqx
from terraflow.util.field import ScalarField
from terraflow.mesh.scvt.field import (
    SCVTFieldLike,
    SCVTField1D,
    SCVTField2D,
    SCVTField3D, 
    SCVTFields
)
from terraflow.dynamics.abstract import AbstractEquationAuxiliaryDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.interface import AdvectionAuxiliaryDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.pgf.interface import PressureGradientForceDataset
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.coriolis.trisk import TriSKCoriolisForceData
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.damp.interface import DampDataset


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.config import UserConfig
    from terraflow.model.abstract import ModelMode
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate

    INT    = typing.Int
    FLT    = typing.Float
    Array  = typing.Array
    PyTree = typing.PyTree


class State(SCVTFields):
    mode    : ModelMode   = eqx.field(default_factory=lambda: ModelMode  (), static=True)
    un      : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    gz      : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    gz_lev  : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    dmg     : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    dmg_lev : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    mg      : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    mg_lev  : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    mgs     : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    pt      : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    ph      : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    ph_lev  : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    phs     : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    # Nonhydrostatic variables
    w       : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    w_lev   : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    p       : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D()) 
    p_lev   : SCVTField3D = eqx.field(default_factory=lambda: SCVTField3D())
    ps      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    # Total diagnostics
    tm      : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    tqv     : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    te      : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    te_ke   : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    te_ie   : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    te_pe   : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    tpe     : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    tpt     : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    tq      : ScalarField = eqx.field(default_factory=lambda: ScalarField())
    
    @classmethod
    def from_coord(cls, mode: ModelMode, cfg: UserConfig, mesh: SCVTMesh, dev: AbstractDevice):
        fields = state_init(mode, cfg, mesh, dev=dev)
        return cls.from_fields(fields=fields, mode=mode)


class Tendency(SCVTFields):
    mode            : ModelMode   = eqx.field(default_factory=lambda: ModelMode  (), static=True)
    dgzdt           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dmgsdt          : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    dptdt           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dundt           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dundt_coriolis  : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dundt_grad_ke   : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dundt_pgf       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dundt_wedundeta : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())

    @classmethod
    def from_coord(cls, mode: ModelMode, cfg: UserConfig, mesh: SCVTMesh, dev: AbstractDevice):
        fields = tend_init(mode, cfg, mesh, dev=dev)
        return cls.from_fields(fields=fields, mode=mode)


class Static(SCVTFields):
    mode       : ModelMode   = eqx.field(default_factory=lambda: ModelMode  (), static=True)
    landmask   : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    # Topography
    gzs        : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    dzsdn      : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    # Reference surface pressure
    ref_ps     : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    ref_ps_smth: SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    ref_ps_perb: SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())

    @classmethod
    def from_coord(cls, mode: ModelMode, cfg: UserConfig, mesh: SCVTMesh, dev: AbstractDevice):
        fields = static_init(mode, cfg, mesh, dev=dev)
        return cls.from_fields(fields=fields, mode=mode)


class Auxiliary(SCVTFields):
    mode         : ModelMode   = eqx.field(default_factory=lambda: ModelMode  (), static=True)
    # Smagorinsky damping variables
    smag_coef    : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    smag_coef_lev: SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    # Laplacian damping variables
    g_1d         : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    fn_1d        : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    df_1d        : SCVTField1D = eqx.field(default_factory=lambda: SCVTField1D())
    g_2d         : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    fn_2d        : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    # Other Variables
    u            : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    v            : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    un_lev       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    ut           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    we_lev       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    div          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    div2         : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D()) 
    vor          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    pv           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    pv_edge      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dmg_edge     : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D()) 
    dmg_vtx      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    ke           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    ke_vtx       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    mfn          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    mfn_lev      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    mfz          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    mfz_lev      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    mfz_lev_edge : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dmf          : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dmf_lev      : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    adv_w_lev    : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    adv_gz_lev   : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    t            : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    tv           : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    rhod         : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    pk_lev       : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    # Tendencies from damping processes
    dundt_damp   : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dwdt_damp    : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())
    dptdt_damp   : SCVTField2D = eqx.field(default_factory=lambda: SCVTField2D())

    @classmethod
    def from_coord(cls, mode: ModelMode, cfg: UserConfig, mesh: SCVTMesh, dev: AbstractDevice):
        fields = aux_init(mode, cfg, mesh, dev=dev)
        return cls.from_fields(fields=fields, mode=mode)


class EquationAuxDataset(AbstractEquationAuxiliaryDataset):
    pgf     : PressureGradientForceDataset
    coriolis: TriSKCoriolisForceData
    adv     : AdvectionAuxiliaryDataset
    damp    : DampDataset

    @classmethod
    def empty(cls) -> Self:
        pgf      = PressureGradientForceDataset.empty()
        coriolis = TriSKCoriolisForceData.empty()
        adv      = AdvectionAuxiliaryDataset.empty()
        damp     = DampDataset.empty()
        return cls(
            initialized=False,
            pgf=pgf,
            coriolis=coriolis,
            adv=adv,
            damp=damp
        )

    @classmethod
    def initialize(
        cls,
        mode: ModelMode,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        ref_ps: FLT[Array, "ncell"],
        dev: AbstractDevice
    ) -> Self:
        if not mode.is_advection():
            pgf = PressureGradientForceDataset.initialize(
                mode=mode,
                cfg=cfg,
                para=para,
                mesh=mesh,
                vert_coord=vert_coord,
                ref_ps=ref_ps,
                dev=dev
            )
            coriolis = TriSKCoriolisForceData.from_coord(
                mesh=mesh,
                dev=dev
            )
        else:
            pgf = PressureGradientForceDataset.empty()
            coriolis = TriSKCoriolisForceData.empty() 
        adv = AdvectionAuxiliaryDataset.initialize(
            cfg=cfg,
            para=para,
            mesh=mesh,
            dev=dev
        )
        if not mode.is_advection():
            damp = DampDataset.initialize(
                cfg=cfg,
                para=para,
                mesh=mesh,
                dev=dev
            )
        else:
            damp = DampDataset.empty() 
        return cls(
            initialized=True,
            pgf=pgf,
            coriolis=coriolis,
            adv=adv,
            damp=damp            
        )


def state_init(
    mode: ModelMode,
    cfg: UserConfig,
    mesh: SCVTMesh,
    dev: AbstractDevice
) -> dict[str, SCVTFieldLike | ScalarField]:
    fields: dict[str, SCVTFieldLike | ScalarField] = {}
    if cfg.time_scheme == "wrfrk3" or cfg.time_scheme == "pc2":
        nstate = 3
    elif cfg.time_scheme == "euler" or cfg.time_scheme == "rk2":
        nstate = 2
    else:
        raise ValueError(f"state_init: unsupported time_scheme {cfg.time_scheme}!")
    
    fields["un"] = SCVTField3D.from_mesh(
        name="un",
        long_name="Normal wind component",
        units="m/s",
        loc="edge",
        mesh=mesh,
        dev=dev,
        is_expand_dim0=True,
        expand_dim0_len=nstate,
        output="h1"
    )
    fields["gz"] = SCVTField3D.from_mesh(
        name="gz",
        long_name="Geopotential",
        units="m2 / s2",
        loc="cell",
        mesh=mesh,
        dev=dev,
        is_expand_dim0=True,
        expand_dim0_len=nstate,
        output="h0"
    )
    fields["gz_lev"] = SCVTField3D.from_mesh(
        name="gz_lev",
        long_name="Geopotential",
        units="m2 / s2",
        loc="lev",
        mesh=mesh,
        dev=dev,
        is_expand_dim0=True,
        expand_dim0_len=nstate,
        output="h0"
    )
    fields["dmg"] = SCVTField3D.from_mesh(
        name="dmg",
        long_name="Dry-air weight",
        units="Pa",
        loc="cell",
        mesh=mesh,
        dev=dev,
        is_expand_dim0=True,
        expand_dim0_len=nstate,
        output="h1"
    )
    if mode.is_baroclinic() or mode.is_advection():
        fields["dmg_lev"] = SCVTField3D.from_mesh(
            name="dmg_lev",
            long_name="Dry-air weight",
            units="Pa",
            loc="lev",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output=""
        )
    if mode.is_baroclinic() or mode.is_advection():
        if mode.is_moist():
            pt_long_name = "Modified Potential temperature"
        else:
            pt_long_name = "Potential temperature"
        fields["pt"] = SCVTField3D.from_mesh(
            name="pt",
            long_name=pt_long_name,
            units="K",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0"
        )
    fields["mg"] = SCVTField3D.from_mesh(
        name="mg",
        long_name="Dry-air weight",
        units="Pa",
        loc="cell",
        mesh=mesh,
        dev=dev,
        is_expand_dim0=True,
        expand_dim0_len=nstate,
        output=""
    )
    if mode.is_baroclinic() or mode.is_advection():
        fields["mg_lev"] = SCVTField3D.from_mesh(
            name="mg_lev",
            long_name="Dry-air weight",
            units="Pa",
            loc="lev",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output=""
        )
    if mode.is_baroclinic() or mode.is_advection():
        fields["mgs"] = SCVTField2D.from_mesh(
            name="mgs",
            long_name="Dry-air weight on surface",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            at_sfc=True,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h1"
        )
    if mode.is_baroclinic() or mode.is_advection(): 
        fields["ph"] = SCVTField3D.from_mesh(
            name="ph",
            long_name="Hydrostatic pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection(): 
        fields["ph_lev"] = SCVTField3D.from_mesh(
            name="ph_lev",
            long_name="Hydrostatic pressure",
            units="Pa",
            loc="lev",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h1"
        )
    if mode.is_baroclinic(): 
        fields["phs"] = SCVTField2D.from_mesh(
            name="phs",
            long_name="Hydrostatic pressure on surface",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            at_sfc=True,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0"
        )    
    if mode.is_nonhydrostatic():
        fields["w"] = SCVTField3D.from_mesh(
            name="w",
            long_name="Vertical wind speed",
            units="m s-1",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output=""
        )
        fields["w_lev"] = SCVTField3D.from_mesh(
            name="w_lev",
            long_name="Vertical wind speed",
            units="m s-1",
            loc="lev",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0"
        )
    if mode.is_baroclinic():
        fields["p"] = SCVTField3D.from_mesh(
            name="p",
            long_name="Pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0" if mode.nonhydrostatic else ""
        )
        fields["p_lev"] = SCVTField3D.from_mesh(
            name="p_lev",
            long_name="Pressure",
            units="Pa",
            loc="lev",
            mesh=mesh,
            dev=dev,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h1" if mode.nonhydrostatic else ""
        )
        fields["ps"] = SCVTField2D.from_mesh(
            name="ps",
            long_name="Surface pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            at_sfc=True,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="" if mode.nonhydrostatic else ""
        )
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        fields["tm"] = ScalarField.initialize(
            name="tm",
            long_name="Total mass",
            units="kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
    if mode.is_baroclinic() and mode.is_moist():
        fields["tqv"] = ScalarField.initialize(
            name="tqv",
            long_name="Total water vapor",
            units="kg/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        fields["tpe"] = ScalarField.initialize(
            name="tpe",
            long_name="Total potential vorticity enstrophy",
            units="J/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
        fields["te"] = ScalarField.initialize(
            name="te",
            long_name="Total energy",
            units="J/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
        fields["te_ke"] = ScalarField.initialize(
            name="te_ke",
            long_name="Total energy due to kinetic energy",
            units="J/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="",
            dev=dev
        )
        fields["te_pe"] = ScalarField.initialize(
            name="te_pe",
            long_name="Total energy due to potential energy",
            units="J/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="",
            dev=dev
        )
    if mode.is_baroclinic():
        fields["te_ie"] = ScalarField.initialize(
            name="te_ie",
            long_name="Total energy due to internal energy",
            units="J/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="",
            dev=dev
        )
        fields["tpt"] = ScalarField.initialize(
            name="tpt",
            long_name="Total potential temperature",
            units="K",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
    if mode.is_advection():
        fields["tq"] = ScalarField.initialize(
            name="tq",
            long_name="Total tracer dry mixing ratio",
            units="m3 kg/kg",
            is_static=False,
            is_expand_dim0=True,
            expand_dim0_len=nstate,
            output="h0",
            dev=dev
        )
    return fields
        
        
def tend_init(
    mode: ModelMode,
    cfg: UserConfig,
    mesh: SCVTMesh,
    dev: AbstractDevice
) -> dict[str, SCVTFieldLike]:
    fields: dict[str, SCVTFieldLike] = {}
    if mode.is_barotropic():
        fields["dgzdt"] = SCVTField2D.from_mesh(
            name="dgzdt",
            long_name="Dynamic tendency of gz",
            units="m2 / s3",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if mode.is_baroclinic():
        fields["dmgsdt"] = SCVTField1D.from_mesh(
            name="dmgsdt",
            long_name="Dynamic tendency of mgs",
            units="Pa / s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
        fields["dptdt"] = SCVTField2D.from_mesh(
            name="dptdt",
            long_name="Dynamic tendency of pt",
            units="K / s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if mode.is_barotropic() or mode.is_baroclinic():
        fields["dundt"] = SCVTField2D.from_mesh(
            name="dundt",
            long_name="Dynamic tendency of un",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
        fields["dundt_coriolis"] = SCVTField2D.from_mesh(
            name="dundt_coriolis",
            long_name="Dynamic tendency of un due to Coriolis force",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1" if cfg.output_h1_tend else ""
        )
        fields["dundt_grad_ke"] = SCVTField2D.from_mesh(
            name="dundt_grad_ke",
            long_name="Dynamic tendency of un due to kinetic gradient",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1" if cfg.output_h1_tend else ""
        )
        fields["dundt_pgf"] = SCVTField2D.from_mesh(
            name="dundt_pgf",
            long_name="Dynamic tendency of un due to PGF",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1" if cfg.output_h1_tend else ""
        )
    if mode.is_baroclinic():
        fields["dundt_wedundeta"] = SCVTField2D.from_mesh(
            name="dundt_wedundeta",
            long_name="Dynamic tendency of un due to vertical advection",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1" if cfg.output_h1_tend else ""
        )
    return fields


def static_init(
    mode: ModelMode,
    cfg: UserConfig,
    mesh: SCVTMesh,
    dev: AbstractDevice
) -> dict[str, SCVTFieldLike]:
    fields: dict[str, SCVTFieldLike] = {}
    if mode.is_barotropic() or mode.is_baroclinic() or mode.is_advection():
        fields["gzs"] = SCVTField1D.from_mesh(
            name="gzs",
            long_name="Surface geopotential",
            units="m2 / s2",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_static=True,
            output="h0"
        )
    if mode.is_baroclinic():
        fields["landmask"] = SCVTField1D.from_mesh(
            name="landmask",
            long_name="Land mask",
            units="1",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_static=True,
            output="h0"
        )
        fields["dzsdn"] = SCVTField1D.from_mesh(
            name="dzsdn",
            long_name="Gradient of zs",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            is_static=True,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection3d():
        fields["ref_ps"] = SCVTField1D.from_mesh(
            name="ref_ps",
            long_name="Reference surface pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_static=False,
            output="h1"
        )
        fields["ref_ps_smth"] = SCVTField1D.from_mesh(
            name="ref_ps_smth",
            long_name="Smoothed reference surface pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_static=False,
            output="h1"
        )
        fields["ref_ps_perb"] = SCVTField1D.from_mesh(
            name="ref_ps_perb",
            long_name="Perturbation of reference surface pressure",
            units="Pa",
            loc="cell",
            mesh=mesh,
            dev=dev,
            is_static=False,
            output="h1"
        )
    return fields


def aux_init(
    mode: ModelMode,
    cfg: UserConfig,
    mesh: SCVTMesh,
    dev: AbstractDevice
) -> dict[str, SCVTFieldLike]:
    fields: dict[str, SCVTFieldLike] = {}
    if cfg.use_smag_damp:
        fields["smag_coef"] = SCVTField2D.from_mesh(
            name            ='smag_coef'                          ,
            long_name       ='Coefficient for Smagorinsky damping',
            units           ='s-2'                                ,
            loc             ='edge'                               ,
            mesh            =mesh                                 ,
            dev             =dev                                  ,
        )
        if mode.is_nonhydrostatic():
            fields["smag_coef_lev"] = SCVTField2D.from_mesh(
                name="smag_coef_lev",
                long_name       ='Coefficient for Smagorinsky damping at half lev' , 
                units           ='s-2'                                             , 
                loc             ='lev_edge'                                        , 
                mesh            =mesh                                              ,
                dev             =dev
            )
    if cfg.use_laplace_damp or cfg.use_div_damp or cfg.use_smag_damp:
        fields["dundt_damp"] = SCVTField2D.from_mesh(
            name="dundt_damp",
            long_name="Tendency of normal wind due to damping",
            units="m / s2",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if (cfg.use_laplace_damp or cfg.use_smag_damp) and mode.is_nonhydrostatic():
        fields["dwdt_damp"] = SCVTField2D.from_mesh(
            name="dwdt_damp",
            long_name="Tendency of vertical wind due to damping",
            units="m / s2",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if cfg.use_smag_damp:
        fields["dptdt_damp"] = SCVTField2D.from_mesh(
            name="dptdt_damp",
            long_name="Tendency of temperature due to damping",
            units="K / s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    fields["g_1d"] = SCVTField1D.from_mesh(
        name="f_1d",
        long_name="Temporal array for 1D Laplacian damping",
        units="",
        loc="edge",
        mesh=mesh,
        dev=dev,
        output=""
    )
    fields["fn_1d"] = SCVTField1D.from_mesh(
        name="fn_1d",
        long_name="Temporal array for 1D Laplacian damping",
        units="",
        loc="edge",
        mesh=mesh,
        dev=dev,
        output=""
    )
    fields["df_1d"] = SCVTField1D.from_mesh(
        name="df_1d",
        long_name="Temporal array for 1D Laplacian damping",
        units="",
        loc="cell",
        mesh=mesh,
        dev=dev,
        output=""
    )
    if cfg.use_laplace_damp or cfg.use_smag_damp:
        fields["g_2d"] = SCVTField2D.from_mesh(
            name="g_2d",
            long_name="Temporal array for 2D Laplacian damping",
            units="",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output=""
        )
        fields["fn_2d"] = SCVTField2D.from_mesh(
            name="fn_2d",
            long_name="Temporal array for 2D Laplacian damping",
            units="",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output=""
        )
    if mode.is_baroclinic() or mode.is_advection():
        fields["t"] = SCVTField2D.from_mesh(
            name="t",
            long_name="Temperature",
            units="K",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
    if mode.is_baroclinic() or mode.is_advection():
        fields["tv"] = SCVTField2D.from_mesh(
            name="tv",
            long_name="Virtual temperature",
            units="K",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if mode.is_baroclinic():
        fields["rhod"] = SCVTField2D.from_mesh(
            name="rhod",
            long_name="Dry-air density",
            units="kg / m3",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
        fields["pk_lev"] = SCVTField2D.from_mesh(
            name="pk_lev",
            long_name="Exner function on half levels",
            units="Pa**kappa",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output=""
        )
    fields["we_lev"] = SCVTField2D.from_mesh(
        name="we_lev",
        long_name="Vertical coordinate speed",
        units="m/s",
        loc="lev",
        mesh=mesh,
        dev=dev,
        output=""
    )
    if not mode.is_advection():
        fields["u"] = SCVTField2D.from_mesh(
            name="u",
            long_name="Zonal wind component",
            units="m/s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
        fields["v"] = SCVTField2D.from_mesh(
            name="v",
            long_name="Meridional wind component",
            units="m/s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
        fields["ut"] = SCVTField2D.from_mesh(
            name="ut",
            long_name="Tangential wind component",
            units="m/s",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
        fields["ke"] = SCVTField2D.from_mesh(
            name="ke",
            long_name="Kinetic energy",
            units="m2 / s2",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h1" if not mode.is_advection() else ""
        )
        fields["ke_vtx"] = SCVTField2D.from_mesh(
            name="ke_vtx",
            long_name="Kinetic energy",
            units="m2 / s2",
            loc="vtx",
            mesh=mesh,
            dev=dev,
            output="h1" if not mode.is_advection() else ""
        )
        fields["pv_edge"] = SCVTField2D.from_mesh(
            name="pv_edge",
            long_name="Potential vorticity",
            units="Pa-1 s-1",
            loc="edge",
            mesh=mesh,
            dev=dev,
            output="h1" if not mode.is_advection() else ""
        )
    fields["dmg_edge"] = SCVTField2D.from_mesh(
        name="dmg_edge",
        long_name="Dry-air weight",
        units="Pa",
        loc="edge",
        mesh=mesh,
        dev=dev,
        output="h1"
    )
    fields["dmg_vtx"] = SCVTField2D.from_mesh(
        name="dmg_vtx",
        long_name="Dry-air weight",
        units="Pa",
        loc="vtx",
        mesh=mesh,
        dev=dev,
        output=""
    )
    if mode.is_baroclinic() or mode.is_advection():
        fields["mfz_lev"] = SCVTField2D.from_mesh(
            name="mfz_lev",
            long_name="Vertical mass flux",
            units="Pa s-1",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if mode.is_nonhydrostatic():
        fields["mfz"] = SCVTField2D.from_mesh(
            name="mfz",
            long_name="Vertical mass flux",
            units="Pa s-1",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output=""
        )
        fields["mfn_lev"] = SCVTField2D.from_mesh(
            name="mfn_lev",
            long_name="Normal mass flux on half levels",
            units="Pa m s-1",
            loc="lev_edge",
            mesh=mesh,
            dev=dev,
            output=""
        )
        fields["dmf_lev"] = SCVTField2D.from_mesh(
            name="dmf_lev",
            long_name="Mass flux divergence on half levels",
            units="kg / (m s)",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output=""
        )
        fields["adv_w_lev"] = SCVTField2D.from_mesh(
            name="adv_w_lev",
            long_name="Advective tendency of half-level vertical wind",
            units="m / s2",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
        fields["adv_gz_lev"] = SCVTField2D.from_mesh(
            name="adv_gz_lev",
            long_name="Advective tendency of half-level geopotential",
            units="m2 / s3",
            loc="lev",
            mesh=mesh,
            dev=dev,
            output="h1"
        )
    if mode.is_baroclinic():
        fields["mfz_lev_edge"] = SCVTField2D.from_mesh(
            name="mfz_lev_edge",
            long_name="Vertical mass flux",
            units="Pa s-1",
            loc="lev_edge",
            mesh=mesh,
            dev=dev,
            output=""
        )
    fields["mfn"] = SCVTField2D.from_mesh(
        name="mfn",
        long_name="Normal mass flux",
        units="Pa m s-1",
        loc="edge",
        mesh=mesh,
        dev=dev,
        output="h1"
    )
    if not mode.is_advection():
        fields["vor"] = SCVTField2D.from_mesh(
            name="vor",
            long_name="Relative vorticity",
            units="1/s",
            loc="vtx",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
    fields["pv"] = SCVTField2D.from_mesh(
        name="pv",
        long_name="Potential vorticity",
        units="Pa-1 s-1",
        loc="vtx",
        mesh=mesh,
        dev=dev,
        output="h1" if not mode.is_advection() else ""
    )
    if not mode.is_advection():
        fields["div"] = SCVTField2D.from_mesh(
            name="div",
            long_name="Divergence",
            units="1/s",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
    if cfg.use_div_damp and cfg.div_damp_order == 4:
        fields["div2"] = SCVTField2D.from_mesh(
            name="div",
            long_name="Gradient of divergence",
            units="m-1 s-1",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )        
    if cfg.use_div_damp and cfg.div_damp_order == 4:
        fields["div2"] = SCVTField2D.from_mesh(
            name="div2",
            long_name="Gradient of divergence",
            units="m-1 s-1",
            loc="cell",
            mesh=mesh,
            dev=dev,
            output="h0"
        )
    fields["dmf"] = SCVTField2D.from_mesh(
        name="dmf",
        long_name="Mass flux divergence",
        units="kg / (m s)",
        loc="cell",
        mesh=mesh,
        dev=dev,
        output="h1"
    )       
    if mode.is_hydrostatic() and cfg.uv_vert_adv_type == 1:
        fields["un_lev"] =SCVTField2D.from_mesh(
            name="un_lev",
            long_name="Normal wind component",
            units="m/s",
            loc="lev_edge",
            mesh=mesh,
            dev=dev,
            output=""
        )
    if mode.is_nonhydrostatic():
        fields["un_lev"] =SCVTField2D.from_mesh(
            name="un_lev",
            long_name="Normal wind component",
            units="m/s",
            loc="lev_edge",
            mesh=mesh,
            dev=dev,
            output=""
        )
        
    return fields
