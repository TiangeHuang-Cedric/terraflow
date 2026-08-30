from __future__ import annotations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    overload,
    Literal,
    Any,
    Optional,
    Union,
    Self
)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import netCDF4 as nc4

from terraflow.util import typing, device, log
from terraflow.util.history.setup.mesh import SCVTHistorySetup
from terraflow.util.uxarray import UGRIDGrid
from terraflow.mesh.abstract import AbstractUnstructuredMesh
from terraflow.mesh.scvt.decomp import scvt_decomp_run
from terraflow.mesh.scvt.parallel_types import Proc
from terraflow.mesh.scvt.halo import SCVTHalo
from terraflow.mesh.scvt.geometry import (
    calc_kite_area,
    correct_normal_vectors,
)
from terraflow.prepare.readers.grist_mesh import GRISTRefMesh
from terraflow.vert_coord.gtc.onelayer import OneLayerVertCoordinate


if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.vert_coord.gtc.hybrid_coord import AbstractGeneralizedTerrainFollowingCoordinate
    import numpy as np
    
FLT       = typing.Float
INT       = typing.Int
Array     = typing.Array
IntType   = typing.IntType
FloatType = typing.FloatType


class SCVTMesh(AbstractUnstructuredMesh):
    parent           : SCVTMesh = eqx.field(default=None)
    halo             : SCVTHalo = eqx.field(default=None)
    id               : int = eqx.field(default=0, static=True)
    hori_hw          : int = eqx.field(default=2, static=True)
    lev_hw           : int = eqx.field(default=3, static=True)
    n_cell           : int = eqx.field(default=0, static=True)
    n_vtx            : int = eqx.field(default=0, static=True)
    n_edge           : int = eqx.field(default=0, static=True)
    full_nlev        : int = eqx.field(default=0, static=True)
    half_nlev        : int = eqx.field(default=0, static=True)
    maxnnb_cell      : int = eqx.field(default=0, static=True)
    maxnnb_vtx       : int = eqx.field(default=0, static=True)
    glevel           : int = eqx.field(default=0, static=True)
    cell_ids         : int = eqx.field(default=0, static=True)
    cell_ide         : int = eqx.field(default=0, static=True)
    vtx_ids          : int = eqx.field(default=0, static=True)
    vtx_ide          : int = eqx.field(default=0, static=True)
    edge_ids         : int = eqx.field(default=0, static=True)
    edge_ide         : int = eqx.field(default=0, static=True)
    full_kds         : int = eqx.field(default=0, static=True)
    full_kde         : int = eqx.field(default=0, static=True)
    half_kds         : int = eqx.field(default=0, static=True)
    half_kde         : int = eqx.field(default=0, static=True)
    cell_ims         : int = eqx.field(default=0, static=True)
    cell_ime         : int = eqx.field(default=0, static=True)
    vtx_ims          : int = eqx.field(default=0, static=True)
    vtx_ime          : int = eqx.field(default=0, static=True)
    edge_ims         : int = eqx.field(default=0, static=True)
    edge_ime         : int = eqx.field(default=0, static=True)
    full_kms         : int = eqx.field(default=0, static=True)
    full_kme         : int = eqx.field(default=0, static=True)
    half_kms         : int = eqx.field(default=0, static=True)
    half_kme         : int = eqx.field(default=0, static=True)
    cell_ins         : int = eqx.field(default=0, static=True)
    cell_ine         : int = eqx.field(default=0, static=True)
    vtx_ins          : int = eqx.field(default=0, static=True)
    vtx_ine          : int = eqx.field(default=0, static=True)
    edge_ins         : int = eqx.field(default=0, static=True)
    edge_ine         : int = eqx.field(default=0, static=True)
    cell_ibs         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    cell_ibe         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    vtx_ibs          : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    vtx_ibe          : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    edge_ibs         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    edge_ibe         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    cell_ihs         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    cell_ihe         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    vtx_ihs          : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    vtx_ihe          : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    edge_ihs         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    edge_ihe         : tuple[int, ...] = eqx.field(default=(0,0,0), static=True)
    full_dlev        : FLT[Array, "full_nlev-1"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    half_dlev        : FLT[Array, "half_nlev-1"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_lon         : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_lat         : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_xyz         : FLT[Array, "n_cell 3"   ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_lon          : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_lat          : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_xyz          : FLT[Array, "n_vtx 3"    ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_lon         : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_lat         : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_xyz         : FLT[Array, "n_edge 3"   ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_lon    : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_lat    : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_xyz    : FLT[Array, "n_edge 3"   ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    full_lev         : FLT[Array, "full_nlev"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    half_lev         : FLT[Array, "half_nlev"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_cos_lon     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_sin_lon     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_cos_lat     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_sin_lat     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_cos_lon      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_sin_lon      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_cos_lat      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_sin_lat      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_cos_lon     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_sin_lon     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_cos_lat     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_sin_lat     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_cos_lon: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_sin_lon: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_cos_lat: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_sin_lat: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    # For output
    cell_lon_deg     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_lat_deg     : FLT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_lon_deg      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_lat_deg      : FLT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_lon_deg     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_lat_deg     : FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_lon_deg: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_lat_deg: FLT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    global_cell_id   : INT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    global_vtx_id    : INT[Array, "n_vtx"      ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    global_edge_id   : INT[Array, "n_edge"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_proc_idmap  : INT[Array, "n_cell"     ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Topology
    cell_nnb         : INT[Array, "n_cell"            ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_nb          : INT[Array, "n_cell maxnnb_cell"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_ed          : INT[Array, "n_cell maxnnb_cell"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_nr          : INT[Array, "n_cell maxnnb_cell"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_of_cell      : INT[Array, "n_cell maxnnb_cell"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    
    vtx_nnb          : INT[Array, "n_vtx"             ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_nb           : INT[Array, "n_vtx maxnnb_vtx"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_ed           : INT[Array, "n_vtx maxnnb_vtx"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_nr           : INT[Array, "n_vtx maxnnb_vtx"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_of_vtx      : INT[Array, "n_vtx maxnnb_vtx"  ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    
    edge_nr          : FLT[Array, "n_edge 3"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    edge_tg          : FLT[Array, "n_edge 3"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    cell_of_edge     : INT[Array, "n_edge 2"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    
    dual_edge_nr     : FLT[Array, "n_edge 3"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    dual_edge_tg     : FLT[Array, "n_edge 3"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    vtx_of_edge      : INT[Array, "n_edge 2"]           = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Area for weighting
    total_area       : FloatType                        = eqx.field(default_factory=lambda: jnp.zeros(0))
    area_cell        : FLT[Array, "n_cell"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    area_vtx         : FLT[Array, "n_vtx" ]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    area_edge        : FLT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    kite_area_cell   : FLT[Array, "n_cell maxnnb_cell"] = eqx.field(default_factory=lambda: jnp.zeros(0))
    kite_area_vtx    : FLT[Array, "n_vtx  maxnnb_vtx" ] = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Area for global weighting operations
    area_vtx_wgt     : FLT[Array, "n_vtx" ]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    area_edge_wgt    : FLT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Edge length
    le               : FLT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    de               : FLT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Aux indicators for operators
    edpnr_eddtg      : INT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    edptg_eddnr      : INT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    edp12_edptg      : INT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    edp12_eddnr      : INT[Array, "n_edge"]             = eqx.field(default_factory=lambda: jnp.zeros(0))
    # Coriolis parameter
    f_vtx            : FLT[Array, "n_vtx"]              = eqx.field(default_factory=lambda: jnp.zeros(0))

    @staticmethod
    def alias_name() -> str:
        return "scvt"

    def get_coords(self, loc: str, layout: str, with_halo: bool = False) -> dict[str, Array]:
        if layout == "triskc":
            if loc == "cell" or loc == "lev":
                if with_halo:
                    i_s, i_e = self.cell_ims, self.cell_ime
                    k_s, k_e = self.full_kms, self.full_kme
                else:
                    i_s, i_e = self.cell_ids, self.cell_ide
                    k_s, k_e = self.full_kds, self.full_kde
                return dict(
                    xyz=self.cell_xyz[i_s:i_e+1, :],
                    lat=self.cell_lat[i_s:i_e+1],
                    lon=self.cell_lon[i_s:i_e+1],
                    lev=self.full_lev[k_s:k_e+1] if loc == "cell" else self.half_lev[k_s:k_e+1]
                )
            elif loc == "vtx" or loc == "lev_vtx":
                if with_halo:
                    i_s, i_e = self.vtx_ims , self.vtx_ime
                    k_s, k_e = self.full_kms, self.full_kme
                else:
                    i_s, i_e = self.vtx_ids , self.vtx_ide
                    k_s, k_e = self.full_kds, self.full_kde
                return dict(
                    xyz=self.vtx_xyz[i_s:i_e+1, :],
                    lat=self.vtx_lat[i_s:i_e+1],
                    lon=self.vtx_lon[i_s:i_e+1],
                    lev=self.full_lev[k_s:k_e+1] if loc == "vtx"  else self.half_lev[k_s:k_e+1]
                )
            elif loc == "edge" or loc == "lev_edge":
                if with_halo:
                    i_s, i_e = self.edge_ims, self.edge_ime
                    k_s, k_e = self.full_kms, self.full_kme
                else:
                    i_s, i_e = self.edge_ids, self.edge_ide
                    k_s, k_e = self.full_kds, self.full_kde
                return dict(
                    xyz=self.dual_edge_xyz[i_s:i_e+1, :],
                    lat=self.dual_edge_lat[i_s:i_e+1],
                    lon=self.dual_edge_lon[i_s:i_e+1],
                    nr=self.edge_nr[i_s:i_e+1, :],
                    tg=self.edge_tg[i_s:i_e+1, :],
                    lev=self.full_lev[k_s:k_e+1] if loc == "edge" else self.half_lev[k_s:k_e+1]
                )
            else:
                raise ValueError(f"Unsupported location {loc} for layout {layout}!")
        else:
            raise ValueError(f"Unsupported layout {layout}!")
        
    def to_ugrid(self, use_dual: bool = False, **kwargs) -> UGRIDGrid:
        return UGRIDGrid.from_scvt_mesh(self, use_dual=use_dual, **kwargs)

    def history_builder(
        self,
        file_path: str,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        *args,
        **kwargs
    ) -> SCVTHistorySetup:
        return SCVTHistorySetup(
            file_path=file_path,
            mesh=self,
            output_type=output_type,
            float_type=float_type,
            int_type=int_type,
        )
    
    @overload
    def get_index_range(
        self,
        loc: str,
        with_lev_idx: Literal[True],
        with_halo: bool = False,
    ) -> tuple[tuple[IntType, IntType], tuple[IntType, IntType]]:
        ...
                
    @overload
    def get_index_range(
        self,
        loc: str,
        with_lev_idx: Literal[False] = False,
        with_halo: bool = False,
    ) -> tuple[IntType, IntType]:
        ...
    
    def get_index_range(
        self,
        loc: str,
        with_lev_idx: bool = False,
        with_halo: bool = False,
    ) -> Union[tuple[IntType, IntType], tuple[tuple[IntType, IntType], tuple[IntType, IntType]]]:
        if with_halo:
            if loc == "cell" or loc == "lev":
                i_s = self.cell_ims; i_e = self.cell_ime
            elif loc == "vtx" or loc == "lev_vtx":
                i_s = self.vtx_ims ; i_e = self.vtx_ime
            elif loc == "edge" or loc == "lev_edge":
                i_s = self.edge_ims; i_e = self.edge_ime
            else:
                raise ValueError(f"Unsupported location: {loc}!")
        else:
            if loc == "cell" or loc == "lev":
                i_s = self.cell_ids; i_e = self.cell_ide
            elif loc == "vtx" or loc == "lev_vtx":
                i_s = self.vtx_ids ; i_e = self.vtx_ide
            elif loc == "edge" or loc == "lev_edge":
                i_s = self.edge_ids; i_e = self.edge_ide
            else:
                raise ValueError(f"Unsupported location: {loc}!")
        
        if not with_lev_idx:
            return (i_s, i_e)
        else:
            if loc == "cell" or loc == "vtx" or loc == "edge":
                k_s = self.full_kms; k_e = self.full_kme
            elif loc == "lev" or loc == "lev_vtx" or loc == "lev_edge":
                k_s = self.half_kms; k_e = self.half_kme
            else:
                raise ValueError(f"Unsupported location: {loc}!")
            return (i_s, i_e), (k_s, k_e)
    
    @classmethod
    def init_global_from_ref_mesh(
        cls,
        ref_mesh_type: str,
        para: PlanetParameterSet,
        vert_coord: Optional[AbstractGeneralizedTerrainFollowingCoordinate] = None,
        hori_hw: int = 3,
        lev_hw: int = 3,
        grist_mesh_0d_file: Optional[str] = None, 
        grist_mesh_2d_file: Optional[str] = None, 
        dev: device.AbstractDevice = device.CPU(),
    ) -> Self:
        int_type = dev.int_type
        radius = para.radius.unwrap()
        # ====================================================
        # Initilize one layer hybrid vert coord if vert_coord is None
        if vert_coord is None:
            vert_coord = OneLayerVertCoordinate(dev=dev)
        # Read from reference mesh
        if ref_mesh_type == "grist":
            if grist_mesh_0d_file is None or grist_mesh_2d_file is None:
                raise ValueError("Must provide 'grist_mesh_0d_file' and 'grist_mesh_2d_file' when initializing grist mesh")
            grist_mesh = GRISTRefMesh(
                mesh_0d_file=grist_mesh_0d_file, 
                mesh_2d_file=grist_mesh_2d_file,
                dev=dev,
            )
            fields: dict[str, Any] = dict(
                n_cell       =grist_mesh.nv,
                n_vtx        =grist_mesh.nt,
                n_edge       =grist_mesh.ne,
                maxnnb_cell  =grist_mesh.maxnnb_plg,
                maxnnb_vtx   =grist_mesh.maxnnb_tri,
                glevel       =grist_mesh.glevel,
                cell_lon     =grist_mesh.vtx_lon,
                cell_lat     =grist_mesh.vtx_lat,
                cell_xyz     =grist_mesh.vtx_p * radius,
                vtx_lon      =grist_mesh.tri_cc_lon,
                vtx_lat      =grist_mesh.tri_cc_lat,
                vtx_xyz      =grist_mesh.tri_cc_p * radius,
                edge_lon     =grist_mesh.edp_cc_lon,
                edge_lat     =grist_mesh.edp_cc_lat,
                edge_xyz     =grist_mesh.edp_cc_p * radius,
                dual_edge_lon=grist_mesh.edt_cc_lon,
                dual_edge_lat=grist_mesh.edt_cc_lat,
                dual_edge_xyz=grist_mesh.edt_cc_p * radius,
                cell_nnb     =grist_mesh.vtx_nnb,
                cell_nb      =grist_mesh.vtx_nb,
                cell_ed      =grist_mesh.vtx_ed,
                vtx_of_cell  =grist_mesh.vtx_tr,
                vtx_nb       =grist_mesh.tri_nb,
                vtx_ed       =grist_mesh.tri_ed,
                cell_of_vtx  =grist_mesh.tri_v,
                edge_nr      =grist_mesh.edp_nr,
                edge_tg      =grist_mesh.edp_tg,
                cell_of_edge =grist_mesh.edt_v,
                dual_edge_nr =grist_mesh.edt_nr,
                dual_edge_tg =grist_mesh.edt_tg,
                vtx_of_edge  =grist_mesh.edp_v,
                area_cell    =grist_mesh.plg_area * radius ** 2,
                area_vtx     =grist_mesh.tri_area * radius ** 2,
                le           =grist_mesh.edp_len * radius,
                de           =grist_mesh.edt_len * radius,
            )
        else:
            raise ValueError(f"Unsupported ref_mesh_type: {ref_mesh_type}")
       
        # ====================================================
        # Global initialize
        fields["id"      ] = 0
        fields["hori_hw" ] = hori_hw
        fields["lev_hw"  ] = lev_hw
        fields["cell_ins"] = 0
        fields["cell_ine"] = fields["n_cell"] - 1
        fields["vtx_ins" ] = 0 
        fields["vtx_ine" ] = fields["n_vtx" ] - 1
        fields["edge_ins"] = 0
        fields["edge_ine"] = fields["n_edge"] - 1
        fields["cell_ids"] = 0
        fields["cell_ide"] = fields["n_cell"] - 1
        fields["vtx_ids" ] = 0 
        fields["vtx_ide" ] = fields["n_vtx" ] - 1
        fields["edge_ids"] = 0
        fields["edge_ide"] = fields["n_edge"] - 1    
        fields["cell_ims"] = 0
        fields["cell_ime"] = fields["n_cell"] - 1
        fields["vtx_ims" ] = 0 
        fields["vtx_ime" ] = fields["n_vtx" ] - 1
        fields["edge_ims"] = 0
        fields["edge_ime"] = fields["n_edge"] - 1
        fields["cell_ibs"] = tuple(0 for _ in range(hori_hw))
        fields["cell_ibe"] = tuple(fields["n_cell"] - 1 for _ in range(hori_hw))
        fields["vtx_ibs" ] = tuple(0 for _ in range(hori_hw))
        fields["vtx_ibe" ] = tuple(fields["n_vtx" ] - 1 for _ in range(hori_hw))
        fields["edge_ibs"] = tuple(0 for _ in range(hori_hw))
        fields["edge_ibe"] = tuple(fields["n_edge"] - 1 for _ in range(hori_hw))
        fields["cell_ihs"] = tuple(0 for _ in range(hori_hw))
        fields["cell_ihe"] = tuple(fields["n_cell"] - 1 for _ in range(hori_hw))
        fields["vtx_ihs" ] = tuple(0 for _ in range(hori_hw))
        fields["vtx_ihe" ] = tuple(fields["n_vtx" ] - 1 for _ in range(hori_hw))
        fields["edge_ihs"] = tuple(0 for _ in range(hori_hw))
        fields["edge_ihe"] = tuple(fields["n_edge"] - 1 for _ in range(hori_hw))
        fields["full_kds"] = 0
        fields["full_kde"] = vert_coord.full_nlev - 1
        fields["half_kds"] = 0
        fields["half_kde"] = vert_coord.half_nlev - 1
        fields["full_kms"] = 0
        fields["full_kme"] = vert_coord.full_nlev - 1
        fields["half_kms"] = 0
        fields["half_kme"] = vert_coord.half_nlev - 1
        
        # Convert lon range from [-pi, pi] to [0, 2pi] if needed
        fields["cell_lon"     ] = jnp.where(fields["cell_lon"     ] < 0, fields["cell_lon"     ] + jnp.pi*2, fields["cell_lon"     ])
        fields["vtx_lon"      ] = jnp.where(fields["vtx_lon"      ] < 0, fields["vtx_lon"      ] + jnp.pi*2, fields["vtx_lon"      ])
        fields["edge_lon"     ] = jnp.where(fields["edge_lon"     ] < 0, fields["edge_lon"     ] + jnp.pi*2, fields["edge_lon"     ])
        fields["dual_edge_lon"] = jnp.where(fields["dual_edge_lon"] < 0, fields["dual_edge_lon"] + jnp.pi*2, fields["dual_edge_lon"]) 
      
        fields["cell_lon_deg"     ] = jnp.rad2deg(fields["cell_lon"     ])
        fields["cell_lat_deg"     ] = jnp.rad2deg(fields["cell_lat"     ])
        fields["vtx_lon_deg"      ] = jnp.rad2deg(fields["vtx_lon"      ])
        fields["vtx_lat_deg"      ] = jnp.rad2deg(fields["vtx_lat"      ])
        fields["edge_lon_deg"     ] = jnp.rad2deg(fields["edge_lon"     ])
        fields["edge_lat_deg"     ] = jnp.rad2deg(fields["edge_lat"     ])
        fields["dual_edge_lon_deg"] = jnp.rad2deg(fields["dual_edge_lon"])
        fields["dual_edge_lat_deg"] = jnp.rad2deg(fields["dual_edge_lat"])

        fields["full_nlev"] = vert_coord.full_nlev
        fields["half_nlev"] = vert_coord.half_nlev    
        fields["full_lev" ] = vert_coord.full_lev
        fields["full_dlev"] = vert_coord.full_dlev
        fields["half_lev" ] = vert_coord.half_lev
        fields["half_dlev"] = vert_coord.half_dlev

        fields["global_cell_id"] = jnp.arange(fields["n_cell"], dtype=int_type)
        fields["global_vtx_id" ] = jnp.arange(fields["n_vtx" ], dtype=int_type)
        fields["global_edge_id"] = jnp.arange(fields["n_edge"], dtype=int_type)
        fields["vtx_nnb"       ] = jnp.full(
            shape=(fields["n_vtx"],),
            fill_value=fields["maxnnb_vtx"],
            device=dev(),
            dtype=int_type,
        )
        fields["total_area"       ] = jnp.sum(fields["area_cell"])
        # ====================================================
        return cls(**fields)
  
    @classmethod
    def from_parent(
        cls,
        parent: "SCVTMesh",
        para: PlanetParameterSet,
        id: Optional[int] = None,
        dev: device.AbstractDevice = device.CPU()
    ) -> "SCVTMesh":
        int_type   = dev.int_type
        float_type = dev.float_type
        omega = para.omega.unwrap()
        # ====================================================
        # Decompose parent global mesh into local mesh
        scvt_proc = scvt_decomp_run(
            mesh=parent,
            proc=dev.proc,
            hori_hw=int(parent.hori_hw),
        )
        halo = SCVTHalo(proc=scvt_proc, mesh=parent, dev=dev)
        
        l2g_cell_id     = jnp.asarray(scvt_proc.l2g_cell_id    , device=dev(), dtype=int_type)
        l2g_vtx_id      = jnp.asarray(scvt_proc.l2g_vtx_id     , device=dev(), dtype=int_type)
        l2g_edge_id     = jnp.asarray(scvt_proc.l2g_edge_id    , device=dev(), dtype=int_type)
        g2l_cell_id     = jnp.asarray(scvt_proc.g2l_cell_id    , device=dev(), dtype=int_type)
        g2l_vtx_id      = jnp.asarray(scvt_proc.g2l_vtx_id     , device=dev(), dtype=int_type)
        g2l_edge_id     = jnp.asarray(scvt_proc.g2l_edge_id    , device=dev(), dtype=int_type)
        cell_proc_idmap = jnp.asarray(scvt_proc.cell_proc_idmap, device=dev(), dtype=int_type)
        # ====================================================
        # Prepare parent mesh
        parent = eqx.tree_at(
            where=lambda m: m.cell_proc_idmap,
            pytree=parent,
            replace=cell_proc_idmap
        )
        # ====================================================
        # Determine domain indices
        # haloN -> haloN-1 -> ... -> halo1 -> bdry1 -> bdry2 -> ... -> bdryN -> inner 
        cell_ins = 0; cell_ine = scvt_proc.n_cell_inner - 1
        vtx_ins  = 0; vtx_ine  = scvt_proc.n_vtx_inner  - 1
        edge_ins = 0; edge_ine = scvt_proc.n_edge_inner - 1
        cell_ids = 0; cell_ide = scvt_proc.n_cell_cmpt - 1
        vtx_ids  = 0; vtx_ide  = scvt_proc.n_vtx_cmpt  - 1
        edge_ids = 0; edge_ide = scvt_proc.n_edge_cmpt - 1
        cell_ims = 0; cell_ime = scvt_proc.n_cell - 1
        vtx_ims  = 0; vtx_ime  = scvt_proc.n_vtx  - 1
        edge_ims = 0; edge_ime = scvt_proc.n_edge - 1
        
        # Determine bdry indices
        cell_ibs_list = [0] * parent.hori_hw
        cell_ibe_list = [0] * parent.hori_hw
        vtx_ibs_list  = [0] * parent.hori_hw
        vtx_ibe_list  = [0] * parent.hori_hw
        edge_ibs_list = [0] * parent.hori_hw
        edge_ibe_list = [0] * parent.hori_hw
        if (scvt_proc.nngb > 0):
            cell_ibs_list[-1] = cell_ine + 1
            cell_ibe_list[-1] = cell_ine + scvt_proc.n_cell_bdry[-1]
            vtx_ibs_list [-1] = vtx_ine  + 1
            vtx_ibe_list [-1] = vtx_ine  + scvt_proc.n_vtx_bdry [-1]
            edge_ibs_list[-1] = edge_ine + 1
            edge_ibe_list[-1] = edge_ine + scvt_proc.n_edge_bdry[-1]
        else:
            cell_ibs_list[-1] = cell_ids
            cell_ibe_list[-1] = cell_ide
            vtx_ibs_list [-1] = vtx_ids
            vtx_ibe_list [-1] = vtx_ide
            edge_ibs_list[-1] = edge_ids
            edge_ibe_list[-1] = edge_ide
        for hw in range(parent.hori_hw - 2, -1, -1):
            cell_ibs_list[hw] = scvt_proc.n_cell_bdry[hw+1] + cell_ibs_list[hw+1]
            cell_ibe_list[hw] = scvt_proc.n_cell_bdry[hw  ] + cell_ibe_list[hw+1]
            vtx_ibs_list [hw] = scvt_proc.n_vtx_bdry [hw+1] + vtx_ibs_list [hw+1]
            vtx_ibe_list [hw] = scvt_proc.n_vtx_bdry [hw  ] + vtx_ibe_list [hw+1]            
            edge_ibs_list[hw] = scvt_proc.n_edge_bdry[hw+1] + edge_ibs_list[hw+1]
            edge_ibe_list[hw] = scvt_proc.n_edge_bdry[hw  ] + edge_ibe_list[hw+1]
        
        if scvt_proc.nngb > 0:
            assert cell_ibs_list[-1] == cell_ine + 1
            assert vtx_ibs_list [-1] == vtx_ine  + 1
            assert edge_ibs_list[-1] == edge_ine + 1
            for hw in range(parent.hori_hw - 2, -1, -1):
                assert cell_ibs_list[hw] == cell_ibe_list[hw+1] + 1
                assert vtx_ibs_list [hw] == vtx_ibe_list [hw+1] + 1
                assert edge_ibs_list[hw] == edge_ibe_list[hw+1] + 1
            assert cell_ibe_list[0 ] == cell_ide
            assert vtx_ibe_list [0 ] == vtx_ide
            assert edge_ibe_list[0 ] == edge_ide
        
        # Determine halo indices 
        cell_ihs_list = [0] * parent.hori_hw
        cell_ihe_list = [0] * parent.hori_hw
        vtx_ihs_list  = [0] * parent.hori_hw
        vtx_ihe_list  = [0] * parent.hori_hw
        edge_ihs_list = [0] * parent.hori_hw
        edge_ihe_list = [0] * parent.hori_hw
        if (scvt_proc.nngb > 0):
            cell_ihs_list[0] = cell_ibe_list[0] + 1
            cell_ihe_list[0] = cell_ibe_list[0] + scvt_proc.n_cell_halo[0]
            vtx_ihs_list [0] = vtx_ibe_list [0] + 1
            vtx_ihe_list [0] = vtx_ibe_list [0] + scvt_proc.n_vtx_halo [0]
            edge_ihs_list[0] = edge_ibe_list[0] + 1
            edge_ihe_list[0] = edge_ibe_list[0] + scvt_proc.n_edge_halo[0]
        else:
            cell_ihs_list[0] = cell_ids
            cell_ihe_list[0] = cell_ide
            vtx_ihs_list [0] = vtx_ids
            vtx_ihe_list [0] = vtx_ide
            edge_ihs_list[0] = edge_ids
            edge_ihe_list[0] = edge_ide
        for hw in range(1, parent.hori_hw, 1):
            cell_ihs_list[hw] = scvt_proc.n_cell_halo[hw-1] + cell_ihs_list[hw-1]
            cell_ihe_list[hw] = scvt_proc.n_cell_halo[hw  ] + cell_ihe_list[hw-1]
            vtx_ihs_list [hw] = scvt_proc.n_vtx_halo [hw-1] + vtx_ihs_list [hw-1]
            vtx_ihe_list [hw] = scvt_proc.n_vtx_halo [hw  ] + vtx_ihe_list [hw-1]            
            edge_ihs_list[hw] = scvt_proc.n_edge_halo[hw-1] + edge_ihs_list[hw-1]
            edge_ihe_list[hw] = scvt_proc.n_edge_halo[hw  ] + edge_ihe_list[hw-1]
            
        cell_ibs = tuple(cell_ibs_list); cell_ibe = tuple(cell_ibe_list)
        vtx_ibs  = tuple(vtx_ibs_list ); vtx_ibe  = tuple(vtx_ibe_list )
        edge_ibs = tuple(edge_ibs_list); edge_ibe = tuple(edge_ibe_list)
        cell_ihs = tuple(cell_ihs_list); cell_ihe = tuple(cell_ihe_list)
        vtx_ihs  = tuple(vtx_ihs_list ); vtx_ihe  = tuple(vtx_ihe_list )
        edge_ihs = tuple(edge_ihs_list); edge_ihe = tuple(edge_ihe_list)
        
        if scvt_proc.nngb > 0:
            assert cell_ihs_list[0] == cell_ide + 1
            assert vtx_ihs_list [0] == vtx_ide  + 1
            assert edge_ihs_list[0] == edge_ide + 1
            for hw in range(1, parent.hori_hw, 1):
                assert cell_ihs_list[hw] == cell_ihe_list[hw-1] + 1
                assert vtx_ihs_list [hw] == vtx_ihe_list [hw-1] + 1
                assert edge_ihs_list[hw] == edge_ihe_list[hw-1] + 1
            assert cell_ihe_list[-1] == cell_ime
            assert vtx_ihe_list [-1] == vtx_ime
            assert edge_ihe_list[-1] == edge_ime
        # ====================================================
        fields: dict[str, Any] = dict(
            parent=parent,
            halo=halo,
            id=int(scvt_proc.id_model if id is None else id),
            hori_hw=int(scvt_proc.hori_hw),
            lev_hw=parent.lev_hw,
            n_cell=int(scvt_proc.n_cell),
            n_vtx=int(scvt_proc.n_vtx),
            n_edge=int(scvt_proc.n_edge),
            full_nlev=parent.full_nlev,
            half_nlev=parent.half_nlev,
            maxnnb_cell=parent.maxnnb_cell,
            maxnnb_vtx=parent.maxnnb_vtx,
            glevel=parent.glevel,
            cell_ids=cell_ids,
            cell_ide=cell_ide,
            cell_ims=cell_ims,
            cell_ime=cell_ime,
            cell_ins=cell_ins,
            cell_ine=cell_ine,
            cell_ibs=cell_ibs,
            cell_ibe=cell_ibe,
            cell_ihs=cell_ihs,
            cell_ihe=cell_ihe,
            vtx_ids=vtx_ids,
            vtx_ide=vtx_ide,
            vtx_ims=vtx_ims,
            vtx_ime=vtx_ime,
            vtx_ins=vtx_ins,
            vtx_ine=vtx_ine,
            vtx_ibs=vtx_ibs,
            vtx_ibe=vtx_ibe,
            vtx_ihs=vtx_ihs,
            vtx_ihe=vtx_ihe,
            edge_ids=edge_ids,
            edge_ide=edge_ide,
            edge_ims=edge_ims,
            edge_ime=edge_ime,
            edge_ins=edge_ins,
            edge_ine=edge_ine,
            edge_ibs=edge_ibs,
            edge_ibe=edge_ibe,
            edge_ihs=edge_ihs,
            edge_ihe=edge_ihe,
            full_kds=parent.full_kds,
            full_kde=parent.full_kde,
            half_kds=parent.half_kds,
            half_kde=parent.half_kde,
            full_kms=parent.full_kms,
            full_kme=parent.full_kme,
            half_kms=parent.half_kms,
            half_kme=parent.half_kme,
            full_lev=parent.full_lev,
            half_lev=parent.half_lev,
            full_dlev=parent.full_dlev,
            half_dlev=parent.half_dlev,
            global_cell_id=l2g_cell_id,
            global_vtx_id=l2g_vtx_id,
            global_edge_id=l2g_edge_id,
            cell_lon=parent.cell_lon[l2g_cell_id],
            cell_lat=parent.cell_lat[l2g_cell_id],
            cell_xyz=parent.cell_xyz[l2g_cell_id],
            vtx_lon=parent.vtx_lon[l2g_vtx_id],
            vtx_lat=parent.vtx_lat[l2g_vtx_id],
            vtx_xyz=parent.vtx_xyz[l2g_vtx_id],
            edge_lon=parent.edge_lon[l2g_edge_id],
            edge_lat=parent.edge_lat[l2g_edge_id],
            edge_xyz=parent.edge_xyz[l2g_edge_id],
            dual_edge_lon=parent.dual_edge_lon[l2g_edge_id],
            dual_edge_lat=parent.dual_edge_lat[l2g_edge_id],
            dual_edge_xyz=parent.dual_edge_xyz[l2g_edge_id],
            cell_nnb=parent.cell_nnb[l2g_cell_id],
            vtx_nnb=parent.vtx_nnb[l2g_vtx_id],
            cell_nb=remap_g2l(parent.cell_nb[l2g_cell_id], g2l_cell_id),
            cell_ed=remap_g2l(parent.cell_ed[l2g_cell_id], g2l_edge_id),
            vtx_of_cell=remap_g2l(parent.vtx_of_cell[l2g_cell_id], g2l_vtx_id),
            vtx_nb=remap_g2l(parent.vtx_nb[l2g_vtx_id], g2l_vtx_id),
            vtx_ed=remap_g2l(parent.vtx_ed[l2g_vtx_id], g2l_edge_id),
            cell_of_vtx=remap_g2l(parent.cell_of_vtx[l2g_vtx_id], g2l_cell_id),
            cell_of_edge=remap_g2l(parent.cell_of_edge[l2g_edge_id], g2l_cell_id),
            vtx_of_edge=remap_g2l(parent.vtx_of_edge[l2g_edge_id], g2l_vtx_id),
            edge_nr=parent.edge_nr[l2g_edge_id],
            edge_tg=parent.edge_tg[l2g_edge_id],
            dual_edge_nr=parent.dual_edge_nr[l2g_edge_id],
            dual_edge_tg=parent.dual_edge_tg[l2g_edge_id],
            area_cell=parent.area_cell[l2g_cell_id],
            area_vtx=parent.area_vtx[l2g_vtx_id], 
            le=parent.le[l2g_edge_id],
            de=parent.de[l2g_edge_id], 
            cell_lon_deg=parent.cell_lon_deg[l2g_cell_id],
            cell_lat_deg=parent.cell_lat_deg[l2g_cell_id],
            vtx_lon_deg=parent.vtx_lon_deg[l2g_vtx_id],
            vtx_lat_deg=parent.vtx_lat_deg[l2g_vtx_id],
            edge_lon_deg=parent.edge_lon_deg[l2g_edge_id],
            edge_lat_deg=parent.edge_lat_deg[l2g_edge_id],
            dual_edge_lon_deg=parent.dual_edge_lon_deg[l2g_edge_id],
            dual_edge_lat_deg=parent.dual_edge_lat_deg[l2g_edge_id],
        )
        fields["cell_cos_lon"     ] = jnp.cos(fields["cell_lon"     ])
        fields["cell_sin_lon"     ] = jnp.sin(fields["cell_lon"     ])
        fields["cell_cos_lat"     ] = jnp.cos(fields["cell_lat"     ])
        fields["cell_sin_lat"     ] = jnp.sin(fields["cell_lat"     ])
        fields["vtx_cos_lon"      ] = jnp.cos(fields["vtx_lon"      ])
        fields["vtx_sin_lon"      ] = jnp.sin(fields["vtx_lon"      ])
        fields["vtx_cos_lat"      ] = jnp.cos(fields["vtx_lat"      ])
        fields["vtx_sin_lat"      ] = jnp.sin(fields["vtx_lat"      ])
        fields["edge_cos_lon"     ] = jnp.cos(fields["edge_lon"     ])
        fields["edge_sin_lon"     ] = jnp.sin(fields["edge_lon"     ])
        fields["edge_cos_lat"     ] = jnp.cos(fields["edge_lat"     ])
        fields["edge_sin_lat"     ] = jnp.sin(fields["edge_lat"     ])
        fields["dual_edge_cos_lon"] = jnp.cos(fields["dual_edge_lon"])
        fields["dual_edge_sin_lon"] = jnp.sin(fields["dual_edge_lon"])
        fields["dual_edge_cos_lat"] = jnp.cos(fields["dual_edge_lat"])
        fields["dual_edge_sin_lat"] = jnp.sin(fields["dual_edge_lat"])
        fields["f_vtx"            ] = 2 * omega * fields["vtx_sin_lat"] 
        fields["area_edge"        ] = fields["le"] * fields["de"] * 0.5
        fields["total_area"       ] = parent.total_area
        fields["area_vtx_wgt"     ] = fields["area_vtx"]
        fields["area_edge_wgt"    ] = fields["area_edge"]

        cell_owner = np.asarray(scvt_proc.cell_proc_idmap, dtype=np.int32)
        local_cell_of_vtx = np.asarray(fields["cell_of_vtx"], dtype=np.int64)
        local_cell_of_edge = np.asarray(fields["cell_of_edge"], dtype=np.int64)
        global_cell_id = np.asarray(fields["global_cell_id"], dtype=np.int64)
        area_vtx_wgt = np.asarray(fields["area_vtx_wgt"]).copy()
        area_edge_wgt = np.asarray(fields["area_edge_wgt"]).copy()
        if scvt_proc.nngb > 0 and fields["vtx_ibs"]:
            start, end = fields["vtx_ibs"][0], fields["vtx_ibe"][0]
            for i in range(start, end + 1):
                local_cells = local_cell_of_vtx[i]
                valid = local_cells >= 0
                owners = cell_owner[global_cell_id[local_cells[valid]]]
                divisor = 1 + int(np.count_nonzero(owners != scvt_proc.id_model))
                area_vtx_wgt[i] = area_vtx_wgt[i] / divisor
        if scvt_proc.nngb > 0 and fields["edge_ibs"]:
            start, end = fields["edge_ibs"][0], fields["edge_ibe"][0]
            for i in range(start, end + 1):
                local_cells = local_cell_of_edge[i]
                valid = local_cells >= 0
                owners = cell_owner[global_cell_id[local_cells[valid]]]
                divisor = 1 + int(np.count_nonzero(owners != scvt_proc.id_model))
                area_edge_wgt[i] = area_edge_wgt[i] / divisor
        fields["area_vtx_wgt"] = jnp.asarray(area_vtx_wgt, dtype=fields["area_vtx"].dtype)
        fields["area_edge_wgt"] = jnp.asarray(area_edge_wgt, dtype=fields["area_edge"].dtype)
        
        fields["kite_area_cell"] = jnp.zeros(
            (cell_ime-cell_ims+1, fields["maxnnb_cell"]),
            device=dev(),
            dtype=float_type  
        )
        fields["kite_area_vtx"] = jnp.zeros(
            (vtx_ime-vtx_ims+1, fields["maxnnb_vtx"]),
            device=dev(),
            dtype=float_type  
        )        
        fields["cell_nr"] = jnp.zeros(
            (cell_ime-cell_ims+1, fields["maxnnb_cell"]),
            device=dev(),
            dtype=int_type
        )
        fields["vtx_nr"] = jnp.zeros(
            (vtx_ime-vtx_ims+1, fields["maxnnb_vtx"]),
            device=dev(),
            dtype=int_type
        )
        fields["edptg_eddnr"] = jnp.zeros(
            (edge_ime-edge_ims+1),
            device=dev(),
            dtype=int_type
        )
        fields["edpnr_eddtg"] = jnp.zeros(
            (edge_ime-edge_ims+1),
            device=dev(),
            dtype=int_type
        ) 
        fields["edp12_edptg"] = jnp.zeros(
            (edge_ime-edge_ims+1),
            device=dev(),
            dtype=int_type
        )
        fields["edp12_eddnr"] = jnp.zeros(
            (edge_ime-edge_ims+1),
            device=dev(),
            dtype=int_type
        )
        self = cls(**fields)
        self = calc_kite_area(para, self)
        self = correct_normal_vectors(para, self)
        return self
    
    @classmethod
    def from_grist(
        cls,
        mesh_0d_file: str, 
        mesh_2d_file: str,
        para: PlanetParameterSet,
        vert_coord: Optional[AbstractGeneralizedTerrainFollowingCoordinate] = None,
        hori_hw: int = 3,
        lev_hw: int = 3,
        dev: device.AbstractDevice = device.CPU()
    ) -> "SCVTMesh":
        global_mesh = cls.init_global_from_ref_mesh(
            ref_mesh_type="grist",
            vert_coord=vert_coord,
            para=para,
            hori_hw=hori_hw,
            lev_hw=lev_hw,
            grist_mesh_0d_file=mesh_0d_file,
            grist_mesh_2d_file=mesh_2d_file,
            dev=dev
        )
        return cls.from_parent(
            parent=global_mesh,
            para=para,
            id=0,
            dev=dev,
        )
        
    def lon_range(self) -> tuple[float, ...]:
        
        min_lon = min(
            self.cell_lon_deg     .min(), 
            self.vtx_lon_deg      .min(), 
            self.edge_lon_deg     .min(),
            self.dual_edge_lon_deg.min()
        )
        max_lon = max(
            self.cell_lon_deg     .max(), 
            self.vtx_lon_deg      .max(),
            self.edge_lon_deg     .max(),
            self.dual_edge_lon_deg.max()
        )
        
        if bool(jnp.allclose(min_lon, 0)) and bool(jnp.allclose(max_lon, 360)):
            # cyclic = True
            cell_lon_deg = jnp.where(
                self.cell_lon_deg >= 180.0, 
                self.cell_lon_deg - 360.0, 
                self.cell_lon_deg
            )
            vtx_lon_deg = jnp.where(
                self.vtx_lon_deg >= 180.0, 
                self.vtx_lon_deg - 360.0, 
                self.vtx_lon_deg
            )
            edge_lon_deg = jnp.where(
                self.edge_lon_deg >= 180.0, 
                self.edge_lon_deg - 360.0, 
                self.edge_lon_deg
            )
            dual_edge_lon_deg = jnp.where(
                self.dual_edge_lon_deg >= 180.0, 
                self.dual_edge_lon_deg - 360.0, 
                self.dual_edge_lon_deg
            )
            min_lon = min(
                cell_lon_deg     .min(), 
                vtx_lon_deg      .min(), 
                edge_lon_deg     .min(),
                dual_edge_lon_deg.min()
            )
            max_lon = max(
                cell_lon_deg     .max(), 
                vtx_lon_deg      .max(),
                edge_lon_deg     .max(),
                dual_edge_lon_deg.max()
            )
            if bool(jnp.allclose(min_lon, -180)) and bool(jnp.allclose(max_lon, 180)):
                min_lon = 0.0
                max_lon = 360.0
            else:
                min_lon = min_lon + 360.0
                max_lon = max_lon
        else:
            # cyclic = False
            pass
            
        return float(min_lon), float(max_lon)
    
    def lat_range(self) -> tuple[float, ...]:
        min_lat = min(
            self.cell_lat_deg     .min(), 
            self.vtx_lat_deg      .min(), 
            self.edge_lat_deg     .min(),
            self.dual_edge_lat_deg.min()
        )
        max_lat = max(
            self.cell_lat_deg     .max(), 
            self.vtx_lat_deg      .max(), 
            self.edge_lat_deg     .max(),
            self.dual_edge_lat_deg.max()
        )
        return float(min_lat), float(max_lat)
        
    def output(
        self,
        proc: Proc,
        output_path: str = "",
        file_name: str = "scvt_mesh"
    ):
        halo      = self.halo
        file_name = file_name + "." + str(proc.id_model) + ".nc"
        output_file = Path(output_path).joinpath(file_name)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        nc = self.cell_ime - self.cell_ims + 1
        nv = self.vtx_ime  - self.vtx_ims  + 1
        ne = self.edge_ime - self.edge_ims + 1

        nngb          = max(1, int(halo.nngb))
        max_send_cell = max(1, int(halo.max_send_n_cell))
        max_send_vtx  = max(1, int(halo.max_send_n_vtx ))
        max_send_edge = max(1, int(halo.max_send_n_edge))
        max_recv_cell = max(1, int(halo.max_recv_n_cell))
        max_recv_vtx  = max(1, int(halo.max_recv_n_vtx ))
        max_recv_edge = max(1, int(halo.max_recv_n_edge))

        send_ncell = np.zeros(nngb, dtype=np.int32)
        send_nvtx  = np.zeros(nngb, dtype=np.int32)
        send_nedge = np.zeros(nngb, dtype=np.int32)
        recv_ncell = np.zeros(nngb, dtype=np.int32)
        recv_nvtx  = np.zeros(nngb, dtype=np.int32)
        recv_nedge = np.zeros(nngb, dtype=np.int32)
        send_cell  = np.zeros((nngb, max_send_cell), dtype=np.int32)
        send_vtx   = np.zeros((nngb, max_send_vtx ), dtype=np.int32)
        send_edge  = np.zeros((nngb, max_send_edge), dtype=np.int32)
        recv_cell  = np.zeros((nngb, max_recv_cell), dtype=np.int32)
        recv_vtx   = np.zeros((nngb, max_recv_vtx ), dtype=np.int32)
        recv_edge  = np.zeros((nngb, max_recv_edge), dtype=np.int32)

        if halo.nngb > 0:
            send_ncell[:halo.nngb] = np.asarray(halo.send_n_cell, dtype=np.int32)
            send_nvtx [:halo.nngb] = np.asarray(halo.send_n_vtx , dtype=np.int32)
            send_nedge[:halo.nngb] = np.asarray(halo.send_n_edge, dtype=np.int32)
            recv_ncell[:halo.nngb] = np.asarray(halo.recv_n_cell, dtype=np.int32)
            recv_nvtx [:halo.nngb] = np.asarray(halo.recv_n_vtx , dtype=np.int32)
            recv_nedge[:halo.nngb] = np.asarray(halo.recv_n_edge, dtype=np.int32)
            send_cell [:halo.nngb, :halo.max_send_n_cell] = np.asarray(halo.send_cell_id, dtype=np.int32)
            send_vtx  [:halo.nngb, :halo.max_send_n_vtx ] = np.asarray(halo.send_vtx_id , dtype=np.int32)
            send_edge [:halo.nngb, :halo.max_send_n_edge] = np.asarray(halo.send_edge_id, dtype=np.int32)
            recv_cell [:halo.nngb, :halo.max_recv_n_cell] = np.asarray(halo.recv_cell_id, dtype=np.int32)
            recv_vtx  [:halo.nngb, :halo.max_recv_n_vtx ] = np.asarray(halo.recv_vtx_id , dtype=np.int32)
            recv_edge [:halo.nngb, :halo.max_recv_n_edge] = np.asarray(halo.recv_edge_id, dtype=np.int32)
        
        if proc.is_root():
            log.logger.info("Output SCVT mesh to file.")
        
        ds = nc4.Dataset(output_file, mode="w", format="NETCDF4")
        ds.createDimension("nc"           , int(nc))
        ds.createDimension("nv"           , int(nv))
        ds.createDimension("ne"           , int(ne))
        ds.createDimension("maxnnb_cell"  , int(self.maxnnb_cell))
        ds.createDimension("maxnnb_vtx"   , int(self.maxnnb_vtx))
        ds.createDimension("nnb_of_proc"  , nngb)
        ds.createDimension("max_send_cell", max_send_cell)
        ds.createDimension("max_send_vtx" , max_send_vtx)
        ds.createDimension("max_send_edge", max_send_edge)
        ds.createDimension("max_recv_cell", max_recv_cell)
        ds.createDimension("max_recv_vtx" , max_recv_vtx)
        ds.createDimension("max_recv_edge", max_recv_edge)
        ds.createDimension("hori_hw"      , int(self.hori_hw))
        ds.createDimension("one"          , 1)
        ds.createDimension("two"          , 2)
        ds.createDimension("three"        , 3)
        ds.createDimension("six"          , 6)

        int_type       = "i4"
        float_type     = "f8"
        cell_ins       = ds.createVariable("cell_ins", int_type, ("one"    ,))
        cell_ins.long_name = "Start index of inner cell"
        cell_ins.units = ""
        cell_ine       = ds.createVariable("cell_ine", int_type, ("one"    ,))
        cell_ine.long_name = "End index of inner cell"
        cell_ine.units = ""
        vtx_ins        = ds.createVariable("vtx_ins" , int_type, ("one"    ,))
        vtx_ins.long_name = "Start index of inner vertex"
        vtx_ins.units = ""
        vtx_ine        = ds.createVariable("vtx_ine" , int_type, ("one"    ,))
        vtx_ine.long_name = "End index of inner vertex"
        vtx_ine.units = ""
        edge_ins       = ds.createVariable("edge_ins", int_type, ("one"    ,))
        edge_ins.long_name = "Start index of inner edge"
        edge_ins.units = ""
        edge_ine       = ds.createVariable("edge_ine", int_type, ("one"    ,))
        edge_ine.long_name = "End index of inner edge"
        edge_ine.units = ""
        cell_ibs       = ds.createVariable("cell_ibs", int_type, ("hori_hw",))
        cell_ibs.long_name = "Start index of bdry cell"
        cell_ibs.units = ""
        cell_ibe       = ds.createVariable("cell_ibe", int_type, ("hori_hw",))
        cell_ibe.long_name = "End index of bdry cell"
        cell_ibe.units = ""
        vtx_ibs        = ds.createVariable("vtx_ibs" , int_type, ("hori_hw",))
        vtx_ibs.long_name = "Start index of bdry vertex"
        vtx_ibs.units = ""
        vtx_ibe        = ds.createVariable("vtx_ibe" , int_type, ("hori_hw",))
        vtx_ibe.long_name = "End index of bdry vertex"
        vtx_ibe.units = ""
        edge_ibs       = ds.createVariable("edge_ibs", int_type, ("hori_hw",))
        edge_ibs.long_name = "Start index of bdry edge"
        edge_ibs.units = ""
        edge_ibe       = ds.createVariable("edge_ibe", int_type, ("hori_hw",))
        edge_ibe.long_name = "End index of bdry edge"
        edge_ibe.units = ""
        cell_ids       = ds.createVariable("cell_ids", int_type, ("one"    ,))
        cell_ids.long_name = "Start index of compute cell"
        cell_ids.units = ""
        cell_ide       = ds.createVariable("cell_ide", int_type, ("one"    ,))
        cell_ide.long_name = "End index of compute cell"
        cell_ide.units = ""
        vtx_ids        = ds.createVariable("vtx_ids" , int_type, ("one"    ,))
        vtx_ids.long_name = "Start index of compute vertex"
        vtx_ids.units = ""
        vtx_ide        = ds.createVariable("vtx_ide" , int_type, ("one"    ,))
        vtx_ide.long_name = "End index of computed vertex"
        vtx_ide.units = ""
        edge_ids       = ds.createVariable("edge_ids", int_type, ("one"    ,))
        edge_ids.long_name = "Start index of compute edge"
        edge_ids.units = ""
        edge_ide       = ds.createVariable("edge_ide", int_type, ("one"    ,))
        edge_ide.long_name = "End index of compute edge"
        edge_ide.units = ""
        cell_ihs       = ds.createVariable("cell_ihs", int_type, ("hori_hw",))
        cell_ihs.long_name = "Start index of halo cell"
        cell_ihs.units = ""
        cell_ihe       = ds.createVariable("cell_ihe", int_type, ("hori_hw",))
        cell_ihe.long_name = "End index of halo cell"
        cell_ihe.units = ""
        vtx_ihs        = ds.createVariable("vtx_ihs" , int_type, ("hori_hw",))
        vtx_ihs.long_name = "Start index of halo vertex"
        vtx_ihs.units = ""
        vtx_ihe        = ds.createVariable("vtx_ihe" , int_type, ("hori_hw",))
        vtx_ihe.long_name = "End index of halo vertex"
        vtx_ihe.units = ""
        edge_ihs       = ds.createVariable("edge_ihs", int_type, ("hori_hw",))
        edge_ihs.long_name = "Start index of halo edge"
        edge_ihs.units = ""
        edge_ihe       = ds.createVariable("edge_ihe", int_type, ("hori_hw",))
        edge_ihe.long_name = "End index of halo edge"
        edge_ihe.units = ""
        cell_ims       = ds.createVariable("cell_ims", int_type, ("one"    ,))
        cell_ims.long_name = "Start index of memory cell"
        cell_ims.units = ""
        cell_ime       = ds.createVariable("cell_ime", int_type, ("one"    ,))
        cell_ime.long_name = "End index of memory cell"
        cell_ime.units = ""
        vtx_ims        = ds.createVariable("vtx_ims" , int_type, ("one"    ,))
        vtx_ims.long_name = "Start index of memory vertex"
        vtx_ims.units = ""
        vtx_ime        = ds.createVariable("vtx_ime" , int_type, ("one"    ,))
        vtx_ime.long_name = "End index of memory vertex"
        vtx_ime.units = ""
        edge_ims       = ds.createVariable("edge_ims", int_type, ("one"    ,))
        edge_ims.long_name = "Start index of memory edge"
        edge_ims.units = ""
        edge_ime       = ds.createVariable("edge_ime", int_type, ("one"    ,))
        edge_ime.long_name = "End index of memory edge"
        edge_ime.units = ""

        cell_lon       = ds.createVariable("cell_lon"      , float_type, ("nc",              ))
        cell_lon.long_name = "Longitude of cell center"
        cell_lon.units = "radians"
        cell_lat       = ds.createVariable("cell_lat"      , float_type, ("nc",              ))
        cell_lat.long_name = "Latitude of cell center"
        cell_lat.units = "radians"
        edge_lon       = ds.createVariable("edge_lon"      , float_type, ("ne",              ))
        edge_lon.long_name = "Longitude of edge point"
        edge_lon.units = "radians"
        edge_lat       = ds.createVariable("edge_lat"      , float_type, ("ne",              ))
        edge_lat.long_name = "Latitude of edge point"
        edge_lat.units = "radians"
        dual_edge_lon  = ds.createVariable("dual_edge_lon" , float_type, ("ne",              ))
        dual_edge_lon.long_name = "Longitude of dual edge point"
        dual_edge_lon.units = "radians"
        dual_edge_lat  = ds.createVariable("dual_edge_lat" , float_type, ("ne",              ))
        dual_edge_lat.long_name = "Latitude of dual edge point"
        dual_edge_lat.units = "radians"
        vtx_lon        = ds.createVariable("vtx_lon"       , float_type, ("nv",              ))
        vtx_lon.long_name = "Longitude of vertex point"
        vtx_lon.units = "radians"
        vtx_lat        = ds.createVariable("vtx_lat"       , float_type, ("nv",              ))
        vtx_lat.long_name = "Latitude of vertex point"
        vtx_lat.units = "radians"
        cell_xyz       = ds.createVariable("cell_xyz"      , float_type, ("nc", "three"      ))
        cell_xyz.long_name = "Cartesian coordinates of cell center"
        cell_xyz.units = "m"
        edge_xyz       = ds.createVariable("edge_xyz"      , float_type, ("ne", "three"      ))
        edge_xyz.long_name = "Cartesian coordinates of edge point"
        edge_xyz.units = "m"
        dual_edge_xyz  = ds.createVariable("dual_edge_xyz" , float_type, ("ne", "three"      ))
        dual_edge_xyz.long_name = "Cartesian coordinates of dual edge point"
        dual_edge_xyz.units = "m"
        vtx_xyz        = ds.createVariable("vtx_xyz"       , float_type, ("nv", "three"      ))
        vtx_xyz.long_name = "Cartesian coordinates of vertex point"
        vtx_xyz.units = "m"

        cell_nnb       = ds.createVariable("cell_nnb"      , int_type  , ("nc",              ))
        cell_nnb.long_name = "Number of Neighboring cells of each cell"
        cell_nnb.units = ""
        cell_nb        = ds.createVariable("cell_nb"       , int_type  , ("nc", "maxnnb_cell"))
        cell_nb.long_name = "Neighboring cells of each cell"
        cell_nb.units = ""
        cell_ed        = ds.createVariable("cell_ed"       , int_type  , ("nc", "maxnnb_cell"))
        cell_ed.long_name = "Edges of each cell"
        cell_ed.units = ""
        cell_nr        = ds.createVariable("cell_nr"       , int_type  , ("nc", "maxnnb_cell"))
        cell_nr.long_name = "Normal direction for each edge of cell"
        cell_nr.units = ""
        vtx_of_cell    = ds.createVariable("vtx_of_cell"   , int_type  , ("nc", "maxnnb_cell"))
        vtx_of_cell.long_name = "Vertices of each cell"
        vtx_of_cell.units = ""
        area_cell      = ds.createVariable("area_cell"     , float_type, ("nc",              ))
        area_cell.long_name = "Area of each cell"
        area_cell.units = "m2"
        kite_area_cell = ds.createVariable("kite_area_cell", float_type, ("nc", "maxnnb_cell"))
        kite_area_cell.long_name = "Kite area of each cell"
        kite_area_cell.units = "m2"

        vtx_nb         = ds.createVariable("vtx_nb"        , int_type  , ("nv", "maxnnb_vtx" ))
        vtx_nb.long_name = "Neighboring vertices of each vertex"
        vtx_nb.units = ""
        vtx_ed         = ds.createVariable("vtx_ed"        , int_type  , ("nv", "maxnnb_vtx" ))
        vtx_ed.long_name = "Edges of each vertex"
        vtx_ed.units = ""
        vtx_nr         = ds.createVariable("vtx_nr"        , int_type  , ("nv", "maxnnb_vtx" ))
        vtx_nr.long_name = "Normal direction for each edge of vertex"
        vtx_nr.units = ""
        cell_of_vtx    = ds.createVariable("cell_of_vtx"   , int_type  , ("nv", "maxnnb_vtx" ))
        cell_of_vtx.long_name = "Cells of each vertex"
        cell_of_vtx.units = ""
        area_vtx       = ds.createVariable("area_vtx"      , float_type, ("nv",              ))
        area_vtx.long_name = "Area of each vertex"
        area_vtx.units = "m2"
        kite_area_vtx  = ds.createVariable("kite_area_vtx" , float_type, ("nv", "maxnnb_vtx" ))
        kite_area_vtx.long_name = "Kite area of each vertex"
        kite_area_vtx.units = "m2"

        edge_nr        = ds.createVariable("edge_nr"       , float_type, ("ne", "three"      ))
        edge_nr.long_name = "Cartesian coordinate in normal direction of each edge"
        edge_nr.units = ""
        edge_tg        = ds.createVariable("edge_tg"       , float_type, ("ne", "three"      ))
        edge_tg.long_name = "Cartesian coordinate in tangent direction of each edge"
        edge_tg.units = ""
        edge_vtx       = ds.createVariable("edge_vtx"      , int_type  , ("ne", "two"        ))
        edge_vtx.long_name = "Endpoints of each edge"
        edge_vtx.units = ""
        le             = ds.createVariable("le"            , float_type, ("ne",              ))
        le.long_name = "Length of each edge"
        le.units = "m"
        dual_edge_nr   = ds.createVariable("dual_edge_nr"  , float_type, ("ne", "three"      ))
        dual_edge_nr.long_name = "Cartesian coordinate in normal direction of each dual edge"
        dual_edge_nr.units = ""
        dual_edge_tg   = ds.createVariable("dual_edge_tg"  , float_type, ("ne", "three"      ))
        dual_edge_tg.long_name = "Cartesian coordinate in tangent direction of each dual edge"
        dual_edge_tg.units = ""
        dual_edge_vtx  = ds.createVariable("dual_edge_vtx" , int_type  , ("ne", "two"        ))
        dual_edge_vtx.long_name = "Endpoints of dual edge"
        dual_edge_vtx.units = ""
        de             = ds.createVariable("de"            , float_type, ("ne",              ))
        de.long_name = "Length of each dual edge"
        de.units = "m"

        send_ncell_var = ds.createVariable("send_ncell"    , int_type  , ("nnb_of_proc",                ))
        send_ncell_var.long_name = "Send number of cell sites"
        send_ncell_var.units = ""
        send_nvtx_var  = ds.createVariable("send_nvtx"     , int_type  , ("nnb_of_proc",                ))
        send_nvtx_var.long_name = "Send number of vertex sites"
        send_nvtx_var.units = ""
        send_nedge_var = ds.createVariable("send_nedge"    , int_type  , ("nnb_of_proc",                ))
        send_nedge_var.long_name = "Send number of edge sites"
        send_nedge_var.units = ""
        recv_ncell_var = ds.createVariable("recv_ncell"    , int_type  , ("nnb_of_proc",                ))
        recv_ncell_var.long_name = "Recv number of cell sites"
        recv_ncell_var.units = ""
        recv_nvtx_var  = ds.createVariable("recv_nvtx"     , int_type  , ("nnb_of_proc",                ))
        recv_nvtx_var.long_name = "Recv number of vertex sites"
        recv_nvtx_var.units = ""
        recv_nedge_var = ds.createVariable("recv_nedge"    , int_type  , ("nnb_of_proc",                ))
        recv_nedge_var.long_name = "Recv number of edge sites"
        recv_nedge_var.units = ""
        send_cell_var  = ds.createVariable("send_cell"     , int_type  , ("nnb_of_proc", "max_send_cell"))
        send_cell_var.long_name = "Send cell sites"
        send_cell_var.units = ""
        send_vtx_var   = ds.createVariable("send_vtx"      , int_type  , ("nnb_of_proc", "max_send_vtx" ))
        send_vtx_var.long_name = "Send vertex sites"
        send_vtx_var.units = ""
        send_edge_var  = ds.createVariable("send_edge"     , int_type  , ("nnb_of_proc", "max_send_edge"))
        send_edge_var.long_name = "Send edge sites"
        send_edge_var.units = ""
        recv_cell_var  = ds.createVariable("recv_cell"     , int_type  , ("nnb_of_proc", "max_recv_cell"))
        recv_cell_var.long_name = "Recv cell sites"
        recv_cell_var.units = ""
        recv_vtx_var   = ds.createVariable("recv_vtx"      , int_type  , ("nnb_of_proc", "max_recv_vtx" ))
        recv_vtx_var.long_name = "Recv vertex sites"
        recv_vtx_var.units = ""
        recv_edge_var  = ds.createVariable("recv_edge"     , int_type  , ("nnb_of_proc", "max_recv_edge"))
        recv_edge_var.long_name = "Recv edge sites"
        recv_edge_var.units = ""

        cell_ins      [:] = np.asarray([self.cell_ins]    )
        cell_ine      [:] = np.asarray([self.cell_ine]    )
        vtx_ins       [:] = np.asarray([self.vtx_ins ]    )
        vtx_ine       [:] = np.asarray([self.vtx_ine ]    )
        edge_ins      [:] = np.asarray([self.edge_ins]    )
        edge_ine      [:] = np.asarray([self.edge_ine]    )
        cell_ibs      [:] = np.asarray( self.cell_ibs     )
        cell_ibe      [:] = np.asarray( self.cell_ibe     )
        vtx_ibs       [:] = np.asarray( self.vtx_ibs      )
        vtx_ibe       [:] = np.asarray( self.vtx_ibe      )
        edge_ibs      [:] = np.asarray( self.edge_ibs     )
        edge_ibe      [:] = np.asarray( self.edge_ibe     )
        cell_ids      [:] = np.asarray([self.cell_ids]    )
        cell_ide      [:] = np.asarray([self.cell_ide]    )
        vtx_ids       [:] = np.asarray([self.vtx_ids ]    )
        vtx_ide       [:] = np.asarray([self.vtx_ide ]    )
        edge_ids      [:] = np.asarray([self.edge_ids]    )
        edge_ide      [:] = np.asarray([self.edge_ide]    )
        cell_ihs      [:] = np.asarray( self.cell_ihs     )
        cell_ihe      [:] = np.asarray( self.cell_ihe     )
        vtx_ihs       [:] = np.asarray( self.vtx_ihs      )
        vtx_ihe       [:] = np.asarray( self.vtx_ihe      )
        edge_ihs      [:] = np.asarray( self.edge_ihs     )
        edge_ihe      [:] = np.asarray( self.edge_ihe     )
        cell_ims      [:] = np.asarray([self.cell_ims]    )
        cell_ime      [:] = np.asarray([self.cell_ime]    )
        vtx_ims       [:] = np.asarray([self.vtx_ims ]    )
        vtx_ime       [:] = np.asarray([self.vtx_ime ]    )
        edge_ims      [:] = np.asarray([self.edge_ims]    )
        edge_ime      [:] = np.asarray([self.edge_ime]    )

        cell_lon      [:] = np.asarray(self.cell_lon      )
        cell_lat      [:] = np.asarray(self.cell_lat      )
        edge_lon      [:] = np.asarray(self.edge_lon      )
        edge_lat      [:] = np.asarray(self.edge_lat      )
        dual_edge_lon [:] = np.asarray(self.dual_edge_lon )
        dual_edge_lat [:] = np.asarray(self.dual_edge_lat )
        vtx_lon       [:] = np.asarray(self.vtx_lon       )
        vtx_lat       [:] = np.asarray(self.vtx_lat       )
        cell_xyz      [:] = np.asarray(self.cell_xyz      )
        edge_xyz      [:] = np.asarray(self.edge_xyz      )
        dual_edge_xyz [:] = np.asarray(self.dual_edge_xyz )
        vtx_xyz       [:] = np.asarray(self.vtx_xyz       )

        cell_nnb      [:] = np.asarray(self.cell_nnb      )
        cell_nb       [:] = np.asarray(self.cell_nb       )
        cell_ed       [:] = np.asarray(self.cell_ed       )
        cell_nr       [:] = np.asarray(self.cell_nr       )
        vtx_of_cell   [:] = np.asarray(self.vtx_of_cell   )
        area_cell     [:] = np.asarray(self.area_cell     )
        kite_area_cell[:] = np.asarray(self.kite_area_cell)

        vtx_nb        [:] = np.asarray(self.vtx_nb        )
        vtx_ed        [:] = np.asarray(self.vtx_ed        )
        vtx_nr        [:] = np.asarray(self.vtx_nr        )
        cell_of_vtx   [:] = np.asarray(self.cell_of_vtx   )
        area_vtx      [:] = np.asarray(self.area_vtx      )
        kite_area_vtx [:] = np.asarray(self.kite_area_vtx )

        edge_nr       [:] = np.asarray(self.edge_nr       )
        edge_tg       [:] = np.asarray(self.edge_tg       )
        edge_vtx      [:] = np.asarray(self.vtx_of_edge   )
        le            [:] = np.asarray(self.le            )
        dual_edge_nr  [:] = np.asarray(self.dual_edge_nr  )
        dual_edge_tg  [:] = np.asarray(self.dual_edge_tg  )
        dual_edge_vtx [:] = np.asarray(self.cell_of_edge  )
        de            [:] = np.asarray(self.de            )

        send_ncell_var[:] = send_ncell
        send_nvtx_var [:] = send_nvtx
        send_nedge_var[:] = send_nedge
        recv_ncell_var[:] = recv_ncell
        recv_nvtx_var [:] = recv_nvtx
        recv_nedge_var[:] = recv_nedge
        send_cell_var [:] = send_cell
        send_vtx_var  [:] = send_vtx
        send_edge_var [:] = send_edge
        recv_cell_var [:] = recv_cell
        recv_vtx_var  [:] = recv_vtx
        recv_edge_var [:] = recv_edge
        ds.close()
        
        if proc.is_root():
            log.logger.success(f"Successfully output scvt mesh into netCDF file.")
        
def remap_g2l(
    local_id: INT[Array, "npoints ..."], 
    g2l_id: INT[Array, "npoints"]
) -> Array:
    valid = local_id >= 0
    mapped = g2l_id[local_id]
    return jnp.where(valid, mapped, -1)
