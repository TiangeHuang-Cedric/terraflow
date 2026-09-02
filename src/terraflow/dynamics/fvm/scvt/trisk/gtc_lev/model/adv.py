from __future__ import annotations
from typing import (
    TYPE_CHECKING, 
    overload, 
    Literal, 
    Union,
    Optional, 
    Callable
)
import equinox as eqx
import numpy as np


from terraflow.util.datetime import DateTimeManager
from terraflow.model.abstract import AbstractModel
from terraflow.model.config import Const
from terraflow.mesh.scvt.field import SCVTFields
from terraflow.vert_coord.gtc.onelayer import OneLayerVertCoordinate
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import adv_time_schemes
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.interface import (
    adv_run_mass,
    adv_run_tracers_fn
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.common import (
    print_logo,
    diagnose
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.test_case.adv import (
    solid_rotation as adv_sr,
    dcmip12 as adv_dcmip12,
    deform as adv_dc
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.history import build_history
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import Tracer
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


if TYPE_CHECKING:
    import uxarray as uxr
    
    
    from terraflow.util import typing
    from terraflow.util.device import AbstractDevice
    from terraflow.util.uxarray import UGRIDDataset
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.model.abstract import ModelMode
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.history import (
        UGRIDHistoryset,
        SCVTHistoryset
    )
    from terraflow.test_case.abstract import AbstractAdvectionModelTestCase
    
    
    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array
    
    
class Adv2DModel(AbstractModel[Block, SCVTFields]):
    block          : Block
    adv_run_tracers: Callable[[Block, int  , int], Block] = eqx.field(static=True)
    set_ic         : Callable[[Block            ], Block] = eqx.field(static=True)
    set_uv         : Callable[[float, Block, int], Block] = eqx.field(static=True) 
    
    def __init__(
        self, 
        mode: ModelMode,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        test_case: AbstractAdvectionModelTestCase,
        dev: AbstractDevice
    ):
        if dev.is_root(): print_logo()
        if dev.is_root(): cfg.__print__()
        # =================================================================
        if test_case.alias_name == "adv_sr":
            test_case   = adv_sr.SolidRotationTestCase(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        elif test_case.alias_name == "adv_dc1":
            test_case = adv_dc.DeformationalFlowCase1(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        elif test_case.alias_name == "adv_dc2":
            test_case = adv_dc.DeformationalFlowCase2(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        elif test_case.alias_name == "adv_dc3":
            test_case = adv_dc.DeformationalFlowCase3(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        elif test_case.alias_name == "adv_dc4":
            test_case = adv_dc.DeformationalFlowCase4(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        else:
            raise ValueError(f"Unsupported test case {test_case.alias_name} for adv model!")
        # =================================================================
        self.mode  = mode
        self.const = Const(cfg)
        self.cfg   = cfg
        self.time_manager = DateTimeManager(
            start_time=cfg.start_time,
            end_time=cfg.end_time,
            run_time=cfg.run_time,
            dt_dyn=cfg.dt_dyn,
            dt_adv=cfg.dt_adv,
            save_interval=cfg.save_interval,
            log_every=cfg.log_every,
            dev=dev
        )
        # =================================================================
        adv_time_integrator = adv_time_schemes.adv_scheme_init(cfg)
        self.adv_run_tracers = adv_run_tracers_fn(adv_time_integrator)
        # =================================================================
        tracers_meta = test_case.init_tracers_meta(cfg.dt_adv_seconds)
        block = Block.from_coord(
            mode=self.mode, 
            const=self.const,
            cfg=self.cfg,
            para=para,
            mesh=mesh,
            vert_coord=OneLayerVertCoordinate(dev=dev),
            tracers=Tracer.initialize(
                mode=mode,
                cfg=cfg,
                mesh=mesh,
                tracers_meta=tracers_meta,
                dev=dev
            ),
            dev=dev
        )
        block = self.set_ic(block)
        elapsed_seconds = block.elapsed_seconds
        block = self.set_uv(elapsed_seconds, block, block.old)
        self.block = block.from_parent_after_ic()
        # =================================================================

    @overload
    def build_history(
        self,
        mode: Literal["ugrid"],
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> UGRIDHistoryset:
        ...

    @overload
    def build_history(
        self,
        mode: Literal["scvt"] = "scvt",
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> SCVTHistoryset:
        ...
        
    def build_history(
        self,
        mode: str = "scvt",
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> Union[UGRIDHistoryset, SCVTHistoryset]:
        return build_history(
            model=self,
            mode=mode,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h1_opt=h1_opt,
            h2_opt=h2_opt,
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type
        )
        
    def terraflow_run(self, block: Block) -> Block:
        block = adv_run_mass(block, block.old, block.new, block.dt_dyn)
        elapsed_seconds = block.elapsed_seconds
        dt_dyn = block.dt_dyn
        block = self.set_uv(elapsed_seconds + dt_dyn, block, block.new)
        block = block.time_advance()
        block = self.adv_run_tracers(block, block.new, block.old)
        return block
    
    def terraflow_diagnose(self, block: Block) -> Block:
        return diagnose(block, "old")
    
    def trajectory_diag_tools(
        self,
        trajectory: dict[str, Array],
        use_dual: bool = False
    ) -> UGRIDDataset:
        return self.block.trajectory_diag_tools(
            trajectory=trajectory,
            use_dual=use_dual
        )
    
    def to_dict(self, itime_str: str = "old") -> dict[str, Array]:
        return self.block.to_dict(itime_str=itime_str)
    
    def to_uxarray(
        self,
        use_dual: bool = False,
        itime_str: str = "old",
    ) -> uxr.UxDataset:
        return self.to_ugrid(
            use_dual=use_dual,
            itime_str=itime_str
        ).to_uxarray()

    def to_ugrid(self, use_dual: bool = False, itime_str: str = "old") -> UGRIDDataset:
        return self.block.to_ugrid(
            use_dual=use_dual, 
            itime_str=itime_str
        )


class Adv3DModel(AbstractModel[Block, SCVTFields]):
    block          : Block
    adv_run_tracers: Callable[[Block, int  , int], Block] = eqx.field(static=True)
    set_ic         : Callable[[Block            ], Block] = eqx.field(static=True)
    set_uv         : Callable[[float, Block, int], Block] = eqx.field(static=True) 
    
    def __init__(
        self, 
        mode: ModelMode,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        test_case: AbstractAdvectionModelTestCase,
        dev: AbstractDevice
    ):
        if dev.is_root(): print_logo()
        if dev.is_root(): cfg.__print__()
        # =================================================================
        if test_case.alias_name == "adv_dcmip12":
            test_case = adv_dcmip12.DCMIP12TestCase(para)
            self.set_ic = test_case.set_ic
            self.set_uv = test_case.set_uv
        else:
            raise ValueError(f"Unsupported test case {test_case.alias_name} for adv model!")
        # =================================================================
        self.mode  = mode
        self.const = Const(cfg)
        self.cfg   = cfg
        self.time_manager = DateTimeManager(
            start_time=cfg.start_time,
            end_time=cfg.end_time,
            run_time=cfg.run_time,
            dt_dyn=cfg.dt_dyn,
            dt_adv=cfg.dt_adv,
            save_interval=cfg.save_interval,
            log_every=cfg.log_every,
            dev=dev
        )
        # =================================================================
        adv_time_integrator = adv_time_schemes.adv_scheme_init(cfg)
        self.adv_run_tracers = adv_run_tracers_fn(adv_time_integrator)
        # =================================================================
        tracers_meta = test_case.init_tracers_meta(cfg.dt_adv_seconds)
        block = Block.from_coord(
            mode=self.mode, 
            const=self.const,
            cfg=self.cfg,
            para=para,
            mesh=mesh,
            vert_coord=vert_coord,
            tracers=Tracer.initialize(
                mode=mode,
                cfg=cfg,
                mesh=mesh,
                tracers_meta=tracers_meta,
                dev=dev
            ),
            dev=dev
        )
        block = self.set_ic(block)
        block = self.set_uv(block.elapsed_seconds, block, block.old)
        self.block = block.from_parent_after_ic()
        # =================================================================

    @overload
    def build_history(
        self,
        mode: Literal["ugrid"],
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> UGRIDHistoryset:
        ...

    @overload
    def build_history(
        self,
        mode: Literal["scvt"] = "scvt",
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> SCVTHistoryset:
        ...
        
    def build_history(
        self,
        mode: str = "scvt",
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h1_opt: dict = {"use_dual": False},
        h2_opt: dict = {},
        output_ngroups: int = 0,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
    ) -> Union[UGRIDHistoryset, SCVTHistoryset]:
        return build_history(
            model=self,
            mode=mode,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h1_opt=h1_opt,
            h2_opt=h2_opt,
            output_ngroups=output_ngroups,
            dev=dev,
            float_type=float_type,
            int_type=int_type
        )
            
    def terraflow_run(self, block: Block) -> Block:
        block = adv_run_mass(block, block.old, block.new, block.dt_dyn)
        elapsed_seconds = block.elapsed_seconds
        dt_dyn = block.dt_dyn
        block = self.set_uv(elapsed_seconds + dt_dyn, block, block.new)
        block = block.time_advance()
        block = self.adv_run_tracers(block, block.new, block.old)
        return block
    
    def terraflow_diagnose(self, block: Block) -> Block:
        return diagnose(block, "old")
    
    def trajectory_diag_tools(
        self,
        trajectory: dict[str, Array],
        use_dual: bool = False
    ) -> UGRIDDataset:
        return self.block.trajectory_diag_tools(
            trajectory=trajectory,
            use_dual=use_dual
        )
        
    def to_dict(self, itime_str: str = "old") -> dict[str, Array]:
        return self.block.to_dict(itime_str=itime_str)

    def to_uxarray(
        self,
        use_dual: bool = False,
        itime_str: str = "old",
    ) -> uxr.UxDataset:
        return self.to_ugrid(
            use_dual=use_dual,
            itime_str=itime_str
        ).to_uxarray()

    def to_ugrid(self, use_dual: bool = False, itime_str: str = "old") -> UGRIDDataset:
        return self.block.to_ugrid(
            use_dual=use_dual, 
            itime_str=itime_str
        )
