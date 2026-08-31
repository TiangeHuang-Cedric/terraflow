from __future__ import annotations
from typing import (
    TYPE_CHECKING, 
    overload, 
    Literal, 
    Union,
    Optional, 
    Callable
)
import math
import equinox as eqx
import numpy as np


from terraflow.util.datetime import DateTimeManager
from terraflow.model.abstract import AbstractModel
from terraflow.model.config import Const
from terraflow.mesh.scvt.field import SCVTFields
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev import time_schemes
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv import adv_time_schemes
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.adv.interface import adv_run_tracers_fn
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.model.common import (
    print_logo,
    space_operators,
    diagnose
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.damp.interface import (
    damp_prepare,
    damp_run
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.test_case import (
    mountain_wave as mz,
    rossby_haurwitz_wave_3d as rh,
    steady_state as ss
)
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.prepare.bkg import prepare_bkg
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.prepare.topo import prepare_topo
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.history import build_history
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.tracer import TracerMeta, Tracer
from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.block import Block


if TYPE_CHECKING:
    import uxarray as uxr
    
    
    from terraflow.util import typing
    from terraflow.util.device import AbstractDevice
    from terraflow.util.uxarray import UGRIDDataset
    from terraflow.planet import PlanetParameterSet
    from terraflow.model.abstract import ModelMode
    from terraflow.model.config import UserConfig
    from terraflow.mesh.scvt.mesh_type import SCVTMesh
    from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
    from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase
    from terraflow.prepare.readers.abstract import (
        AbstractBackgroundReader,
        AbstractTopographyReader
    )
    from terraflow.dynamics.fvm.scvt.trisk.gtc_lev.history import (
        UGRIDHistoryset,
        SCVTHistoryset
    )
    
    
    INT = typing.Int
    FLT = typing.Float
    Array = typing.Array

 
class BaroclinicModel(AbstractModel[Block, SCVTFields]):
    block          : Block
    space_operators: Callable[[Block, int, int, float], Block] = eqx.field(static=True)
    adv_run_tracers: Callable[[Block, int, int       ], Block] = eqx.field(static=True)
    set_ic         : Callable[[Block                 ], Block] = eqx.field(static=True) 

    def __init__(
        self, 
        mode: ModelMode,
        cfg: UserConfig,
        para: PlanetParameterSet,
        mesh: SCVTMesh,
        vert_coord: AbstractGeneralizedTerrainFollowingCoordinate,
        dev: AbstractDevice,
        test_case: Optional[AbstractBaroclinicModelTestCase] = None,
        bkg_reader: Optional[AbstractBackgroundReader] = None,
        topo_reader: Optional[AbstractTopographyReader] = None
    ):
        if dev.is_root(): print_logo()
        if dev.is_root(): cfg.__print__()
        # =================================================================
        if test_case is None and bkg_reader is None and topo_reader is None:
            raise ValueError("User must provide a test_case or readers for barocclinic models!")
        # =================================================================
        if test_case is not None and bkg_reader is None and topo_reader is None:
            if test_case.alias_name == "mz":
                test_case = mz.MountainWaveTestCase(para)
                self.set_ic = test_case.set_ic
            elif test_case.alias_name == "rh":
                test_case = rh.RossbyHaurwitzWaveTestCase(para)
                self.set_ic = test_case.set_ic
            elif test_case.alias_name == "ss":
                test_case = ss.SCVTSteadyStateTestCase(para)
                self.set_ic = test_case.set_ic
            else:
                raise ValueError(f"Unsupported test case {test_case.alias_name} for baroclinic model!")
        else:
            self.set_ic = lambda b: b
        # =================================================================
        if test_case is None and bkg_reader is not None and topo_reader is not None:
            min_lon, max_lon = mesh.lon_range()
            min_lat, max_lat = mesh.lat_range()
            min_lon = max(0  , math.floor(min_lon))
            min_lat = max(-90, math.floor(min_lat))
            max_lon = min(360, math.ceil (max_lon))
            max_lat = min(90 , math.ceil (max_lat))
            topo_reader.load_data(
                min_lon=min_lon,
                max_lon=max_lon,
                min_lat=min_lat,
                max_lat=max_lat
            )
            if bkg_reader.cfg.hori_interp_backend == "esmf":
                bkg_reader.load_data() 
            else:
                bkg_reader.load_data(
                    min_lon=min_lon, 
                    max_lon=max_lon, 
                    min_lat=min_lat, 
                    max_lat=max_lat
                )
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
            bkg_reader=bkg_reader,
            dev=dev
        )
        # =================================================================
        time_integrator = time_schemes.time_scheme_init(cfg)
        self.space_operators = time_integrator(
            space_operators, 
            cfg.use_forward_backward
        )
        # =================================================================
        adv_time_integrator = adv_time_schemes.adv_scheme_init(cfg)
        self.adv_run_tracers = adv_run_tracers_fn(adv_time_integrator)
        # =================================================================
        if bkg_reader is not None:
            tracers_meta = TracerMeta.from_reader(
                mode=mode,
                para=para,
                reader=bkg_reader,
                dt_adv=cfg.dt_adv_seconds
            )
        else:
            tracers_meta = {}
        tracers = Tracer.add_tracers(
            mode=mode,
            cfg=cfg,
            mesh=mesh,
            tracers_meta=tracers_meta
        )        
        block = Block.from_coord(
            mode=self.mode, 
            const=self.const,
            cfg=self.cfg,
            para=para,
            mesh=mesh,
            vert_coord=vert_coord,
            tracers=tracers,
            dev=dev
        )
        if test_case is not None:
            block = self.set_ic(block)
        elif bkg_reader is not None and topo_reader is not None:
            block = prepare_topo(block, topo_reader)
            topo_reader.unload()
            block = prepare_bkg (block, bkg_reader )
            bkg_reader .unload()
        self.block = block.from_parent_after_ic()

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
        block = damp_prepare(block, block.old, block.dt_dyn)
        block = self.space_operators(block, block.old, block.new, block.dt_dyn)
        block = damp_run(block, block.new, block.dt_dyn)
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
