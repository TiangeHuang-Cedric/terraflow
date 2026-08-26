from typing import Self
import yaml
import dataclasses
import datetime


@dataclasses.dataclass
class Const:
    total_substeps    : int   = 0
    adv_total_substeps: int   = 0
    
    def __init__(
        self,
        cfg: "UserConfig"
    ):
        if cfg.time_scheme == "euler":
            self.total_substeps = 1
        elif cfg.time_scheme == "rk2":
            self.total_substeps = 2
        elif cfg.time_scheme == "pc2":
            self.total_substeps = 3
        elif cfg.time_scheme == "wrfrk3":
            self.total_substeps = 3
        else:
            raise ValueError(f"Unsupported time scheme {cfg.time_scheme}")
        
        if cfg.adv_time_scheme == "euler":
            self.adv_total_substeps = 1
        elif cfg.adv_time_scheme == "rk2":
            self.adv_total_substeps = 2
        elif cfg.adv_time_scheme == "pc2":
            self.adv_total_substeps = 3
        elif cfg.adv_time_scheme == "wrfrk3":
            self.adv_total_substeps = 3
        else:
            raise ValueError(f"Unsupported time scheme {cfg.time_scheme}")


@dataclasses.dataclass(frozen=True)
class UserConfig:
    # Model control
    start_time          : str        = ""
    end_time            : str        = ""
    run_time            : str        = ""
    dt_dyn              : str        = ""
    dt_adv              : str        = ""
    save_interval       : str        = "1 hours"
    log_every           : str        = "1 hours"
   
    use_forward_backward: bool       = True
    time_scheme         : str        = "wrfrk3"
    adv_time_scheme     : str        = "wrfrk3"
    output_h1_tend      : bool       = False
    init_hydrostatic_gz : bool       = False
    
    pgf_scheme          : str        = "ptb"
    ke_scheme           : int        = 2
    ke_cell_wgt         : float      = 0.9
    coriolis_scheme     : str        = "trisk"

    # pv advection
    pv_adv_scheme       : str        = "upwind"
    upwind_order_pv     : int        = 3
    upwind_optimized_pv : bool       = True
    tri_stencil_type    : int        = 2

    # Background advection
    bg_adv_scheme       : str        = "upwind"
    upwind_order_bg     : int        = 3
    upwind_optimized_bg : bool       = True

    # Potential Temperature advection
    pt_adv_scheme       : str        = "upwind"

    # Tracer batches advection
    adv_scheme          : str        = "upwind"
    
    # Momentum advection
    uv_vert_adv_type    : int        = 1

    # Advection configuration
    upwind_order        : int        = 3
    upwind_order_h      : int        = -1
    upwind_order_v      : int        = -1
    upwind_wgt          : float      = 1.0
    upwind_wgt_pv       : float      = 1.0
    upwind_optimized    : bool       = True
    limiter_type        : str        = "fct"
    fct_regression_h    : bool       = False
    fct_lo_scheme_h     : str        = "upwind1"
    fct_regression_v    : bool       = False
    fct_lo_scheme_v     : str        = "upwind1"
    adv_hvsplit_type    : int        = 1

    # Damp
    use_smag_damp       : bool       = False
    smag_damp_coef      : float      = 0.015
    use_laplace_damp    : bool       = False
    use_div_damp        : bool       = False
    div_damp_cycles     : int        = 1
    div_damp_order      : int        = 2
    div_damp_top        : float      = 1
    div_damp_k0         : int        = 10
    div_damp_coef2      : float      = 1.0 / 128.0
    div_damp_coef4      : float      = 0.01

    def __post_init__(self):
        if self.end_time == "" and self.run_time == "":
            raise ValueError("UserConfig: end_time and run_time are empty strings at the same time!")
        if self.dt_dyn == "" and self.dt_adv == "":
            raise ValueError("UserConfig: dt_dyn and dt_adv are empty strings at the same time!")        
        if self.dt_dyn == "":
            object.__setattr__(self, "dt_dyn"        , self.dt_adv      )
        if self.dt_adv == "":
            object.__setattr__(self, "dt_adv"        , self.dt_dyn      )
        if self.upwind_order_h == -1:
            object.__setattr__(self, "upwind_order_h", self.upwind_order)
        if self.upwind_order_v == -1:
            object.__setattr__(self, "upwind_order_v", self.upwind_order)
        self.__print__()

    def __print__(self):
        print("====================== Model Config ========================")
        print(f"  use_forward_backward       = {self.use_forward_backward }")
        print(f"  time_scheme                = {self.time_scheme          }")
        print(f"  adv_time_scheme            = {self.adv_time_scheme      }")
        print(f"  pgf_scheme                 = {self.pgf_scheme           }")
        print(f"  ke_scheme                  = {self.ke_scheme            }")
        print(f"  ke_cell_wg                 = {self.ke_cell_wgt          }")
        print(f"  coriolis_scheme            = {self.coriolis_scheme      }")
        print(f"  pv_adv_scheme              = {self.pv_adv_scheme        }")
        print(f"  upwind_order_pv            = {self.upwind_order_pv      }")
        print(f"  upwind_optimized_pv        = {self.upwind_optimized_pv  }")
        print(f"  bg_adv_scheme              = {self.bg_adv_scheme        }")
        print(f"  upwind_order_bg            = {self.upwind_order_bg      }")
        print(f"  upwind_optimized_bg        = {self.upwind_optimized_bg  }")
        print(f"  pt_adv_scheme              = {self.pt_adv_scheme        }")
        print(f"  adv_scheme                 = {self.adv_scheme           }")
        print(f"  uv_vert_adv_type           = {self.uv_vert_adv_type     }")
        print(f"  upwind_order               = {self.upwind_order         }")
        print(f"  upwind_order_h             = {self.upwind_order_h       }")
        print(f"  upwind_order_v             = {self.upwind_order_v       }")
        print(f"  upwind_wgt                 = {self.upwind_wgt           }")
        print(f"  upwind_wgt_pv              = {self.upwind_wgt_pv        }")
        print(f"  upwind_optimized           = {self.upwind_optimized     }")
        print(f"  limiter_type               = {self.limiter_type         }")
        print(f"  fct_regression_h           = {self.fct_regression_h     }")
        print(f"  fct_lo_scheme_h            = {self.fct_lo_scheme_h      }")
        print(f"  fct_regression_v           = {self.fct_regression_v     }")
        print(f"  fct_lo_scheme_v            = {self.fct_lo_scheme_v      }")
        print(f"  adv_hvsplit_type           = {self.adv_hvsplit_type     }")
        print(f"  use_smag_damp              = {self.use_smag_damp        }")
        print(f"  smag_damp_coef             = {self.smag_damp_coef       }")
        print(f"  use_laplace_damp           = {self.use_laplace_damp     }")
        print(f"  use_div_damp               = {self.use_div_damp         }")
        print(f"  div_damp_cycles            = {self.div_damp_cycles      }")
        print(f"  div_damp_order             = {self.div_damp_order       }") 
        print(f"  div_damp_top               = {self.div_damp_top         }") 
        print(f"  div_damp_k0                = {self.div_damp_k0          }") 
        print(f"  div_damp_coef2             = {self.div_damp_coef2       }") 
        print(f"  div_damp_coef4             = {self.div_damp_coef4       }") 
        print("============================================================")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)

    def to_dict(self, keep_bool: bool = True) -> dict:
        if keep_bool:
            return dataclasses.asdict(self)
        else: 
            res = dataclasses.asdict(self)
            for k, v in res.items():
                if isinstance(v, bool):
                    res[k] = int(v)
            return res
    
    @property
    def dt_dyn_seconds(self) -> float:
        dt_dyn_value = float(self.dt_dyn.split()[0])
        dt_dyn_unit = self.dt_dyn.split()[1]
        return datetime.timedelta(**{dt_dyn_unit: dt_dyn_value}).total_seconds()
    
    @property
    def dt_adv_seconds(self) -> float:
        dt_adv_value = float(self.dt_adv.split()[0])
        dt_adv_unit = self.dt_adv.split()[1]
        return datetime.timedelta(**{dt_adv_unit: dt_adv_value}).total_seconds()