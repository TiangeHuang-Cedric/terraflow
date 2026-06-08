from typing import Self
import yaml
import dataclasses


@dataclasses.dataclass
class ModelConfig:
    pgf_scheme: str = "ptb"
    ke_scheme: str = "gassmann_2013"
    ke_gamma: float = 0.9
    coriolis_scheme: str = "trisk"

    # pv advection
    pv_adv_scheme: str = "pure_upwind"
    upwind_order_pv: int = 3
    upwind_use_mbs_pv: bool = True
    upwind_optimized_pv: bool = True

    # Background advection
    bg_adv_scheme: str = "upwind"
    upwind_order_bg: int = 3
    upwind_optimized_bg: bool = True

    # Potential Temperature advection
    pt_adv_scheme: str = "upwind"

    # Tracer batches advection
    adv_scheme: str = "upwind"

    # Advection configuration
    upwind_order: int = 3
    upwind_order_h: int = -1
    upwind_order_v: int = -1
    upwind_wgt: float = 1.0
    upwind_use_mbs: bool = True
    upwind_optimized: bool = True
    wrf_order_h: int = 3
    wrf_use_mbs: bool = True
    wrf_optimized: bool = True
    limiter_type: str = "fct"
    fct_regression_h: bool = True
    fct_lo_scheme_h: str = "upwind1"
    fct_regression_v: bool = True
    fct_lo_scheme_v: str = "upwind1"

    # Damp
    use_smag_damp: bool = False
    smag_damp_coef: float = 0.015
    use_sponge_layer: bool = False
    sponge_layer_k0: int = 5
    sponge_layer_coef: float = 2.0

    def __post_init__(self):
        if self.upwind_order_h == -1:
            object.__setattr__(self, "upwind_order_h", self.upwind_order)
        if self.upwind_order_v == -1:
            object.__setattr__(self, "upwind_order_v", self.upwind_order)

    def __print__(self):
        print("====================== Model Config ========================")
        print(f"  pgf_scheme                 = {self.pgf_scheme}")
        print(f"  ke_scheme                  = {self.ke_scheme}")
        print(f"  ke_gamma                   = {self.ke_gamma}")
        print(f"  coriolis_scheme            = {self.coriolis_scheme}")
        print(f"  pv_adv_scheme              = {self.pv_adv_scheme}")
        print(f"  upwind_order_pv            = {self.upwind_order_pv}")
        print(f"  upwind_use_mbs_pv          = {self.upwind_use_mbs_pv}")
        print(f"  upwind_optimized_pv        = {self.upwind_optimized_pv}")
        print(f"  bg_adv_scheme              = {self.bg_adv_scheme}")
        print(f"  upwind_order_bg            = {self.upwind_order_bg}")
        print(f"  upwind_optimized_bg        = {self.upwind_optimized_bg}")
        print(f"  pt_adv_scheme              = {self.pt_adv_scheme}")
        print(f"  adv_scheme                 = {self.adv_scheme}")
        print(f"  upwind_order               = {self.upwind_order}")
        print(f"  upwind_order_h             = {self.upwind_order_h}")
        print(f"  upwind_order_v             = {self.upwind_order_v}")
        print(f"  upwind_wgt                 = {self.upwind_wgt}")
        print(f"  upwind_use_mbs             = {self.upwind_use_mbs}")
        print(f"  upwind_optimized           = {self.upwind_optimized}")
        print(f"  wrf_order_h                = {self.wrf_order_h}")
        print(f"  wrf_use_mbs                = {self.wrf_use_mbs}")
        print(f"  wrf_optimized              = {self.wrf_optimized}")
        print(f"  limiter_type               = {self.limiter_type}")
        print(f"  fct_regression_h           = {self.fct_regression_h}")
        print(f"  fct_lo_scheme_h            = {self.fct_lo_scheme_h}")
        print(f"  fct_regression_v           = {self.fct_regression_v}")
        print(f"  fct_lo_scheme_v            = {self.fct_lo_scheme_v}")
        print(f"  use_smag_damp              = {self.use_smag_damp}")
        print(f"  smag_damp_coef             = {self.smag_damp_coef}")
        print(f"  use_sponge_layer           = {self.use_sponge_layer}")
        print(f"  sponge_layer_k0            = {self.sponge_layer_k0}")
        print(f"  sponge_layer_coef          = {self.sponge_layer_coef}")
        print("============================================================")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
