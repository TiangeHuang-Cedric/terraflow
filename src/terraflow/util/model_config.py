from typing import Self
import yaml
import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    pgf_scheme: str = "ptb"
    ke_scheme: str = "gassmann_2013"
    ke_gamma: float = 0.9
    coriolis_scheme: str = "trisk"
    # pv advection
    pv_adv_scheme: str = "upwind"
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

    def __post_init__(self):
        if self.upwind_order_h == -1:
            object.__setattr__(self, "upwind_order_h", self.upwind_order)
        if self.upwind_order_v == -1:
            object.__setattr__(self, "upwind_order_v", self.upwind_order)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
