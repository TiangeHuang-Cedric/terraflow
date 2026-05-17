from typing import Self
import yaml
import dataclasses

from terraflow.util import log


@dataclasses.dataclass
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

    def __init__(self, *args, **kwargs):
        for field in dataclasses.fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])
        self.__post_init__()

    def __post_init__(self):
        if self.upwind_order_h == -1:
            object.__setattr__(self, "upwind_order_h", self.upwind_order)
        if self.upwind_order_v == -1:
            object.__setattr__(self, "upwind_order_v", self.upwind_order)

    def __print__(self):
        log.logger.info(
            "########################################################"
        )
        log.logger.info(
            "--Model Config--"
        )
        log.logger.info(
            f"  pgf_scheme: {self.pgf_scheme}"
        )
        log.logger.info(
            f"  ke_scheme: {self.ke_scheme}, ke_gamma: {self.ke_gamma}"
        )
        log.logger.info(
            f"  coriolis_scheme: {self.coriolis_scheme}"
        )
        # =======================================================
        # PV advection
        log.logger.info(
            f"  pv_adv_scheme: {self.pv_adv_scheme}"
        )
        log.logger.info(
            f"  upwind_order_pv: {self.upwind_order_pv}"
        )
        log.logger.info(
            f"  upwind_use_mbs_pv: {self.upwind_use_mbs_pv}"
        )
        log.logger.info(
            f"  upwind_optimized_pv: {self.upwind_optimized_pv}"
        )
        # =======================================================
        # Background advection
        log.logger.info(
            "  "
            f"bg_adv_scheme: {self.bg_adv_scheme}, "
            f"upwind_order_bg: {self.upwind_order_bg}, "
            f"upwind_optimized_bg: {self.upwind_optimized_bg}"
        )
        # =======================================================
        # Potential Temperature advection
        log.logger.info(
            f"  pt_adv_scheme: {self.pt_adv_scheme}"
        )
        # =======================================================
        # Tracer batches advection
        log.logger.info(
            f"  adv_scheme: {self.adv_scheme}"
        )
        log.logger.info(
            "  "
            f"upwind_order: {self.upwind_order}, "
            f"upwind_order_h: {self.upwind_order_h}, "
            f"upwind_order_v: {self.upwind_order_v}"
        )
        log.logger.info(
            "  "
            f"upwind_wgt: {self.upwind_wgt}, "
            f"upwind_use_mbs: {self.upwind_use_mbs}, "
            f"upwind_optimized: {self.upwind_optimized}"
        )
        log.logger.info(
            "  "
            f"wrf_order_h: {self.wrf_order_h}, "
            f"wrf_use_mbs: {self.wrf_use_mbs}, "
            f"wrf_optimized: {self.wrf_optimized}"
        )
        log.logger.info(
            "  "
            f"limiter_type: {self.limiter_type}, "
            f"fct_regression_h: {self.fct_regression_h}, "
            f"fct_lo_scheme_h: {self.fct_lo_scheme_h}, "
            f"fct_regression_v: {self.fct_regression_v}, "
            f"fct_lo_scheme_v: {self.fct_lo_scheme_v}"
        )
        log.logger.info(
            "--Model Config--"
        )
        log.logger.info(
            "########################################################"
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
