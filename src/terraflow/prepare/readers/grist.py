from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import xarray as xr

from terraflow.util import typing, log
from terraflow.prepare.readers.abstract import (
    AbstractReader,
    AbstractReaderConfig
)
if TYPE_CHECKING:
    from terraflow.util.device import AbstractDevice
    from terraflow.planet import PlanetParameterSet

Float = typing.Float
Array = typing.Array


@dataclass
class GRISTStaticDataReaderConfig(AbstractReaderConfig):
    # zs grad filter parameters
    use_zs_grad_filter   : bool = False
    zs_grad_filter_type  : str = "laplacian2"  # "laplacian1" or "laplacian2"
    zs_grad_filter_order : int = 2
    zs_grad_filter_coef  : float = 1.0e6
    zs_grad_filter_cycles: int = 100
    topo_max_slope       : float = 0.12

    def __print__(self):
        print("================ GRIST Static Data Reader Config ==================")
        print    (f"  use_zs_grad_filter         = {self.use_zs_grad_filter      }")
        if self.zs_grad_filter_type == "laplacian1":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type     }")
            print(f"  zs_grad_filter_order       = {self.zs_grad_filter_order    }")
            print(f"  zs_grad_filter_coef        = {self.zs_grad_filter_coef     }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles   }")
            print(f"  topo_max_slope             = {self.topo_max_slope          }")
        elif self.zs_grad_filter_type == "laplacian2":
            print(f"  zs_grad_filter_type        = {self.zs_grad_filter_type     }")
            print(f"  zs_grad_filter_cycles      = {self.zs_grad_filter_cycles   }")
        print("===================================================================")


@dataclass
class GRISTBackgroundReaderConfig(AbstractReaderConfig):
    # "jax" or "native"
    vert_interp_backend: str = "jax"

    # "native" or "grist"
    vert_interp_type: str = "native"

    def __print__(self):
        print("================ GRIST Background Reader Config ==================")
        print(f"  vert_interp_backend        = {self.vert_interp_backend}")
        print(f"  vert_interp_type           = {self.vert_interp_type}")
        print("==================================================================")


class GRISTStaticDataReader(AbstractReader[GRISTStaticDataReaderConfig]):
    lon: Float[Array, "ncol"]
    lat: Float[Array, "ncol"]
    phis: Float[Array, "ncol"]
    lu_index: Float[Array, "ncol"]
    lu_index_frac: Float[Array, "lu_cat"]
    soiltop: Float[Array, "ncol"]
    soiltop_frac: Float[Array, "ncol"]
    mask: Float[Array, "ncol"]
    soiltemp: Float[Array, "ncol"]
    snoalb: Float[Array, "ncol"]
    greenfrac: Float[Array, "ncol month"]
    albedo: Float[Array, "ncol month"]
    varsso: Float[Array, "ncol"]
    con: Float[Array, "ncol"]
    oa1: Float[Array, "ncol"]
    oa2: Float[Array, "ncol"]
    oa3: Float[Array, "ncol"]
    oa4: Float[Array, "ncol"]
    ol1: Float[Array, "ncol"]
    ol2: Float[Array, "ncol"]
    ol3: Float[Array, "ncol"]
    ol4: Float[Array, "ncol"]
    var2d: Float[Array, "ncol"]

    def __init__(
        self,
        static_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: GRISTStaticDataReaderConfig = GRISTStaticDataReaderConfig()
    ):
        super().__init__(
            para=para,
            cfg=cfg,
            dev=dev
        )
        self.static_file = static_file
        log.logger.info(f"Use GRIST static file {static_file} as static data source.")
        self.extract_data()

    def extract_data(self):
        float_type = self.dev.float_type
        ds = xr.open_dataset(self.static_file, engine="netcdf4")
        self.lon = jnp.array(
            ds["lon"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["lat"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.phis = jnp.array(
            ds["PHIS"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lu_index = jnp.array(
            ds["LU_INDEX"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lu_index_frac = jnp.array(
            ds["LU_INDEX_FRAC"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.soiltop = jnp.array(
            ds["SOILTOP"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.soiltop_frac = jnp.array(
            ds["SOILTOP_FRAC"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.mask = jnp.array(
            ds["MASK"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.soiltemp = jnp.array(
            ds["SOILTEMP"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.snoalb = jnp.array(
            ds["SNOALB"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.greenfrac = jnp.array(
            ds["GREENFRAC"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.albedo = jnp.array(
            ds["ALBEDO"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.varsso = jnp.array(
            ds["VARSSO"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.con = jnp.array(
            ds["CON"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.oa1 = jnp.array(
            ds["OA1"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.oa2 = jnp.array(
            ds["OA2"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.oa3 = jnp.array(
            ds["OA3"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.oa4 = jnp.array(
            ds["OA4"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ol1 = jnp.array(
            ds["OL1"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ol2 = jnp.array(
            ds["OL2"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ol3 = jnp.array(
            ds["OL3"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.ol4 = jnp.array(
            ds["OL4"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.var2d = jnp.array(
            ds["VAR2D"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )

    @property
    def reader_type(self) -> str:
        return "grist"


class GRISTBackgroundReader(AbstractReader[GRISTBackgroundReaderConfig]):
    time_str: str
    lon: Float[Array, "ncells"]
    lat: Float[Array, "ncells"]
    plev: Float[Array, "plev"]
    plev_lev: Float[Array, "plev_lev ncells"]
    nlev_eff: Float[Array, "ncells"]
    ps: Float[Array, "ncells"]
    q: Float[Array, "ncells plev"]
    skintemp: Float[Array, "ncells"]
    snow: Float[Array, "ncells"]
    snowh: Float[Array, "ncells"]
    soilh: Float[Array, "ncells"]
    t: Float[Array, "ncells plev"]
    u: Float[Array, "ncells plev"]
    v: Float[Array, "ncells plev"]
    xice: Float[Array, "ncells"]
    soiltemp: Float[Array, "ncells nsoil_levels"]
    nsoil_levels: Float[Array, "nsoil_levels"]
    soilmoist: Float[Array, "ncells nsoil_levels"]

    def __init__(
        self,
        bkg_file: str,
        para: PlanetParameterSet,
        dev: AbstractDevice,
        cfg: GRISTBackgroundReaderConfig = GRISTBackgroundReaderConfig(),
    ):
        super().__init__(
            para=para,
            cfg=cfg,
            dev=dev
        )
        self.bkg_file = bkg_file
        log.logger.info(f"Use GRIST background file {bkg_file} as background data source.")
        self.extract_data()

    def extract_data(self):
        float_type = self.dev.float_type
        ds = xr.open_dataset(self.bkg_file, engine="netcdf4")
        # ------------- Load data -------------
        self.lon = jnp.array(
            ds["lon"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.lat = jnp.array(
            ds["lat"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.plev = jnp.array(
            ds["plev"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.time_str = str(ds["time"].values)
        self.ps = jnp.array(
            ds["PS"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.q = jnp.array(
            ds["Q"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.skintemp = jnp.array(
            ds["SKINTEMP"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.snow = jnp.array(
            ds["SNOW"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.snowh = jnp.array(
            ds["SNOWH"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.soilh = jnp.array(
            ds["SOILH"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.t = jnp.array(
            ds["T"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.u = jnp.array(
            ds["U"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.v = jnp.array(
            ds["V"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.xice = jnp.array(
            ds["XICE"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        )
        self.soiltemp = jnp.array(
            ds["SoilTemp"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.nsoil_levels = jnp.array(
            ds["nSoilLevels"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        self.soilmoist = jnp.array(
            ds["SoilMoist"].values,
            device=jax.devices("cpu")[0],
            dtype=float_type
        ).T
        if self.cfg.vert_interp_type == "grist":
            # ------------- GRIST Post process -------------
            plev_in = self.plev
            nlev = plev_in.shape[0]
            ncol = self.ps.shape[0]

            # Check effective full level number
            mask = plev_in[:, None] >= self.ps[None, :]
            first_ge_idx = jnp.argmax(mask, axis=0)
            self.nlev_eff = jnp.where(
                jnp.any(mask, axis=0),
                first_ge_idx + 1,
                nlev
            )  # (ncol,)

            # Evaluate dry-mass coordinate at ERA pressure level and surface
            face_idx = jnp.arange(nlev + 1, dtype=jnp.int32)[:, None]  # (nlev+1, 1)
            nlev_eff = self.nlev_eff[None, :]  # (nlev+1, ncol)

            plev_lev = jnp.zeros((nlev + 1, ncol), dtype=plev_in.dtype)
            plev_mid = jnp.pad(
                0.5 * (plev_in[:-1, None] + plev_in[1:, None]),
                ((1, 1), (0, 0)),
                mode='constant'
            )  # (nlev+1, 1)
            plev_lev = jnp.where(
                (face_idx >= 1) & (face_idx < nlev_eff),
                plev_mid,
                plev_lev
            )
            # if ps<pface(nlev_eff), let it be, good assumption! no problem till now...
            plev_lev = jnp.where(
                face_idx == nlev_eff,
                self.ps[None, :],
                plev_lev
            )

            # hpface(1)=pface(1)=zeros
            # hpface(k)=hpface(k-1)+dp(k-1)*(1-q(k-1)), k=2..nlev_eff+1.
            dp = plev_lev[1:, :] - plev_lev[:-1, :]
            lev_idx = jnp.arange(nlev, dtype=jnp.int32)[:, None]  # (nlev, ncol)
            valid_lev = lev_idx <= nlev_eff - 1  # (nlev, ncol)
            dhp = jnp.where(valid_lev, dp, 0.0)
            self.plev_lev = jnp.cumulative_sum(
                dhp * (1.0 - self.q),
                axis=0,
                include_initial=True
            )  # (nlev+1, ncol)
            self.plev = 0.5 * (self.plev_lev[:-1, :] + self.plev_lev[1:, :])

    @property
    def reader_type(self) -> str:
        return "grist"
