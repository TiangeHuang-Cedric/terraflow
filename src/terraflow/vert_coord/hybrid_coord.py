from __future__ import annotations
from typing import Self, TYPE_CHECKING
import jax.numpy as jnp
import equinox as eqx

from terraflow.util import typing, device
from terraflow.vert_coord.abstract import AbstractLorenzCoordinate
from terraflow.util.history.setup.vert import HybridVerticalCoordinateSetup

if TYPE_CHECKING:
    from terraflow.planet import PlanetParameterSet
    from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
    import numpy as np

IntType = typing.IntType
Float = typing.Float
Array = typing.Array

PTOP = 2.194e2
LOCAL_PTOP = 0.0
HYBRID_COORD_P0 = 1.0e5


class HybridVertCoordinate(AbstractLorenzCoordinate):
    """
    HybridVertCoordinate represents a vertical coordinate system following the hybrid staggering.
    """
    hw: IntType = eqx.field(static=True)
    full_kds: IntType = eqx.field(static=True)
    full_kde: IntType = eqx.field(static=True)
    full_kms: IntType = eqx.field(static=True)
    full_kme: IntType = eqx.field(static=True)
    half_kds: IntType = eqx.field(static=True)
    half_kde: IntType = eqx.field(static=True)
    half_kms: IntType = eqx.field(static=True)
    half_kme: IntType = eqx.field(static=True)
    p0: Float[Array, "1"]
    ptop: Float[Array, "1"]
    hyai: Float[Array, "nlev+1"]
    hybi: Float[Array, "nlev+1"]
    hyam: Float[Array, "nlev"]
    hybm: Float[Array, "nlev"]
    full_lev: Float[Array, "nlev"]
    half_lev: Float[Array, "nlev+1"]
    full_dlev: Float[Array, "nlev"]
    half_dlev: Float[Array, "nlev+1"]

    def __init__(
        self,
        hw: IntType,
        full_kds: IntType,
        full_kde: IntType,
        full_kms: IntType,
        full_kme: IntType,
        half_kds: IntType,
        half_kde: IntType,
        half_kms: IntType,
        half_kme: IntType,
        p0: Float[Array, "1"],
        ptop: Float[Array, "1"],
        hyai: Float[Array, "nlev+1"],
        hybi: Float[Array, "nlev+1"],
        hyam: Float[Array, "nlev"],
        hybm: Float[Array, "nlev"],
        full_lev: Float[Array, "nlev"],
        half_lev: Float[Array, "nlev+1"]
    ):
        self.hw = hw
        self.full_kds = full_kds
        self.full_kde = full_kde
        self.full_kms = full_kms
        self.full_kme = full_kme
        self.half_kds = half_kds
        self.half_kde = half_kde
        self.half_kms = half_kms
        self.half_kme = half_kme
        self.p0 = p0
        self.ptop = ptop
        self.hyai = hyai
        self.hybi = hybi
        self.hyam = hyam
        self.hybm = hybm
        self.full_lev = full_lev
        self.half_lev = half_lev
        # =========================================================
        # full dlev
        full_dlev = self.half_lev[1:] - self.half_lev[:-1]
        # =========================================================
        # half dlev
        half_dlev = jnp.zeros_like(self.half_lev)
        # =========================================
        # k = 1
        tmp = self.full_lev[0] - self.half_lev[0]
        half_dlev = half_dlev.at[0].set(tmp)
        # =========================================
        # k = 2, ..., nlev
        tmp = self.full_lev[1:]-self.full_lev[:-1]
        half_dlev = half_dlev.at[1:-1].set(tmp)
        # =========================================
        # k = nlev + 1
        tmp = self.half_lev[-1] - self.full_lev[-1]
        half_dlev = half_dlev.at[-1].set(tmp)
        # =========================================================
        self.full_dlev = full_dlev
        self.half_dlev = half_dlev

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the hybrid vertical coordinate system."""
        return "hybrid"

    @property
    def nlev(self) -> int:
        return self.hyai.shape[0] - 1

    @property
    def n_full_lev(self) -> int:
        return self.nlev

    @property
    def n_half_lev(self) -> int:
        return self.nlev + 1

    @classmethod
    def init_global(
        cls,
        p0: Float[Array, "1"],
        ptop: Float[Array, "1"],
        hyai: Float[Array, "nlev+1"],
        hybi: Float[Array, "nlev+1"],
        hyam: Float[Array, "nlev"],
        hybm: Float[Array, "nlev"],
        full_lev: Float[Array, "nlev"],
        half_lev: Float[Array, "nlev+1"]
    ) -> Self:
        """Initialize the HybridVertCoordinate with global coefficients."""
        return cls(
            hw=0,
            full_kds=0,
            full_kde=hyam.shape[0] - 1,
            full_kms=0,
            full_kme=hyam.shape[0] - 1,
            half_kds=0,
            half_kde=hyai.shape[0] - 1,
            half_kms=0,
            half_kme=hyai.shape[0] - 1,
            p0=p0,
            ptop=ptop,
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    def get_layout(self) -> HybridVertCoordLayout:
        from terraflow.layout.hybrid_vert_coord import HybridVertCoordLayout
        return HybridVertCoordLayout()

    def get_full_lev(self) -> Float[Array, "nlev"]:
        return self.full_lev

    def get_half_lev(self) -> Float[Array, "nlev+1"]:
        return self.half_lev

    def history_builder(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> HybridVerticalCoordinateSetup:
        return HybridVerticalCoordinateSetup(
            file_path=file_path,
            vert=self,
            float_type=float_type,
            int_type=int_type
        )

    def get_idx(self, entity: str, domain: str) -> tuple[IntType, IntType]:
        if entity == "full":
            if domain == "domain":
                k_s, k_e = self.full_kds, self.full_kde
            elif domain == "memory":
                k_s, k_e = self.full_kms, self.full_kme
            else:
                raise ValueError(f"Invalid domain: {domain}")
        elif entity == "half":
            if domain == "domain":
                k_s, k_e = self.half_kds, self.half_kde
            elif domain == "memory":
                k_s, k_e = self.half_kms, self.half_kme
            else:
                raise ValueError(f"Invalid domain: {domain}")
        else:
            raise ValueError(f"Invalid entity: {entity}")
        return k_s, k_e

    def calc_mg(
        self,
        mgs: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        hyam = self.hyam[ks:ke+1, None]
        hybm = self.hybm[ks:ke+1, None]
        mgs = mgs[None, ...]
        p0 = self.p0
        ptop = LOCAL_PTOP
        return hyam * (p0 - ptop) + hybm * (mgs - ptop) + ptop

    def calc_mg_lev(
        self,
        mgs: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        hyai = self.hyai[ks:ke+1, None]
        hybi = self.hybi[ks:ke+1, None]
        mgs = mgs[None, ...]
        p0 = self.p0
        ptop = LOCAL_PTOP
        return hyai * (p0 - ptop) + hybi * (mgs - ptop) + ptop

    def calc_dmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        hybm = self.hybm[ks:ke+1, None]
        return hybm * dmgsdt

    def calc_dmgdt_lev(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        hybi = self.hybi[ks:ke+1, None]
        dmgsdt = dmgsdt[None, :]
        return hybi * dmgsdt

    def calc_ddmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        hybi = self.hybi[ks:ke+1, None]
        dmgsdt = dmgsdt[None, :]
        tmp = hybi[ks+1:ke, ...] - hybi[ks:ke+1, ...]
        return tmp * dmgsdt

    def fill_halo(self, field: Array, is_index: bool = False, *args, **kwargs) -> Array:
        return field

    @classmethod
    def test_l15(
        cls,
        dev: device.AbstractDevice,
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 15-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 15-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l15
        hyai, hybi = hybrid_coord_test_l15(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def test_l15_2(
        cls,
        dev: device.AbstractDevice,
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 15-level coefficients (variant 2).

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 15-level coefficients (variant 2).
        """
        from .hybrid_coord_test import hybrid_coord_test_l15_2
        hyai, hybi = hybrid_coord_test_l15_2(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def test_l26(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 26-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 26-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l26
        hyai, hybi = hybrid_coord_test_l26(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def test_l30(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 30-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 30-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l30
        hyai, hybi = hybrid_coord_test_l30(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def dcmip12(
        cls,
        nlev: int,
        para: PlanetParameterSet,
        dev: device.AbstractDevice,
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using DCMIP1-2 coefficients.

        Args:
            nlev (int): Number of vertical levels.
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with DCMIP1-2 coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_dcmip12
        p0 = HYBRID_COORD_P0
        ptop, hyai, hybi = hybrid_coord_dcmip12(
            p0=p0,
            nlev=nlev,
            para=para,
            dev=dev
        )
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def ecmwf_l50(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using ECMWF 50-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with ECMWF 50-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_ecmwf_l50
        hyai, hybi = hybrid_coord_ecmwf_l50(dev)
        p0 = 1e5  # ECMWF standard surface pressure in Pa
        hyai = hyai / p0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def ecmwf_l90(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using ECMWF 90-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with ECMWF 90-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_ecmwf_l90
        hyai, hybi = hybrid_coord_ecmwf_l90(dev)
        p0 = 1e5  # ECMWF standard surface pressure in Pa
        hyai = hyai / p0
        ptop = p0 * hybi[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def wrf_l26(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 26-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 26-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l26
        hyai, hybi = hybrid_coord_wrf_l26(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=jnp.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def wrf_l32(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 32-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 32-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l32
        hyai, hybi = hybrid_coord_wrf_l32(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=jnp.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def wrf_l60(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 60-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 60-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l60
        hyai, hybi = hybrid_coord_wrf_l60(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=jnp.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def wrf_l64(
        cls,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 64-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 64-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l64
        hyai, hybi = hybrid_coord_wrf_l64(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=jnp.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def cam_l32(
        cls,
        *,
        p0: float = HYBRID_COORD_P0,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using CAM 32-level coefficients.

        Args:
            p0 (float): Reference pressure at the surface (in Pa).
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with CAM 32-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_cam_l32
        hyai, hybi = hybrid_coord_cam_l32(dev)
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )

    @classmethod
    def cam_l30(
        cls,
        *,
        p0: float = HYBRID_COORD_P0,
        dev: device.AbstractDevice
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using CAM 30-level coefficients.

        Args:
            p0 (float): Reference pressure at the surface (in Pa).
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with CAM 30-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_cam_l30
        hyai, hybi = hybrid_coord_cam_l30(dev)
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=jnp.array(p0, dtype=dev.float_type),
            ptop=jnp.array(ptop, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev
        )


def calc_hybrid_midpoint_coeff(
    hyai: Float[Array, "nlev+1"],
    hybi: Float[Array, "nlev+1"]
) -> tuple[Float[Array, "nlev"], Float[Array, "nlev"]]:
    """
    Calculate hybrid coefficients at layer midpoints.

    Args:
        hyai (Array): Hybrid A coefficients at layer interfaces.
        hybi (Array): Hybrid B coefficients at layer interfaces.
    Returns:
        hyam (Array): Hybrid A coefficients at layer midpoints.
        hybm (Array): Hybrid B coefficients at layer midpoints.
    """
    hyam = 0.5 * (hyai[:-1] + hyai[1:])
    hybm = 0.5 * (hybi[:-1] + hybi[1:])
    return hyam, hybm


def calc_full_and_half_levels(
    hyai: Float[Array, "nlev+1"],
    hybi: Float[Array, "nlev+1"],
    hyam: Float[Array, "nlev"],
    hybm: Float[Array, "nlev"]
) -> tuple[Float[Array, "nlev"], Float[Array, "nlev+1"]]:
    """
    Calculate full and half levels from hybrid coefficients.

    Args:
        hyai (Array): Hybrid A coefficients at layer interfaces.
        hybi (Array): Hybrid B coefficients at layer interfaces.
        hyam (Array): Hybrid A coefficients at layer midpoints.
        hybm (Array): Hybrid B coefficients at layer midpoints.
    Returns:
        full_lev (Array): Full level coefficients.
        half_lev (Array): Half level coefficients.
    """
    full_lev = hyam + hybm
    half_lev = hyai + hybi
    return full_lev, half_lev
