from __future__ import annotations
from typing import Self, TYPE_CHECKING


from terraflow.util import typing, device
from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
from terraflow.util.history.setup.vert import HybridVerticalCoordinateSetup

if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet
    from terraflow.util.device import AbstractDevice
    import numpy as np

IntType = typing.IntType
Float = typing.Float
Array = typing.Array

PTOP = 2.194e2

LOCAL_PTOP = 0.0

HYBRID_COORD_P0 = 1.0e5


class HybridVertCoordinate(AbstractGeneralizedTerrainFollowingCoordinate):
    """
    HybridVertCoordinate represents a vertical coordinate system following the hybrid staggering.
    """
    p0        : Float[Array, "1"]
    ptop      : Float[Array, "1"]
    local_ptop: Float[Array, "1"]
    hyai      : Float[Array, "nlev+1"]
    hybi      : Float[Array, "nlev+1"]
    hyam      : Float[Array, "nlev"]
    hybm      : Float[Array, "nlev"]

    def __init__(
        self,
        p0: Float[Array, "1"],
        ptop: Float[Array, "1"],
        local_ptop: Float[Array, "1"],
        hyai: Float[Array, "nlev+1"],
        hybi: Float[Array, "nlev+1"],
        hyam: Float[Array, "nlev"],
        hybm: Float[Array, "nlev"],
        full_lev: Float[Array, "nlev"],
        half_lev: Float[Array, "nlev+1"],
        lev_hw: int,
        dev: AbstractDevice
    ):
        bkd = dev.backend
        if lev_hw < 0:
            raise ValueError(f"lev_hw must be non-negative, got {lev_hw}.")
        self.lev_hw = lev_hw
        self.dev = dev
        self.p0 = p0
        self.ptop = ptop
        self.local_ptop = local_ptop

        full_nlev = int(full_lev.shape[0])
        half_nlev = int(half_lev.shape[0])
        if half_nlev != full_nlev + 1:
            raise ValueError(
                "Hybrid vertical coordinate requires half_lev to have one "
                f"more level than full_lev, got {half_nlev} and {full_nlev}."
            )
        if hyam.shape[0] != full_nlev or hybm.shape[0] != full_nlev:
            raise ValueError(
                "Hybrid midpoint coefficients must match full_lev length."
            )
        if hyai.shape[0] != half_nlev or hybi.shape[0] != half_nlev:
            raise ValueError(
                "Hybrid interface coefficients must match half_lev length."
            )

        def zero_pad(value):
            return bkd.concatenate(
                [
                    bkd.zeros(lev_hw, dtype=value.dtype, device=value.device),
                    value,
                    bkd.zeros(lev_hw, dtype=value.dtype, device=value.device)
                ],
                axis=0
            )

        def edge_pad(value):
            return bkd.concatenate(
                [
                    bkd.full(
                        (lev_hw,),
                        value[0],
                        dtype=value.dtype,
                        device=value.device
                    ),
                    value,
                    bkd.full(
                        (lev_hw,),
                        value[-1],
                        dtype=value.dtype,
                        device=value.device
                    )
                ],
                axis=0
            )

        if full_nlev > 1:
            full_dlev = half_lev[1:] - half_lev[:-1]
        else:
            full_dlev = bkd.full(
                (1,),
                1,
                dtype=full_lev.dtype,
                device=full_lev.device
            )

        half_dlev = bkd.zeros_like(half_lev)
        half_dlev = bkd.slice_update(half_dlev, 0, full_lev[0] - half_lev[0])
        if full_nlev > 1:
            half_dlev = bkd.slice_update(
                half_dlev,
                slice(1, full_nlev),
                full_lev[1:] - full_lev[:-1]
            )
        half_dlev = bkd.slice_update(
            half_dlev,
            full_nlev,
            half_lev[-1] - full_lev[-1]
        )

        hyai = zero_pad(hyai)
        hybi = zero_pad(hybi)
        hyam = zero_pad(hyam)
        hybm = zero_pad(hybm)
        full_lev  = zero_pad(full_lev)
        half_lev  = zero_pad(half_lev)
        full_dlev = edge_pad(full_dlev)
        half_dlev = edge_pad(half_dlev)

        self.hyai = hyai
        self.hybi = hybi
        self.hyam = hyam
        self.hybm = hybm
        self.full_lev = full_lev
        self.half_lev = half_lev
        self.full_nlev = full_nlev
        self.half_nlev = half_nlev
        self.full_dlev = full_dlev
        self.half_dlev = half_dlev

    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the hybrid vertical coordinate system."""
        return "hybrid"

    @classmethod
    def init_global(
        cls,
        p0: Float[Array, "1"],
        ptop: Float[Array, "1"],
        local_ptop: Float[Array, "1"],
        hyai: Float[Array, "nlev+1"],
        hybi: Float[Array, "nlev+1"],
        hyam: Float[Array, "nlev"],
        hybm: Float[Array, "nlev"],
        full_lev: Float[Array, "nlev"],
        half_lev: Float[Array, "nlev+1"],
        lev_hw: int,
        dev: AbstractDevice
    ) -> Self:
        """Initialize the HybridVertCoordinate with global coefficients."""
        return cls(
            p0=p0,
            ptop=ptop,
            local_ptop=local_ptop,
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    def get_full_lev(self) -> Float[Array, "nlev"]:
        return self.full_lev

    def get_half_lev(self) -> Float[Array, "nlev+1"]:
        return self.half_lev

    def history_builder(
        self,
        file_path: str,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> HybridVerticalCoordinateSetup:
        return HybridVerticalCoordinateSetup(
            file_path=file_path,
            vert=self,
            output_type=output_type,
            float_type=float_type,
            int_type=int_type
        )

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
        ptop = self.local_ptop
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
        ptop = self.local_ptop
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
        hybi_l = self.hybi[ks:ke, None]
        hybi_r = self.hybi[ks+1:ke+1, None]
        dmgsdt = dmgsdt[None, :]
        return (hybi_r - hybi_l) * dmgsdt

    @classmethod
    def test_l15(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 15-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 15-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l15
        bkd = dev.backend
        hyai, hybi = hybrid_coord_test_l15(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def test_l15_2(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 15-level coefficients (variant 2).

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 15-level coefficients (variant 2).
        """
        from .hybrid_coord_test import hybrid_coord_test_l15_2
        bkd = dev.backend
        hyai, hybi = hybrid_coord_test_l15_2(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def test_l26(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 26-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 26-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l26
        bkd = dev.backend
        hyai, hybi = hybrid_coord_test_l26(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def test_l30(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using test 30-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with test 30-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_test_l30
        bkd = dev.backend
        hyai, hybi = hybrid_coord_test_l30(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )
        
    @classmethod
    def schar_l40(cls, dev: device.AbstractDevice, lev_hw: int = 0) -> Self:
        from .hybrid_coord_test import hybrid_coord_schar_l40
        bkd = dev.backend
        hyai, hybi = hybrid_coord_schar_l40(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )
            
    @classmethod
    def dcmip31_l10(cls, dev: device.AbstractDevice, lev_hw: int = 0) -> Self:
        from .hybrid_coord_test import hybrid_coord_dcmip31_l10
        bkd = dev.backend
        hyai, hybi = hybrid_coord_dcmip31_l10(dev)
        p0 = HYBRID_COORD_P0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def dcmip12(
        cls,
        nlev: int,
        para: PlanetParameterSet,
        dev: device.AbstractDevice,
        lev_hw: int = 0
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
        bkd = dev.backend
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
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def ecmwf_l50(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using ECMWF 50-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with ECMWF 50-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_ecmwf_l50
        bkd = dev.backend
        hyai, hybi = hybrid_coord_ecmwf_l50(dev)
        p0 = 1e5  # ECMWF standard surface pressure in Pa
        hyai = hyai / p0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def ecmwf_l90(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using ECMWF 90-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with ECMWF 90-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_ecmwf_l90
        bkd = dev.backend
        hyai, hybi = hybrid_coord_ecmwf_l90(dev)
        p0 = 1e5  # ECMWF standard surface pressure in Pa
        hyai = hyai / p0
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def wrf_l26(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 26-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 26-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l26
        bkd = dev.backend
        hyai, hybi = hybrid_coord_wrf_l26(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=bkd.array(PTOP, dtype=dev.float_type),
            local_ptop=bkd.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def wrf_l32(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 32-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 32-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l32
        bkd = dev.backend
        hyai, hybi = hybrid_coord_wrf_l32(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=bkd.array(PTOP, dtype=dev.float_type),
            local_ptop=bkd.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def wrf_l60(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 60-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 60-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l60
        bkd = dev.backend
        hyai, hybi = hybrid_coord_wrf_l60(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=bkd.array(PTOP, dtype=dev.float_type),
            local_ptop=bkd.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def wrf_l64(
        cls,
        dev: device.AbstractDevice,
        lev_hw: int = 0
    ) -> Self:
        """
        Create a HybridVertCoordinate instance using WRF 64-level coefficients.

        Args:
            dev (AbstractDevice): Device to create arrays on.
        Returns:
            HybridVertCoordinate: An instance of HybridVertCoordinate with WRF 64-level coefficients.
        """
        from .hybrid_coord_test import hybrid_coord_wrf_l64
        bkd = dev.backend
        hyai, hybi = hybrid_coord_wrf_l64(dev)
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(HYBRID_COORD_P0, dtype=dev.float_type),
            ptop=bkd.array(PTOP, dtype=dev.float_type),
            local_ptop=bkd.array(PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def cam_l32(
        cls,
        *,
        p0: float = HYBRID_COORD_P0,
        dev: device.AbstractDevice,
        lev_hw: int = 0
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
        bkd = dev.backend
        hyai, hybi = hybrid_coord_cam_l32(dev)
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
        )

    @classmethod
    def cam_l30(
        cls,
        *,
        p0: float = HYBRID_COORD_P0,
        dev: device.AbstractDevice,
        lev_hw: int = 0
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
        bkd = dev.backend
        hyai, hybi = hybrid_coord_cam_l30(dev)
        ptop = p0 * hyai[0]
        hyam, hybm = calc_hybrid_midpoint_coeff(hyai, hybi)
        full_lev, half_lev = calc_full_and_half_levels(hyai, hybi, hyam, hybm)
        return cls.init_global(
            p0=bkd.array(p0, dtype=dev.float_type),
            ptop=bkd.array(ptop, dtype=dev.float_type),
            local_ptop=bkd.array(LOCAL_PTOP, dtype=dev.float_type),
            hyai=hyai,
            hybi=hybi,
            hyam=hyam,
            hybm=hybm,
            full_lev=full_lev,
            half_lev=half_lev,
            lev_hw=lev_hw,
            dev=dev
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
