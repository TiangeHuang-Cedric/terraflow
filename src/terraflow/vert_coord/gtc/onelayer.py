from __future__ import annotations
from typing import TYPE_CHECKING


from terraflow.util.device import AbstractDevice
from terraflow.vert_coord.abstract import AbstractGeneralizedTerrainFollowingCoordinate
from terraflow.util.history.setup.vert import OneLayerVerticalCoordinateSetup


if TYPE_CHECKING:
    from terraflow.util import typing
    import numpy as np

    IntType = typing.IntType
    Float = typing.Float
    Array = typing.Array


class OneLayerVertCoordinate(AbstractGeneralizedTerrainFollowingCoordinate):
    
    def __init__(self, dev: AbstractDevice):
        bkd = dev.backend
        self.dev       = dev
        self.full_nlev = 1
        self.half_nlev = 2
        self.full_lev  = bkd.zeros(
            self.full_nlev, 
            device=dev(), 
            dtype=dev.float_type
        )
        self.half_lev  = bkd.zeros(
            self.half_nlev, 
            device=dev(), 
            dtype=dev.float_type
        )
        self.full_dlev = bkd.zeros(
            self.full_nlev,
            device=dev(), 
            dtype=dev.float_type           
        )
        self.half_dlev = bkd.zeros(
            self.half_nlev,
            device=dev(), 
            dtype=dev.float_type
        )
        
    @staticmethod
    def alias_name() -> str:
        """Return the alias name of the vertical coordinate system."""
        return "one_layer"
    
    def history_builder(
        self, 
        file_path: str, 
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> OneLayerVerticalCoordinateSetup:
        return OneLayerVerticalCoordinateSetup(
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
        raise NotImplementedError
    
    def calc_mg_lev(
        self,
        mgs: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        raise NotImplementedError
    
    def calc_dmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        raise NotImplementedError
    
    def calc_dmgdt_lev(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        raise NotImplementedError

    def calc_ddmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        raise NotImplementedError    