from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING
import equinox as eqx


from terraflow.util import typing
from terraflow.util.history.setup.vert import AbstractVertHistorySetup


if TYPE_CHECKING:
    import numpy as np
    
IntType = typing.IntType
Float = typing.Float
Array = typing.Array


VMST = TypeVar("VMST", bound=AbstractVertHistorySetup)


class AbstractVertCoordinate(eqx.Module, Generic[VMST], ABC):

    @staticmethod
    @abstractmethod
    def alias_name() -> str:
        """Return the alias name of the vertical coordinate system."""
        raise NotImplementedError

    @abstractmethod
    def history_builder(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        *args,
        **kwargs
    ) -> VMST:
        raise NotImplementedError


class AbstractGeneralizedTerrainFollowingCoordinate(AbstractVertCoordinate):
    full_nlev: int
    half_nlev: int
    full_lev: Float[Array, "nlev"]
    half_lev: Float[Array, "nlev+1"]
    full_dlev: Float[Array, "nlev"]
    half_dlev: Float[Array, "nlev+1"]
    
    @abstractmethod
    def calc_mg(
        self,
        mgs: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        raise NotImplementedError
    
    @abstractmethod
    def calc_mg_lev(
        self,
        mgs: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        raise NotImplementedError
    
    @abstractmethod
    def calc_dmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        raise NotImplementedError
    
    @abstractmethod
    def calc_dmgdt_lev(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev+1 npoints ..."]:
        raise NotImplementedError
    
    @abstractmethod
    def calc_ddmgdt(
        self,
        dmgsdt: Float[Array, "npoints ..."],
        ks: IntType,
        ke: IntType,
    ) -> Float[Array, "nlev npoints ..."]:
        raise NotImplementedError