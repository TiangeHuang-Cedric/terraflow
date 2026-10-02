from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Generic,
    TypeVar,
    Self,
    TYPE_CHECKING
)


from terraflow.util import equinox as eqx
from terraflow.util.field import AbstractFields
from terraflow.util.planet import PlanetParameterSet
from terraflow.mesh.abstract import AbstractMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate


if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.util.backend import AbstractBackend
    from terraflow.util.device import AbstractDevice
    from terraflow.model.abstract import ModelMode
    from terraflow.model.config import UserConfig, Const


    FLT = typing.Float
    Array = typing.Array

HM       = TypeVar("HM"      , bound=AbstractMesh                      )
VM       = TypeVar("VM"      , bound=AbstractVertCoordinate            )
STATE    = TypeVar("STATE"   , bound="AbstractFields"                  )
STATIC   = TypeVar("STATIC"  , bound="AbstractFields"                  )
TEND     = TypeVar("TEND"    , bound="AbstractFields"                  )
AUX      = TypeVar("AUX"     , bound="AbstractFields"                  )
ADVBATCH = TypeVar("ADVBATCH", bound="AbstractFields"                  )
TRACERS  = TypeVar("TRACERS" , bound="AbstractFields"                  )
AUXDS    = TypeVar("AUXDS"   , bound="AbstractEquationAuxiliaryDataset")
BLOCK    = TypeVar("BLOCK"   , bound="AbstractBlock"                   )



class AbstractEquationAuxiliaryData(eqx.Module, ABC):
    initialized: bool = eqx.field(static=True)

    def has_initialized(self) -> bool:
        return self.initialized

    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        raise NotImplementedError


class AbstractEquationAuxiliaryDataset(eqx.Module, ABC):
    initialized: bool = eqx.field(static=True)
    
    def has_initialized(self) -> bool:
        return self.initialized
     
    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        raise NotImplementedError


class AbstractBlock(
    ABC,
    eqx.Module,
    Generic[
        HM, VM,
        STATE, TEND, STATIC, AUX, ADVBATCH, TRACERS, AUXDS
    ]
):
    elapsed_seconds: float | FLT[Array, ""]
    mode           : ModelMode  = eqx.field(static=True)
    const          : Const      = eqx.field(static=True)
    cfg            : UserConfig = eqx.field(static=True) 
    dt_dyn         : float      = eqx.field(static=True)
    dt_adv         : float      = eqx.field(static=True)
    old            : int
    new            : int
    dev            : AbstractDevice     = eqx.field(static=True)
    para           : PlanetParameterSet = eqx.field(static=True)
    mesh           : HM
    vert_coord     : VM
    state          : STATE
    tend           : TEND
    static         : STATIC
    aux            : AUX
    adv_batch_bg   : ADVBATCH
    adv_batch_pt   : ADVBATCH
    adv_batches    : dict[str, ADVBATCH]
    adv_batch_nh   : ADVBATCH
    tracers        : TRACERS
    equaux         : AUXDS

    @abstractmethod
    def reset(self, *args, **kwargs) -> Self:
        raise NotImplementedError
    
    @abstractmethod
    def trajectory_post_process(self) -> dict[str, Array]:
        raise NotImplementedError
    
    @property
    def backend(self) -> AbstractBackend:
        return self.dev.backend

    def time_advance(self) -> Self:
        elapsed_seconds = self.elapsed_seconds + self.dt_dyn
        return self.dev.backend.tree_at(
            lambda b: (b.old, b.new, b.elapsed_seconds),
            self,
            (self.new, self.old, elapsed_seconds)
        )

    def get_time_idx(self, itime_str: str) -> int:
        if itime_str == "old":
            return self.old
        elif itime_str == "new":
            return self.new
        else:
            raise ValueError(
                f"Invalid time index '{itime_str}'. Expected 'old' or 'new'."
            )
