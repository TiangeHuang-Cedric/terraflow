from __future__ import annotations
from typing import (
    Self,
    Any,
    Optional,
    TYPE_CHECKING
)
from abc import ABC, abstractmethod
from pathlib import Path
import netCDF4 as nc4
import numpy as np


from terraflow.util import log, datetime as tfdatetime
from terraflow.util.datetime import (
    DateTimeManager,
    perf_runtime,
    perf_write_history,
    format_seconds
)


if TYPE_CHECKING:
    from terraflow.model.abstract import AbstractModel
    from terraflow.util.backend import AbstractBackend
    from terraflow.util.device import AbstractDevice
    from terraflow.util.field import AbstractFields


FLOAT_STR2NP = {
    "r4": np.float32,
    "r8": np.float64,
}

INT_STR2NP = {
    "i4": np.int32,
    "i8": np.int64,
}

FLOAT_NP2NC = {
    np.float32: "f4",
    np.float64: "f8"
}

INT_NP2NC = {
    np.int32: "i4",
    np.int64: "i8"
}


class AbstractHistoryset(ABC):
    backend: AbstractBackend
    dev: AbstractDevice
    time_manager: DateTimeManager
    case_name: str
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    built: bool

    @abstractmethod
    def __init__(
        self,
        model: AbstractModel,
        case_name: str = "terraflow",
        file_path: str = "",
        float_type: Optional[str] = None,
        int_type: Optional[str] = None,
        **kwargs
    ) -> None:
        self.built = False
        self.backend = model.backend
        self.dev = model.device
        self.time_manager = model.time_manager
        self.case_name = case_name
        if float_type is None:
            self.float_type = np.float32
        elif float_type in FLOAT_STR2NP:
            self.float_type = FLOAT_STR2NP[float_type]
        else:
            raise ValueError(f"Unsupported float type: {float_type}")
        if int_type is None:
            self.int_type = np.int32
        elif int_type in INT_STR2NP:
            self.int_type = INT_STR2NP[int_type]
        else:
            raise ValueError(f"Unsupported int type: {int_type}")

    @abstractmethod
    def setup(self, block: Any) -> Self:
        raise NotADirectoryError
    
    @abstractmethod
    def get_all_histories(self, with_h2: bool = False) -> list[str]:
        raise NotImplementedError
    
    def is_root(self) -> bool:
        return self.dev.is_root()
    
    def sync(self) -> None:
        self.dev.proc.comm_model.Barrier()
    
    def has_built(self) -> bool:
        return self.built

    def output_diag(self, itimestep: int) -> None:
        self.backend.io_callback(
            self.log_diag,
            None,
            itimestep,
            ordered=True
        )

    def output_prepare(self, save_interval_idx: int) -> None:
        
        def _prepare_for_output(save_interval_idx: int) -> None:
            if not self.is_root():
                return None
            perf_runtime()

            log.logger.log(
                "RUNTIME",
                f"==> {self.time_manager.curr_time4save(save_interval_idx)}  "
                f"{format_seconds(tfdatetime.elapsed_seconds_in_integration, precision=16)}  "
                f"{format_seconds(tfdatetime.elapsed_secondes_in_real_world, precision=16)}"
            )
            return None

        self.backend.io_callback(
            _prepare_for_output,
            None,
            save_interval_idx,
            ordered=True
        )

    @abstractmethod
    def output(
        self,
        block: Any,
        itime_str: str,
        save_idx: int,
        first_call: bool = False
    ) -> None:
        raise NotImplementedError

    def log_diag(self, itimestep: int) -> None:
        if not self.is_root():
            return None
        perf_runtime()

        log.logger.log(
            "RUNTIME",
            f"==> {self.time_manager.curr_time4itimestep(itimestep)}  "
            f"{format_seconds(tfdatetime.elapsed_seconds_in_integration, precision=16)}  "
            f"{format_seconds(tfdatetime.elapsed_secondes_in_real_world, precision=16)}"
        )
        return None

    def log_write_elapsed_seconds(self) -> None:
        if not self.is_root():
            return None
        log.logger.success(
            f"Time cost {format(perf_write_history(), '.16f')} seconds."
        )
        return None

    def log_start_write(self, history: str) -> None:
        if not self.is_root():
            return None
        log.logger.info(
            f"Write {history} file."
        )

    def log_done_write(self, history: str, cost_time: float) -> None:
        if not self.is_root():
            return None
        log.logger.success(
            f"Done write {history} file cost {format(cost_time, '.16f')} seconds."
        )
        return None


class AbstractHistory(ABC):
    backend: AbstractBackend
    file_path: str
    time_manager: DateTimeManager
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    nc_float_type: str
    nc_int_type: str
    built: bool

    @abstractmethod
    def __init__(
        self,
        model: AbstractModel,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> None:
        self.backend = model.backend
        self.history_name = history_name
        self.case_name = case_name
        self.file_path = str(
            Path(file_path) /
            f"{case_name}.{history_name}.nc"
        )
        self.float_type = float_type
        self.int_type = int_type
        self.nc_float_type = FLOAT_NP2NC[self.float_type]
        self.nc_int_type = INT_NP2NC[self.int_type]
        self.built = False

    @abstractmethod
    def setup(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    def end_setup(self):
        self.built = True

    def has_built(self) -> bool:
        return self.built

    def open_dataset(self) -> nc4.Dataset:
        return nc4.Dataset(self.file_path, mode="r")

    @abstractmethod
    def write_fields(
        self,
        fields: AbstractFields,
        *args,
        **kwargs
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_fields(self, fields: Any) -> None:
        raise NotImplementedError
