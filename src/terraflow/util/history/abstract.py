from __future__ import annotations
from typing import (
    TypeVar,
    Type,
    Self,
    Generic,
    Optional,
    TYPE_CHECKING
)
from abc import ABC, abstractmethod
from pathlib import Path
import time
import datetime
import netCDF4 as nc4
import numpy as np
import jax.numpy as jnp
import jax.experimental

from terraflow.util import log, datetime as tfdatetime, uxarray as tuxr
from terraflow.util.device import AbstractDevice
from terraflow.util.datetime import (
    DateTimeManager,
    perf_runtime,
    perf_write_history,
    format_seconds
)
from terraflow.util.history.setup.base import GlobalAttributesSetup, TimeSetup
from terraflow.util.history.setup.mesh import AbstractMeshHistorySetup
from terraflow.util.history.setup.vert import AbstractVertHistorySetup
from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.mesh.scvt.regridder import Regridder
from terraflow.mesh.scvt.interp import interp_plev
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.layout.field import Field, FieldMetaData, StaggeredFieldMetaData, NumericMetaData

if TYPE_CHECKING:
    from terraflow.util import typing
    from terraflow.discretization.abstract import (
        AbstractEquationSet,
        EquationMode,
        AbstractFields,
        AbstractBlock,
        AbstractTracerMetaData
    )
    from terraflow.solver.abstract import AbstractSolver

    Int = typing.Int
    Array = typing.Array

UM = TypeVar("UM", bound=AbstractUnstructuredMesh)
HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
HY = TypeVar("HY", bound='AbstractHistory')
SHY = TypeVar("SHY", bound='Unstructured2StructuredHistory')


STRUCTURED_HISTORY_LIST = ["h0", "h2"]


FLOAT_JNP2NP = {
    jnp.float32: np.float32,
    jnp.float64: np.float64,
}

INT_JNP2NP = {
    jnp.int32: np.int32,
    jnp.int64: np.int64,
}

FLOAT_NP2NC = {
    np.float32: "f4",
    np.float64: "f8"
}

INT_NP2NC = {
    np.int32: "i4",
    np.int64: "i8"
}

UNSTRUCTURED_SPATIAL_DIMS = {
    "cell", "vtx", "edge",
    "n_face", "n_node", "n_edge"
}

UGRID_SPATIAL_DIMS = {
    "n_face", "n_node", "n_edge"
}

LEV_DIMS = {"lev", "ilev"}

VARNAMES_USING_LOG_INTERP = {"gz", "gz_lev", "t"}

PLEV_NAMES = {"ph", "ph_lev"}


class AbstractHistoryset(ABC, Generic[HY, SHY]):
    h0: HY
    h1: HY
    h2: SHY
    output_h0: bool
    output_h1: bool
    output_h2: bool
    time_manager: DateTimeManager
    case_name: str
    dt_dyn: datetime.timedelta
    save_interval: datetime.timedelta
    save_interval_value: float
    save_interval_unit: str
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    _built: bool

    @abstractmethod
    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem,
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        h0_opt: dict = {},
        h1_opt: dict = {},
        h2_opt: dict = {},
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
        **kwargs
    ) -> None:
        self._built = False
        self.time_manager = solver.time_manager
        self.dt_dyn = solver.dt_dyn
        self.save_interval = solver.save_interval
        self.save_interval_value = solver.save_interval_value
        self.save_interval_unit = solver.save_interval_unit
        self.case_name = case_name
        self.output_h0 = output_h0
        self.output_h1 = output_h1
        self.output_h2 = output_h2
        if dev is not None:
            if float_type is None and int_type is None:
                self.float_type = FLOAT_JNP2NP[dev.float_type]
                self.int_type = INT_JNP2NP[dev.int_type]
            else:
                raise ValueError(
                    "Specifying both dev and type parameters caused confusion errors"
                )
        if float_type is None:
            self.float_type = np.float32
        elif float_type in FLOAT_JNP2NP.values():
            self.float_type = float_type
        else:
            raise ValueError(f"Unsupported float type: {float_type}")
        if int_type is None:
            self.int_type = np.int32
        elif int_type in INT_JNP2NP.values():
            self.int_type = int_type
        else:
            raise ValueError(f"Unsupported int type: {int_type}")
        if self.output_h0:
            self.h0 = self.history_type()(
                solver=solver,
                coord=coord,
                history_name="h0",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h0_opt,
                **kwargs,
            )
            log.logger.info(f"History will output h0: {self.h0.file_path}")
        if self.output_h1:
            self.h1 = self.history_type()(
                solver=solver,
                coord=coord,
                history_name="h1",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h1_opt,
                **kwargs,
            )
            log.logger.info(f"History will output h1: {self.h1.file_path}")
        if self.output_h2:
            self.h2 = self.structured_history_type()(
                solver=solver,
                coord=coord,
                history_name="h2",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h2_opt,
                **kwargs,
            )
            log.logger.info(f"History will output h2: {self.h2.file_path}")

    def setup(self, block: AbstractBlock) -> Self:
        if self.has_built():
            return self
        self.setup_h0(block)
        self.setup_h1(block)
        self.setup_h2(block)
        self._built = True
        return self

    @staticmethod
    @abstractmethod
    def history_type() -> Type[HY]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def structured_history_type() -> Type[SHY]:
        raise NotImplementedError

    def get_all_histories(self, with_h2: bool = False) -> list[str]:
        histories = []
        if self.output_h0:
            histories.append(self.h0.file_path)
        if self.output_h1:
            histories.append(self.h1.file_path)
        if self.output_h2 and with_h2:
            histories.append(self.h2.file_path)
        return histories

    def has_built(self) -> bool:
        return self._built

    def output_diag(self, itimestep: int) -> None:
        jax.experimental.io_callback(
            self.log_diag,
            None,
            itimestep,
            ordered=True
        )

    def output_prepare(self, save_interval_idx: int) -> None:
        jax.experimental.io_callback(
            self._prepare_for_output,
            None,
            save_interval_idx,
            ordered=True
        )

    def output(
        self,
        block: AbstractBlock,
        itime_str: str,
        save_idx: int,
        first_call: bool = False
    ) -> None:
        """
        Write the fields and diagnostics to the history file.

        Args:
            old_block (AbstractBlock): The block before the time step (include old state).
            new_block (AbstractBlock): The block after the time step (include new state, old tend and old aux).
            save_idx (int): The index of the current save interval.
            first_call (bool): Whether the output function is called for the first time.
                If True, it will write the static fields to the history file.

        Returns:
            None
        """
        if itime_str == "old":
            jax.experimental.io_callback(
                self._output_old,
                None,
                block,
                save_idx,
                first_call,
                ordered=True
            )
        elif itime_str == "new":
            jax.experimental.io_callback(
                self._output_new,
                None,
                block,
                save_idx,
                first_call,
                ordered=True
            )
        else:
            raise ValueError(
                f"Invalid itime_str: {itime_str}, expected 'old' or 'new'."
            )

    def log_diag(self, itimestep: int) -> None:
        perf_runtime()

        log.logger.log(
            "RUNTIME",
            f"==> {self.time_manager.curr_time4itimestep(self.dt_dyn, itimestep)}  "
            f"{format_seconds(tfdatetime.elapsed_seconds_in_integration, precision=16)}  "
            f"{format_seconds(tfdatetime.elapsed_secondes_in_real_world, precision=16)}"
        )
        return None

    def _prepare_for_output(self, save_interval_idx: int) -> None:
        perf_runtime()

        log.logger.log(
            "RUNTIME",
            f"==> {self.time_manager.curr_time4save(self.save_interval, save_interval_idx)}  "
            f"{format_seconds(tfdatetime.elapsed_seconds_in_integration, precision=16)}  "
            f"{format_seconds(tfdatetime.elapsed_secondes_in_real_world, precision=16)}"
        )
        return None

    def log_write_elapsed_seconds(self) -> None:
        log.logger.success(
            f"Time cost {format(perf_write_history(), '.16f')} seconds."
        )
        return None

    def log_start_write(self, history: str) -> None:
        log.logger.info(
            f"Write {history} file."
        )

    def log_done_write(self, history: str, cost_time: float) -> None:
        log.logger.success(
            f"Done write {history} file cost {format(cost_time, '.16f')} seconds."
        )
        return None

    def _output_old(
        self,
        block: AbstractBlock,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:
        return self._output_in_time(
            block=block,
            itime_str="old",
            save_interval_idx=save_interval_idx,
            first_call=first_call
        )

    def _output_new(
        self,
        block: AbstractBlock,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:
        return self._output_in_time(
            block=block,
            itime_str="new",
            save_interval_idx=save_interval_idx,
            first_call=first_call
        )

    def _output_in_time(
        self,
        block: AbstractBlock,
        itime_str: str,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:

        self.log_write_elapsed_seconds()

        def _set_time_var(h: HY | SHY):
            ds = nc4.Dataset(h.file_path, mode="a")
            time_var = ds.variables["time"]
            current_length = len(time_var)
            interval = save_interval_idx * self.save_interval_value
            time_var[current_length] = interval
            ds.close()

        if self.output_h0:
            self.log_start_write("h0")
            t0 = time.perf_counter()
            _set_time_var(self.h0)
            self.write_h0(block, itime_str=itime_str, first_call=first_call)
            t1 = time.perf_counter()
            self.log_done_write("h0", t1 - t0)
        if self.output_h1:
            self.log_start_write("h1")
            t0 = time.perf_counter()
            _set_time_var(self.h1)
            self.write_h1(block, itime_str=itime_str)
            t1 = time.perf_counter()
            self.log_done_write("h1", t1 - t0)
        if self.output_h2:
            self.log_start_write("h2")
            t0 = time.perf_counter()
            _set_time_var(self.h2)
            self.write_h2(block, itime_str=itime_str)
            t1 = time.perf_counter()
            self.log_done_write("h2", t1 - t0)
        return None

    def setup_h0(self, block: AbstractBlock) -> None:
        if self.output_h0:
            self.h0.setup(block.state)
            self.h0.setup(block.tend)
            self.h0.setup(block.static)
            self.h0.setup(block.aux)
            self.h0.setup(block.tracers)
            self.h0.setup(block.adv_batch_bg)
            self.h0.setup(block.adv_batch_pt)
            for batch in block.adv_batches.values():
                self.h0.setup(batch)
            self.h0.end_setup()

    def setup_h1(self, block: AbstractBlock) -> None:
        if self.output_h1:
            self.h1.setup(block.state)
            self.h1.setup(block.tend)
            self.h1.setup(block.static)
            self.h1.setup(block.aux)
            self.h1.setup(block.adv_batch_bg)
            self.h1.setup(block.adv_batch_pt)
            for batch in block.adv_batches.values():
                self.h1.setup(batch)
            self.h1.end_setup()

    def setup_h2(self, block: AbstractBlock) -> None:
        if self.output_h2:
            if self.output_h0:
                self.h2.setup(block.state)
                self.h2.setup(block.tend)
                self.h2.setup(block.static)
                self.h2.setup(block.aux)
                self.h2.setup(block.tracers)
                self.h2.setup(block.adv_batch_bg)
                self.h2.setup(block.adv_batch_pt)
                for batch in block.adv_batches.values():
                    self.h2.setup(batch)
                self.h2.end_setup()
            else:
                raise ValueError(
                    "output_h0 is False, cannot setup h2."
                )

    def write_h0(self, block: AbstractBlock, itime_str: str, first_call: bool = False):
        itime = block.get_time_idx(itime_str=itime_str)
        self.h0.write_fields(block.state, expand_idx=itime)
        self.h0.write_fields(block.tend)
        if first_call:
            self.h0.write_fields(block.static)
        self.h0.write_fields(block.aux)
        for batch in block.adv_batches.values():
            batch_idx = batch.idx
            itime_batch = batch.get_time_idx(itime_str=itime_str)
            self.h0.write_fields(
                block.tracers,
                expand_idx=itime_batch,
                tracer_idx=batch_idx
            )
        self.h0.write_fields(block.adv_batch_bg)
        self.h0.write_fields(block.adv_batch_pt)
        for batch in block.adv_batches.values():
            self.h0.write_fields(batch)

    def write_h1(self, block: AbstractBlock, itime_str: str):
        itime = block.get_time_idx(itime_str=itime_str)
        self.h1.write_fields(block.state, expand_idx=itime)
        self.h1.write_fields(block.tend)
        self.h1.write_fields(block.static)
        self.h1.write_fields(block.aux)
        self.h1.write_fields(block.adv_batch_bg)
        self.h1.write_fields(block.adv_batch_pt)
        for batch in block.adv_batches.values():
            self.h1.write_fields(batch)

    def write_h2(self, block: AbstractBlock, itime_str: str):
        itime = block.get_time_idx(itime_str=itime_str)
        ph, ph_lev = block.get_plev()
        ph, ph_lev = self.h2.update_plev(ph=ph, ph_lev=ph_lev, itime=itime)
        if self.output_h0:
            self.h2.write_fields(
                block.state,
                expand_idx=itime,
                ph=ph,
                ph_lev=ph_lev
            )
            self.h2.write_fields(block.tend, ph=ph, ph_lev=ph_lev)
            self.h2.write_fields(block.static, ph=ph, ph_lev=ph_lev)
            self.h2.write_fields(block.aux, ph=ph, ph_lev=ph_lev)
            for batch in block.adv_batches.values():
                batch_idx = batch.idx
                itime_batch = batch.get_time_idx(itime_str=itime_str)
                self.h2.write_fields(
                    block.tracers,
                    expand_idx=itime_batch,
                    tracer_idx=batch_idx,
                    ph=ph,
                    ph_lev=ph_lev
                )
            self.h2.write_fields(block.adv_batch_bg, ph=ph, ph_lev=ph_lev)
            self.h2.write_fields(block.adv_batch_pt, ph=ph, ph_lev=ph_lev)
            for batch in block.adv_batches.values():
                self.h2.write_fields(batch, ph=ph, ph_lev=ph_lev)
        else:
            raise ValueError(
                "output_h0 is False, cannot write h2."
            )


class UGRIDHistoryset(
    AbstractHistoryset["UGRIDHistory", "Unstructured2StructuredHistory"]
):

    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        use_dual: bool = False,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            solver=solver,
            coord=coord,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"use_dual": use_dual},
            h1_opt={"use_dual": use_dual},
            dev=dev,
            float_type=float_type,
            int_type=int_type,
            **kwargs
        )

    @staticmethod
    def history_type() -> Type["UGRIDHistory"]:
        return UGRIDHistory

    @staticmethod
    def structured_history_type() -> Type["Unstructured2StructuredHistory"]:
        return Unstructured2StructuredHistory


class UnstructuredHistoryset(
    AbstractHistoryset["UnstructuredHistory", "Unstructured2StructuredHistory"]
):

    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        equation: AbstractEquationSet,
        case_name: str = "terraflow",
        file_path: str = "",
        output_h0: bool = True,
        output_h1: bool = False,
        output_h2: bool = False,
        dev: Optional[AbstractDevice] = None,
        float_type: Optional[type[np.float32] | type[np.float64]] = None,
        int_type: Optional[type[np.int32] | type[np.int64]] = None,
        **kwargs
    ) -> None:
        mode: EquationMode = equation.mode
        with_edge = False if not mode.is_advection() and not mode.is_baroclinic() else True
        with_vtx = True if not mode.is_advection() else False
        super().__init__(
            solver=solver,
            coord=coord,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"with_vtx": with_vtx, "with_edge": with_edge},
            h1_opt={"with_vtx": with_vtx, "with_edge": True},
            dev=dev,
            float_type=float_type,
            int_type=int_type,
            **kwargs,
        )

    @staticmethod
    def history_type() -> Type["UnstructuredHistory"]:
        return UnstructuredHistory

    @staticmethod
    def structured_history_type() -> Type["Unstructured2StructuredHistory"]:
        return Unstructured2StructuredHistory


class AbstractHistory(ABC):
    file_path: str
    time_manager: DateTimeManager
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    nc_float_type: str
    nc_int_type: str
    _built: bool

    @abstractmethod
    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> None:
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
        ds = nc4.Dataset(self.file_path, mode="w", format="NETCDF4")
        ds.sync()
        ds.close()
        self._built = False

    @abstractmethod
    def setup(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    def end_setup(self):
        self._built = True

    def has_built(self) -> bool:
        return self._built

    def open_dataset(self) -> nc4.Dataset:
        return nc4.Dataset(self.file_path, mode="r")

    def write_fields(
        self,
        fields: AbstractFields,
        expand_idx: Optional[Int[Array, ""] | int] = None,
        tracer_idx: Optional[Int[Array, "ntracers"]] = None,
        **kwargs
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        for field_name in fields.keys():
            field = getattr(fields, field_name)
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if expand_idx is None:
                data = np.asarray(field.unwrap(), dtype=self.float_type)
            else:
                data = np.asarray(
                    field.unwrap()[expand_idx, ...],
                    dtype=self.float_type
                )
            if field.meta.is_static:
                ds.variables[field.meta.name][...] = data
            if not getattr(field.meta, "is_tracer", False):
                ds.variables[field.meta.name][idx, ...] = data
            else:
                if hasattr(fields, "tracers_meta"):
                    tracers_meta: dict[str, AbstractTracerMetaData] = getattr(
                        fields, "tracers_meta"
                    )
                else:
                    raise ValueError(
                        "tracers_meta is required for tracer fields."
                    )
                if tracer_idx is None:
                    raise ValueError(
                        "tracer_idx is required for tracer fields."
                    )
                for q_idx, tracer_name in enumerate(tracers_meta.keys()):
                    if q_idx in tracer_idx:
                        ds.variables[tracer_name][idx, ...] = data[q_idx, ...]
        ds.close()
        return None

    @abstractmethod
    def add_fields(self, fields: AbstractFields) -> None:
        raise NotImplementedError


class UGRIDHistory(AbstractHistory):
    use_dual: bool
    attr: GlobalAttributesSetup
    time: TimeSetup
    mesh: AbstractMeshHistorySetup
    vert: AbstractVertHistorySetup

    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        use_dual: bool = False,
    ) -> None:
        super().__init__(
            solver=solver,
            coord=coord,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.use_dual = use_dual
        self.attr = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=solver.equation.config2ncattrs()
        )
        self.time = TimeSetup(
            file_path=self.file_path,
            start_time_str=solver.time_manager.start_time_str,
            save_interval_unit=solver.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh = coord.mesh.history_builder(
            file_path=self.file_path,
            output_type="ugrid",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert = coord.vert.history_builder(
            file_path=self.file_path,
            output_type="mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )

    def setup(self, fields: AbstractFields) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup(use_dual=self.use_dual)
        if not self.vert.has_built():
            self.vert.setup()
        if (
            self.attr.has_built()
            and self.time.has_built()
            and self.mesh.has_built()
            and self.vert.has_built()
            and not self._built
        ):
            self.add_fields(fields)
        return self

    def add_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if isinstance(field.meta, FieldMetaData):
                dsxr = tuxr.format_terraflow2ugrid(
                    field.data, field.meta.to_dict(), use_dual=self.use_dual, expand_idx=0,
                )
                var_dims = tuple(str(dim) for dim in dsxr.dims)
                if not field.meta.is_static:
                    var_dims = tuple(("time",) + var_dims)
                if not field.meta.is_tracer:
                    v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                    v.mesh = dsxr.attrs.get("mesh", "")
                    v.location = dsxr.attrs.get("location", "")
                    v.coordinates = dsxr.attrs.get("coordinates", "")
                    v.units = dsxr.attrs.get("units", "")
                    v.long_name = dsxr.attrs.get("long_name", "")
                    v.description = dsxr.attrs.get("description", "")
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, AbstractTracerMetaData] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            dsxr.attrs.get("long_name", "")
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_description = (
                            dsxr.attrs.get("description", "")
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = dsxr.attrs.get("units", "")
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.mesh = dsxr.attrs.get("mesh", "")
                        v.location = dsxr.attrs.get("location", "")
                        v.coordinates = dsxr.attrs.get("coordinates", "")
                        v.units = q_units
                        v.long_name = q_long_name
                        v.description = q_description
            elif isinstance(field.meta, NumericMetaData):
                var_dims = ("time",) if not field.meta.is_static else ()
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.units = field.meta.units
                v.long_name = field.meta.long_name
                v.description = field.meta.description
            else:
                raise ValueError(
                    f"Unsupported field meta data type: {type(field.meta)}"
                )
        ds.close()
        return None


class UnstructuredHistory(AbstractHistory):
    file_path: str
    time_manager: DateTimeManager
    attr: GlobalAttributesSetup
    time: TimeSetup
    mesh: AbstractMeshHistorySetup
    vert: AbstractVertHistorySetup
    _built: bool

    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        history_name: str,
        case_name: str,
        file_path: str,
        with_vtx: bool,
        with_edge: bool,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
    ) -> None:
        super().__init__(
            solver=solver,
            coord=coord,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.attr = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=solver.equation.config2ncattrs()
        )
        self.time = TimeSetup(
            file_path=self.file_path,
            start_time_str=solver.time_manager.start_time_str,
            save_interval_unit=solver.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh = coord.mesh.history_builder(
            file_path=self.file_path,
            output_type="unstructured",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert = coord.vert.history_builder(
            file_path=self.file_path,
            output_type="mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.with_vtx = with_vtx
        self.with_edge = with_edge

    def setup(self, fields: AbstractFields) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup(
                with_vtx=self.with_vtx,
                with_edge=self.with_edge
            )
        if not self.vert.has_built():
            self.vert.setup()
        if (
            self.attr.has_built()
            and self.time.has_built()
            and self.mesh.has_built()
            and self.vert.has_built()
            and not self._built
        ):
            self.add_fields(fields)
        return self

    def add_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if isinstance(field.meta, FieldMetaData):
                var_dims = field.meta.dims
                if not field.meta.is_static:
                    var_dims = tuple(("time",) + field.meta.dims)
                if not field.meta.is_tracer:
                    v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                    v.hori_location = field.meta.hori_location
                    v.vert_location = field.meta.vert_location
                    v.coordinates = field.meta.coordinates
                    v.units = field.meta.units
                    v.long_name = field.meta.long_name
                    v.description = field.meta.description
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, AbstractTracerMetaData] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            field.meta.long_name
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_description = (
                            field.meta.description
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = field.meta.units
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.hori_location = field.meta.hori_location
                        v.vert_location = field.meta.vert_location
                        v.coordinates = field.meta.coordinates
                        v.units = q_units
                        v.long_name = q_long_name
                        v.description = q_description
            elif isinstance(field.meta, NumericMetaData):
                var_dims = ("time",) if not field.meta.is_static else ()
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.units = field.meta.units
                v.long_name = field.meta.long_name
                v.description = field.meta.description
            else:
                raise ValueError(
                    f"Unsupported field meta data type: {type(field.meta)}"
                )
        ds.close()
        return None


class Unstructured2StructuredHistory(AbstractHistory):
    file_path: str
    time_manager: DateTimeManager
    attr: GlobalAttributesSetup
    time: TimeSetup
    mesh: AbstractMeshHistorySetup
    vert: AbstractVertHistorySetup
    coord: CoordinateSystem
    structured_lon: np.ndarray
    structured_lat: np.ndarray
    structured_lev: np.ndarray
    hori_regrid_method: str
    hori_regrid_backend: str
    vert_interp: bool
    _regridders: dict[tuple[str, Optional[int]], Regridder]
    _built: bool

    def __init__(
        self,
        solver: AbstractSolver,
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        hori_regrid_method: str = "conserve_1st",
        hori_regrid_backend: str = "esmf",
        vert_interp: bool = True,
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
        lon: Optional[np.ndarray | Array] = None,
        lat: Optional[np.ndarray | Array] = None,
        plev: np.ndarray | Array = np.array([10, 20, 50, 100, 200, 300, 500, 700, 850, 925, 1000]),
        plev_unit: str = "hPa"
    ) -> None:
        super().__init__(
            solver=solver,
            coord=coord,
            history_name=history_name,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.attr = GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type,
            attrs=solver.equation.config2ncattrs()
        )
        self.time = TimeSetup(
            file_path=self.file_path,
            start_time_str=solver.time_manager.start_time_str,
            save_interval_unit=solver.save_interval_unit,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.mesh = coord.mesh.history_builder(
            file_path=self.file_path,
            output_type="structured",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.vert = coord.vert.history_builder(
            file_path=self.file_path,
            output_type="plev" if vert_interp else "mlev",
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.coord = coord
        self.hori_regrid_method = hori_regrid_method
        self.hori_regrid_backend = hori_regrid_backend
        self.vert_interp = vert_interp

        if lon is not None and lat is not None:
            self.structured_lon = np.array(lon)
            self.structured_lat = np.array(lat)
            self.structured_lev = np.array(plev)
        else:
            self.structured_lon = np.arange(
                lon_west, lon_east, east_west_resolution
            )
            self.structured_lat = np.arange(
                lat_south, lat_north, south_north_resolution
            )
            if plev_unit.lower() == "hpa" or plev_unit.lower() == "mb":
                self.structured_lev = np.array(plev, dtype=np.float64) * 100.0
            elif plev_unit.lower() == "pa":
                self.structured_lev = np.array(plev, dtype=np.float64)
            else:
                raise ValueError(
                    f"Unsupported plev_unit: {plev_unit}, expected 'hPa', 'Pa', or 'mb'."
                )
            self.structured_lev = np.sort(self.structured_lev)

        self._regridders = {}

    def setup(self, fields: AbstractFields) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup(
                dst_lon=self.structured_lon,
                dst_lat=self.structured_lat,
            )
        if not self.vert.has_built():
            self.vert.setup(
                lev=self.structured_lev,
                lev_unit="Pa",
            )
        if (
            self.attr.has_built()
            and self.time.has_built()
            and self.mesh.has_built()
            and self.vert.has_built()
            and not self._built
        ):
            self.add_fields(fields)
        return self

    def add_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            meta = field.meta
            if meta.output not in STRUCTURED_HISTORY_LIST:
                continue
            if meta.output == "":
                continue
            if meta.name in PLEV_NAMES and self.vert_interp:
                continue

            if isinstance(meta, StaggeredFieldMetaData):
                var_dims = scvt2structured_var_dims(meta, self.vert_interp)
                if not meta.is_static:
                    var_dims = tuple(("time",) + var_dims)

                if not meta.is_tracer:
                    v = ds.createVariable(
                        str(meta.name), var_dtype, var_dims
                    )
                    v.setncattr("mesh", meta.mesh_name)
                    v.setncattr("units", meta.units)
                    v.setncattr("long_name", meta.long_name)
                    v.setncattr("description", meta.description)
                    v.setncattr("coordinates", "lon lat")
                else:
                    if hasattr(fields, "tracers_meta"):
                        tracers_meta: dict[str, AbstractTracerMetaData] = getattr(
                            fields, "tracers_meta"
                        )
                    else:
                        raise ValueError(
                            "tracers_meta is required for tracer fields."
                        )
                    for tracer_name, tracer_meta in tracers_meta.items():
                        q_name = tracer_name
                        q_long_name = (
                            meta.long_name
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_description = (
                            meta.description
                            + " of "
                            + tracer_meta.long_name.lower()
                        )
                        q_units = meta.units
                        v = ds.createVariable(q_name, var_dtype, var_dims)
                        v.setncattr("mesh", meta.mesh_name)
                        v.setncattr("coordinates", "lon lat")
                        v.setncattr("units", q_units)
                        v.setncattr("long_name", q_long_name)
                        v.setncattr("description", q_description)
            elif isinstance(meta, NumericMetaData):
                is_static = meta.is_static
                var_dims = () if is_static else ("time",)
                v = ds.createVariable(
                    str(meta.name), var_dtype, var_dims
                )
                v.setncattr("units", meta.units)
                v.setncattr("long_name", meta.long_name)
                v.setncattr("description", meta.description)
            else:
                raise ValueError(
                    f"Unsupported meta data type for variable {meta.name}: {type(meta)}."
                )
            # Set regridder for field if applicable
            if isinstance(meta, FieldMetaData):
                ndbound = self._metadata_ndbound(field)
                self._get_regridder(meta=meta, ndbound=ndbound)
        ds.close()
        return None

    def write_fields(
        self,
        fields: AbstractFields,
        expand_idx: Optional[Int[Array, ""] | int] = None,
        tracer_idx: Optional[Int[Array, "ntracers"]] = None,
        ph: Optional[np.ndarray] = None,
        ph_lev: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        expand_idx = int(expand_idx) if expand_idx is not None else None
        var_dtype = self.float_type
        # Write remapped FieldMetaData fields
        for _, field in fields.items():
            meta = field.meta
            if not isinstance(meta, StaggeredFieldMetaData):
                continue
            if meta.output not in STRUCTURED_HISTORY_LIST or meta.output == "":
                continue
            if meta.name in PLEV_NAMES and self.vert_interp:
                continue

            if expand_idx is None:
                field_data = field.unwrap()
            else:
                field_data = field.unwrap()[expand_idx, ...]
            field_data = self._remap_field(
                field_data=field_data,
                meta=meta,
                ph=ph,
                ph_lev=ph_lev,
            )
            if meta.is_static:
                ds.variables[meta.name][...] = np.array(
                    field_data,
                    dtype=var_dtype
                )
            elif not meta.is_tracer:
                ds.variables[meta.name][idx, ...] = np.array(
                    field_data,
                    dtype=var_dtype
                )
            else:
                if hasattr(fields, "tracers_meta"):
                    tracers_meta: dict[str, AbstractTracerMetaData] = getattr(
                        fields, "tracers_meta"
                    )
                else:
                    raise ValueError(
                        "tracers_meta is required for tracer fields."
                    )
                if tracer_idx is None:
                    raise ValueError(
                        "tracer_idx is required for tracer fields."
                    )
                tracer_indices = set(
                    np.asarray(tracer_idx).reshape(-1).astype(int).tolist()
                )
                for q_idx, tracer_name in enumerate(tracers_meta.keys()):
                    if q_idx in tracer_indices:
                        ds.variables[tracer_name][idx, ...] = np.array(
                            field_data[q_idx, ...],
                            dtype=var_dtype
                        )
        # Write NumericMetaData fields
        for _, field in fields.items():
            if not isinstance(field.meta, NumericMetaData):
                continue
            if field.meta.output not in STRUCTURED_HISTORY_LIST or field.meta.output == "":
                continue
            if expand_idx is None:
                field_data = field.unwrap()
            else:
                field_data = field.unwrap()[expand_idx, ...]
            field_name = field.meta.name
            if field.meta.is_static:
                ds.variables[field_name][...] = np.array(
                    field_data, dtype=var_dtype
                )
            else:
                ds.variables[field_name][idx] = np.array(
                    field_data, dtype=var_dtype
                )
        ds.close()
        return None

    def _get_regridder(
        self,
        meta: StaggeredFieldMetaData,
        ndbound: Optional[int],
    ) -> Regridder:
        loc = meta.hori_location
        key = (loc, ndbound)
        if key not in self._regridders:
            lon, lat = self.structured_lon, self.structured_lat
            self._regridders[key] = Regridder(
                src=Regridder.create_scvt_info(
                    coord=self.coord,
                    loc=loc,
                ),
                dst=Regridder.create_latlon_info(
                    lon=jnp.array(lon),
                    lat=jnp.array(lat)
                ),
                backend=self.hori_regrid_backend,
                ndbounds=None if ndbound is None else (ndbound,),
                method=self.hori_regrid_method,
                extrap=True,
                esmf_conserve_opt={"norm_opt": "fracarea"},
                precompute_weights=True,
            )
        return self._regridders[key]

    def _metadata_ndbound(self, field: Field) -> Optional[int]:
        meta = field.meta
        data = field.unwrap()

        # For sfc or 2D variables, ndbound
        if meta.vert_location == "" or meta.vert_location == "sfc":
            ndbound = None
        # For 3D variables, ndbound is the product of leading dimensions (excluding spatial dimensions)
        else:
            dims = meta.dims
            ndbound = data.shape[-2] if len(dims) >= 2 else 1

        return ndbound

    def _horizontal_remap(
        self,
        field_data: Array,
        meta: StaggeredFieldMetaData,
        ndbound: Optional[int],
    ) -> np.ndarray:
        regridder = self._get_regridder(meta=meta, ndbound=ndbound)
        if meta.is_tracer:
            return np.stack(
                [
                    np.array(
                        regridder.regrid(field_data[q_idx, ...]),
                        copy=True,
                    )
                    for q_idx in range(field_data.shape[0])
                ]
            )
        return np.array(regridder.regrid(field_data), copy=True)

    def _remap_field(
        self,
        field_data: Array,
        meta: StaggeredFieldMetaData,
        ph: Optional[np.ndarray] = None,
        ph_lev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if meta.vert_location in ("", "sfc"):
            if meta.is_tracer:
                if field_data.ndim == 3 and field_data.shape[1] == 1:
                    field_data = jnp.squeeze(field_data, axis=1)
                elif field_data.ndim != 2:
                    raise ValueError(
                        f"Tracer variable {meta.name} has shape {field_data.shape}, "
                        f"expected (ntracers, npoints) for {meta.vert_location!r}."
                    )
            else:
                if field_data.ndim == 2 and field_data.shape[0] == 1:
                    field_data = jnp.squeeze(field_data, axis=0)
                elif field_data.ndim != 1:
                    raise ValueError(
                        f"Variable {meta.name} has shape {field_data.shape}, "
                        f"expected (npoints,) for {meta.vert_location!r}."
                    )
            ndbound = None
        else:
            # (ntracer, nlev, npoints)
            if meta.is_tracer:
                if field_data.ndim != 3:
                    raise ValueError(
                        f"Variable {meta.name} is a tracer with vert_location "
                        f"{meta.vert_location}, but has shape {field_data.shape}, "
                        f"which does not match metadata dimensions {meta.dims}."
                    )
                ndbound = field_data.shape[1]
            # (nlev, npoints)
            else:
                ndbound = field_data.shape[0]
        # =================================================
        # Horizontal regridding
        if meta.name in PLEV_NAMES:
            if ph is None or ph_lev is None:
                raise ValueError(
                    f"Pressure fields ph and ph_lev are required for variable {meta.name}."
                )
            else:
                if meta.name == "ph":
                    return ph
                elif meta.name == "ph_lev":
                    return ph_lev
                else:
                    raise ValueError(
                        f"Unsupported pressure variable {meta.name}."
                    )
        hori_remapped = self._horizontal_remap(
            field_data=field_data,
            meta=meta,
            ndbound=ndbound,
        )

        if meta.vert_location == "" or meta.vert_location == "sfc" or not self.vert_interp:
            return hori_remapped
        # =================================================
        if meta.vert_location == "cell":
            pi = ph
        elif meta.vert_location == "lev":
            pi = ph_lev
        else:
            raise ValueError(
                f"Unsupported vert_location {meta.vert_location} for variable {meta.name}."
            )
        # Vertical interpolation if applicable
        if meta.is_tracer:
            vert_remapped = np.stack(
                [
                    np.asarray(
                        interp_plev(
                            coord=self.coord,
                            pi=jnp.asarray(pi),
                            xi=jnp.asarray(hori_remapped[q_idx, ...]),
                            po=jnp.asarray(self.structured_lev),
                            xo_vert_loc=meta.vert_location,
                            logp=meta.name in VARNAMES_USING_LOG_INTERP,
                        )
                    )
                    for q_idx in range(hori_remapped.shape[0])
                ]
            )
            return vert_remapped
        vert_remapped = interp_plev(
            coord=self.coord,
            pi=jnp.asarray(pi),
            xi=jnp.asarray(hori_remapped),
            po=jnp.asarray(self.structured_lev),
            xo_vert_loc=meta.vert_location,
            logp=meta.name in VARNAMES_USING_LOG_INTERP,
        )
        return np.asarray(vert_remapped)

    def update_plev(
        self,
        ph: Optional[Field],
        ph_lev: Optional[Field],
        itime: Int[Array, ""] | int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if ph is None or ph_lev is None:
            return None, None
        ph_data = ph.unwrap()[itime, ...]
        ph_lev_data = ph_lev.unwrap()[itime, ...]
        ph_remapped = self._horizontal_remap(
            field_data=ph_data,
            meta=ph.meta,
            ndbound=ph_data.shape[0] if ph_data.ndim >= 2 else None
        )
        ph_lev_remapped = self._horizontal_remap(
            field_data=ph_lev_data,
            meta=ph_lev.meta,
            ndbound=ph_lev_data.shape[0] if ph_lev_data.ndim >= 2 else None
        )
        return ph_remapped, ph_lev_remapped


def scvt2structured_var_dims(meta: FieldMetaData, vert_interp: bool) -> tuple[str, ...]:
    # For sfc variable -> (lat, lon)
    # For variables with vertical dimension -> (lev, lat, lon)
    # For tracer 3D variables -> (ntracers, lev, lat, lon)
    # For tracer 2D variables -> (ntracers, lat, lon)
    dims = tuple(str(dim) for dim in meta.dims)

    # For sfc variable
    if len(dims) == 1:
        leading_dims = ()
    # For variables >= 2D
    elif len(dims) >= 2:
        # For variables with vertical dimension
        if vert_interp:
            if dims[-2] in LEV_DIMS:
                dims = dims[:-2] + ("lev",) + dims[-1:]

    # Process leading dimensions for spatial variables
    if len(dims) > 0 and (dims[-1] in UGRID_SPATIAL_DIMS or dims[-1] in UNSTRUCTURED_SPATIAL_DIMS):
        leading_dims = dims[:-1]
    else:
        leading_dims = dims

    # Process leading dimensions for tracer variables
    if meta.is_tracer and leading_dims[:1] == ("ntracers",):
        leading_dims = leading_dims[1:]
    return tuple(leading_dims + ("lat", "lon"))
