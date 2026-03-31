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
from loguru import logger
import netCDF4 as nc4
import numpy as np
import jax.numpy as jnp
import jax.experimental
import xarray as xr

from terraflow.util import uxarray as tuxr
from terraflow.util.device import AbstractDevice
import terraflow.util.history.setup as st
from terraflow.util.datetime import DateTimeManager
from terraflow.mesh.abstract import AbstractMesh, AbstractUnstructuredMesh
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.layout.abstract import AbstractLayout, CoordinateSystem
from terraflow.layout.field import FieldMetaData, NumericMetaData

if TYPE_CHECKING:
    from terraflow.discretization.abstract import (
        AbstractFields,
        AbstractBlock,
    )

UM = TypeVar("UM", bound=AbstractUnstructuredMesh)
HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)
VLO = TypeVar("VLO", bound=AbstractLayout)
HY = TypeVar("HY", bound='AbstractHistory')
SHY = TypeVar("SHY", bound='AbstractHistory')


STRUCTURED_HISTORY_LIST = ["h0", "h1", "h2"]


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


class AbstractHistoryset(ABC, Generic[HY, SHY]):
    h0: HY
    h1: HY
    h2: SHY
    output_h0: bool
    output_h1: bool
    output_h2: bool
    time_manager: DateTimeManager
    case_name: str
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    _built: bool

    @abstractmethod
    def __init__(
        self,
        time_manager: DateTimeManager,
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
        self.time_manager = time_manager
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
            self.float_type = np.float64
        elif float_type in FLOAT_JNP2NP.values():
            self.float_type = float_type
        else:
            raise ValueError(f"Unsupported float type: {float_type}")
        if int_type is None:
            self.int_type = np.int64
        elif int_type in INT_JNP2NP.values():
            self.int_type = int_type
        else:
            raise ValueError(f"Unsupported int type: {int_type}")
        if self.output_h0:
            self.h0 = self.history_type()(
                time_manager=time_manager,
                history_name="h0",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h0_opt,
                **kwargs,
            )
            logger.info(f"History will output h0: {self.h0.file_path}")
        if self.output_h1:
            self.h1 = self.history_type()(
                time_manager=time_manager,
                history_name="h1",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h1_opt,
                **kwargs,
            )
            logger.info(f"History will output h1: {self.h1.file_path}")
        if self.output_h2:
            self.h2 = self.structured_history_type()(
                time_manager=time_manager,
                history_name="h2",
                case_name=case_name,
                file_path=file_path,
                float_type=self.float_type,
                int_type=self.int_type,
                **h2_opt,
                **kwargs,
            )
            logger.info(f"History will output h2: {self.h2.file_path}")

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

    def output_prepare(self, save_idx: int) -> None:
        jax.experimental.io_callback(
            self.prepare_for_advance,
            None,
            save_idx,
            ordered=True
        )

    def output(
        self,
        block: AbstractBlock,
        itime: int,
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
        jax.experimental.io_callback(
            self.advance_forward_in_time,
            None,
            block,
            itime,
            save_idx,
            first_call,
            ordered=True
        )

    def prepare_for_advance(self, save_interval_idx: int) -> None:
        logger.info(f"==> {self.time_manager.curr_time(save_interval_idx)}")
        return None

    def log_done_write(self, history: str = "history") -> None:
        logger.success(f"Done write {history} file.")
        return None

    def advance_forward_in_time(
        self,
        block: AbstractBlock,
        itime: int,
        save_interval_idx: int,
        first_call: bool = False
    ) -> None:

        def _set_time_var(h: HY | SHY):
            ds = nc4.Dataset(h.file_path, mode="a")
            time_var = ds.variables["time"]
            current_length = len(time_var)
            interval = save_interval_idx * h.time_manager.save_interval_value
            time_var[current_length] = interval
            ds.close()
        if self.output_h0:
            _set_time_var(self.h0)
            self.write_h0(block, itime=itime, first_call=first_call)
            self.log_done_write("h0")
        if self.output_h1:
            _set_time_var(self.h1)
            self.write_h1(block, itime=itime)
            self.log_done_write("h1")
        if self.output_h2:
            _set_time_var(self.h2)
            self.write_h2(block, itime=itime)
            self.log_done_write("h2")
        return None

    def setup_h0(self, block: AbstractBlock) -> None:
        if self.output_h0:
            self.h0.setup([block.state[0], block.static, block.aux])

    def setup_h1(self, block: AbstractBlock) -> None:
        if self.output_h1:
            self.h1.setup(
                [
                    block.state[0],
                    block.static,
                    block.tend,
                    block.aux,
                    block.adv_batch_pt
                ]
            )

    def setup_h2(self, block: AbstractBlock) -> None:
        if self.output_h2:
            h2_list = []
            if self.output_h0 and not self.output_h1:
                h2_list = h2_list + [block.state, block.static, block.aux]
            if self.output_h0 and self.output_h1:
                h2_list = h2_list + [
                    block.state[0],
                    block.tend,
                    block.static,
                    block.aux,
                    block.adv_batch_pt
                ]
            if not self.output_h0 and self.output_h1:
                h2_list = h2_list + [
                    block.state[0],
                    block.tend,
                    block.static,
                    block.aux,
                    block.adv_batch_pt
                ]
            if not self.output_h0 and not self.output_h1:
                h2_list = []
            if h2_list:
                self.h2.setup(h2_list)
            else:
                raise ValueError(
                    "output_h0 and output_h1 are both False, cannot setup h2."
                )

    def write_h0(self, block: AbstractBlock, itime: int, first_call: bool = False):
        if first_call:
            self.h0.write_fields(block.static)
        self.h0.write_fields(block.get_state(itime))
        self.h0.write_fields(block.aux)

    def write_h1(self, block: AbstractBlock, itime: int):
        self.h1.write_fields(block.get_state(itime))
        self.h1.write_fields(block.tend)
        self.h1.write_fields(block.static)
        self.h1.write_fields(block.aux)
        self.h1.write_fields(block.adv_batch_pt)

    def write_h2(self, block: AbstractBlock, itime: int):
        if self.output_h0 and not self.output_h1:
            self.h2.write_fields(block.get_state(itime))
            self.h2.write_fields(block.static)
            self.h2.write_fields(block.aux)
        if self.output_h0 and self.output_h1:
            self.h2.write_fields(block.get_state(itime))
            self.h2.write_fields(block.tend)
            self.h2.write_fields(block.static)
            self.h2.write_fields(block.aux)
            self.h2.write_fields(block.adv_batch_pt)
        if not self.output_h0 and self.output_h1:
            self.h2.write_fields(block.get_state(itime))
            self.h2.write_fields(block.tend)
            self.h2.write_fields(block.static)
            self.h2.write_fields(block.aux)
            self.h2.write_fields(block.adv_batch_pt)
        if not self.output_h0 and not self.output_h1:
            raise ValueError(
                "output_h0 and output_h1 are both False, cannot write h2."
            )


class UGRIDHistoryset(
    AbstractHistoryset["UGRIDHistory", "Unstructured2StructuredHistory"]
):

    def __init__(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        time_manager: DateTimeManager,
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
            coord=coord,
            time_manager=time_manager,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"use_dual": use_dual},
            h1_opt={"use_dual": use_dual},
            h2_opt={},
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
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        time_manager: DateTimeManager,
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
        super().__init__(
            coord=coord,
            time_manager=time_manager,
            case_name=case_name,
            file_path=file_path,
            output_h0=output_h0,
            output_h1=output_h1,
            output_h2=output_h2,
            h0_opt={"with_vtx": True},
            h1_opt={"with_vtx": True},
            h2_opt={},
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
        time_manager: DateTimeManager,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ) -> None:
        self.time_manager = time_manager
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

    def has_built(self) -> bool:
        return self._built

    def open_dataset(self) -> nc4.Dataset:
        return nc4.Dataset(self.file_path, mode="r")

    def write_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        for field_name in fields.keys():
            field = getattr(fields, field_name)
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            data = np.asarray(field.unwrap(), dtype=self.float_type)
            if field.meta.is_static:
                ds.variables[field.meta.name][...] = data
            else:
                ds.variables[field.meta.name][idx, ...] = data
        ds.close()
        return None

    @abstractmethod
    def add_fields(self, fields: AbstractFields) -> None:
        raise NotImplementedError


class UGRIDHistory(AbstractHistory):
    use_dual: bool
    attr: st.base.GlobalAttributesSetup
    time: st.base.TimeSetup
    mesh: st.mesh.AbstractMeshHistorySetup
    vert: st.vert.AbstractVertHistorySetup

    def __init__(
        self,
        coord: CoordinateSystem[HM, VM, HLO, VLO],
        time_manager: DateTimeManager,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        use_dual: bool = False,
    ) -> None:
        super().__init__(
            history_name=history_name,
            time_manager=time_manager,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.use_dual = use_dual
        self.attr = st.base.GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.time = st.base.TimeSetup(
            file_path=self.file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
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
            float_type=self.float_type,
            int_type=self.int_type
        )

    def setup(self, fields_list: list[AbstractFields]) -> Self:
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
        if self.attr.has_built() and self.time.has_built() and \
           self.mesh.has_built() and self.vert.has_built():
            for fields in fields_list:
                self.add_fields(fields)
            self._built = True
        return self

    def add_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            if field.meta.output != self.history_name or field.meta.output == "":
                continue
            if isinstance(field.meta, FieldMetaData):
                dsxr = tuxr.format_terraflow2ugrid(
                    field.data, field.meta.to_dict(), use_dual=self.use_dual
                )
                var_dims = tuple(str(dim) for dim in dsxr.dims)
                if not field.meta.is_static:
                    var_dims = tuple(("time",) + var_dims)
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.mesh = dsxr.attrs.get("mesh", "")
                v.location = dsxr.attrs.get("location", "")
                v.coordinates = dsxr.attrs.get("coordinates", "")
                v.units = dsxr.attrs.get("units", "")
                v.long_name = dsxr.attrs.get("long_name", "")
                v.description = dsxr.attrs.get("description", "")
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
    attr: st.base.GlobalAttributesSetup
    time: st.base.TimeSetup
    mesh: st.mesh.AbstractMeshHistorySetup
    vert: st.vert.AbstractVertHistorySetup
    _built: bool

    def __init__(
        self,
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        time_manager: DateTimeManager,
        history_name: str,
        case_name: str,
        file_path: str,
        with_vtx: bool,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
    ) -> None:
        super().__init__(
            history_name=history_name,
            time_manager=time_manager,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.attr = st.base.GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.time = st.base.TimeSetup(
            file_path=self.file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
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
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.with_vtx = with_vtx

    def setup(self, fields_list: list[AbstractFields]) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup(with_vtx=self.with_vtx)
        if not self.vert.has_built():
            self.vert.setup()
        if self.attr.has_built() and self.time.has_built() and \
           self.mesh.has_built() and self.vert.has_built():
            for fields in fields_list:
                self.add_fields(fields)
            self._built = True
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
                v = ds.createVariable(field.meta.name, var_dtype, var_dims)
                v.hori_location = field.meta.hori_location
                v.vert_location = field.meta.vert_location
                v.coordinates = field.meta.coordinates
                v.units = field.meta.units
                v.long_name = field.meta.long_name
                v.description = field.meta.description
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
    attr: st.base.GlobalAttributesSetup
    time: st.base.TimeSetup
    mesh: st.mesh.AbstractMeshHistorySetup
    vert: st.vert.AbstractVertHistorySetup
    coord: CoordinateSystem
    remap_method: str
    remap_to: str
    lon_west: float
    lon_east: float
    lat_south: float
    lat_north: float
    east_west_resolution: float
    south_north_resolution: float
    _built: bool

    def __init__(
        self,
        coord: CoordinateSystem[UM, VM, HLO, VLO],
        time_manager: DateTimeManager,
        history_name: str,
        case_name: str,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        remap_method: str = "nearest_neighbor",
        remap_to: str = "faces",
        lon_west: float = -180,
        lon_east: float = 180,
        lat_south: float = -90,
        lat_north: float = 90,
        east_west_resolution: float = 1.0,
        south_north_resolution: float = 1.0,
    ) -> None:
        super().__init__(
            history_name=history_name,
            time_manager=time_manager,
            case_name=case_name,
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.attr = st.base.GlobalAttributesSetup(
            file_path=self.file_path,
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.time = st.base.TimeSetup(
            file_path=self.file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
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
            float_type=self.float_type,
            int_type=self.int_type
        )
        self.coord = coord
        self.remap_method = remap_method
        self.remap_to = remap_to
        self.lon_west = lon_west
        self.lon_east = lon_east
        self.lat_south = lat_south
        self.lat_north = lat_north
        self.east_west_resolution = east_west_resolution
        self.south_north_resolution = south_north_resolution

    def setup(self, fields_list: list[AbstractFields]) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup(
                remap_to=self.remap_to,
                lon_west=self.lon_west,
                lon_east=self.lon_east,
                lat_south=self.lat_south,
                lat_north=self.lat_north,
                east_west_resolution=self.east_west_resolution,
                south_north_resolution=self.south_north_resolution,
            )
        if not self.vert.has_built():
            self.vert.setup()
        if self.attr.has_built() and self.time.has_built() and \
           self.mesh.has_built() and self.vert.has_built():
            for fields in fields_list:
                self.add_fields(fields)
            self._built = True
        return self

    def write_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        staggered_var_ds = {}
        for _, field in fields.items():
            if not isinstance(field.meta, FieldMetaData):
                continue
            staggered_var_ds[field.meta.name] = tuxr.format_terraflow2ugrid(
                value=field.data,
                meta=field.meta.to_dict(),
            )
        if staggered_var_ds:
            staggered_ds = xr.Dataset(data_vars=staggered_var_ds)
            uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
            _, fields_remapped = tuxr.UGRIDDataset.from_xarray(
                ds=staggered_ds,
                uxgrid=uxgrid
            ).to_structured_grid(
                remap_method=self.remap_method,
                remap_to=self.remap_to,
                lon_west=self.lon_west,
                lon_east=self.lon_east,
                lat_south=self.lat_south,
                lat_north=self.lat_north,
                east_west_resolution=self.east_west_resolution,
                south_north_resolution=self.south_north_resolution,
                with_reshaped_ds=True
            )
        else:
            fields_remapped = xr.Dataset()
        var_dtype = self.float_type
        for field_name in fields_remapped.keys():
            field = fields_remapped.variables[field_name]
            output = field.attrs.get("output", None)
            if output not in STRUCTURED_HISTORY_LIST or output == "":
                continue
            data = fields_remapped.variables[field_name].data
            if field.attrs.get("is_static", False):
                ds.variables[str(field_name)][...] = np.array(
                    data,
                    dtype=var_dtype
                )
            else:
                ds.variables[str(field_name)][idx, ...] = np.array(
                    data,
                    dtype=var_dtype
                )
        for _, field in fields.items():
            if not isinstance(field.meta, NumericMetaData):
                continue
            if field.meta.output not in STRUCTURED_HISTORY_LIST or field.meta.output == "":
                continue
            field_name = field.meta.name
            if field.meta.is_static:
                ds.variables[field_name][...] = np.array(
                    field.data, dtype=var_dtype)
            else:
                ds.variables[field_name][idx] = np.array(
                    field.data, dtype=var_dtype)
        ds.close()
        return None

    def add_fields(self, fields: AbstractFields) -> None:
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        staggered_var_ds = {}
        for _, field in fields.items():
            if not isinstance(field.meta, FieldMetaData):
                continue
            staggered_var_ds[field.meta.name] = tuxr.format_terraflow2ugrid(
                value=field.data,
                meta=field.meta.to_dict(),
            )
        if staggered_var_ds:
            staggered_ds = xr.Dataset(data_vars=staggered_var_ds)
            uxgrid = self.coord.mesh.to_ugrid().to_uxgrid()
            remapped_ds = tuxr.UGRIDDataset.from_xarray(
                ds=staggered_ds,
                uxgrid=uxgrid
            ).to_structured_grid(
                remap_method=self.remap_method,
                remap_to=self.remap_to,
                lon_west=self.lon_west,
                lon_east=self.lon_east,
                lat_south=self.lat_south,
                lat_north=self.lat_north,
                east_west_resolution=self.east_west_resolution,
                south_north_resolution=self.south_north_resolution,
                with_reshaped_ds=True
            )[1]
        else:
            remapped_ds = xr.Dataset()
        var_dtype = self.nc_float_type
        for _, field in fields.items():
            meta = field.meta
            if meta.output not in STRUCTURED_HISTORY_LIST or meta.output == "":
                logger.info(
                    f"Skip variable {meta.name} for history {self.history_name}."
                )
                continue
            if isinstance(meta, FieldMetaData):
                if meta.name not in remapped_ds.data_vars:
                    logger.info(
                        f"Skip variable {meta.name} for history {self.history_name}: remapped field not found."
                    )
                    continue
                da = remapped_ds.data_vars[meta.name]
                is_static = da.attrs.get("is_static", False)
                if len(da.dims) == 2:
                    var_dims = ("lat", "lon")
                elif len(da.dims) == 3:
                    var_dims = (str(da.dims[0]), "lat", "lon")
                else:
                    raise ValueError(
                        f"Variable {meta.name} has unsupported number of dimensions: {len(da.dims)}."
                    )
                if not is_static:
                    var_dims = tuple(("time",) + var_dims)
                v = ds.createVariable(
                    str(meta.name), var_dtype, var_dims
                )
                v.setncattr("mesh", da.attrs.get("mesh", ""))
                v.setncattr("location", da.attrs.get("location", ""))
                v.setncattr("units", da.attrs.get("units", ""))
                v.setncattr("long_name", da.attrs.get("long_name", ""))
                v.setncattr("description", da.attrs.get("description", ""))
                v.setncattr(
                    "coordinates",
                    " ".join([str(k) for k in da.coords.keys()])
                )
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
        ds.close()
        return None
