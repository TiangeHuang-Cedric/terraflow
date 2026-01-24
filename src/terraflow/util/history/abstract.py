from typing import TypeVar, Self, Optional
from abc import ABC, abstractmethod
import sys
from loguru import logger
import netCDF4 as nc4
import numpy as np

from terraflow.util import uxarray as tuxr
import terraflow.util.history.setup as st
from terraflow.util.datetime import DateTimeManager
from terraflow import mesh
from terraflow import vert_coord
from terraflow import layout
from terraflow.layout.abstract import CoordinateSystem
from terraflow.discretization.abstract import (
    DiagnosticVariableSet,
    AbstractArrayBasedVariableSet,
    AbstractFieldBasedVariableSet,
    AbstractQuantityBasedVariableSet,
    AbstractVariableSet,
    AbstractBlock,
    AbstractDiagnosticTools
)

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda r: r["level"].name == "INFO",
    format=("{time:HH:mm:ss} | "
            "<level>{level:<7}</level> | "
            "<blue>{message}</blue>"),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    sys.stderr,
    level="SUCCESS",
    filter=lambda r: r["level"].name == "SUCCESS",
    format=("<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<green>{message}</green>"),
    colorize=True,
    backtrace=False,
    diagnose=False,
)


UM = TypeVar("UM", bound=mesh.AbstractUnstructuredMesh)
HM = TypeVar("HM", bound=mesh.AbstractMesh)
VM = TypeVar("VM", bound=vert_coord.AbstractVertCoordinate)
LO = TypeVar("LO", bound=layout.AbstractLayout)


class AbstractHistory(ABC):
    file_path: str
    time_manager: DateTimeManager
    _built: bool
    diag_tool: Optional[AbstractDiagnosticTools]

    @abstractmethod
    def __init__(
        self,
        time_manager: DateTimeManager,
        file_path: str = "terraflow.nc",
        diag_tool: Optional[AbstractDiagnosticTools] = None,
    ) -> None:
        ds = nc4.Dataset(file_path, mode="w", format="NETCDF4")
        ds.sync()
        ds.close()
        self.file_path = file_path
        self.time_manager = time_manager
        self._built = False
        self.diag_tool = diag_tool

    @abstractmethod
    def setup(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    def has_diag_output(self) -> bool:
        return self.diag_tool is not None

    def has_built(self) -> bool:
        return self._built

    def open_dataset(self) -> nc4.Dataset:
        return nc4.Dataset(self.file_path, mode="r")

    def prepare_for_advance(self, save_interval_idx: int) -> None:
        logger.info(f"==> {self.time_manager.curr_time(save_interval_idx)}")
        return None

    def log_done_write(self) -> None:
        logger.success("Done write history file")
        return None

    def advance_forward_in_time(
        self,
        save_interval_idx: int,
        fields: AbstractVariableSet,
        diag: Optional[AbstractVariableSet] = None
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        time_var = ds.variables["time"]
        current_length = len(time_var)
        interval = save_interval_idx * self.time_manager.save_interval_value
        time_var[current_length] = interval
        ds.close()
        self.write_vars(fields)
        if diag is not None:
            self.write_vars(diag)
        self.log_done_write()
        return None

    def write_vars(self, fields: AbstractVariableSet) -> None:
        if isinstance(fields, AbstractFieldBasedVariableSet):
            self._write_fields(fields)
        elif isinstance(fields, AbstractArrayBasedVariableSet | DiagnosticVariableSet):
            self._write_arrays(fields)
        elif isinstance(fields, AbstractQuantityBasedVariableSet):
            self._write_quantities(fields)
        return None

    def _write_fields(
        self,
        fields: AbstractFieldBasedVariableSet
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        for field_name in fields.keys():
            data = getattr(fields, field_name).unwrap()
            data = np.asarray(data)
            ds.variables[field_name][idx, ...] = data
        ds.close()
        return None

    def _write_arrays(
        self,
        fields: AbstractArrayBasedVariableSet | DiagnosticVariableSet
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        for field_name in fields.keys():
            data = getattr(fields, field_name)
            data = np.asarray(data)
            ds.variables[field_name][idx, ...] = data
        ds.close()
        return None

    def _write_quantities(
        self,
        fields: AbstractQuantityBasedVariableSet
    ) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        for field_name in fields.keys():
            data = getattr(fields, field_name).value
            data = np.asarray(data)
            ds.variables[field_name][idx, ...] = data
        ds.close()
        return None

    def add_vars(
        self,
        fields: AbstractVariableSet
    ) -> None:
        if isinstance(fields, AbstractFieldBasedVariableSet):
            self._add_fields(fields)
        elif isinstance(fields, AbstractArrayBasedVariableSet | DiagnosticVariableSet):
            self._add_arrays(fields)
        elif isinstance(fields, AbstractQuantityBasedVariableSet):
            self._add_quantities(fields)
        return None

    @abstractmethod
    def _add_fields(self, fields: AbstractFieldBasedVariableSet) -> None:
        raise NotImplementedError

    @abstractmethod
    def _add_arrays(
        self,
        fields: AbstractArrayBasedVariableSet | DiagnosticVariableSet
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _add_quantities(self, fields: AbstractQuantityBasedVariableSet) -> None:
        raise NotImplementedError


class UGRIDHistory(AbstractHistory):
    file_path: str
    time_manager: DateTimeManager
    diag_tool: Optional[AbstractDiagnosticTools]
    use_dual: bool
    attr: st.base.GlobalAttributesSetup
    time: st.base.TimeSetup
    mesh: st.mesh.AbstractMeshHistorySetup
    vert: st.vert.AbstractVertHistorySetup
    _built: bool

    def __init__(
        self,
        coord: CoordinateSystem[HM, VM, LO],
        time_manager: DateTimeManager,
        file_path: str = "terraflow.nc",
        diag_tool: Optional[AbstractDiagnosticTools] = None,
        use_dual: bool = False,
    ) -> None:
        super().__init__(
            time_manager=time_manager,
            file_path=file_path,
            diag_tool=diag_tool
        )
        self.use_dual = use_dual
        self.attr = st.base.GlobalAttributesSetup(file_path)
        self.time = st.base.TimeSetup(
            file_path=file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
        )
        self.mesh = coord.mesh.history_builder(file_path, output_type="ugrid")
        self.vert = coord.vert.history_builder(file_path)

    def setup(self, block: AbstractBlock) -> Self:
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
            self.add_vars(block.state)
            self.add_vars(block.static)
            if self.diag_tool is not None:
                diag = self.diag_tool.to_diag(
                    state=block.state,
                    tend=block.tend,
                    aux=block.aux
                )
                self.add_vars(diag)
            self._built = True
        return self

    def _add_fields(self, fields: AbstractFieldBasedVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name, field in fields.items():
            dsxr = tuxr.format_terraflow2ugrid(
                field.data, field.meta.to_dict(), use_dual=self.use_dual
            )
            var_dims = tuple(("time",) + tuple(str(dim) for dim in dsxr.dims))
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.mesh = dsxr.attrs.get("mesh", "")
            v.location = dsxr.attrs.get("location", "")
            v.coordinates = dsxr.attrs.get("coordinates", "")
            v.units = dsxr.attrs.get("units", "")
            v.long_name = dsxr.attrs.get("long_name", "")
            v.description = dsxr.attrs.get("description", "")
        ds.close()
        return None

    def _add_arrays(self, fields: AbstractArrayBasedVariableSet | DiagnosticVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name, field in fields.items():
            meta = fields.meta[field_name]
            dsxr = tuxr.format_terraflow2ugrid(
                field, meta.to_dict(), use_dual=self.use_dual
            )
            var_dims = tuple(("time",) + tuple(str(dim) for dim in dsxr.dims))
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.mesh = dsxr.attrs.get("mesh", "")
            v.location = dsxr.attrs.get("location", "")
            v.coordinates = dsxr.attrs.get("coordinates", "")
            v.units = dsxr.attrs.get("units", "")
            v.long_name = dsxr.attrs.get("long_name", "")
            v.description = dsxr.attrs.get("description", "")
        ds.close()
        return None

    def _add_quantities(self, fields: AbstractQuantityBasedVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name, field in fields.items():
            meta = fields.meta[field_name]
            dsxr = tuxr.format_terraflow2ugrid(
                field.value, meta.to_dict(), use_dual=self.use_dual
            )
            var_dims = tuple(("time",) + tuple(str(dim) for dim in dsxr.dims))
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.mesh = dsxr.attrs.get("mesh", "")
            v.location = dsxr.attrs.get("location", "")
            v.coordinates = dsxr.attrs.get("coordinates", "")
            v.units = dsxr.attrs.get("units", "")
            v.long_name = dsxr.attrs.get("long_name", "")
            v.description = dsxr.attrs.get("description", "")
        ds.close()
        return None


class UnstructuredHistory(AbstractHistory):
    file_path: str
    time_manager: DateTimeManager
    diag: Optional[AbstractDiagnosticTools]
    attr: st.base.GlobalAttributesSetup
    time: st.base.TimeSetup
    mesh: st.mesh.AbstractMeshHistorySetup
    vert: st.vert.AbstractVertHistorySetup
    _built: bool

    def __init__(
        self,
        coord: CoordinateSystem[UM, VM, LO],
        time_manager: DateTimeManager,
        file_path: str = "terraflow.nc",
        diag_tool: Optional[AbstractDiagnosticTools] = None,
    ) -> None:
        super().__init__(
            time_manager=time_manager,
            file_path=file_path,
            diag_tool=diag_tool
        )
        self.attr = st.base.GlobalAttributesSetup(file_path)
        self.time = st.base.TimeSetup(
            file_path=file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
        )
        self.mesh = coord.mesh.history_builder(
            file_path, output_type="unstructured"
        )
        self.vert = coord.vert.history_builder(file_path)

    def setup(self, block: AbstractBlock) -> Self:
        if self.has_built():
            return self
        if not self.attr.has_built():
            self.attr.setup()
        if not self.time.has_built():
            self.time.setup()
        if not self.mesh.has_built():
            self.mesh.setup()
        if not self.vert.has_built():
            self.vert.setup()
        if self.attr.has_built() and self.time.has_built() and \
           self.mesh.has_built() and self.vert.has_built():
            self.add_vars(block.state)
            self.add_vars(block.static)
            if self.diag_tool is not None:
                diag = self.diag_tool.to_diag(
                    state=block.state,
                    tend=block.tend,
                    aux=block.aux
                )
                self.add_vars(diag)
            self._built = True
        return self

    def _add_fields(self, fields: AbstractVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name, field in fields.items():
            var_dims = tuple(("time",) + field.meta.dims)
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.hori_location = field.meta.hori_location
            v.vert_location = field.meta.vert_location
            v.coordinates = field.meta.coordinates
            v.units = field.meta.units
            v.long_name = field.meta.long_name
            v.description = field.meta.description
        ds.close()
        return None

    def _add_arrays(self, fields: AbstractArrayBasedVariableSet | DiagnosticVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name in fields.keys():
            meta = fields.meta[field_name]
            var_dims = tuple(("time",) + meta.dims)
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.hori_location = meta.hori_location
            v.vert_location = meta.vert_location
            v.coordinates = meta.coordinates
            v.units = meta.units
            v.long_name = meta.long_name
            v.description = meta.description
        ds.close()
        return None

    def _add_quantities(self, fields: AbstractQuantityBasedVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        for field_name in fields.keys():
            meta = fields.meta[field_name]
            var_dims = tuple(("time",) + meta.dims)
            var_dtype = "f8"
            v = ds.createVariable(field_name, var_dtype, var_dims)
            v.hori_location = meta.hori_location
            v.vert_location = meta.vert_location
            v.coordinates = meta.coordinates
            v.units = meta.units
            v.long_name = meta.long_name
            v.description = meta.description
        ds.close()
        return None


class Unstructured2StructuredHistory(AbstractHistory):
    file_path: str
    time_manager: DateTimeManager
    diag_tool: Optional[AbstractDiagnosticTools]
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
        coord: CoordinateSystem[UM, VM, LO],
        time_manager: DateTimeManager,
        file_path: str = "terraflow.nc",
        diag_tool: Optional[AbstractDiagnosticTools] = None,
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
            time_manager=time_manager,
            file_path=file_path,
            diag_tool=diag_tool
        )
        self.attr = st.base.GlobalAttributesSetup(file_path)
        self.time = st.base.TimeSetup(
            file_path=file_path,
            start_time_str=time_manager.start_time_str,
            save_interval_unit=time_manager.save_interval_unit,
        )
        self.mesh = coord.mesh.history_builder(
            file_path=file_path,
            output_type="structured"
        )
        self.vert = coord.vert.history_builder(file_path)
        self.coord = coord
        self.remap_method = remap_method
        self.remap_to = remap_to
        self.lon_west = lon_west
        self.lon_east = lon_east
        self.lat_south = lat_south
        self.lat_north = lat_north
        self.east_west_resolution = east_west_resolution
        self.south_north_resolution = south_north_resolution

    def setup(self, block: AbstractBlock) -> Self:
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
            self.add_vars(block.state)
            self.add_vars(block.static)
            if self.diag_tool is not None:
                diag = self.diag_tool.to_diag(
                    state=block.state,
                    tend=block.tend,
                    aux=block.aux
                )
                self.add_vars(diag)
            self._built = True
        return self

    def write_vars(self, fields: AbstractVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a")
        idx = len(ds.variables["time"]) - 1
        _, fields_remapped = fields.to_ugrid(self.coord).to_structured_grid(
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
        for field_name in fields_remapped.keys():
            data = fields_remapped.variables[field_name].data
            ds.variables[str(field_name)][idx, ...] = np.array(data)
        ds.close()
        return None

    def add_vars(self, fields: AbstractVariableSet) -> None:
        ds = nc4.Dataset(self.file_path, mode="a", format="NETCDF4")
        remapped_ds = fields.to_ugrid(self.coord).to_structured_grid(
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
        for var_name, da in remapped_ds.data_vars.items():
            var_dtype = "f8"
            v = ds.createVariable(
                str(var_name), var_dtype, ("time", "lat", "lon"))
            for attr_name, attr_value in da.attrs.items():
                v.setncattr(attr_name, attr_value)
            v.setncattr(
                "coordinates",
                " ".join([str(k) for k in da.coords.keys()])
            )
        ds.close()
        return None
