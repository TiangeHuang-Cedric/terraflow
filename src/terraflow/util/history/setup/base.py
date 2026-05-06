from abc import ABC, abstractmethod
from typing import Optional, Any
import netCDF4 as nc4
import datetime
import numpy as np

FLOAT_NP2NC = {
    np.float32: "f4",
    np.float64: "f8"
}

INT_NP2NC = {
    np.int32: "i4",
    np.int64: "i8"
}


class AbstractHistorySetup(ABC):
    file_path: str
    float_type: type[np.float32] | type[np.float64]
    int_type: type[np.int32] | type[np.int64]
    nc_float_type: str
    nc_int_type: str
    _built: bool

    def __init__(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        self.file_path = file_path
        self.float_type = float_type
        self.int_type = int_type
        self.nc_float_type = FLOAT_NP2NC[self.float_type]
        self.nc_int_type = INT_NP2NC[self.int_type]
        self._built = False

    def has_built(self) -> bool:
        return self._built

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAttributesSetup(AbstractHistorySetup):
    title: str
    source: str
    history: str
    Conventions: str
    institution: str
    attrs: dict[str, Any]

    def __init__(
        self,
        file_path: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
        title: str = "TerraFlow Structured Grid History File",
        source: str = "TerraFlow",
        history: Optional[str] = None,
        Conventions: str = "CF-1.8",
        institution: str = "TerraFlow Development Project",
        attrs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type,
        )
        self.title = title
        self.source = source
        if history is None:
            history = f"Created on {datetime.datetime.now().isoformat()}"
        self.history = history
        self.Conventions = Conventions
        self.institution = institution
        if attrs is not None:
            self.attrs = attrs
        else:
            self.attrs = {}

    def setup(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, 'a')
        ds.title = self.title
        ds.source = self.source
        ds.history = f"Created on {datetime.datetime.now().isoformat()}"
        ds.Conventions = self.Conventions
        ds.institution = self.institution
        for key, value in self.attrs.items():
            ds.setncattr(key, value)
        ds.close()
        self._built = True


class TimeSetup(AbstractHistorySetup):
    file_path: str
    save_interval_unit: str
    start_time_str: str
    _built: bool

    def __init__(
        self,
        file_path: str,
        start_time_str: str,
        save_interval_unit: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64],
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.save_interval_unit = save_interval_unit
        self.start_time_str = start_time_str

    def setup(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, 'a')
        ds.createDimension("time", None)
        time = ds.createVariable("time", "f8", ("time",))
        time.long_name = "Time"
        time.units = f"{self.save_interval_unit} since {self.start_time_str}"
        ds.close()
        self._built = True
