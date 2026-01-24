from abc import ABC, abstractmethod
from typing import Optional
import netCDF4 as nc4
import datetime


class AbstractHistorySetup(ABC):
    file_path: str
    _built: bool

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._built = False

    def has_built(self) -> bool:
        return self._built

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAttributesSetup(AbstractHistorySetup):
    file_path: str
    title: str
    source: str
    history: str
    Conventions: str
    institution: str
    _built: bool

    def __init__(
        self,
        file_path: str,
        title: str = "TerraFlow Structured Grid History File",
        source: str = "TerraFlow",
        history: Optional[str] = None,
        Conventions: str = "CF-1.8",
        institution: str = "TerraFlow Development Project",
    ):
        super().__init__(file_path=file_path)
        self.title = title
        self.source = source
        if history is None:
            history = f"Created on {datetime.datetime.now().isoformat()}"
        self.history = history
        self.Conventions = Conventions
        self.institution = institution

    def setup(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, 'a')
        ds.title = self.title
        ds.source = self.source
        ds.history = f"Created on {datetime.datetime.now().isoformat()}"
        ds.Conventions = self.Conventions
        ds.institution = self.institution
        ds.close()
        self._built = True


class TimeSetup(AbstractHistorySetup):
    file_path: str
    save_interval_unit: str
    start_time_str: str
    _built: bool

    def __init__(self, file_path: str, start_time_str: str, save_interval_unit: str):
        super().__init__(file_path=file_path)
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
