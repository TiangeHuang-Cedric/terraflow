from __future__ import annotations
from typing import TYPE_CHECKING
import netCDF4 as nc4

from terraflow.util.history.setup.base import AbstractHistorySetup

if TYPE_CHECKING:
    from terraflow.vert_coord.gtc.onelayer import OneLayerVertCoordinate
    from terraflow.vert_coord.gtc.hybrid_coord import HybridVertCoordinate
    import numpy as np


class AbstractVertHistorySetup(AbstractHistorySetup):
    pass


class OneLayerVerticalCoordinateSetup(AbstractVertHistorySetup):

    def __init__(
        self,
        file_path: str,
        vert: OneLayerVertCoordinate,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.vert = vert
        self.output_type = output_type

    def setup(self, **kwargs):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, "a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("lev", self.vert.full_nlev)
        ds.createDimension("ilev", self.vert.half_nlev)
        # Create variables
        ds.createVariable("lev", "f8", ("lev",))
        ds.createVariable("ilev", "f8", ("ilev",))
        # Assign data
        lev = ds.variables["lev"]
        lev[:] = self.vert.full_lev
        lev.long_name = "Full vertical level"
        lev.units = "Pa"
        lev.positive = "down"
        ilev = ds.variables["ilev"]
        ilev[:] = self.vert.half_lev
        ilev.long_name = "Half vertical level"
        ilev.units = "Pa"
        ilev.positive = "down"
        ds.close()
        self._built = True        


class HybridVerticalCoordinateSetup(AbstractVertHistorySetup):

    def __init__(
        self,
        file_path: str,
        vert: HybridVertCoordinate,
        output_type: str,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.vert = vert
        self.output_type = output_type

    def setup(self, **kwargs):
        if self.has_built():
            return
        if self.output_type == "mlev":
            self.mlev()
        elif self.output_type == "plev":
            self.plev(**kwargs)
        else:
            raise ValueError(f"Unsupported output_type: {self.output_type}")

    def mlev(self) -> None:
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, "a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("lev", self.vert.full_nlev)
        ds.createDimension("ilev", self.vert.half_nlev)
        # Create variables
        ds.createVariable("lev", "f8", ("lev",))
        ds.createVariable("ilev", "f8", ("ilev",))
        # Assign data
        lev = ds.variables["lev"]
        lev[:] = self.vert.full_lev
        lev.long_name = "Full vertical level"
        lev.units = "Pa"
        lev.positive = "down"
        ilev = ds.variables["ilev"]
        ilev[:] = self.vert.half_lev
        ilev.long_name = "Half vertical level"
        ilev.units = "Pa"
        ilev.positive = "down"
        ds.close()
        self._built = True

    def plev(self, lev: np.ndarray, lev_unit: str) -> None:
        if self.has_built():
            return
        if lev_unit not in ["Pa", "hPa", "mb"]:
            raise ValueError(f"Unsupported pressure level unit: {lev_unit}")
        if lev_unit == "hPa" or lev_unit == "mb":
            lev = lev * 100.0  # Convert to Pa
        if lev[0] > lev[1]:
            positive = "up"
        else:
            positive = "down"
        ds = nc4.Dataset(self.file_path, "a", format="NETCDF4")
        ds.createDimension("lev", len(lev))
        plev = ds.createVariable("lev", "f8", ("lev",))
        plev[:] = lev
        plev.long_name = "Pressure level"
        plev.units = "Pa"
        plev.positive = positive
        ds.close()
        self._built = True
