from __future__ import annotations
from typing import TYPE_CHECKING
import netCDF4 as nc4

from terraflow.util.history.setup.base import AbstractHistorySetup

if TYPE_CHECKING:
    from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate
    from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate
    import numpy as np


class AbstractVertHistorySetup(AbstractHistorySetup):
    pass


class OneLayerVerticalCoordinateSetup(AbstractVertHistorySetup):

    def __init__(
        self,
        file_path: str,
        vert: OneLayerVertCoordinate,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.vert = vert

    def setup(self):
        if self.has_built():
            return
        self._built = True


class HybridVerticalCoordinateSetup(AbstractVertHistorySetup):

    def __init__(
        self,
        file_path: str,
        vert: HybridVertCoordinate,
        float_type: type[np.float32] | type[np.float64],
        int_type: type[np.int32] | type[np.int64]
    ):
        super().__init__(
            file_path=file_path,
            float_type=float_type,
            int_type=int_type
        )
        self.vert = vert

    def setup(self):
        if self.has_built():
            return
        ds = nc4.Dataset(self.file_path, "a", format="NETCDF4")
        # Create dimensions
        ds.createDimension("lev", self.vert.nlev)
        ds.createDimension("ilev", self.vert.nlev + 1)
        # Create variables
        ds.createVariable("lev", "f8", ("lev",))
        ds.createVariable("ilev", "f8", ("ilev",))
        # Assign data
        lev = ds.variables["lev"]
        lev[:] = self.vert.get_full_lev()
        lev.long_name = "Full vertical level"
        lev.units = "Pa"
        lev.positive = "down"
        ilev = ds.variables["ilev"]
        ilev[:] = self.vert.get_half_lev()
        ilev.long_name = "Half vertical level"
        ilev.units = "Pa"
        ilev.positive = "down"
        ds.close()
        self._built = True
