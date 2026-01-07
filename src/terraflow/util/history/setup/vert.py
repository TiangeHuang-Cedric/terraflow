from __future__ import annotations
from typing import TYPE_CHECKING

from terraflow.util.history.setup.base import AbstractHistorySetup

if TYPE_CHECKING:
    from terraflow.vert_coord.abstract import OneLayerVertCoordinate


class AbstractVertHistorySetup(AbstractHistorySetup):
    pass


class OneLayerVerticalCoordinateSetup(AbstractVertHistorySetup):

    def __init__(self, file_path: str, vert: OneLayerVertCoordinate):
        super().__init__(file_path=file_path)
        self.vert = vert

    def setup(self):
        if self.has_built():
            return
        self._built = True
