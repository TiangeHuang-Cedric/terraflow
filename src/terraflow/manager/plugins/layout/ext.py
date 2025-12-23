from typing import Any, Dict, Tuple, Union
import polars as pl
from polars._typing import PolarsDataType

from terraflow.manager.abstract import AbstractBaseDataFrame
from terraflow.manager.abstract import AbstractDataFrameExt


class LayoutDataFrame(AbstractBaseDataFrame):

    @property
    def ext(self) -> "LayoutDataFrameExt":
        return LayoutDataFrameExt(self)

    @staticmethod
    def pack(value: pl.DataFrame) -> "LayoutDataFrame":
        return LayoutDataFrame(value)

    def unpack(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())


class LayoutDataFrameExt(AbstractDataFrameExt[LayoutDataFrame]):

    def set_dataframe_kind(self) -> None:
        setattr(self, '__dataframe__', 'layout')

    @staticmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        return {
            "name": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "horizontal_mesh": {"polars": pl.Utf8, "type": str, "default": ""},
            "vertical_coordinate": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "factory": {"polars": pl.Object, "type": object, "default": None},
            "alias": {"polars": pl.Utf8, "type": str, "default": ""},
        } | AbstractDataFrameExt.base_meta()

    @staticmethod
    def unique_columns() -> Tuple[str, ...]:
        return ("horizontal_mesh", "vertical_coordinate", "alias")

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any
    ) -> pl.DataFrame:
        """Register or upsert a mesh row using generic meta/default flow."""
        return super().register(overwrite=overwrite, **kwargs)
