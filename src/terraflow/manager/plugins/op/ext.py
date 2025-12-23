from typing import Any, Dict, Tuple, Union, Optional
import polars as pl
from polars._typing import PolarsDataType

from terraflow.manager.abstract import AbstractBaseDataFrame
from terraflow.manager.abstract import AbstractDiscretizationDataFrameExt


class OperatorDataFrame(AbstractBaseDataFrame):

    @property
    def ext(self) -> "OperatorDataFrameExt":
        return OperatorDataFrameExt(self)

    @staticmethod
    def pack(value: pl.DataFrame) -> "OperatorDataFrame":
        return OperatorDataFrame(value)

    def unpack(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())


class OperatorDataFrameExt(AbstractDiscretizationDataFrameExt[OperatorDataFrame]):

    def __init__(self, df: Optional[OperatorDataFrame] = None) -> None:
        if df is None:
            df = OperatorDataFrame(pl.DataFrame())
        super().__init__(df)

    def set_dataframe_kind(self) -> None:
        setattr(self, '__dataframe__', 'op')

    @staticmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        return {
            "name": {"polars": pl.Utf8, "type": str, "default": ""},
            "method": {"polars": pl.Utf8, "type": str, "default": ""},
            "layout": {"polars": pl.Utf8, "type": str, "default": ""},
            "horizontal_mesh": {"polars": pl.Utf8, "type": str, "default": ""},
            "vertical_coordinate": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "factory": {"polars": pl.Object, "type": object, "default": None},
            "group": {"polars": pl.Utf8, "type": str, "default": ""},
            "alias": {"polars": pl.Utf8, "type": str, "default": ""},
        } | AbstractDiscretizationDataFrameExt.extra_meta()

    @staticmethod
    def unique_columns() -> Tuple[str, ...]:
        return ("method", "layout", "horizontal_mesh", "vertical_coordinate", "group", "alias")

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Register or upsert an operator row"""
        return super().register(overwrite=overwrite, **kwargs)
