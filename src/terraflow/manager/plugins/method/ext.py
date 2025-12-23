from typing import Any, Dict, Tuple, Union
import polars as pl
from polars._typing import PolarsDataType

from terraflow.manager.abstract import AbstractBaseDataFrame
from terraflow.manager.abstract import AbstractDataFrameExt


class MethodDataFrame(AbstractBaseDataFrame):

    @property
    def ext(self) -> "MethodDataFrameExt":
        return MethodDataFrameExt(self)

    @staticmethod
    def pack(value: pl.DataFrame) -> "MethodDataFrame":
        return MethodDataFrame(value)

    def unpack(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())


class MethodDataFrameExt(AbstractDataFrameExt[MethodDataFrame]):

    def set_dataframe_kind(self) -> None:
        setattr(self, '__dataframe__', 'method')

    @staticmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        return {
            "name": {"polars": pl.Utf8, "type": str, "default": ""},
            "alias": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "factory": {"polars": pl.Object, "type": object, "default": None},
        } | AbstractDataFrameExt.base_meta()

    @staticmethod
    def unique_columns() -> Tuple[str, ...]:
        return ("alias",)

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any
    ) -> pl.DataFrame:
        """Register or upsert a mesh row using generic meta/default flow."""
        return super().register(overwrite=overwrite, **kwargs)
