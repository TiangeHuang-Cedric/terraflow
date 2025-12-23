from typing import Any, Dict, Tuple, Union, Optional
import polars as pl
from polars._typing import PolarsDataType

from terraflow.manager.abstract import AbstractBaseDataFrame
from terraflow.manager.abstract import AbstractDataFrameExt


class EquationSetDataFrame(AbstractBaseDataFrame):

    @property
    def ext(self) -> "EquationSetDataFrameExt":
        return EquationSetDataFrameExt(self)

    @staticmethod
    def pack(value: pl.DataFrame) -> "EquationSetDataFrame":
        return EquationSetDataFrame(value)

    def unpack(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())


class EquationSetDataFrameExt(AbstractDataFrameExt[EquationSetDataFrame]):

    def __init__(self, df: Optional[EquationSetDataFrame] = None) -> None:
        if df is None:
            df = EquationSetDataFrame(pl.DataFrame())
        super().__init__(df)

    def set_dataframe_kind(self) -> None:
        setattr(self, '__dataframe__', 'equset')

    @staticmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        return {
            "name": {"polars": pl.Utf8, "type": str, "default": ""},
            "form": {"polars": pl.Utf8, "type": str, "default": ""},
            "mesh_kind": {"polars": pl.Utf8, "type": str, "default": ""},
            "ode_kind": {"polars": pl.Utf8, "type": str, "default": ""},
            "alias": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "factory": {"polars": pl.Object, "type": object, "default": None},
        } | AbstractDataFrameExt.base_meta()

    @staticmethod
    def unique_columns() -> Tuple[str, ...]:
        return ("form", "mesh_kind", "ode_kind", "alias")

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any
    ) -> pl.DataFrame:
        """Register or upsert an discretization-set row using generic meta/default flow."""
        return super().register(overwrite=overwrite, **kwargs)
