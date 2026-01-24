from typing import Any, Dict, Tuple, Union, Optional
import polars as pl
from polars._typing import PolarsDataType

from terraflow.manager.abstract import AbstractBaseDataFrame
from terraflow.manager.abstract import AbstractDiscretizationDataFrameExt


class EquationAuxiliaryDataFrame(AbstractBaseDataFrame):

    @property
    def ext(self) -> "EquationAuxiliaryDataFrameExt":
        return EquationAuxiliaryDataFrameExt(self)

    @staticmethod
    def pack(value: pl.DataFrame) -> "EquationAuxiliaryDataFrame":
        return EquationAuxiliaryDataFrame(value)

    def unpack(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())


class EquationAuxiliaryDataFrameExt(AbstractDiscretizationDataFrameExt[EquationAuxiliaryDataFrame]):

    def __init__(self, df: Optional[EquationAuxiliaryDataFrame] = None) -> None:
        if df is None:
            df = EquationAuxiliaryDataFrame(pl.DataFrame())
        super().__init__(df)

    def set_dataframe_kind(self) -> None:
        setattr(self, '__dataframe__', 'equaux')

    @staticmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        return {
            "name": {"polars": pl.Utf8, "type": str, "default": ""},
            "planet": {"polars": pl.Utf8, "type": str, "default": ""},
            "component": {"polars": pl.Utf8, "type": str, "default": ""},
            "equset": {"polars": pl.Utf8, "type": str, "default": ""},
            "form": {"polars": pl.Utf8, "type": str, "default": ""},
            "ode_kind": {"polars": pl.Utf8, "type": str, "default": ""},
            "method": {"polars": pl.Utf8, "type": str, "default": ""},
            "layout": {"polars": pl.Utf8, "type": str, "default": ""},
            "horizontal_mesh": {"polars": pl.Utf8, "type": str, "default": ""},
            "vertical_coordinate": {"polars": pl.Utf8, "type": str, "default": ""},
            "description": {"polars": pl.Utf8, "type": str, "default": ""},
            "factory": {"polars": pl.Object, "type": object, "default": None},
            "alias": {"polars": pl.Utf8, "type": str, "default": ""},
        } | AbstractDiscretizationDataFrameExt.extra_meta()

    @staticmethod
    def unique_columns() -> Tuple[str, ...]:
        return (
            "planet", "component", "equset", "form", "ode_kind",
            "method", "layout", "horizontal_mesh", "vertical_coordinate",
            "alias"
        )

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any
    ) -> pl.DataFrame:
        """Register or upsert a term row"""
        return super().register(overwrite=overwrite, **kwargs)
