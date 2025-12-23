from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, cast, Type, TypeVar, Generic
from typing import Iterable, Sequence, Tuple, Dict, Any
import polars as pl
from polars._typing import (
    PolarsDataType,
    JoinStrategy,
    JoinValidation,
    MaintainOrderJoin,
)


DataFrame = TypeVar("DataFrame", bound="AbstractBaseDataFrame")


class AbstractBaseDataFrame(pl.DataFrame, ABC):
    @classmethod
    def _wrap(cls: Type[DataFrame], df: pl.DataFrame) -> DataFrame:
        """
        Convert pl.DataFrame to the specific subclass type.
        This is the key method for type preservation.
        """
        # Create instances of the same type through the _from_pydf internal method
        if hasattr(df, '_df'):
            # polars internally uses the _df attribute to store PyDataFrame
            return cast(DataFrame, cls._from_pydf(df._df))
        return cast(DataFrame, cls(df))

    def clone(self: DataFrame) -> DataFrame:
        """Clone the DataFrame, preserving subclass type."""
        return type(self)._wrap(super().clone())

    # Override common DataFrame methods to preserve type
    def select(self, *exprs, **named_exprs):
        """Select columns, preserving subclass type."""
        result = super().select(*exprs, **named_exprs)
        return type(self)._wrap(result)

    def filter(self, *predicates, **kwargs):
        """Filter rows, preserving subclass type."""
        result = super().filter(*predicates, **kwargs)
        return type(self)._wrap(result)

    def with_columns(self, *exprs, **named_exprs):
        """Add or replace columns, preserving subclass type."""
        result = super().with_columns(*exprs, **named_exprs)
        return type(self)._wrap(result)

    def drop(self, *columns, **kwargs):
        """Drop columns, preserving subclass type."""
        result = super().drop(*columns, **kwargs)
        return type(self)._wrap(result)

    def sort(self, by, *more_by, **kwargs):
        """Sort DataFrame, preserving subclass type."""
        result = super().sort(by, *more_by, **kwargs)
        return type(self)._wrap(result)

    def unique(self, subset=None, **kwargs):
        """Get unique rows, preserving subclass type."""
        result = super().unique(subset=subset, **kwargs)
        return type(self)._wrap(result)

    def head(self, n: int = 5):
        """Get first n rows, preserving subclass type."""
        result = super().head(n)
        return type(self)._wrap(result)

    def tail(self, n: int = 5):
        """Get last n rows, preserving subclass type."""
        result = super().tail(n)
        return type(self)._wrap(result)

    def slice(self, offset: int, length: Optional[int] = None):
        """Slice DataFrame, preserving subclass type."""
        result = super().slice(offset, length)
        return type(self)._wrap(result)

    # Override to preserve subclass type
    def join(  # type: ignore[override]
        self,
        other: pl.DataFrame,
        *,
        on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        how: JoinStrategy = "inner",
        left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        nulls_equal: bool = False,
        coalesce: bool | None = None,
        maintain_order: MaintainOrderJoin | None = None,
    ):
        """Join with another DataFrame, preserving subclass type."""
        result = super().join(
            other,
            on=on,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
            validate=validate,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
        )
        return type(self)._wrap(result)

    def group_by(self, *by, **kwargs):
        """Group by columns - returns GroupBy object."""
        # Note: group_by returns a GroupBy object, not a DataFrame
        # We'll wrap its result methods instead
        gb = super().group_by(*by, **kwargs)
        return self._wrap_groupby(gb)

    def _wrap_groupby(self, gb):
        """Wrap GroupBy object to preserve type in aggregation results."""
        original_agg = gb.agg

        def wrapped_agg(*args, **kwargs):
            result = original_agg(*args, **kwargs)
            if isinstance(result, pl.DataFrame):
                return type(self)._wrap(result)
            return result

        gb.agg = wrapped_agg
        return gb

    @property
    @abstractmethod
    def ext(self) -> Any:
        """Extension accessor for custom DataFrame operations."""
        pass

    @staticmethod
    @abstractmethod
    def pack(value: pl.DataFrame) -> "AbstractBaseDataFrame":
        pass

    @abstractmethod
    def unpack(self) -> pl.DataFrame:
        pass


class AbstractDataFrameExt(ABC, Generic[DataFrame]):
    """Extended Polars DataFrame with additional helper methods."""
    _df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self._df = df
        self._loaded: bool = False
        self._db_loaded: bool = False
        self._db_path: Dict[str, str] = {}
        self._manifest_path: Dict[str, str] = {}

    def add_meta(self) -> DataFrame:
        self.set_dataframe_kind()
        if self._df.is_empty():
            blank: Dict[str, pl.Series] = {}
            for col, spec in self.meta().items():
                dtype = cast(PolarsDataType, spec.get("polars", pl.Utf8))
                blank[col] = pl.Series(name=col, values=[], dtype=dtype)
            self._df = cast(DataFrame, self._df.pack(pl.DataFrame(blank)))
        else:
            # Ensure DataFrame has correct columns and dtypes
            meta = self.meta()
            schema = self._df.schema
            if schema == meta:
                return self._df
            if set(schema.keys()) - set(meta.keys()):
                raise ValueError(
                    "DataFrame has extra columns not declared in meta."
                )
            if set(meta.keys()) - set(schema.keys()):
                print(
                    "DataFrame is missing some columns declared in meta."
                )
                df = self._df
                for col, spec in meta.items():
                    dtype = cast(PolarsDataType, spec.get("polars", pl.Utf8))
                    if col not in df.columns:
                        self._df = cast(DataFrame, df.pack(df.with_columns(
                            pl.Series(name=col, values=[], dtype=dtype)
                        )))
                    else:
                        self._df = cast(DataFrame, df.pack(df.with_columns(
                            pl.col(col).cast(dtype).alias(col)
                        )))
        return self._df

    @abstractmethod
    def set_dataframe_kind(self) -> None:
        """Set the kind of DataFrame this extension represents."""
        if not hasattr(self, '__dataframe__'):
            setattr(self, '__dataframe__', 'undefined')

    @staticmethod
    def base_meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        """Return the base metadata dictionary common to all table factories."""
        return {
            "module": {"polars": pl.Utf8, "type": str, "default": ""},
            "qualname": {"polars": pl.Utf8, "type": str, "default": ""},
            "source": {"polars": pl.Utf8, "type": str, "default": ""},
            "sourceline": {"polars": pl.Int64, "type": int, "default": -1},
            "unique_id": {"polars": pl.Utf8, "type": str, "default": ""},
        }

    @staticmethod
    @abstractmethod
    def meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        """Return the metadata dictionary for this table factory."""
        return AbstractDataFrameExt.base_meta()

    def get_meta_by_attr(
        self,
        attr: str,
        specific: Optional[str | Iterable[str]] = None
    ) -> Dict[str, Any]:
        """
        Get a dict of meta attribute values for all or specific keys.

        Args:
            attr: The attribute name to retrieve from the meta dict.
            specific: Optional key or iterable of keys to filter; if None, retrieves for all keys.

        Returns:
            A dict mapping keys to their corresponding attribute values or None if not found.

        Usage:
            get_meta_by_attr('default') -> {'alias': '', 'name': '', ...}
            get_meta_by_attr('type', specific='alias') -> {'alias': str}
            get_meta_by_attr('type', specific=['alias', 'name']) -> {'alias': str, 'name': str}
        """
        if specific is not None and isinstance(specific, str):
            specific = [specific]
        out: Dict[str, Any] = {}
        iters = specific if specific is not None else self.meta().keys()
        for k in iters:
            spec = self.meta().get(k, {})
            if isinstance(spec, dict) and attr in spec:
                out[k] = spec[attr]
            else:
                out[k] = None
        return out

    def _ensure_input_iterables_are_the_same(
        self,
        a: Iterable[Any],
        b: Iterable[Any],
    ) -> None:
        """Ensure that kwargs keys match required keys exactly."""
        if a == b:
            return
        else:
            sa, sb = set(a), set(b)
            extra_a = sa - sb
            extra_b = sb - sa

            if extra_a and not extra_b:
                raise ValueError(
                    f"Iterable a has extra items not in b: \n"
                    f"{sorted(extra_a)}"
                )
            if extra_b and not extra_a:
                raise ValueError(
                    f"Iterable b has extra items not in a: \n"
                    f"{sorted(extra_b)}"
                )
            if extra_a and extra_b:
                raise ValueError(
                    "Input iterables do not match exactly: \n"
                    f"Different items in a but not in b: {sorted(extra_a)}\n"
                    f"Different items in b but not in a: {sorted(extra_b)}"
                )

    @staticmethod
    def join_keys(keys: Iterable[str], sep: str = "::") -> str:
        """
        Using sep to join the given keys into a string.

        Args:
            keys: Iterable of keys to join.
            sep: Separator string.

        Returns:
            Joined string of keys.

        Usage:
            join_keys(['a', 'b', 'c'], sep='::') -> 'a::b::c'
        """
        return sep.join(str(k) for k in keys)

    def combine_str_columns(
        self,
        cols: Iterable[str],
        sep: str = "::"
    ) -> pl.Series:
        """Combine multiple columns into a single Series with values joined by sep."""
        exprs = [pl.col(c).cast(pl.Utf8) for c in cols]
        combined = pl.concat_str(exprs, separator=sep).alias("_combined")
        return self._df.select(combined).to_series()

    def _has(
        self,
        cond: Dict[str, object] | Tuple[str, object]
    ) -> bool:
        if isinstance(cond, dict):
            mask = None
            for k, v in cond.items():
                m = (pl.col(k) == v)
                mask = m if mask is None else (mask & m)
            out = self._df.filter(mask) if mask is not None else self._df
            return out.height > 0
        else:
            k, v = cond
            return self._df.filter(pl.col(k) == v).height > 0

    def _upsert_row(
        self,
        params: Dict[str, Any],
        unique_id: str,
        overwrite: bool,
        unique_cond: Optional[Union[Dict[str, Any], Tuple[str, Any]]] = None,
    ) -> pl.DataFrame:
        """Upsert a single row by unique ID; enforce metadata equality if not overwriting."""
        # Build condition from either explicit unique_cond or fallback to (unique_id, value)
        if unique_cond is None:
            key_val = params[unique_id]
            cond: Union[Dict[str, Any], Tuple[str, Any]] = (
                unique_id, key_val)
        else:
            cond = unique_cond

        # Helper to build a boolean mask expr
        def _mask_expr(cond_in: Union[Dict[str, Any], Tuple[str, Any]]):
            if isinstance(cond_in, dict):
                expr = pl.lit(True)
                for k, v in cond_in.items():
                    expr = expr & (pl.col(k) == v)
                return expr
            else:
                k2, v2 = cond_in
                return pl.col(k2) == v2

        exists = self._has(cond)
        if exists:
            if not overwrite:
                mask = _mask_expr(cond)
                item_tb = self._df.filter(mask)
                for col in item_tb.columns:
                    val = item_tb.select(pl.col(col)).to_series().item()
                    new_val = params.get(col)
                    if val != new_val:
                        raise ValueError(
                            f"Item {col} has been already registered before "
                            f"with a different value.\n"
                            f"Existing value for '{col}': {val}, \n"
                            f"New value for '{col}': {new_val}"
                        )
            # remove existing rows that match cond
            mask = _mask_expr(cond)
            df = self._df
            self._df = cast(DataFrame, self._df.pack(df.filter(~mask)))

        # Build row following DataFrame's schema/dtypes
        row_data: dict[str, object] = {}
        for col, dtype in zip(self._df.columns, self._df.dtypes):
            val = params.get(col, None)
            if dtype == pl.Object:
                row_data[col] = pl.Series(
                    name=col, values=[val], dtype=pl.Object)
            else:
                row_data[col] = [val]
        new_row = pl.DataFrame(row_data)
        return pl.concat([self._df, new_row], how="vertical_relaxed")

    def _get_default_ids(self, cols: Iterable[str]) -> Dict[str, str]:
        if not cols:
            raise ValueError("Input columns returned empty")
        defaults = self.get_meta_by_attr("default")
        defaults = {c: str(defaults.get(c, "")) for c in cols}
        return {c: str(defaults.get(c, "")) for c in cols}

    def _generate_ids(self, cols: Sequence[str]) -> list[str]:
        if not cols:
            raise ValueError("Input columns returned empty")
        if len(cols) == 1:
            return self._df[cols[0]].cast(pl.Utf8).to_list()
        gen_ids = self.combine_str_columns(cols, sep="::").to_list()
        return gen_ids

    @staticmethod
    @abstractmethod
    def unique_columns() -> Tuple[str, ...]:
        """Return the tuple of column names forming the unique key; default to ('alias',)."""
        return ("alias",)

    def get_default_unique_id(self) -> Dict[str, str]:
        """
        Get the default unique ID names and its values.

        Args:
            out_format: 'dict' or 'tuple'

        Returns:
            If out_format is 'dict', returns a dict of {column_name: default_value}.
        """
        cols = tuple(self.unique_columns())
        return self._get_default_ids(cols)

    def generate_unique_ids(self) -> list[str]:
        """Generate unique keys for each row in the DataFrame based on unique columns."""
        cols = list(self.unique_columns())
        gen_ids = self._generate_ids(cols)
        frame_ids = self._df["unique_id"].to_list()
        if gen_ids != frame_ids:
            raise ValueError(
                "The generated unique IDs do not match "
                "the existing 'unique_id' column in the DataFrame."
            )
        return gen_ids

    def make_unique_id(self, **kwargs) -> str:
        """Generate a unique ID based on unique columns."""
        default_id = self.get_default_unique_id()
        self._ensure_input_iterables_are_the_same(default_id, kwargs)
        default = self.get_meta_by_attr(
            "default", specific=default_id.keys()
        )
        default.update(kwargs)
        return self.join_keys(default.values())

    @abstractmethod
    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any
    ) -> pl.DataFrame:
        """Register or upsert a row using meta defaults and composite unique condition.

        Behavior:
        - Validate input keys against meta().
        - Merge defaults from meta() with provided kwargs to form params.
        - Require all unique_columns() to be explicitly provided in kwargs.
        - Delegate existence check and upsert to _upsert_row using a composite condition.
        """
        # Validate fields: only accept keys declared in meta()
        meta = self.meta()
        unknown = set(kwargs) - set(meta.keys())
        if unknown:
            raise ValueError(
                f"Unknown meta fields for registration: {sorted(unknown)}"
            )

        # Merge defaults with provided kwargs (meta-declared keys only)
        defaults = self.get_meta_by_attr("default")
        params: Dict[str, Any] = {
            k: kwargs.get(k, defaults.get(k)) for k in meta.keys()
        }

        # Unique constraints: require all unique columns to be explicitly provided
        cols = list(self.unique_columns())
        if not cols:
            raise ValueError("unique_columns() returned empty")
        missing = [c for c in cols if c not in kwargs]
        if missing:
            raise ValueError(f"Unique columns must be provided: {missing}")

        # Composite unique condition handled in _upsert_row
        cond = {c: params[c] for c in cols}
        # placeholder; not used when unique_cond is provided
        placeholder_key = cols[0]
        result = self._upsert_row(
            params,
            unique_id=placeholder_key,
            overwrite=overwrite,
            unique_cond=cond,
        )
        # Update the internal DataFrame with the result
        self._df = cast(DataFrame, self._df.pack(result))
        return self._df


class AbstractDiscretizationDataFrameExt(AbstractDataFrameExt[DataFrame], ABC):

    @staticmethod
    def extra_meta() -> Dict[str, Dict[str, Union[PolarsDataType, object]]]:
        base_meta = AbstractDataFrameExt.base_meta()
        extra = {
            "group_id": {"polars": pl.Utf8, "type": str, "default": ""},
        }
        return base_meta | extra

    def register(
        self,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Register or upsert an operator row"""
        return super().register(overwrite=overwrite, **kwargs)

    def group_columns(self) -> Tuple[str, ...]:
        """
        Return the tuple of column names forming the group key.
        Default to all but the last unique column.
        """
        cols = list(self.unique_columns())
        if not cols:
            raise ValueError("unique_columns() returned empty")
        return tuple(cols[:-1])  # all but last

    def get_default_group_id(self) -> Dict[str, str]:
        """
        Get the default group ID names and its values.

        Returns:
            A dict of {column_name: default_value}.
        """
        cols = tuple(self.group_columns())
        return self._get_default_ids(cols)

    def generate_group_ids(self) -> list[str]:
        """Generate group IDs for each row in the DataFrame based on group columns."""
        cols = list(self.group_columns())
        gen_ids = self._generate_ids(cols)
        frame_ids = self._df["group_id"].to_list()
        if gen_ids != frame_ids:
            raise ValueError(
                "The generated group IDs do not match "
                "the existing 'group_id' column in the DataFrame."
            )
        return gen_ids

    def make_group_id(self, **kwargs) -> str:
        """Generate a unique group ID."""
        default_id = self.get_default_group_id()
        self._ensure_input_iterables_are_the_same(default_id, kwargs)
        default = self.get_meta_by_attr("default", specific=default_id.keys())
        default.update(kwargs)
        return self.join_keys(default.values())
