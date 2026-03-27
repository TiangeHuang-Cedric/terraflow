from __future__ import annotations
from abc import ABC
import inspect
import importlib
from typing import Optional, Union, Type
from typing import Dict, List, Any, Callable
from pathlib import Path
import platform
import os
import importlib.metadata as metadata
import polars as pl

from terraflow.manager.plugins.equterm.ext import EquationTermDataFrame, EquationTermDataFrameExt
from terraflow.manager.plugins.equaux.ext import EquationAuxiliaryDataFrame, EquationAuxiliaryDataFrameExt
from terraflow.manager.plugins.diag.ext import DiagnosticMethodsDataFrame, DiagnosticMethodsDataFrameExt

from terraflow.util.tools import camel_to_snake

DataFrame = Union[
    EquationTermDataFrame,
    EquationAuxiliaryDataFrame,
    DiagnosticMethodsDataFrame,
]

DataFrameExt = Union[
    EquationTermDataFrameExt,
    EquationAuxiliaryDataFrameExt,
    DiagnosticMethodsDataFrameExt,
]


def tables_set() -> Dict[str, Type[DataFrame]]:
    return {
        "equterm": EquationTermDataFrame,
        "equaux": EquationAuxiliaryDataFrame,
        "diag": DiagnosticMethodsDataFrame,
    }


def tables_ext_set() -> Dict[str, Type[DataFrameExt]]:
    return {
        "equterm": EquationTermDataFrameExt,
        "equaux": EquationAuxiliaryDataFrameExt,
        "diag": DiagnosticMethodsDataFrameExt,
    }


class TableRegistrar:
    """Helper class for table-specific registration."""

    def __init__(self, catalog: "Catalog", table: str):
        self._catalog = catalog
        self._table = table

    def register(
        self,
        belong: Optional[str] = None,
        *,
        factory: Optional[Callable[..., Any]] = None,
        overwrite: bool = False,
        **meta: Any
    ) -> None:
        """Register a function/class to this specific table.

        Args:
            belong: Optional belong string (e.g., "fvm::stagc::odm")
            factory: The function or class to register (keyword-only)
            overwrite: Whether to overwrite existing registration
            **meta: Additional metadata (e.g., group, alias, kind, etc.)
        """
        self._catalog.register(
            self._table, factory,
            belong, overwrite=overwrite, **meta
        )


class Catalog:
    """Holds pure DataFrames per component and provides a unified register API."""

    def __init__(self) -> None:
        tbs = self.create_empty_tables()
        self._tables: Dict[str, DataFrame] = tbs
        self._loaded: bool = False
        self._db_loaded: bool = False

        # Create table-specific registrars as properties
        self.equterm = TableRegistrar(self, "equterm")
        self.equaux = TableRegistrar(self, "equaux")
        self.diag = TableRegistrar(self, "diag")

    def __getitem__(self, name: str) -> DataFrame:
        return self.table(name)

    def __setitem__(self, name: str, value: pl.DataFrame) -> None:
        if name not in self._tables:
            raise ValueError(f"Unknown table '{name}'")
        self._tables[name] = self.pack(name, value)

    def pack(self, name: str, value: pl.DataFrame) -> DataFrame:
        """Pack a DataFrame into the specified table."""
        if name not in self._tables:
            raise ValueError(f"Unknown table '{name}'")
        return tables_set()[name](value)

    def unpack(self, name: str) -> pl.DataFrame:
        """Unpack the specified table as a DataFrame."""
        if name not in self._tables:
            raise ValueError(f"Unknown table '{name}'")
        return pl.DataFrame(self._tables[name].to_dict())

    def clear(self) -> None:
        """Clear all tables to empty DataFrames."""
        self._tables = self.create_empty_tables()
        self._loaded = False
        self._db_loaded = False

    @staticmethod
    def create_empty_tables() -> Dict[str, DataFrame]:
        """Create empty table factories for all registered tables."""
        tbs: Dict[str, DataFrame] = {}
        for name, factory in tables_set().items():
            tbs[name] = factory().ext.add_meta()
        return tbs

    @staticmethod
    def _package_root() -> Path:
        pkg = importlib.import_module("terraflow")
        pkg_file = getattr(pkg, "__file__", None)
        if pkg_file is None:
            raise RuntimeError("Cannot determine terraflow package location")
        return Path(pkg_file).resolve().parent

    @staticmethod
    def _factory_to_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__",
                           getattr(value, "__name__", ""))
        if module and qualname:
            return f"{module}:{qualname}"
        return str(value)

    @staticmethod
    def _import_symbol(module: str, qualname: str) -> Any:
        mod = importlib.import_module(module)
        attr: Any = mod
        for part in qualname.split("."):
            if not part:
                continue
            attr = getattr(attr, part)
        return attr

    @staticmethod
    def _serialize_factory_column(df: pl.DataFrame) -> pl.DataFrame:
        if "factory" not in df.columns:
            return df
        serialized = [
            Catalog._factory_to_string(val) for val in df["factory"].to_list()
        ]
        return df.with_columns(
            pl.Series(name="factory", values=serialized, dtype=pl.Utf8)
        )

    @staticmethod
    def _deserialize_factory_column(df: pl.DataFrame) -> pl.DataFrame:
        if "factory" not in df.columns:
            return df
        if "module" not in df.columns or "qualname" not in df.columns:
            # Cannot restore without module/qualname metadata
            return df
        modules: List[str] = df["module"].to_list()
        qualnames: List[str] = df["qualname"].to_list()
        cache: Dict[tuple[str, str], Any] = {}
        restored: list[Any] = []
        for module, qualname in zip(modules, qualnames):
            if not module or not qualname:
                restored.append(None)
                continue
            rel_qualname = qualname
            if rel_qualname.startswith(module):
                rel_qualname = rel_qualname[len(module):].lstrip(".")
            key = (module, rel_qualname)
            if key not in cache:
                cache[key] = Catalog._import_symbol(module, rel_qualname)
            restored.append(cache[key])
        return df.with_columns(
            pl.Series(name="factory", values=restored, dtype=pl.Object)
        )

    def _prepare_table_for_disk(self, table: DataFrame) -> pl.DataFrame:
        df = table.unpack()
        return self._serialize_factory_column(df)

    def _restore_table_from_disk(self, name: str, df: pl.DataFrame) -> DataFrame:
        df = self._deserialize_factory_column(df)
        return self.pack(name, df)

    @staticmethod
    def default_database_dir(usr_path: bool = False) -> Path:
        """Get the default database directory path."""
        version = metadata.version("terraflow")
        if usr_path:
            system = platform.system().lower()
            if system in ("darwin", "linux"):
                return Path.home() / ".terraflow" / version
            base = Path(
                os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")
            )
            return base / "terraflow" / version
        else:
            package_root = Catalog._package_root()
            return package_root / ".terraflow" / version

    def build_database(
        self,
        path: Union[None, Path | str] = None,
        usr_path: bool = False
    ) -> None:
        """Build the registration database from current tables."""
        if not self.has_loaded():
            raise RuntimeError(
                "Registration has not been loaded yet."
            )
        if path is None:
            path = self.default_database_dir(usr_path=usr_path)
        path = Path(path)
        path_dict: Dict[str, str] = {}
        path.mkdir(parents=True, exist_ok=True)
        for tname, tb in self._tables.items():
            table_path = str(path / f"{tname}.parquet")
            path_dict[tname] = str(table_path)
            df_to_disk = self._prepare_table_for_disk(tb)
            df_to_disk.write_parquet(table_path)
        self._db_path: Dict[str, str] = path_dict

    def build_manifest(
        self,
        path: Union[None, Path | str] = None,
        usr_path: bool = False
    ) -> None:
        """Build the registration manifest from current tables."""
        if not self.has_loaded():
            raise RuntimeError(
                "Registration has not been loaded yet."
            )
        if path is None:
            path = self.default_database_dir(usr_path=usr_path)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path_dict: Dict[str, str] = {}
        import json
        for tname, tb in self._tables.items():
            manifest_path = path / f"{tname}.manifest"
            path_dict[tname] = str(manifest_path)
            manifest = {
                "name": tname,
                "count": tb.height,
                "ids": tb["unique_id"].to_list(),
                "terraflow_version": metadata.version("terraflow"),
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
        self._manifest_path: Dict[str, str] = path_dict

    def to_disk(
        self,
        path: Union[None, str, Path] = None,
        usr_path: bool = False,
    ) -> None:
        """Save the registration database to disk."""
        if not self.has_loaded():
            raise RuntimeError(
                "Registration has not been loaded yet."
            )
        if path is None:
            path = self.default_database_dir(usr_path=usr_path)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.build_database(path=path, usr_path=usr_path)
        self.build_manifest(path=path, usr_path=usr_path)

    def has_loaded(self) -> bool:
        """Check if the registration database has been loaded."""
        return self._loaded

    def load_source(self) -> str:
        """Get the source from which the registration database was loaded."""
        if not self.has_loaded():
            raise RuntimeError(
                "Registration has not been loaded yet."
            )
        else:
            return "database" if self._db_loaded else "entry_points"

    def _load_from_database(
        self,
        path: Union[None, str, Path] = None,
        usr_path: bool = False
    ) -> None:
        """Load the registration database into the model tables."""
        if self._db_loaded:
            return
        if path is None:
            path = self.default_database_dir(usr_path=usr_path)
        path = Path(path)
        for tname, tb in self._tables.items():
            table_path = path / f"{tname}.parquet"
            if not table_path.exists():
                raise FileNotFoundError(
                    f"Registration database file for table '{tname}' "
                    f"not found at '{table_path}'."
                )
            tb = pl.read_parquet(str(table_path))
            self._tables[tname] = self._restore_table_from_disk(tname, tb)
        self._db_loaded = True
        self._loaded = True

    def _load_from_entry_points(self) -> None:
        """Load registration from entry points into the model tables."""
        import importlib.metadata
        import traceback
        ep_group = "terraflow.registrations"

        any_loaded = False
        for ep in importlib.metadata.entry_points(group=ep_group):
            try:
                fn = ep.load()
                # Pass the Catalog instance (self) to the registration function
                fn(self)
                any_loaded = True
            except Exception as e:
                print(f"Error loading entry point '{ep.name}': {e}")
                traceback.print_exc()

        if not any_loaded:
            raise RuntimeError(
                f"No registration entry points found in group '{ep_group}'."
            )

        self._loaded = True
        self._db_loaded = False

    def load_from(
        self,
        path: Union[None, str, Path] = None,
        source: str = "database",
        usr_path: bool = False
    ) -> None:
        """
        Load registration from both database and entry points.

        Args:
            path: The path to the registration database.
            source: The source from which to load the registration.
            usr_path: Whether to use the user-specific path.

        Raises:
            ValueError: If both path and usr_path are specified when loading from database.
            ValueError: If path is provided when loading from entry points.
            ValueError: If usr_path is True when loading from entry points.
            ValueError: If an unknown source is specified.

        Usage:
            load_from(path="/path/to/db", source="database", usr_path=False)
            load_from(source="entry_points")
            load_from()  # defaults to database with default path in package
        """
        if source == "database":
            if path is not None and usr_path:
                raise ValueError(
                    "Cannot specify both path and usr_path when loading from database."
                )
            self._load_from_database(path=path, usr_path=usr_path)
        elif source == "entry_points":
            if path is not None:
                raise ValueError(
                    "Path should not be provided when loading from entry points."
                )
            if usr_path:
                raise ValueError(
                    "usr_path should not be True when loading from entry points."
                )
            self._load_from_entry_points()
        else:
            raise ValueError(
                f"Unknown source '{source}' for loading registration."
            )

    def ensure_loaded(self) -> bool:
        """Ensure that the registration is loaded either from the database or entry points."""
        if not self._loaded:
            try:
                self.load_from(source="database")
            except Exception:
                try:
                    self.load_from(source="entry_points")
                except Exception as e:
                    raise RuntimeError(
                        "Failed to load registration from both database and entry points."
                    ) from e
        return self.has_loaded()

    def table(self, name: str) -> DataFrame:
        if name not in self._tables:
            raise ValueError(f"Unknown table '{name}'")
        self.ensure_loaded()
        return self._tables[name]

    def view(self, name: str) -> DataFrame:
        self.ensure_loaded()
        return self.table(name).clone()

    def register(
        self,
        table: str,
        factory: Optional[Callable[..., Any]] = None,
        belong: Optional[str] = None,
        *,
        overwrite: bool = False,
        **meta: Any
    ) -> None:
        """Register a function/class with metadata without modifying the function itself.

        Args:
            factory: The function or class to register
            table: The table name to register into
            belong: Optional belong string (e.g., "fvm::stagc::odm")
            overwrite: Whether to overwrite existing registration
            **meta: Additional metadata (e.g., group, alias, kind, etc.)
        """
        if table not in tables_set():
            raise ValueError(f"Unknown table '{table}'")

        if factory is not None:
            # Prepare metadata using the bind system
            prepared_meta = prepare_metadata(factory, table, belong, **meta)

            obj = self._tables[table]

            # Register returns the updated DataFrame
            updated_df = obj.ext.register(
                factory=factory, overwrite=overwrite, **prepared_meta
            )
        else:
            # Register item those do not have factory (components, planets, etc.)
            obj = self._tables[table]

            # Ensure 'name' is provided in metadata
            name = meta.get("name", None)
            if name is None:
                raise ValueError(
                    "When registering without a factory, 'name' metadata must be provided."
                )

            # Auto-generate 'unique_id' if not provided
            meta["unique_id"] = meta.get("alias", camel_to_snake(name))

            # Register without factory
            updated_df = obj.ext.register(overwrite=overwrite, **meta)
        # Update the table with the returned DataFrame
        self._tables[table] = self.pack(table, updated_df)

    def filter(
        self,
        table: str,
        condition: pl.Expr
    ) -> pl.DataFrame:
        if table not in self._tables:
            raise ValueError(f"Unknown table '{table}'")
        self.ensure_loaded()
        obj = self._tables[table]
        return obj.filter(condition)


_DEFAULT_MODEL = Catalog()


def load_info(path: Union[None, str, Path] = None, source: str = "") -> Catalog:
    """
    Get the default model registration info.

    Args:
        path: Optional path to load registration from
        source: Source to load registration from: "database", "entry_points",
                default is "" which means try database first then entry points

    Returns:
        Catalog: The loaded registration catalog
    """
    model = Catalog()
    if source == "database":
        try:
            model.load_from(source="database", usr_path=True)
        except Exception:
            try:
                model.load_from(source="database", usr_path=False)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load registration from database."
                ) from e
    elif source == "entry_points":
        try:
            model.load_from(source="entry_points")
        except Exception as e:
            raise RuntimeError(
                "Failed to load registration from entry points."
            ) from e
    elif source == "":
        if path is None:
            try:
                model.load_from(source="database", usr_path=True)
            except Exception:
                try:
                    model.load_from(source="database", usr_path=False)
                except Exception:
                    model.load_from(source="entry_points")
        else:
            model.load_from(path=path)
    else:
        raise ValueError(
            f"Unknown source '{source}' for loading registration.")
    return model


def subregister(
    model: Catalog,
    factory: Callable,
) -> Catalog:
    """Register a function or class into the given model's registry.

    The function or class will be registered into the appropriate table
    based on the __table__ and __meta__ attributes set by the bind() decorator.
    """
    table = getattr(factory, "__table__", None)
    meta = getattr(factory, "__meta__", {})
    if table is None:
        raise ValueError(
            "The decorated function/class must have a '__table__' attribute. "
            "Use the 'bind' decorator to specify the target table."
        )
    model.register(factory=factory, **meta)
    return model


def catalog(
    table: str,
    *,
    model: Catalog = _DEFAULT_MODEL,
) -> DataFrame:
    """Get the model table information.

    Parameters:
    - table: table name, e.g. "op", "term", ...
    - model: use this specific model instance if provided
    - fresh: when True and model is None, create a new Catalog
             and return info from it (pending and discovered entries applied)

    Behavior:
    - Always ensures pending registrations are processed and previously
      discovered items are applied to the chosen model.
    """
    return model.view(table)


def navigate(
    table: str,
    *,
    model: Catalog = _DEFAULT_MODEL,
    id_col: Optional[str] = None,
    id_val: Optional[str] = None,
    attr: str = "factory",
    **meta,
) -> list:
    """Navigate to a specific registered item by its unique key."""
    model.ensure_loaded()
    if id_col is not None and id_val is not None:
        df = model.filter(table, pl.col(id_col).is_in([id_val]))
    else:
        df = model[table]
        for k, v in meta.items():
            df = df.filter(pl.col(k) == v)
    if df.height == 0:
        raise ValueError(
            f"No entry found in table '{table}' for criteria: "
            f"{id_col}={id_val}, " +
            ", ".join(f"{k}={v}" for k, v in meta.items())
        )
    return df[attr].to_list()


def prepare_metadata(
    factory: Callable,
    table: str,
    belong: Optional[str] = None,
    **meta: Any
) -> Dict[str, Any]:
    """Prepare metadata for a function without modifying the function itself.

    This function generates registration metadata from a factory and table info
    without attaching anything to the function. The metadata is returned as a dict.

    Args:
        factory: The function or class to prepare metadata for
        table: The name of the catalog table
        belong: Optional belong string (e.g., "fvm::stagc::odm")
        **meta: Additional metadata fields

    Returns:
        Dictionary containing all registration metadata

    Example:
        # In registrations.py
        from terraflow.operator.fvm.odm import base
        metadata = prepare_metadata(
            base.div_at_primal_cell,
            table="op",
            belong="fvm::stagc::odm",
            group="div"
        )
        model.register(base.div_at_primal_cell, table="op", **metadata)
    """
    # Create bind instance to generate metadata
    bind_class_map = {
        "equterm": EquationTermBind,
        "equaux": EquationAuxiliaryBind,
        "diag": DiagBind,
    }

    bind_class = bind_class_map.get(table, AbstractBind)
    bind_instance = bind_class(table, belong, **meta)

    # Extract metadata without modifying the function
    meta_local = dict(bind_instance.meta)

    if bind_instance.belong is not None:
        metadata = bind_instance._build_metadata_with_belong(
            factory, meta_local)
    else:
        metadata = bind_instance._build_metadata_without_belong(
            factory, meta_local)

    return metadata


class AbstractBind(ABC):
    """Abstract base class for bind decorators.

    This class encapsulates the common logic for binding functions to catalog tables.
    Subclasses can override specific methods to customize behavior for different table types.
    """

    def __init__(self, table: str, belong: Optional[str] = None, **meta: Any):
        """Initialize the bind decorator.

        Args:
            table: The name of the catalog table
            belong: Optional belong string (e.g., "fvm::stagc::odm")
            **meta: Additional metadata fields
        """
        self.table = table
        self.belong = belong
        self.meta = dict(meta)

        # Validate table exists
        if table not in tables_set():
            raise ValueError(f"Unknown table '{table}'")

        # Get extension class and create temporary instance
        self.ext_class = tables_ext_set()[table]
        temp_df = tables_set()[table]().ext.add_meta()
        # Use type: ignore to suppress the type checker warning
        # The runtime will ensure the correct type is used based on table name
        self.ext_instance = self.ext_class(temp_df)  # type: ignore[arg-type]

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply the bind decorator to a function.

        Note: This is kept for backward compatibility with decorator syntax,
        but prefer using prepare_metadata() for explicit registration.
        """
        meta_local = dict(self.meta)

        # Build metadata based on whether belong is provided
        if self.belong is not None:
            meta_local = self._build_metadata_with_belong(func, meta_local)
        else:
            meta_local = self._build_metadata_without_belong(func, meta_local)

        # Attach metadata to function (for backward compatibility only)
        setattr(func, "__table__", self.table)
        setattr(func, "__meta__", dict(meta_local))
        return func

    def _build_metadata_with_belong(
        self,
        func: Callable,
        meta_local: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build metadata when belong string is provided.

        This method handles the complex logic of parsing belong strings,
        extracting unique columns, and generating unique_id.
        """
        # Get unique columns for this table
        unique_cols = list(self.ext_instance.unique_columns())

        # Extract unique column values from meta (except 'group')
        unique_vals: Dict[str, Any] = {}
        for col in unique_cols:
            if col == "group":
                continue
            if col in meta_local:
                unique_vals[col] = meta_local.pop(col)

        # Auto-generate alias if not provided
        if "alias" not in unique_vals and "alias" in unique_cols:
            if hasattr(func, "alias_name"):
                alias_name_fn = getattr(func, "alias_name")
                unique_vals["alias"] = alias_name_fn()
            else:
                unique_vals["alias"] = camel_to_snake(func.__name__)

        # Build unique_id from belong + group + alias
        group = meta_local.pop("group", "")
        alias = unique_vals.get("alias", "")
        unique_id_parts = self._build_unique_id_parts(
            group, alias
        )

        # Validate unique_id length
        if len(unique_id_parts) != len(unique_cols):
            raise ValueError(
                f"belong='{self.belong}' + group='{group}' "
                f"produces unique id with {len(unique_id_parts)} parts, "
                f"but table '{self.table}' requires {len(unique_cols)} "
                f"unique columns: {unique_cols}"
            )

        # Map unique_id parts to column values
        col_values = self._map_unique_id_to_columns(
            unique_cols, unique_id_parts)
        meta_local.update(col_values)

        # Add standard metadata fields
        self._add_standard_metadata(func, meta_local)

        # Set unique_id
        unique_id = "::".join(unique_id_parts)
        meta_local["unique_id"] = unique_id

        # Set group_id for Discretization tables
        self._set_group_id(meta_local, unique_id_parts)

        return meta_local

    def _build_unique_id_parts(self, group: str, alias: str) -> list:
        """Build unique_id parts from belong, group, and alias.

        Args:
            group: The group name (may be empty)
            alias: The function alias

        Returns:
            List of unique_id parts
        """
        # self.belong is guaranteed to be non-None when this method is called
        assert self.belong is not None, "belong must not be None"
        belong_parts = self.belong.split("::")
        if group:
            return belong_parts + [group, alias]
        elif alias:
            return belong_parts + [alias]
        else:
            return belong_parts

    def _map_unique_id_to_columns(
        self, unique_cols: list, unique_id_parts: list
    ) -> Dict[str, Any]:
        """Map unique_id parts to column values.

        Subclasses can override this to provide special handling for certain columns.

        Args:
            unique_cols: List of unique column names
            unique_id_parts: List of unique_id parts

        Returns:
            Dictionary mapping column names to values
        """
        return dict(zip(unique_cols, unique_id_parts))

    def _add_standard_metadata(
        self, func: Callable, meta_local: Dict[str, Any]
    ) -> None:
        """Add standard metadata fields (name, description, module, etc.)."""
        meta_local["name"] = func.__name__
        meta_local["description"] = func.__doc__ or ""
        meta_local["module"] = func.__module__
        meta_local["qualname"] = f"{func.__module__}.{func.__qualname__}"

        # Get source file
        try:
            source_file = inspect.getsourcefile(func)
            if source_file is not None:
                meta_local["source"] = str(Path(source_file).resolve())
            else:
                meta_local["source"] = ""
        except Exception:
            meta_local["source"] = ""

        # Get source line number
        try:
            _, lineno = inspect.getsourcelines(func)
            meta_local["sourceline"] = lineno
        except Exception:
            meta_local["sourceline"] = -1

    def _set_group_id(self, meta_local: Dict[str, Any], unique_id_parts: list) -> None:
        """Set group_id for tables that support grouping."""
        if (
            hasattr(self.ext_instance, "group_columns")
            and callable(getattr(self.ext_instance, "group_columns", None))
        ):
            group_cols = getattr(self.ext_instance, "group_columns")()
            meta_local["group_id"] = "::".join(
                unique_id_parts[:len(group_cols)])

    def _build_metadata_without_belong(
        self, func: Callable, meta_local: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build metadata when no belong string is provided.

        This handles the simpler case where metadata is provided directly.
        """
        # Set default values for standard fields
        if "name" not in meta_local or not meta_local["name"]:
            meta_local["name"] = func.__name__
        if "description" not in meta_local or not meta_local["description"]:
            meta_local["description"] = func.__doc__ or ""
        if "alias" not in meta_local or not meta_local["alias"]:
            if hasattr(func, "alias_name"):
                alias_name_fn = getattr(func, "alias_name")
                meta_local["alias"] = alias_name_fn()
            else:
                meta_local["alias"] = camel_to_snake(func.__name__)
        if "module" not in meta_local or not meta_local["module"]:
            meta_local["module"] = func.__module__
        if "qualname" not in meta_local or not meta_local["qualname"]:
            meta_local["qualname"] = f"{func.__module__}.{func.__qualname__}"

        # Get source file
        if "source" not in meta_local or not meta_local["source"]:
            try:
                source_file = inspect.getsourcefile(func)
                if source_file is not None:
                    meta_local["source"] = str(Path(source_file).resolve())
                else:
                    meta_local["source"] = ""
            except Exception:
                meta_local["source"] = ""

        # Get source line number
        if "sourceline" not in meta_local or not meta_local["sourceline"]:
            try:
                _, lineno = inspect.getsourcelines(func)
                meta_local["sourceline"] = lineno
            except Exception:
                meta_local["sourceline"] = -1

        # Generate unique_id using extension's method
        unique_cols = self.ext_instance.unique_columns()
        unique_kwargs = {k: meta_local[k]
                         for k in unique_cols if k in meta_local}
        unique_id = self.ext_instance.make_unique_id(**unique_kwargs)
        meta_local["unique_id"] = unique_id

        # Generate group_id if supported
        if (
            hasattr(self.ext_instance, "make_group_id")
            and callable(getattr(self.ext_instance, "make_group_id", None))
        ):
            if (
                hasattr(self.ext_instance, "group_columns")
                and callable(getattr(self.ext_instance, "group_columns", None))
            ):
                group_cols = getattr(self.ext_instance, "group_columns")()
                group_kwargs = {
                    k: meta_local[k]
                    for k in group_cols if k in meta_local
                }
                group_id = getattr(
                    self.ext_instance,
                    "make_group_id"
                )(**group_kwargs)
                meta_local["group"] = group_id
        return meta_local


class EquationTermBind(AbstractBind):
    """Bind decorator for discretization terms."""
    pass


class EquationAuxiliaryBind(AbstractBind):
    """Bind decorator for auxiliary equation components."""
    pass


class DiagBind(AbstractBind):
    """Bind decorator for diagnostic methods definition."""
    pass
