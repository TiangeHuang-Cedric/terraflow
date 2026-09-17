from __future__ import annotations
from dataclasses import (
    MISSING, 
    dataclass, 
    field as dataclass_field, 
    fields,
    is_dataclass
)
from typing import (
    Any, 
    Callable, 
    Mapping, 
    Sequence, 
    Self, 
    TypeVar, 
    cast, 
    dataclass_transform
)
from weakref import WeakSet


T = TypeVar("T")
PathPart = str | int
Path = Sequence[PathPart]

STATIC_METADATA_KEY = "static"
PytreeRegistrar = Callable[[type], None]

_MODULE_CLASSES: WeakSet[type] = WeakSet()
_PYTREE_REGISTRAR: PytreeRegistrar | None = None


@dataclass(frozen=True)
class _StaticValue:
    kind: str
    value: Any


class _IdentityStaticValue:
    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityStaticValue) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


def _freeze_static(value: Any) -> _StaticValue | _IdentityStaticValue:
    if isinstance(value, list):
        items = tuple(_freeze_static(v) for v in value)
        return _StaticValue("list"     , items)
    if isinstance(value, tuple):
        items = tuple(_freeze_static(v) for v in value) 
        return _StaticValue("tuple"    , items)
    if isinstance(value, dict):
        items = tuple((_freeze_static(k), _freeze_static(v)) for k, v in value.items())
        return _StaticValue("dict"     , items)
    if isinstance(value, set):
        items = tuple(sorted((_freeze_static(v) for v in value), key=repr))
        return _StaticValue("set"      , items)
    if isinstance(value, frozenset):
        items = tuple(sorted((_freeze_static(v) for v in value), key=repr))
        return _StaticValue("frozenset", items)

    try:
        hash(value)
    except TypeError:
        return _IdentityStaticValue(value)
    return _StaticValue("raw", value)


def _unfreeze_static(value: _StaticValue | _IdentityStaticValue) -> Any:
    if isinstance(value, _IdentityStaticValue):
        return value.value

    if value.kind == "raw":
        return value.value
    if value.kind == "list":
        return [_unfreeze_static(v) for v in value.value]
    if value.kind == "tuple":
        return tuple(_unfreeze_static(v) for v in value.value)
    if value.kind == "dict":
        return {
            _unfreeze_static(k): _unfreeze_static(v)
            for k, v in value.value
        }
    if value.kind == "set":
        return {_unfreeze_static(v) for v in value.value}
    if value.kind == "frozenset":
        return frozenset(_unfreeze_static(v) for v in value.value)

    raise ValueError(f"Unknown static value kind: {value.kind}")


def field(
    *,
    static: bool = False,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[str, Any] | None = None,
    kw_only: bool | Any = MISSING,
) -> Any:
    field_metadata = dict(metadata or {})
    field_metadata[STATIC_METADATA_KEY] = bool(static)

    kwargs: dict[str, Any] = {
        "init": init,
        "repr": repr,
        "hash": hash,
        "compare": compare,
        "metadata": field_metadata,
    }
    if default is not MISSING:
        kwargs["default"] = default
    if default_factory is not MISSING:
        kwargs["default_factory"] = default_factory
    if kw_only is not MISSING:
        kwargs["kw_only"] = kw_only

    return dataclass_field(**kwargs)


@dataclass_transform(field_specifiers=(field,))
class Module:

    def __init_subclass__(
        cls,
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        kw_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            kw_only=kw_only,
        )
        _MODULE_CLASSES.add(cls)
        if _PYTREE_REGISTRAR is not None:
            _PYTREE_REGISTRAR(cls)

    def replace(self: Self, **updates: Any) -> Self:
        """Return a shallow structural copy with selected fields replaced."""
        return replace(self, **updates)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], tuple[tuple[str, Any], ...]]]:
        """Flatten dataclass fields using ``field(static=True)`` as pytree metadata."""
        dynamic_names: list[str] = []
        dynamic_values: list[Any] = []
        static_items: list[tuple[str, Any]] = []

        for item in fields(cast(Any, self)):
            value = getattr(self, item.name)
            if item.metadata.get(STATIC_METADATA_KEY, False):
                static_items.append((item.name, _freeze_static(value)))
            else:
                dynamic_names.append(item.name)
                dynamic_values.append(value)

        return tuple(dynamic_values), (tuple(dynamic_names), tuple(static_items))

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[tuple[str, ...], tuple[tuple[str, Any], ...]],
        children: Sequence[Any],
    ) -> Self:
        dynamic_names, static_items = aux_data
        obj = object.__new__(cls)

        for name, value in static_items:
            object.__setattr__(obj, name, _unfreeze_static(value))
        for name, value in zip(dynamic_names, children):
            object.__setattr__(obj, name, value)

        return obj


def replace(obj: T, **updates: Any) -> T:
    """Return a shallow copy of a dataclass/module with fields replaced.

    Unlike ``dataclasses.replace``, this does not call ``__init__``. That makes
    it suitable for immutable-style updates of simulation state objects whose
    constructors may perform expensive setup or require external resources.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("replace() expects a dataclass or Module instance.")

    field_names = {f.name for f in fields(obj)}
    unknown = set(updates) - field_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise AttributeError(f"Unknown field(s) for {type(obj).__name__}: {names}")

    new_obj = object.__new__(type(obj))
    if hasattr(obj, "__dict__"):
        new_obj.__dict__.update(obj.__dict__)
    for name, value in updates.items():
        object.__setattr__(new_obj, name, value)
    return new_obj


def install_pytree_registrar(register: PytreeRegistrar) -> None:
    """Install a backend-specific pytree registrar for all current/future modules."""
    global _PYTREE_REGISTRAR

    _PYTREE_REGISTRAR = register
    for cls in tuple(_MODULE_CLASSES):
        register(cls)


def iter_module_classes() -> tuple[type, ...]:
    return tuple(_MODULE_CLASSES)


__all__ = [
    "Module",
    "Path",
    "PathPart",
    "PytreeRegistrar",
    "STATIC_METADATA_KEY",
    "field",
    "install_pytree_registrar",
    "iter_module_classes",
    "replace",
]
