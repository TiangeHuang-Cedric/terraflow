from functools import wraps
from typing import Optional, Dict, Any


def mutattr(attr: str):
    """Decorator to mutate an attribute of a class method."""

    def _decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            setattr(self, attr, result)
            return result

        return wrapper

    return _decorator


def max_nested_dict_depth(d: Dict | None) -> int:
    """Calculate the maximum depth of the nested dictionary."""
    def _depth(d: Dict | None, current_depth: int) -> int:
        if not isinstance(d, dict) or not d:
            return current_depth
        return max(_depth(v, current_depth + 1) for v in d.values())

    return _depth(d, 0)


def nested_items(d: Dict, sep: str = ".") -> list[tuple[str, Any]]:
    """Get all nested items in a netsed dictionary as (key_path, value) tuples."""
    items = []

    def _collect_items(d: Dict, prefix: str = ""):
        for k, v in d.items():
            full_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                _collect_items(v, full_key)
            else:
                items.append((full_key, v))
    _collect_items(d)
    return items


def nested_keys(d: Dict, sep: str = ".") -> list[str]:
    """Get all nested keys in a nested dictionary."""
    items = nested_items(d, sep)
    return [k for k, v in items]


def nested_values(d: Dict, sep: str = ".") -> list[Any]:
    items = nested_items(d, sep)
    return [v for k, v in items]


class NestedDict:
    """A class to handle nested dictionaries with dot or bracket notation."""

    def __init__(
        self,
        *,
        nest: Optional[int] = None,
        d: Optional[Dict] = None,
        sep: str = "."
    ):
        if d is not None and nest != max_nested_dict_depth(d):
            raise ValueError(
                "The provided data does not match the specified nest depth."
            )
        d = d if d is not None else {}
        self._dict = d
        self._nest = max_nested_dict_depth(d) if nest is None else nest
        self._sep = sep

    def _check_depth(self, parts: list[str]):
        if self._nest is not None and len(parts) > self._nest:
            raise KeyError(
                f"Path depth {len(parts)} exceeds nest={self._nest}")

    def _ensure_path(self, parts: list[str], create_leaf: bool = False):
        self._check_depth(parts)
        node = self._dict
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if create_leaf:
            node.setdefault(parts[-1], {})
        return node

    def __setitem__(self, key: str, value):
        if isinstance(key, str) and self._sep in key:
            parts = [p for p in key.split(self._sep) if p]
            if not parts:
                raise KeyError("Empty key path")
            node = self._ensure_path(parts, create_leaf=False)
            node[parts[-1]] = value
        else:
            self._dict[key] = value

    def __getitem__(self, key: str):
        if isinstance(key, str) and self._sep in key:
            parts = [p for p in key.split(self._sep) if p]
            if not parts:
                raise KeyError("Empty key path")
            node = self._ensure_path(parts, create_leaf=True)
            return node.setdefault(parts[-1], {})
        else:
            return self._dict.setdefault(key, {})

    @property
    def nest(self) -> int | None:
        return self._nest

    def nested_keys(self) -> list[str]:
        return nested_keys(self._dict, self._sep)

    def nested_values(self) -> list[Any]:
        return nested_values(self._dict, self._sep)

    def nested_items(self) -> list[tuple[str, Any]]:
        return nested_items(self._dict, self._sep)

    def to_dict(self) -> Dict:
        return self._dict

    def repo(self):
        def _repo(d: Dict, indent: int = 1):
            for k, v in d.items():
                line = "──" * indent
                line = "└" + line
                if isinstance(v, dict):
                    print(f"  {line}{k}:")
                    indent += 1
                    _repo(v, indent)
                else:
                    print(f"  {line}{k}: {v}")

        print(f"Info: {self.__class__.__name__}")
        _repo(self.to_dict())
        return self.to_dict()


def camel_to_snake(name: str) -> str:
    """Convert a CamelCase string to snake_case string."""
    import re
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
