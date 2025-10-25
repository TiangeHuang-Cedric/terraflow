from token import OP
from typing import Callable, Dict, Set, TypeVar, Generic, Optional, Union, \
    overload
import importlib
import pkgutil
from pathlib import Path

T = TypeVar('T')


class RegistryMeta:
    """Metadata for registered items."""

    def __init__(self, name: str, description: str = "", tag: str = ""):
        self.name = name
        self.description = description
        self.tag = tag


# Global pending registration queue
_pending_registrations = []


class Registry(Generic[T]):
    """Light-weight registry for pluggable simulation components with nested registry support."""
    _registry_type: str
    _description: str
    _factories: Dict[str, Callable[..., T]]
    _meta: Dict[str, RegistryMeta]
    _sub_registries: Dict[str, 'Registry']
    _parent: Optional['Registry']
    _auto_discovered: bool  # Flag to track if auto-discovery has been performed

    def __init__(
        self,
        registry_type: str,
        parent: Optional['Registry'] = None,
        description: str = ""
    ):
        """
        Initialize a Registry.

        Args:
            registry_type: Type/category name for this registry
            parent: Optional parent registry for nested hierarchies
            description: Optional description of this registry
        """
        self._registry_type = registry_type
        self._description = description
        self._factories = {}
        self._meta = {}
        self._sub_registries = {}
        self._parent = parent
        self._auto_discovered = False

    def _ensure_discovered(self):
        """Ensure auto-discovery has been performed (only once)"""
        if not self._auto_discovered:
            self._auto_discover()
            self._auto_discovered = True

    def _auto_discover(self):
        """Auto-discover and import all modules containing registration decorators"""
        # Only run auto-discovery on the root registry
        if self._parent is not None:
            # If this is a sub-registry, delegate to root
            root = self
            while root._parent is not None:
                root = root._parent
            root._ensure_discovered()
            return

        try:
            import terraflow
            package_path = Path(terraflow.__file__).parent

            # Recursively scan all Python modules
            for module_info in pkgutil.walk_packages(
                    [str(package_path)],
                    prefix='terraflow.'
            ):
                try:
                    # Import module (triggers decorator execution)
                    importlib.import_module(module_info.name)
                except Exception:
                    # Silently ignore import errors to avoid breaking on optional dependencies
                    # or modules that require specific initialization order
                    pass

            # Process all pending registrations
            self._process_pending_registrations()

        except Exception:
            pass

    def _process_pending_registrations(self):
        """Process all pending registration items"""
        global _pending_registrations

        for path, name, factory, description, tag in _pending_registrations:
            try:
                # Use navigate from root (self) to get target registry
                target_registry = self.navigate(path)
                # Register directly on target registry
                target_registry.register(
                    name, factory, description=description, tag=tag)
            except Exception:
                pass

        _pending_registrations.clear()

    def __getitem__(self, name: str) -> 'Registry':
        """Allow bracket notation for accessing sub-registries."""
        return self.get_subregistry(name)

    @overload
    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        description: str = "",
        tag: str = "",
    ) -> Callable[..., T]:
        ...

    @overload
    def register(
        self,
        name: str,
        factory: None = None,
        *,
        description: str = "",
        tag: str = "",
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        ...

    def register(
        self,
        name: str,
        factory: Optional[Callable[..., T]] = None,
        *,
        description: str = "",
        tag: str = "",
    ) -> Union[
            Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
        """
        Register a factory function or class. Can be used as a decorator.

        Usage:
            # Direct registration:
            registry.register("MyClass", MyClass)

            # As decorator:
            @registry.register("MyClass", description="...")
            class MyClass:
                pass

        Args:
            name: Unique name for the registered item
            factory: Callable that creates instances (None when used as decorator)
            description: Optional description of the registered item
            tag: Optional tag for categorization

        Returns:
            The factory function (allows use as decorator)
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            key = name
            if key in self._factories:
                raise ValueError(
                    f"{self._registry_type} '{name}' is already registered."
                )
            self._factories[key] = func
            self._meta[key] = RegistryMeta(
                name=name, description=description, tag=tag)
            return func

        if factory is None:
            # Used as @register("name", description="...")
            return decorator
        else:
            # Used as register("name", factory)
            return decorator(factory)

    def has(self, name: str) -> bool:
        """Check if a factory is registered."""
        self._ensure_discovered()  # Auto-discovery
        return name in self._factories

    def available(self) -> tuple[str, ...]:
        """Get names of all registered factories."""
        self._ensure_discovered()  # Auto-discovery
        return tuple(meta.name for meta in self._meta.values()
                     if meta.name in self._factories)

    def create(self, key: str, *args, **kwargs) -> T:
        """Create an instance using a registered factory."""
        self._ensure_discovered()  # Auto-discovery
        lookup_key = key
        if lookup_key not in self._factories:
            raise KeyError(
                f"'{key}' not found in {self._registry_type}. "
                f"Available: {self.available()}"
            )
        return self._factories[lookup_key](*args, **kwargs)

    def get(self, key: str) -> Callable[..., T]:
        """Get a registered factory by name."""
        self._ensure_discovered()  # Auto-discovery
        lookup_key = key
        if lookup_key not in self._factories:
            raise KeyError(
                f"'{key}' not found in {self._registry_type}. "
                f"Available: {self.available()}"
            )
        return self._factories[lookup_key]

    def description(self, name: str) -> str:
        """Get description of a registered item."""
        self._ensure_discovered()  # Auto-discovery
        key = name
        if key not in self._meta:
            raise KeyError(f"'{name}' not found in {self._registry_type}")
        return self._meta[key].description

    def tag(self, name: str) -> str:
        """Get tag of a registered item."""
        self._ensure_discovered()  # Auto-discovery
        key = name
        if key not in self._meta:
            raise KeyError(f"'{name}' not found in {self._registry_type}")
        return self._meta[key].tag

    def lookup_by_tag(self, tag: str) -> Dict[str, Callable[..., T]]:
        """Lookup a registered factory by tag."""
        self._ensure_discovered()  # Auto-discovery
        result = {}
        for key, meta in self._meta.items():
            if meta.tag == tag and key in self._factories:
                result[key] = self._factories[key]
        if not result:
            raise KeyError(
                f"No item with tag '{tag}' found in {self._registry_type}. "
                f"Available tags: {[m.tag for m in self._meta.values()]}"
            )
        return result

    def register_subregistry(
        self,
        name: str,
        registry_type: Optional[str] = None,
        *,
        description: str = "",
        tag: str = "",
    ) -> 'Registry':
        """
        Create and register a sub-registry.

        Args:
            name: Unique name for the sub-registry
            registry_type: Type name for the sub-registry (defaults to name)
            description: Optional description
            tag: Optional tag for categorization

        Returns:
            The newly created sub-registry
        """
        key = name
        if key in self._sub_registries:
            raise ValueError(
                f"Sub-registry '{name}' is already registered in {self._registry_type}."
            )

        sub_type = registry_type or name
        sub_registry = Registry(registry_type=sub_type,
                                parent=self, description=description)
        self._sub_registries[key] = sub_registry
        self._meta[key] = RegistryMeta(
            name=name, description=description, tag=tag)
        return sub_registry

    def get_subregistry(self, name: str) -> 'Registry':
        """
        Get a sub-registry by name.

        Args:
            name: Name of the sub-registry

        Returns:
            The sub-registry
        """
        key = name
        if key not in self._sub_registries:
            # Get the root registry to check auto-discovery status
            root = self
            while root._parent is not None:
                root = root._parent

            # Try auto-discovery before failing (only if not already done on root)
            if not root._auto_discovered:
                root._ensure_discovered()

            # Check again after auto-discovery
            if key not in self._sub_registries:
                raise KeyError(
                    f"Sub-registry '{name}' not found in {self._registry_type}. "
                    f"Available: {self.available_subregistries()}"
                )
        return self._sub_registries[key]

    def has_subregistry(self, name: str) -> bool:
        """Check if a sub-registry exists."""
        return name in self._sub_registries

    def available_subregistries(self) -> tuple[str, ...]:
        """Get names of all registered sub-registries."""
        return tuple(meta.name for meta in self._meta.values()
                     if meta.name in self._sub_registries)

    def navigate(
        self,
        path: str = "",
        *,
        description: str = "",
        tag: str = "",
    ) -> 'Registry':
        """
        Navigate to or create a nested registry path.

        Args:
            path: Path to navigate, can use dot or slash separators
                  (e.g., "equations.atmosphere.shallow_water" or "equations/atmosphere/shallow_water")
            description: Description for the final registry in the path
            tag: Tag for the final registry

        Returns:
            The registry at the end of the path
        """
        if not path:
            return self

        # Support both dot and slash separators
        # Replace slashes with dots for unified processing
        normalized_path = path.replace('/', '.')
        parts = normalized_path.split('.')
        current = self

        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)

            if not current.has_subregistry(part):
                # Create intermediate registry
                desc = description if is_last else f"{part} registry"
                t = tag if is_last else ""
                current = current.register_subregistry(
                    part, description=desc, tag=t)
            else:
                # Get existing sub-registry
                current = current.get_subregistry(part)

        return current

    def register_at(
        self,
        path: str,
        name: Optional[str] = None,
        factory: Optional[Callable[..., T]] = None,
        *,
        description: str = "",
        tag: str = "",
    ) -> Union[
            Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
        """
        Register at a specific path in the registry hierarchy.

        Args:
            path: Dot-separated path to the target registry
            name: Name for registration (uses factory.__name__ if None)
            factory: Factory to register (None when used as decorator)
            description: Optional description
            tag: Optional tag

        Returns:
            The factory (for decorator usage)
        """
        target_registry = self.navigate(path)

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            reg_name = name or func.__name__
            target_registry.register(reg_name, func, description=description,
                                     tag=tag)
            return func

        if factory is None:
            return decorator
        else:
            return decorator(factory)

    def create_from_path(self, full_path: str, *args, **kwargs) -> T:
        """
        Create instance from a full path like "equations.atmosphere.VectorInvariant"
        or "equations/atmosphere/VectorInvariant".

        Args:
            full_path: Full path including the factory name, supports dot or slash separators
            *args, **kwargs: Arguments to pass to the factory

        Returns:
            Created instance
        """
        self._ensure_discovered()  # Auto-discovery

        # Support both dot and slash separators
        normalized_path = full_path.replace('/', '.')
        parts = normalized_path.split('.')
        if len(parts) < 2:
            raise ValueError(
                f"Path must include at least registry and name: '{full_path}'"
            )

        registry_path = '.'.join(parts[:-1])
        factory_name = parts[-1]

        target_registry = self.navigate(registry_path)
        return target_registry.create(factory_name, *args, **kwargs)

    def get_from_path(self, full_path: str) -> Callable[..., T]:
        """
        Get factory from a full path like "equations.atmosphere.VectorInvariant"
        or "equations/atmosphere/VectorInvariant".

        Args:
            full_path: Full path including the factory name, supports dot or slash separators

        Returns:
            The factory function or class
        """
        self._ensure_discovered()  # Auto-discovery

        # Support both dot and slash separators
        normalized_path = full_path.replace('/', '.')
        parts = normalized_path.split('.')
        if len(parts) < 2:
            raise ValueError(
                f"Path must include at least registry and name: '{full_path}'"
            )

        registry_path = '.'.join(parts[:-1])
        factory_name = parts[-1]

        target_registry = self.navigate(registry_path)
        return target_registry.get(factory_name)

    def get_path(self) -> str:
        """Get the full path of this registry."""
        if self._parent is None:
            return self._registry_type

        parent_path = self._parent.get_path()
        return f"{parent_path}.{self._registry_type}"


"""Registry Hierarchy for TerraFlow"""

terraflow = Registry("TerraFlow")


def pre_register(reg: Registry):
    reg.navigate("equation.atmosphere", description="Atmosphere Equations")
    reg.navigate("equation.ocean", description="Ocean Equations")
    reg.navigate("mesh.structured", description="Structured Mesh Types")
    reg.navigate("mesh.unstructured", description="Unstructured Mesh Types")
    reg.navigate(
        "operator.fvm.stagc4odm",
        description="Finite Volume Staggered C-grid Method on Orthogonal Dual Mesh Operators"
    )


def bind(
        path: Optional[str] = None,
        name: Optional[str] = None,
        factory: Optional[Callable[..., T]] = None,
        *,
        description: str = "",
        tag: str = "",
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Lazy registration decorator: collects registration info and registers on first access.

    This avoids circular import issues and eliminates the need to explicitly import all modules.

    Usage:
        @bind("operator.fvm.stagc4odm", tag="divergence")
        def div_at_prime_cell(mesh):
            ...
    """

    def decorator(obj: Callable[..., T]) -> Callable[..., T]:
        reg_name = name or obj.__name__
        # Directly register to the terraflow root registry
        terraflow.register_at(path, reg_name, obj,
                              description=description, tag=tag)
        return obj

    if factory is None:
        return decorator
    else:
        return decorator(factory)


def navigate(path: Optional[str] = None) -> Registry:
    """
    Navigate to or create a nested registry path from the root.

    Args:
        path: The dot-separated path to navigate or create.

    Returns:
        The registry at the specified path.
    """
    return terraflow.navigate(path)


def navigate_by_tag(
    path: Optional[str] = None,
    *,
    tag: str
) -> Dict[str, Callable[..., T]]:
    """
    Navigate to a sub-registry by tag.

    Args:
        path: The dot-separated path to navigate.
        tag: The tag to filter sub-registries.

    Returns:
        The registry at the specified path and tag.
    """
    if path is None:
        base_registry = terraflow
    else:
        base_registry = terraflow.navigate(path)
    return base_registry.lookup_by_tag(tag)
