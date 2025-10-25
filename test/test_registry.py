"""
Comprehensive test suite for the Registry system.

This module provides extensive test coverage for the nested Registry system,
including basic registration, nested hierarchies, path-based navigation,
decorator patterns, and path separator support (dot vs slash notation).

Test Organization:
    - TestRegistryMeta: Tests for metadata structure
    - TestBasicRegistry: Core registration and retrieval functionality
    - TestNestedRegistry: Hierarchical registry features
    - TestDecoratorPattern: Decorator-based registration
    - TestBracketNotation: Dictionary-style access patterns
    - TestPathBasedNavigation: Path string navigation methods
    - TestPathSeparators: Dot (.) vs slash (/) separator support
    - TestRegisterAtPath: Path-based registration convenience methods
    - TestStringRepresentation: Registry object attributes
    - TestIntegration: Complex real-world usage scenarios
    - TestEdgeCases: Error handling and boundary conditions

Coverage Summary:
    This test suite consolidates and extends tests from:
    - test_registry.py: Basic and nested registry functionality
    - test_smart_registry.py: Path-based navigation and registration
    - test_path_separators.py: Separator compatibility (dot/slash)
    
    Total: 58 test cases covering all public API methods and error paths.
"""

import pytest
from terraflow.manager.registry import Registry, RegistryMeta


class TestRegistryMeta:
    """Test cases for RegistryMeta class."""

    def test_init_with_all_parameters(self):
        """Test RegistryMeta initialization with all parameters."""
        meta = RegistryMeta(
            name="test",
            description="test description",
            tag="test_tag"
        )
        assert meta.name == "test"
        assert meta.description == "test description"
        assert meta.tag == "test_tag"

    def test_init_with_minimal_parameters(self):
        """Test RegistryMeta initialization with only required parameters."""
        meta = RegistryMeta(name="test")
        assert meta.name == "test"
        assert meta.description == ""
        assert meta.tag == ""


class TestBasicRegistry:
    """Test cases for basic Registry functionality."""

    def test_init(self):
        """Test Registry initialization."""
        reg = Registry("TestRegistry")
        assert reg._registry_type == "TestRegistry"
        assert len(reg._factories) == 0
        assert len(reg._sub_registries) == 0
        assert reg._parent is None
        assert not reg._auto_discovered

    def test_register_class(self):
        """Test registering a class."""
        reg = Registry("Models")

        class TestModel:
            pass

        result = reg.register("test", TestModel)
        assert result is TestModel
        assert reg.has("test")

    def test_register_function(self):
        """Test registering a function."""
        reg = Registry("Functions")

        def test_func():
            return "test"

        reg.register("func", test_func)
        assert reg.has("func")
        assert reg.get("func") is test_func

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate names raises ValueError."""
        reg = Registry("Models")

        class Model1:
            pass

        class Model2:
            pass

        reg.register("model", Model1)

        with pytest.raises(ValueError, match="already registered"):
            reg.register("model", Model2)

    def test_available_returns_registered_items(self):
        """Test that available() returns list of registered item names."""
        reg = Registry("Models")

        class Model1:
            pass

        class Model2:
            pass

        reg.register("Model1", Model1)
        reg.register("Model2", Model2)

        available = reg.available()
        assert len(available) == 2
        assert "Model1" in available
        assert "Model2" in available

    def test_create_instantiates_registered_class(self):
        """Test that create() instantiates a registered class with arguments."""
        reg = Registry("Models")

        class TestModel:
            def __init__(self, value):
                self.value = value

        reg.register("test", TestModel)
        instance = reg.create("test", value=42)

        assert isinstance(instance, TestModel)
        assert instance.value == 42

    def test_get_nonexistent_raises_keyerror(self):
        """Test that accessing non-existent item raises KeyError."""
        reg = Registry("Models")

        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_description_retrieval(self):
        """Test retrieving description of registered item."""
        reg = Registry("Models")

        class TestModel:
            pass

        reg.register("test", TestModel, description="A test model")
        assert reg.description("test") == "A test model"

    def test_tag_registration_and_retrieval(self):
        """Test registering items with tags and retrieving by tag."""
        reg = Registry("Operations")

        def op1():
            pass

        def op2():
            pass

        def op3():
            pass

        reg.register("op1", op1, tag="math")
        reg.register("op2", op2, tag="math")
        reg.register("op3", op3, tag="string")

        math_ops = reg.lookup_by_tag("math")
        assert len(math_ops) == 2
        assert "op1" in math_ops
        assert "op2" in math_ops


class TestNestedRegistry:
    """Test cases for nested Registry functionality."""

    def test_register_subregistry(self):
        """Test creating a sub-registry."""
        parent = Registry("Parent")
        child = parent.register_subregistry("child")

        assert isinstance(child, Registry)
        assert child._registry_type == "child"
        assert child._parent is parent
        assert parent.has_subregistry("child")

    def test_register_subregistry_with_custom_type(self):
        """Test creating sub-registry with custom registry type."""
        parent = Registry("Parent")
        child = parent.register_subregistry(
            "child",
            registry_type="CustomType"
        )

        assert child._registry_type == "CustomType"
        assert child._parent is parent

    def test_get_subregistry(self):
        """Test retrieving an existing sub-registry."""
        parent = Registry("Parent")
        child = parent.register_subregistry("child")

        retrieved = parent.get_subregistry("child")
        assert retrieved is child

    def test_get_nonexistent_subregistry_raises_keyerror(self):
        """Test that accessing non-existent sub-registry raises KeyError."""
        parent = Registry("Parent")

        with pytest.raises(KeyError, match="Sub-registry"):
            parent.get_subregistry("nonexistent")

    def test_available_subregistries(self):
        """Test listing all available sub-registries."""
        parent = Registry("Parent")
        parent.register_subregistry("child1")
        parent.register_subregistry("child2")

        subs = parent.available_subregistries()
        assert len(subs) == 2
        assert "child1" in subs
        assert "child2" in subs

    def test_multilevel_nesting(self):
        """Test creating deeply nested registry hierarchy."""
        root = Registry("Root")
        level1 = root.register_subregistry("level1")
        level2 = level1.register_subregistry("level2")
        level3 = level2.register_subregistry("level3")

        assert level3._parent is level2
        assert level2._parent is level1
        assert level1._parent is root
        assert root._parent is None

    def test_get_path_single_level(self):
        """Test path retrieval for single-level registry."""
        root = Registry("Root")
        assert root.get_path() == "Root"

    def test_get_path_nested_levels(self):
        """Test path retrieval for nested registry hierarchy."""
        root = Registry("Root")
        level1 = root.register_subregistry("level1")
        level2 = level1.register_subregistry("level2")

        assert root.get_path() == "Root"
        assert level1.get_path() == "Root.level1"
        assert level2.get_path() == "Root.level1.level2"

    def test_register_in_subregistry(self):
        """Test that items registered in sub-registry are isolated from parent."""
        parent = Registry("Parent")
        child = parent.register_subregistry("child")

        class TestClass:
            pass

        child.register("test", TestClass)

        assert child.has("test")
        assert not parent.has("test")

    def test_isolation_between_parent_and_child(self):
        """Test complete isolation of parent and child registrations."""
        parent = Registry("Parent")

        class ParentClass:
            pass

        class ChildClass:
            pass

        parent.register("parent_item", ParentClass)
        child = parent.register_subregistry("child")
        child.register("child_item", ChildClass)

        assert parent.has("parent_item")
        assert not parent.has("child_item")
        assert child.has("child_item")
        assert not child.has("parent_item")


class TestDecoratorPattern:
    """Test cases for decorator-based registration."""

    def test_decorator_usage(self):
        """Test using Registry.register as a decorator."""
        reg = Registry("Models")

        @reg.register("TestModel", description="A test model")
        class TestModel:
            pass

        assert reg.has("TestModel")
        assert reg.get("TestModel") is TestModel
        assert reg.description("TestModel") == "A test model"

    def test_decorator_with_tag(self):
        """Test using decorator with tag parameter."""
        reg = Registry("Operations")

        @reg.register("compute", tag="math")
        def compute():
            return 42

        assert reg.has("compute")
        assert reg.tag("compute") == "math"

    def test_decorator_in_subregistry(self):
        """Test using decorator on sub-registry items."""
        parent = Registry("Parent")
        child = parent.register_subregistry("child")

        @child.register("Item", description="Child item")
        class Item:
            pass

        assert child.has("Item")
        assert child.get("Item") is Item
        assert child.description("Item") == "Child item"


class TestBracketNotation:
    """Test cases for bracket notation access."""

    def test_getitem_subregistry(self):
        """Test accessing sub-registry using bracket notation."""
        parent = Registry("Parent")
        child = parent.register_subregistry("child")

        retrieved = parent["child"]
        assert retrieved is child

    def test_getitem_nonexistent_raises_keyerror(self):
        """Test that bracket notation raises KeyError for non-existent sub-registry."""
        parent = Registry("Parent")

        with pytest.raises(KeyError):
            _ = parent["nonexistent"]

    def test_getitem_subregistry_priority(self):
        """Test that bracket notation only accesses sub-registries.

        Note: The current implementation of __getitem__ only supports
        accessing sub-registries, not factories.
        """
        parent = Registry("Parent")

        class TestClass:
            pass

        parent.register("item", TestClass)
        child = parent.register_subregistry("item")

        # Should return the sub-registry
        result = parent["item"]
        assert result is child


class TestPathBasedNavigation:
    """Test cases for path-based registry navigation."""

    def test_navigate_simple_path(self):
        """Test navigating to a simple single-level path."""
        reg = Registry("Root")

        sub = reg.navigate("level1")
        assert sub._registry_type == "level1"
        assert sub._parent is reg
        assert reg.has_subregistry("level1")

    def test_navigate_nested_path(self):
        """Test navigating to deeply nested path."""
        reg = Registry("Root")

        deep = reg.navigate("a.b.c.d")

        assert deep._registry_type == "d"
        assert deep.get_path() == "Root.a.b.c.d"

        # Verify all intermediate levels were created
        assert reg.has_subregistry("a")
        a = reg.get_subregistry("a")
        assert a.has_subregistry("b")
        b = a.get_subregistry("b")
        assert b.has_subregistry("c")

    def test_navigate_is_idempotent(self):
        """Test that navigating to the same path returns the same object."""
        reg = Registry("Root")

        sub1 = reg.navigate("path.to.registry")
        sub2 = reg.navigate("path.to.registry")

        assert sub1 is sub2

    def test_navigate_with_description(self):
        """Test navigating with description parameter."""
        reg = Registry("Root")

        sub = reg.navigate(
            "models.ml",
            description="Machine Learning Models"
        )

        assert sub._description == "Machine Learning Models"

    def test_navigate_empty_path_returns_self(self):
        """Test that navigating with empty path returns the registry itself."""
        reg = Registry("Root")

        result = reg.navigate("")
        assert result is reg

    def test_navigate_default_path_returns_self(self):
        """Test that navigating without path argument returns the registry itself."""
        reg = Registry("Root")

        result = reg.navigate()
        assert result is reg


class TestPathSeparators:
    """Test cases for dot and slash path separator support."""

    def test_navigate_with_dot_separator(self):
        """Test navigation using dot separator (original style)."""
        reg = Registry("Root")

        sub = reg.navigate("a.b.c")
        assert sub.get_path() == "Root.a.b.c"

    def test_navigate_with_slash_separator(self):
        """Test navigation using slash separator (filesystem style)."""
        reg = Registry("Root")

        sub = reg.navigate("a/b/c")
        assert sub.get_path() == "Root.a.b.c"

    def test_navigate_separators_equivalence(self):
        """Test that dot and slash separators produce identical results."""
        reg = Registry("Root")

        sub1 = reg.navigate("x.y.z")
        sub2 = reg.navigate("x/y/z")

        assert sub1 is sub2
        assert sub1.get_path() == sub2.get_path()

    def test_navigate_mixed_separators(self):
        """Test navigation with mixed dot and slash separators."""
        reg = Registry("Root")

        sub = reg.navigate("a/b.c/d")
        assert sub.get_path() == "Root.a.b.c.d"

    def test_create_from_path_with_dot(self):
        """Test create_from_path using dot separator."""
        reg = Registry("Root")

        class TestClass:
            def __init__(self, value):
                self.value = value

        reg.register_at("items", "Test", TestClass)

        instance = reg.create_from_path("items.Test", value=42)
        assert isinstance(instance, TestClass)
        assert instance.value == 42

    def test_create_from_path_with_slash(self):
        """Test create_from_path using slash separator."""
        reg = Registry("Root")

        class TestClass:
            def __init__(self, value):
                self.value = value

        reg.register_at("items", "Test", TestClass)

        instance = reg.create_from_path("items/Test", value=42)
        assert isinstance(instance, TestClass)
        assert instance.value == 42

    def test_create_from_path_nested_with_slash(self):
        """Test create_from_path with deeply nested path using slash separator."""
        reg = Registry("Root")

        class Model:
            def __init__(self, name):
                self.name = name

        reg.register_at("models.ml.transformers", "BERT", Model)

        instance = reg.create_from_path(
            "models/ml/transformers/BERT",
            name="bert-base"
        )
        assert isinstance(instance, Model)
        assert instance.name == "bert-base"

    def test_get_from_path_with_dot(self):
        """Test get_from_path using dot separator."""
        reg = Registry("Root")

        class TestClass:
            pass

        reg.register_at("items", "Test", TestClass)

        factory = reg.get_from_path("items.Test")
        assert factory is TestClass

    def test_get_from_path_with_slash(self):
        """Test get_from_path using slash separator."""
        reg = Registry("Root")

        class TestClass:
            pass

        reg.register_at("items", "Test", TestClass)

        factory = reg.get_from_path("items/Test")
        assert factory is TestClass

    def test_get_from_path_nested_with_slash(self):
        """Test get_from_path with deeply nested path using slash separator."""
        reg = Registry("Root")

        class DeepClass:
            pass

        reg.register_at("a.b.c", "Deep", DeepClass)

        factory = reg.get_from_path("a/b/c/Deep")
        assert factory is DeepClass


class TestRegisterAtPath:
    """Test cases for path-based registration functionality."""

    def test_register_at_path_with_class_name_inference(self):
        """Test register_at with automatic class name inference."""
        reg = Registry("Root")

        @reg.register_at("models.ml")
        class Transformer:
            pass

        target = reg.navigate("models.ml")
        assert target.has("Transformer")
        assert target.get("Transformer") is Transformer

    def test_register_at_path_with_explicit_name(self):
        """Test register_at with explicitly specified name."""
        reg = Registry("Root")

        @reg.register_at("models.vision", "ResNet50")
        class ResNet:
            pass

        target = reg.navigate("models.vision")
        assert target.has("ResNet50")
        assert not target.has("ResNet")

    def test_register_at_path_direct_call(self):
        """Test register_at using direct function call (not as decorator)."""
        reg = Registry("Root")

        class MyClass:
            pass

        reg.register_at("components.utils", "Tool", MyClass)

        target = reg.navigate("components.utils")
        assert target.has("Tool")
        assert target.get("Tool") is MyClass

    def test_register_at_path_auto_name_direct(self):
        """Test register_at with automatic name inference in direct call."""
        reg = Registry("Root")

        class AutoNamed:
            pass

        reg.register_at("items", factory=AutoNamed)

        target = reg.navigate("items")
        assert target.has("AutoNamed")

    def test_register_at_path_with_description(self):
        """Test register_at with description parameter."""
        reg = Registry("Root")

        @reg.register_at("models", "Test", description="A test model")
        class TestModel:
            pass

        target = reg.navigate("models")
        assert target.description("Test") == "A test model"

    def test_register_at_path_creates_intermediate_registries(self):
        """Test that register_at automatically creates intermediate registries."""
        reg = Registry("Root")

        @reg.register_at("a.b.c.d")
        class DeepClass:
            pass

        # Verify all intermediate levels exist
        assert reg.has_subregistry("a")
        a = reg.get_subregistry("a")
        assert a.has_subregistry("b")
        b = a.get_subregistry("b")
        assert b.has_subregistry("c")
        c = b.get_subregistry("c")
        assert c.has_subregistry("d")
        d = c.get_subregistry("d")
        assert d.has("DeepClass")


class TestStringRepresentation:
    """Test cases for registry object properties and basic attributes."""

    def test_registry_type_attribute(self):
        """Test that registry_type attribute is correctly set."""
        reg = Registry("TestReg")
        assert reg._registry_type == "TestReg"

    def test_registry_with_items_attributes(self):
        """Test registry attributes after adding items and sub-registries."""
        reg = Registry("TestReg")

        class Item1:
            pass

        class Item2:
            pass

        reg.register("item1", Item1)
        reg.register("item2", Item2)
        reg.register_subregistry("child")

        assert len(reg._factories) == 2
        assert len(reg._sub_registries) == 1
        assert reg._registry_type == "TestReg"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_mixed_traditional_and_path_based_usage(self):
        """Test mixing traditional and path-based registration methods."""
        reg = Registry("Root")

        # Traditional approach
        sub1 = reg.register_subregistry("traditional")

        class TraditionalClass:
            pass

        sub1.register("item", TraditionalClass)

        # Path-based approach
        @reg.register_at("modern")
        class ModernClass:
            pass

        # Both approaches work seamlessly
        assert reg.get_subregistry("traditional").has("item")
        assert reg.navigate("modern").has("ModernClass")

        # Path-based access to traditionally created items
        trad_instance = reg.create_from_path("traditional.item")
        assert isinstance(trad_instance, TraditionalClass)

    def test_complex_hierarchical_structure(self):
        """Test building and navigating complex hierarchical registry structure."""
        app = Registry("Application")

        # Create complex hierarchy
        paths = [
            "core.data.loaders",
            "core.data.processors",
            "core.models.ml.supervised",
            "core.models.ml.unsupervised",
            "plugins.visualization.charts",
            "plugins.visualization.maps",
            "plugins.export.pdf",
            "plugins.export.html",
        ]

        for path in paths:
            app.navigate(path)

        # Register components
        @app.register_at("core.data.loaders")
        class CSVLoader:
            pass

        @app.register_at("core.models.ml.supervised", "RandomForest")
        class RF:
            pass

        @app.register_at("plugins.visualization.charts")
        class BarChart:
            pass

        # Verify correct access
        assert app.get_from_path("core.data.loaders.CSVLoader") is CSVLoader
        assert app.get_from_path(
            "core.models.ml.supervised.RandomForest") is RF
        assert app.get_from_path(
            "plugins.visualization.charts.BarChart") is BarChart

        # Verify hierarchy structure
        assert "core" in app.available_subregistries()
        core = app.get_subregistry("core")
        assert "data" in core.available_subregistries()
        assert "models" in core.available_subregistries()

    def test_path_separators_across_all_methods(self):
        """Test that both separators work consistently across all path-based methods."""
        reg = Registry("Root")

        class TestItem:
            def __init__(self, val):
                self.val = val

        # Use dot separator for registration
        reg.register_at("group.subgroup", "Item", TestItem)

        # Use slash separator for navigation
        target1 = reg.navigate("group/subgroup")
        assert target1.has("Item")

        # Use mixed separators for get_from_path
        factory = reg.get_from_path("group/subgroup.Item")
        assert factory is TestItem

        # Use slash separator for create_from_path
        instance = reg.create_from_path("group/subgroup/Item", val=100)
        assert isinstance(instance, TestItem)
        assert instance.val == 100


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_create_from_path_insufficient_parts_raises_error(self):
        """Test that create_from_path raises error for single-part path."""
        reg = Registry("Root")

        with pytest.raises(ValueError, match="at least registry and name"):
            reg.create_from_path("SinglePart")

    def test_get_from_path_insufficient_parts_raises_error(self):
        """Test that get_from_path raises error for single-part path."""
        reg = Registry("Root")

        with pytest.raises(ValueError, match="at least registry and name"):
            reg.get_from_path("SinglePart")

    def test_lookup_by_nonexistent_tag_raises_error(self):
        """Test that lookup_by_tag raises error for non-existent tag."""
        reg = Registry("Root")

        def item():
            pass

        reg.register("item", item, tag="existing")

        with pytest.raises(KeyError, match="No item with tag"):
            reg.lookup_by_tag("nonexistent")

    def test_double_registration_of_subregistry_raises_error(self):
        """Test that registering same sub-registry name twice raises error."""
        reg = Registry("Root")

        reg.register_subregistry("child")

        with pytest.raises(ValueError, match="already registered"):
            reg.register_subregistry("child")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
