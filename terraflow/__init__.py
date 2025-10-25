import terraflow.manager
import terraflow.util
import terraflow.mesh
import terraflow.operator
import terraflow.equation
import terraflow.model

from .manager.registry import bind, navigate, navigate_by_tag
from .manager.registry import terraflow, pre_register

# Pre-create registry structure
pre_register(terraflow)

# Trigger auto-discovery to load all modules immediately
# This ensures all @bind decorated items are registered at import time
terraflow._ensure_discovered()

all = ["bind", "navigate", "navigate_by_tag"]

__version__ = "0.1.0"
