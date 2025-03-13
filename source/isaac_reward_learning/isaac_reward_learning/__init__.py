"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

from . import algorithms, config, modules, runners, storage, utils

__all__ = ["algorithms", "config", "modules", "runners", "storage", "utils"]
