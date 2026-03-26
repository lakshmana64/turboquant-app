"""Helpers for exposing the legacy source tree under the turboquant package."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Mapping


def alias_module(alias: str, target: str) -> ModuleType:
    """Import ``target`` and register it under ``alias`` in ``sys.modules``."""
    module = importlib.import_module(target)
    sys.modules[alias] = module
    return module


def alias_submodules(package_name: str, mapping: Mapping[str, str]) -> None:
    """Expose several target modules under a package namespace."""
    for alias_name, target in mapping.items():
        alias_module(f"{package_name}.{alias_name}", target)
