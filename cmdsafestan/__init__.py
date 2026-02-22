"""Python package for cmdsafestan."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SafeStanResult",
    "SafeStanRuntime",
    "evaluate_model_string",
    "init",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        api = import_module("cmdsafestan.api")
        return getattr(api, name)
    raise AttributeError(f"module 'cmdsafestan' has no attribute {name!r}")
