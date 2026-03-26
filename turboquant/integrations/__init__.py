"""Expose integrations under ``turboquant.integrations``."""

from __future__ import annotations

from integrations import *  # type: ignore[F401,F403]
from integrations import __all__ as _integrations_all

from turboquant._alias import alias_submodules

alias_submodules(
    __name__,
    {
        "huggingface": "integrations.huggingface",
        "plugins": "integrations.plugins",
    },
)

__all__ = list(_integrations_all)
