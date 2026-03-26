"""Expose the core implementation under ``turboquant.core``."""

from __future__ import annotations

from core import *  # type: ignore[F401,F403]
from core import __all__ as _core_all

from turboquant._alias import alias_submodules

alias_submodules(
    __name__,
    {
        "codec": "core.codec",
        "estimator": "core.estimator",
        "mixed_precision": "core.mixed_precision",
        "monitoring": "core.monitoring",
        "optimized": "core.optimized",
        "qjl_projection": "core.qjl_projection",
        "residual": "core.residual",
        "scalar_quant": "core.scalar_quant",
        "streaming": "core.streaming",
    },
)

__all__ = list(_core_all)
