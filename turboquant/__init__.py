"""Public package interface for TurboQuant."""

from __future__ import annotations

from .core.codec import TurboQuantCodec, TurboQuantConfig
from .core.estimator import UnbiasedInnerProductEstimator
from .core.qjl_projection import QJLProjection
from .core.scalar_quant import dequantize_scalar, quantize_scalar
from .sdk.optimize import TurboQuantizer, optimize

__version__ = "0.1.0"
__author__ = "TurboQuant Contributors"

__all__ = [
    "TurboQuantCodec",
    "TurboQuantConfig",
    "TurboQuantizer",
    "optimize",
    "quantize_scalar",
    "dequantize_scalar",
    "QJLProjection",
    "UnbiasedInnerProductEstimator",
]
