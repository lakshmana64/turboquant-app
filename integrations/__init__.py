"""Integration entry points for TurboQuant."""

from .huggingface import TurboQuantAttentionWrapper, apply_turboquant_to_hf_model
from . import plugins

__all__ = [
    "TurboQuantAttentionWrapper",
    "apply_turboquant_to_hf_model",
    "plugins",
]
