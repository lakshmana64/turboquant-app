"""Expose plugins under ``turboquant.integrations.plugins``."""

from __future__ import annotations

from integrations.plugins import *  # type: ignore[F401,F403]
from integrations.plugins import __all__ as _plugins_all

from turboquant._alias import alias_submodules

alias_submodules(
    __name__,
    {
        "__main__": "integrations.plugins.__main__",
        "examples": "integrations.plugins.examples",
        "haystack_plugin": "integrations.plugins.haystack_plugin",
        "langchain_plugin": "integrations.plugins.langchain_plugin",
        "llama_index_plugin": "integrations.plugins.llama_index_plugin",
        "ollama": "integrations.plugins.ollama",
        "ollama_cli": "integrations.plugins.ollama_cli",
        "openai_plugin": "integrations.plugins.openai_plugin",
        "registry": "integrations.plugins.registry",
        "sentence_transformers_plugin": "integrations.plugins.sentence_transformers_plugin",
        "tgi_plugin": "integrations.plugins.tgi_plugin",
        "vllm_plugin": "integrations.plugins.vllm_plugin",
    },
)

__all__ = list(_plugins_all)
