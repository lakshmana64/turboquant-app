"""
TurboQuant plugins for embedding providers and RAG frameworks.

The optional integrations below are imported lazily so the base package can be
installed without the full plugin dependency set.
"""

from .ollama import CompressionResult as CompressionResult
from .ollama import OllamaPlugin as OllamaPlugin
from .ollama import OllamaPluginConfig as OllamaPluginConfig
from .ollama import compress as compress
from .ollama import query as query
from .registry import (
    PluginInfo as PluginInfo,
    PluginRegistry as PluginRegistry,
    get_plugin_info as get_plugin_info,
    get_registry as get_registry,
    list_plugins as list_plugins,
    load_plugin as load_plugin,
)

__all__ = [
    "CompressionResult",
    "OllamaPlugin",
    "OllamaPluginConfig",
    "compress",
    "query",
    "PluginInfo",
    "PluginRegistry",
    "get_registry",
    "list_plugins",
    "load_plugin",
    "get_plugin_info",
]

try:
    from .openai_plugin import OpenAIPlugin as OpenAIPlugin
    from .openai_plugin import OpenAIPluginConfig as OpenAIPluginConfig

    OPENAI_AVAILABLE = True
    __all__.extend(["OpenAIPlugin", "OpenAIPluginConfig", "OPENAI_AVAILABLE"])
except ImportError:
    OPENAI_AVAILABLE = False
    __all__.append("OPENAI_AVAILABLE")

try:
    from .sentence_transformers_plugin import (
        SentenceTransformersPlugin as SentenceTransformersPlugin,
        SentenceTransformersPluginConfig as SentenceTransformersPluginConfig,
    )

    SENTENCE_TRANSFORMERS_AVAILABLE = True
    __all__.extend(
        [
            "SentenceTransformersPlugin",
            "SentenceTransformersPluginConfig",
            "SENTENCE_TRANSFORMERS_AVAILABLE",
        ]
    )
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    __all__.append("SENTENCE_TRANSFORMERS_AVAILABLE")

try:
    from .llama_index_plugin import (
        TurboQuantEmbedding as TurboQuantEmbedding,
        TurboQuantVectorStore as TurboQuantVectorStore,
        create_compressed_index as create_compressed_index,
    )

    LLAMAINDEX_AVAILABLE = True
    __all__.extend(
        [
            "TurboQuantEmbedding",
            "TurboQuantVectorStore",
            "create_compressed_index",
            "LLAMAINDEX_AVAILABLE",
        ]
    )
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    __all__.append("LLAMAINDEX_AVAILABLE")

try:
    from .langchain_plugin import (
        TurboQuantEmbeddings as TurboQuantEmbeddings,
        TurboQuantFAISS as TurboQuantFAISS,
        create_compressed_vectorstore as create_compressed_vectorstore,
    )

    LANGCHAIN_AVAILABLE = True
    __all__.extend(
        [
            "TurboQuantEmbeddings",
            "TurboQuantFAISS",
            "create_compressed_vectorstore",
            "LANGCHAIN_AVAILABLE",
        ]
    )
except ImportError:
    LANGCHAIN_AVAILABLE = False
    __all__.append("LANGCHAIN_AVAILABLE")

try:
    from .vllm_plugin import TurboQuantVLLMAdapter as TurboQuantVLLMAdapter
    from .vllm_plugin import patch_vllm_with_turboquant as patch_vllm_with_turboquant

    VLLM_AVAILABLE = True
    __all__.extend(
        ["TurboQuantVLLMAdapter", "patch_vllm_with_turboquant", "VLLM_AVAILABLE"]
    )
except ImportError:
    VLLM_AVAILABLE = False
    __all__.append("VLLM_AVAILABLE")

try:
    from .tgi_plugin import TurboQuantTGIAdapter as TurboQuantTGIAdapter
    from .tgi_plugin import create_tgi_handler as create_tgi_handler

    TGI_AVAILABLE = True
    __all__.extend(["TurboQuantTGIAdapter", "create_tgi_handler", "TGI_AVAILABLE"])
except ImportError:
    TGI_AVAILABLE = False
    __all__.append("TGI_AVAILABLE")

try:
    from .haystack_plugin import TurboQuantDocumentEmbedder as TurboQuantDocumentEmbedder
    from .haystack_plugin import TurboQuantDocumentStore as TurboQuantDocumentStore

    HAYSTACK_AVAILABLE = True
    __all__.extend(
        ["TurboQuantDocumentStore", "TurboQuantDocumentEmbedder", "HAYSTACK_AVAILABLE"]
    )
except ImportError:
    HAYSTACK_AVAILABLE = False
    __all__.append("HAYSTACK_AVAILABLE")
