"""
Plugin Registry for TurboQuant

Auto-discovery and management of TurboQuant plugins.

Usage:
    from turboquant.integrations.plugins.registry import PluginRegistry
    
    registry = PluginRegistry()
    registry.discover()
    
    # List available plugins
    print(registry.list_plugins())
    
    # Load a plugin
    plugin = registry.load("ollama")
"""

import json
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    
    name: str
    module: str
    class_name: str
    description: str = ""
    version: str = "0.1.0"
    config_class: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "class_name": self.class_name,
            "description": self.description,
            "version": self.version,
            "config_class": self.config_class,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


class PluginRegistry:
    """
    Central registry for TurboQuant plugins.
    
    Features:
        - Auto-discovery of plugins
        - Lazy loading
        - Configuration management
        - Plugin lifecycle management
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._discovered = False
        
        # Register built-in plugins
        self._register_builtin_plugins()
    
    def _register_builtin_plugins(self):
        """Register built-in plugins."""
        plugins = [
            PluginInfo(
                name="ollama",
                module="turboquant.integrations.plugins.ollama",
                class_name="OllamaPlugin",
                description="Ollama embedding compression plugin",
                config_class="OllamaPluginConfig",
                metadata={
                    "category": "embedding",
                    "requires_connection": True,
                    "default_model": "llama3",
                },
            ),
            PluginInfo(
                name="openai",
                module="turboquant.integrations.plugins.openai_plugin",
                class_name="OpenAIPlugin",
                description="OpenAI embedding compression plugin",
                config_class="OpenAIPluginConfig",
                metadata={
                    "category": "embedding",
                    "requires_api_key": True,
                    "default_model": "text-embedding-3-small",
                },
            ),
            PluginInfo(
                name="sentence_transformers",
                module="turboquant.integrations.plugins.sentence_transformers_plugin",
                class_name="SentenceTransformersPlugin",
                description="SentenceTransformers embedding compression plugin",
                config_class="SentenceTransformersPluginConfig",
                metadata={
                    "category": "embedding",
                    "local_model": True,
                    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            ),
            PluginInfo(
                name="langchain",
                module="turboquant.integrations.plugins.langchain_plugin",
                class_name="TurboQuantEmbeddings",
                description="LangChain embedding wrapper with TurboQuant compression",
                metadata={"category": "framework"},
            ),
            PluginInfo(
                name="llama_index",
                module="turboquant.integrations.plugins.llama_index_plugin",
                class_name="TurboQuantEmbedding",
                description="LlamaIndex embedding wrapper with TurboQuant compression",
                metadata={"category": "framework"},
            ),
            PluginInfo(
                name="vllm",
                module="turboquant.integrations.plugins.vllm_plugin",
                class_name="TurboQuantVLLMAdapter",
                description="VLLM KV-cache compression adapter",
                metadata={"category": "serving"},
            ),
            PluginInfo(
                name="tgi",
                module="turboquant.integrations.plugins.tgi_plugin",
                class_name="TurboQuantTGIAdapter",
                description="Text Generation Inference KV-cache compression adapter",
                metadata={"category": "serving"},
            ),
            PluginInfo(
                name="haystack",
                module="turboquant.integrations.plugins.haystack_plugin",
                class_name="TurboQuantDocumentStore",
                description="Haystack document store with TurboQuant compression",
                metadata={"category": "framework"},
            ),
        ]

        for plugin in plugins:
            self.register(plugin)
    
    def register(self, info: PluginInfo):
        """
        Register a plugin.
        
        Args:
            info: Plugin information
        """
        self._plugins[info.name] = info
    
    def unregister(self, name: str):
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            del self._plugins[name]
        if name in self._instances:
            del self._instances[name]
    
    def discover(self, package: str = "turboquant.integrations.plugins"):
        """
        Auto-discover plugins in a package.
        
        Args:
            package: Package to scan
        """
        if self._discovered:
            return
        
        try:
            importlib.import_module(package)
            for info in self._plugins.values():
                try:
                    importlib.import_module(info.module)
                except ImportError:
                    continue
        
        except Exception as e:
            print(f"Warning: Could not discover plugins: {e}")
        
        self._discovered = True
    
    def list_plugins(
        self,
        enabled_only: bool = True,
        category: Optional[str] = None
    ) -> List[str]:
        """
        List registered plugins.
        
        Args:
            enabled_only: Only list enabled plugins
            category: Filter by category
            
        Returns:
            List of plugin names
        """
        names = []
        for name, info in self._plugins.items():
            if enabled_only and not info.enabled:
                continue
            if category and info.metadata.get("category") != category:
                continue
            names.append(name)
        return sorted(names)
    
    def get_info(self, name: str) -> Optional[PluginInfo]:
        """
        Get plugin information.
        
        Args:
            name: Plugin name
            
        Returns:
            PluginInfo or None
        """
        return self._plugins.get(name)
    
    def load(self, name: str, **kwargs) -> Optional[Any]:
        """
        Load and instantiate a plugin.
        
        Args:
            name: Plugin name
            **kwargs: Arguments passed to plugin constructor
            
        Returns:
            Plugin instance or None
        """
        use_cache = not kwargs
        if use_cache and name in self._instances:
            return self._instances[name]
        
        info = self._plugins.get(name)
        if info is None:
            print(f"Error: Plugin '{name}' not found")
            return None
        
        if not info.enabled:
            print(f"Error: Plugin '{name}' is disabled")
            return None
        
        try:
            # Import module
            module = importlib.import_module(info.module)
            
            # Get plugin class
            plugin_class = getattr(module, info.class_name)
            
            # Instantiate
            instance = plugin_class(**kwargs)
            
            # Cache
            if use_cache:
                self._instances[name] = instance
            
            return instance
        
        except ImportError as e:
            print(f"Error loading plugin '{name}': {e}")
            return None
        except Exception as e:
            print(f"Error instantiating plugin '{name}': {e}")
            return None
    
    def unload(self, name: str):
        """
        Unload a plugin instance.
        
        Args:
            name: Plugin name
        """
        if name in self._instances:
            del self._instances[name]
    
    def enable(self, name: str):
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
    
    def disable(self, name: str):
        """Disable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = False
    
    def get_config(self, name: str) -> Optional[Any]:
        """
        Get default config for a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Config instance or None
        """
        info = self._plugins.get(name)
        if info is None or info.config_class is None:
            return None
        
        try:
            module = importlib.import_module(info.module)
            config_class = getattr(module, info.config_class)
            return config_class()
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary."""
        return {
            "plugins": {
                name: info.to_dict()
                for name, info in self._plugins.items()
            },
            "loaded": list(self._instances.keys()),
        }
    
    def save_config(self, path: str):
        """
        Save registry config to file.
        
        Args:
            path: File path
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_config(self, path: str):
        """
        Load registry config from file.
        
        Args:
            path: File path
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        for name, info_dict in data.get("plugins", {}).items():
            info = PluginInfo(**info_dict)
            self.register(info)
    
    def __repr__(self) -> str:
        return f"PluginRegistry(plugins={list(self._plugins.keys())})"


# Global registry instance
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.discover()
    return _registry


def list_plugins() -> List[str]:
    """List available plugins."""
    return get_registry().list_plugins()


def load_plugin(name: str, **kwargs) -> Optional[Any]:
    """Load a plugin by name."""
    return get_registry().load(name, **kwargs)


def get_plugin_info(name: str) -> Optional[PluginInfo]:
    """Get plugin information."""
    return get_registry().get_info(name)
