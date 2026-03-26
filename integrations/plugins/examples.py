"""
Ollama Plugin Examples

Demonstrates various ways to use the Ollama plugin.
"""


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("BASIC USAGE")
    print("=" * 60)
    
    from turboquant.integrations.plugins import OllamaPlugin
    
    # Create plugin
    plugin = OllamaPlugin(model="llama3")
    
    # Connect to Ollama
    if not plugin.connect():
        print("Could not connect to Ollama. Make sure it's running:")
        print("  ollama serve")
        return
    
    print(f"Connected! Plugin: {plugin}")
    
    # Compress a single prompt
    result = plugin.compress("What is machine learning?")
    
    if result:
        print("\nCompression result:")
        print(f"  Original dim: {result.original_dim}")
        print(
            f"  Compression: {result.compression_ratio:.2%} "
            f"({result.compression_factor:.2f}x smaller)"
        )
        print(f"  Correlation: {result.correlation:.4f}")
    
    print()


def example_batch_processing():
    """Batch processing example."""
    print("=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    
    from turboquant.integrations.plugins import OllamaPlugin
    
    plugin = OllamaPlugin(model="llama3")
    
    if not plugin.connect():
        print("Could not connect to Ollama")
        return
    
    prompts = [
        "Explain quantum computing",
        "How does photosynthesis work?",
        "What is the capital of France?",
        "Write a poem about AI",
        "What is 2 + 2?",
    ]
    
    print(f"Compressing {len(prompts)} prompts...")
    results = plugin.compress_batch(prompts)
    
    successful = [r for r in results if r is not None]
    print(f"Successfully compressed: {len(successful)}/{len(prompts)}")
    
    # Query against compressed
    query_text = "Tell me about science"
    matches = plugin.query(query_text, successful, top_k=3)
    
    print(f"\nQuery: '{query_text}'")
    print("Top matches:")
    for match in matches:
        print(f"  {match['rank']}. [{match['score']:.4f}] {match['prompt']}")
    
    print()


def example_convenience_functions():
    """Using convenience functions."""
    print("=" * 60)
    print("CONVENIENCE FUNCTIONS")
    print("=" * 60)
    
    from turboquant.integrations.plugins import compress, query
    
    # Quick compress
    result = compress("Hello world", model="llama3")
    
    if result:
        print(f"Compressed: {result.prompt}")
        print(f"Correlation: {result.correlation:.4f}")
    
    # Quick query
    prompts = [
        "Machine learning basics",
        "Deep learning tutorial",
        "Python programming",
    ]
    
    matches = query(
        "AI and neural networks",
        prompts,
        model="llama3",
        top_k=2
    )
    
    print("\nQuery results:")
    for match in matches:
        print(f"  {match['rank']}. [{match['score']:.4f}] {match['prompt']}")
    
    print()


def example_with_config():
    """Using custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION")
    print("=" * 60)
    
    from turboquant.integrations.plugins import OllamaPlugin, OllamaPluginConfig
    
    # Create config
    config = OllamaPluginConfig(
        model="llama3",
        host="localhost",
        port=11434,
        num_bits=2,  # Lower bits = more compression
        qjl_dim=32,  # Smaller QJL dim
        cache_enabled=True,
    )
    
    plugin = OllamaPlugin(config=config)
    
    if plugin.connect():
        print(f"Config: {config.to_dict()}")
        
        # Compress with caching
        result1 = plugin.compress("Test prompt")
        result2 = plugin.compress("Test prompt")  # From cache
        
        print("\nFirst compression: computed")
        print("Second compression: from cache")
        print(f"Cache size: {len(plugin._cache)}")
    
    print()


def example_plugin_registry():
    """Using the plugin registry."""
    print("=" * 60)
    print("PLUGIN REGISTRY")
    print("=" * 60)
    
    from turboquant.integrations.plugins import (
        get_registry,
        list_plugins,
        load_plugin,
    )
    
    # Get registry
    registry = get_registry()
    
    # List available plugins
    print(f"Available plugins: {list_plugins()}")
    
    # Get plugin info
    info = registry.get_info("ollama")
    if info:
        print("\nOllama plugin info:")
        print(f"  Description: {info.description}")
        print(f"  Version: {info.version}")
        print(f"  Category: {info.metadata.get('category')}")
    
    # Load plugin
    plugin = load_plugin("ollama", model="llama3")
    
    if plugin and plugin.connect():
        print(f"\nLoaded plugin: {plugin}")
    
    print()


def example_caching():
    """Using cache for efficiency."""
    print("=" * 60)
    print("CACHING")
    print("=" * 60)
    
    from turboquant.integrations.plugins import OllamaPlugin
    
    plugin = OllamaPlugin(model="llama3", cache_enabled=True)
    
    if not plugin.connect():
        print("Could not connect")
        return
    
    # Compress same prompt twice
    print("Compressing 'Hello world' twice...")
    
    result1 = plugin.compress("Hello world")
    result2 = plugin.compress("Hello world")
    
    print("First call: fetched from Ollama")
    print("Second call: served from cache")
    print(f"Cache size: {len(plugin._cache)}")
    
    # Save cache
    cache_path = "/tmp/ollama_cache.pkl"
    plugin.save_cache(cache_path)
    print(f"\nCache saved to: {cache_path}")
    
    # Load cache
    plugin.clear_cache()
    print(f"Cache cleared. Size: {len(plugin._cache)}")
    
    plugin.load_cache(cache_path)
    print(f"Cache loaded. Size: {len(plugin._cache)}")
    
    print()


def example_error_handling():
    """Error handling example."""
    print("=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)
    
    from turboquant.integrations.plugins import OllamaPlugin
    
    # Try connecting to non-existent Ollama
    plugin = OllamaPlugin(
        model="llama3",
        host="localhost",
        port=9999  # Wrong port
    )
    
    if not plugin.connect(timeout=2):
        print("Connection failed (expected)")
        print(f"Plugin status: {plugin}")
    
    # Try compressing without connection
    result = plugin.compress("Test")
    print(f"Compress result: {result}")
    
    print()


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TURBOQUANT OLLAMA PLUGIN EXAMPLES")
    print("=" * 60 + "\n")
    
    examples = [
        example_basic_usage,
        example_batch_processing,
        example_convenience_functions,
        example_with_config,
        example_plugin_registry,
        example_caching,
        example_error_handling,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")
    
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
