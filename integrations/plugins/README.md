# TurboQuant Plugins

Plug-and-play integrations for TurboQuant.

## Installation

```bash
# Base package
pip install -e .

# Dashboard / richer demos
pip install -e ".[app]"

# Optional plugin dependencies
pip install -e ".[plugins]"
```

## Quick Start

### Using Ollama Directly

```python
from turboquant.integrations.plugins import OllamaPlugin

# Create and connect
plugin = OllamaPlugin(model="llama3")
plugin.connect()

# Compress a prompt
result = plugin.compress("What is machine learning?")
print(f"Compression: {result.compression_ratio:.2%}")
print(f"Compression factor: {result.compression_factor:.2f}x")
print(f"Correlation: {result.correlation:.4f}")

# Query compressed embeddings
prompts = ["ML basics", "Deep learning", "Python coding"]
results = plugin.compress_batch(prompts)

matches = plugin.query("neural networks", results, top_k=3)
for match in matches:
    print(f"{match['rank']}. [{match['score']:.4f}] {match['prompt']}")
```

### Using Convenience Functions

```python
from turboquant.integrations.plugins import compress, query

# Quick compress
result = compress("Hello world", model="llama3")

# Quick query
matches = query(
    "AI and machine learning",
    ["prompt1", "prompt2", "prompt3"],
    top_k=5
)
```

### Using the Registry

```python
from turboquant.integrations.plugins import load_plugin

# Load plugin by name
plugin = load_plugin("ollama", model="llama3")

# Get plugin info
from turboquant.integrations.plugins import get_registry

registry = get_registry()
print(registry.list_plugins())  # ['haystack', 'langchain', 'llama_index', 'ollama', 'openai', ...]

info = registry.get_info("ollama")
print(info.description)  # "Ollama embedding compression plugin"
```

### Using OpenAI Embeddings

```python
from turboquant.integrations.plugins import OpenAIPlugin

plugin = OpenAIPlugin(model="text-embedding-3-small")
plugin.connect()  # checks OPENAI_API_KEY by default

result = plugin.compress("What is machine learning?")
print(result.compression_factor)
```

### Using SentenceTransformers Locally

```python
from turboquant.integrations.plugins import SentenceTransformersPlugin

plugin = SentenceTransformersPlugin(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

results = plugin.compress_batch(["TurboQuant", "Vector search", "LLM KV cache"])
matches = plugin.query("embedding compression", [r for r in results if r], top_k=2)
print(matches)
```

### Framework and Serving Adapters

```python
from turboquant.integrations.plugins import (
    TurboQuantDocumentStore,
    TurboQuantDocumentEmbedder,
    TurboQuantEmbedding,
    TurboQuantEmbeddings,
    TurboQuantTGIAdapter,
    patch_vllm_with_turboquant,
)
from turboquant.integrations.huggingface import apply_turboquant_to_hf_model
```

Use these surfaces for:

- `TurboQuantEmbeddings` with LangChain vector stores
- `TurboQuantEmbedding` with LlamaIndex
- `TurboQuantDocumentStore` and `TurboQuantDocumentEmbedder` with Haystack
- `patch_vllm_with_turboquant()` for VLLM engine helpers
- `TurboQuantTGIAdapter` for TGI KV-cache hooks
- `apply_turboquant_to_hf_model()` for Hugging Face attention wrappers

## CLI Usage

```bash
# Check connection
python -m turboquant.integrations.plugins.ollama status

# Compress a prompt
python -m turboquant.integrations.plugins.ollama compress "Your prompt"

# Query compressed embeddings
python -m turboquant.integrations.plugins.ollama query "Your query" \
    --prompts "prompt1" "prompt2" "prompt3"

# List available models
python -m turboquant.integrations.plugins.ollama models

# Run benchmark
python -m turboquant.integrations.plugins.ollama benchmark --model llama3
```

## Local Validation Results

Real local validation was run on March 27, 2026 against the Ollama instance on this machine.

### Verified Models

- `nomic-embed-text:latest`
- `llama3:latest`

### Direct Compression Checks

Using `num_bits=4` and `qjl_dim=64`:

- `nomic-embed-text:latest`: `dim=768`, `compression_ratio=12.76%`, `compression_factor=7.84x`, `mse=0.00137484`, `correlation=0.9999999943`
- `llama3:latest`: `dim=4096`, `compression_ratio=12.55%`, `compression_factor=7.97x`, `mse=271.734619`, `correlation=0.9999995254`

### End-To-End Benchmark

Command:

```bash
python integrations/ollama_test.py --model nomic-embed-text:latest --qjl 64 --sq 4
python integrations/ollama_test.py --url http://127.0.0.1:11434 --model llama3:latest --qjl 64 --sq 4
```

Measured output:

- Compression: `12.76%` of FP32 storage (`7.8x smaller`)
- Bits per dim: `4.08`
- Correlation: `0.997205`
- Mean squared error: `40.00656891`
- Mean absolute error: `5.118484`
- Max absolute error: `15.572113`
- Attention MSE: `0.00000207`
- Attention cosine similarity: `0.999992`
- Top-3 agreement: `83.33%`

### `llama3:latest` Benchmark

- Compression: `12.55%` of FP32 storage (`8.0x smaller`)
- Bits per dim: `4.02`
- Correlation: `0.995912`
- Mean squared error: `136294.0625`
- Mean absolute error: `299.779114`
- Max absolute error: `873.741211`
- Attention MSE: `0.00000000`
- Attention cosine similarity: `1.000000`
- Top-3 agreement: `100.00%`

### `llama3:latest` Memory Accounting

| Baseline | Original | Compressed | Effective Factor |
|----------|----------|------------|------------------|
| FP32 bit-budget used by plugin reporting | `16384 B` | `2056 B` | `7.97x` |
| FP16 packed theoretical KV-cache target | `8192 B` | `2056 B` | `3.98x` |
| Current Python runtime tensor storage | `8192 B` | `4112 B` | `1.99x` |

### Query Smoke Test

With the query `"embedding compression methods"`, the compressed retrieval flow ranked `"vector compression for embeddings"` as the top result.

### Baseline Note

The plugin benchmark compares compressed embeddings against FP32 storage.
Core SDK and dashboard compression factors use an FP16 baseline for KV-cache-style reporting.
The current Python runtime stores low-bit indices in byte tensors, so observed in-memory savings are lower than the packed theoretical target until bit-packing is added.

## Configuration

### Environment Variables

```bash
export OLLAMA_MODEL=llama3
export OLLAMA_HOST=localhost
export OLLAMA_PORT=11434
export TURBOQUANT_BITS=4
export TURBOQUANT_QJL_DIM=64
export OPENAI_API_KEY=your-key
```

### Programmatic Config

```python
from turboquant.integrations.plugins import OllamaPluginConfig, OllamaPlugin

config = OllamaPluginConfig(
    model="llama3",
    host="localhost",
    port=11434,
    num_bits=4,
    qjl_dim=64,
    cache_enabled=True,
)

plugin = OllamaPlugin(config=config)
```

## Features

### Compression

- **Lossy compression** with configurable quality
- **Unbiased inner product estimation** for accurate similarity search
- **Batch processing** for efficiency

```python
result = plugin.compress("Prompt", validate=True)
print(f"MSE: {result.mse}")
print(f"Correlation: {result.correlation}")
```

### Caching

```python
# Enable caching
plugin = OllamaPlugin(cache_enabled=True)

# Compress (cached)
result1 = plugin.compress("Hello")  # Fetches from Ollama
result2 = plugin.compress("Hello")  # From cache

# Save/load cache
plugin.save_cache("cache.pkl")
plugin.load_cache("cache.pkl")
```

### Query

```python
# Compress documents
docs = ["Doc 1", "Doc 2", "Doc 3"]
results = plugin.compress_batch(docs)

# Query
matches = plugin.query("Search query", results, top_k=5)
```

## API Reference

### OllamaPlugin

| Method | Description |
|--------|-------------|
| `connect()` | Test connection to Ollama |
| `get_embedding(prompt)` | Fetch embedding from Ollama |
| `compress(prompt, validate)` | Compress a prompt's embedding |
| `compress_batch(prompts)` | Compress multiple prompts |
| `query(query_text, results, top_k)` | Query compressed embeddings |
| `clear_cache()` | Clear compression cache |
| `save_cache(path)` | Save cache to disk |
| `load_cache(path)` | Load cache from disk |
| `get_stats()` | Get plugin statistics |

### CompressionResult

| Attribute | Description |
|-----------|-------------|
| `prompt` | Original prompt text |
| `original_dim` | Original embedding dimension |
| `compression_ratio` | Compressed size / original size |
| `compression_factor` | Original size / compressed size |
| `bits_per_dim` | Bits per dimension after compression |
| `mse` | Mean squared error (if validated) |
| `correlation` | Correlation with original (if validated) |
| `encoded` | Encoded data dict |

## Examples

See `examples.py` for comprehensive examples:

```bash
python -m turboquant.integrations.plugins.examples
```

## Troubleshooting

### Connection Error

```
Error: Could not connect to Ollama at localhost:11434
```

Make sure Ollama is running:
```bash
ollama serve
```

If your environment behaves differently for `localhost`, try `127.0.0.1` explicitly:

```bash
python -m turboquant.integrations.plugins --host 127.0.0.1 status
```

### Model Not Found

```
Error: model 'llama3' not found
```

Pull the model:
```bash
ollama pull llama3
```

### Import Error

```
ModuleNotFoundError: No module named 'requests'
```

Install dependencies:
```bash
pip install requests
```

## Creating New Plugins

To create a new plugin:

1. Create a new module in `integrations/plugins/`
2. Implement a plugin class with similar interface
3. Register in `registry.py`

```python
# my_plugin.py
class MyPlugin:
    def __init__(self, config=None):
        self.config = config
    
    def connect(self):
        return True
    
    def compress(self, data):
        # Implement compression
        pass
```

## License

MIT License
