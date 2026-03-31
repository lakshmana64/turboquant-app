# TurboQuant Plus Integration Guide

## Overview

This guide covers integration with **all TurboQuant features** including the new turboquant_plus modules.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[plugins]"
```

### 2. Choose Your Integration

#### A. TurboQuant Plus Features (NEW)

```python
from core import (
    # Turbo Formats
    create_codec_from_format, TURBO4,
    
    # Asymmetric K/V
    create_asymmetric_cache,
    
    # Layer-Adaptive
    create_layer_adaptive_cache,
    
    # Sparse V
    SparseVDecoder,
    
    # Norm Correction
    NormCorrectedCodec,
)

# Example: Create asymmetric cache
cache = create_asymmetric_cache(
    dim=4096,
    k_format="q8_0",
    v_format="turbo4"
)
```

#### B. Existing Plugins

```python
from turboquant.integrations.plugins import (
    OllamaPlugin,
    OpenAIPlugin,
    SentenceTransformersPlugin,
    LangChainPlugin,
    LlamaIndexPlugin,
)
```

---

## TurboQuant Plus Integrations

### llama.cpp Integration

```python
from integrations.llama_cpp import (
    LlamaCppConfig,
    LlamaCppIntegration,
    create_llama_cpp_integration
)

# Quick setup
integration = create_llama_cpp_integration(
    llama_cpp_path="./llama.cpp",
    model_path="models/qwen2.5-7b.gguf",
    kv_cache_type_k="q8_0",
    kv_cache_type_v="turbo4",
    use_metal=True  # For Apple Silicon
)

# Run inference
result = integration.run_inference(
    prompt="Explain quantization",
    max_tokens=128
)
```

### Ollama Integration (Enhanced)

```python
from core import create_asymmetric_cache

# Use with Ollama embeddings
from integrations.plugins import OllamaPlugin

ollama = OllamaPlugin(model="nomic-embed-text")

# Create compressed cache
cache = create_asymmetric_cache(
    dim=768,
    k_format="turbo4",
    v_format="turbo4"
)

# Process embeddings
embeddings = ollama.encode(["text1", "text2"])
```

---

## Production Deployment

### FastAPI Service (Enhanced)

```python
# service.py now supports turboquant_plus features

from fastapi import FastAPI
from core import create_codec_from_format

app = FastAPI()

@app.post("/encode/turbo4")
async def encode_turbo4(vectors: list):
    codec = create_codec_from_format("turbo4", dim=4096)
    # Process with 3.8x compression
    return {"status": "compressed"}
```

### Docker Deployment

```bash
# Build with all features
docker-compose up --build

# Access services
# - FastAPI: http://localhost:8000
# - Gradio: http://localhost:7860
```

---

## Framework Integrations

### LangChain

```python
from turboquant.integrations.plugins import TurboQuantEmbeddings
from langchain.vectorstores import FAISS

# Use turbo4 compression
embeddings = TurboQuantEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    compression_format="turbo4"  # NEW
)

# Create vector store
vectorstore = FAISS.from_texts(
    texts=["doc1", "doc2"],
    embedding=embeddings
)
```

### LlamaIndex

```python
from turboquant.integrations.plugins import TurboQuantEmbedding
from llama_index import VectorStoreIndex, Document

# Use asymmetric K/V (NEW)
embeddings = TurboQuantEmbedding(
    model="local",
    asymmetric_kv=True,
    k_format="q8_0",
    v_format="turbo4"
)

# Create index
index = VectorStoreIndex.from_documents(
    documents=[Document(text="content")],
    embed_model=embeddings
)
```

### Haystack

```python
from turboquant.integrations.plugins import (
    TurboQuantDocumentStore,
    TurboQuantDocumentEmbedder
)

# Use layer-adaptive compression (NEW)
document_store = TurboQuantDocumentStore(
    embedding_dim=768,
    layer_adaptive=True,
    keep_last_n=8
)

# Add documents
document_store.write_documents(docs)
```

---

## Performance Tips

### 1. Choose Right Format

| Use Case | Format | Compression |
|----------|--------|-------------|
| Maximum compression | turbo2 | 6.4x |
| Balanced | turbo4 | 3.8x |
| Quality-critical | q8_0 | 2.0x |
| Long context | turbo4 + sparse_v | 4.9x speedup |

### 2. Enable Optimizations

```python
# Recommended production config
config = {
    "format": "turbo4",
    "asymmetric_kv": True,
    "sparse_v": True,
    "norm_correction": True,
    "layer_adaptive": True,
    "outlier_handling": True
}
```

### 3. Monitor Performance

```python
from core.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track compression
metrics.track_operation(
    "encode",
    compression_factor=3.8,
    latency_ms=50
)
```

---

## Testing

### Run Integration Tests

```bash
# Test all plugins
pytest tests/test_turboquant_plus_features.py -v

# Test specific integration
python integrations/plugins/ollama_cli.py --model nomic-embed-text
```

### Benchmark Your Setup

```bash
# Test with your LLM
python benchmark_local_llm.py --model llama3:8b

# Test specific features
python benchmark_local_llm.py --features turbo_formats,sparse_v
```

---

## Troubleshooting

### Issue: llama.cpp not found

```bash
# Install llama.cpp with TurboQuant support
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus/llama.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON  # or -DGGML_CUDA=ON
make
```

### Issue: Out of memory with large models

```python
# Use maximum compression
from core import TURBO2, create_codec_from_format

codec = create_codec_from_format("turbo2", dim=4096)
# 6.4x compression reduces memory by 84%
```

### Issue: Quality degradation

```python
# Enable norm correction
from core import NormCorrectedCodec, TurboQuantCodec

base_codec = TurboQuantCodec(4096, config)
codec = NormCorrectedCodec(base_codec, calibrate=True)
# 18.5% MSE improvement
```

---

## Resources

- **Full Documentation**: `TURBOQUANT_PLUS_FEATURES.md`
- **Examples**: `examples/turboquant_plus_examples.py`
- **Benchmarks**: `BENCHMARK_RESULTS.md`
- **Interactive Demo**: `notebooks/turboquant_plus_demo.ipynb`

---

## Support

- **GitHub Issues**: https://github.com/lakshmana64/turboquant-app/issues
- **Discussions**: https://github.com/lakshmana64/turboquant-app/discussions
- **Paper**: https://arxiv.org/abs/2504.19874

**Status**: ✅ Production Ready - March 30, 2026
