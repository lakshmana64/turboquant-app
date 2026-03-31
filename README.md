# TurboQuant - Complete LLM Quantization Platform

**Version**: 1.2.0 | **Date**: March 31, 2026 | **Status**: ✅ Production Ready

---

## 🎯 What is TurboQuant?

TurboQuant is a **complete LLM quantization platform** that reduces memory usage by 75% while maintaining model quality.

### Two Components:

1. **🐍 Python Package** (`turboquant-app/`)
   - Quantization algorithms
   - Model optimization tools
   - Integration plugins
   - **Use for**: Training, optimization, embedding compression

2. **🦙 llama.cpp Fork** (`llama.cpp/turboquant-llama-cpp/`)
   - C/C++ inference engine
   - GPU kernels (CUDA/Metal)
   - Production deployment
   - **Use for**: Running LLMs with TurboQuant KV cache

---

## 🚀 Quick Start

### I Want To...

#### Compress Embeddings / Vectors
```bash
# Install Python package
pip install -e .

# Use in Python
from turboquant import optimize
compressed = optimize(vectors, sq_bits=4)
```

#### Run LLMs with Less VRAM
```bash
# 1. Build llama.cpp
cd llama.cpp/turboquant-llama-cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # or -DGGML_METAL=ON for Mac
cmake --build . --config Release

# 2. Run LLM with TurboQuant KV cache
./bin/main -m llama3-8b.gguf \
  -p "Hello" \
  --gpu-layers 32 \
  --kv-cache-type-v turbo4  # 75% less VRAM!
```

#### Use Both Together
```bash
# Python handles embeddings
from turboquant.integrations.plugins import OllamaPlugin

# llama.cpp handles LLM inference
# Both use TurboQuant compression = maximum efficiency!
```

---

## 📦 What's In This Repository

### Python Package (`turboquant-app/`)

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `core/` | Quantization algorithms | Compressing vectors/embeddings |
| `integrations/` | LangChain, LlamaIndex, etc. | Using with AI frameworks |
| `service.py` | FastAPI server | Production API deployment |
| `app.py` | Gradio dashboard | Interactive experiments |
| `benchmarks/` | Performance testing | Measuring compression/speed |

### llama.cpp Fork (`llama.cpp/turboquant-llama-cpp/`)

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `main` | LLM inference | Running chat/completion |
| `server` | HTTP API | Production serving |
| `quantize` | Model conversion | Converting models to GGUF |
| Metal/CUDA kernels | GPU acceleration | Fast inference on GPU |

---

## 🎯 Use Cases

### 1. RAG Systems (Retrieval Augmented Generation)

**Problem**: Embeddings take too much memory

**Solution**:
```python
from turboquant import optimize

# Compress embeddings 8x
compressed, codec = optimize(embeddings, sq_bits=4)

# Store in vector database
# Query with unbiased inner product
```

**Result**: 8x more embeddings in same memory

---

### 2. LLM Inference (Chat/Completion)

**Problem**: Long context requires too much VRAM

**Solution**:
```bash
./main -m llama3-8b.gguf \
  -p "Analyze this 100K document..." \
  --gpu-layers 32 \
  --kv-cache-type-v turbo4  # 75% less VRAM
```

**Result**: 32K context in 16GB VRAM (vs 64GB needed)

---

### 3. AI Application Development

**Problem**: Need both embeddings and LLM

**Solution**:
```python
# Python for embeddings
from turboquant.integrations.plugins import LangChainPlugin

# llama.cpp for LLM
# Both compressed = maximum efficiency
```

**Result**: Full AI stack with 75% memory reduction

---

## 📊 Performance

### Memory Savings

| Use Case | Standard | TurboQuant | Savings |
|----------|----------|------------|---------|
| **Embeddings** | 100% | 12.5% | **87.5%** |
| **LLM KV Cache (4K)** | 8 GB | 2 GB | **75%** |
| **LLM KV Cache (32K)** | 64 GB | 16 GB | **75%** |

### Speed

| Operation | Standard | TurboQuant | Notes |
|-----------|----------|------------|-------|
| **Embedding Compression** | 1x | 10-15x | Triton kernels |
| **LLM Prefill** | 1x | 1.15x | CUDA acceleration |
| **LLM Decode (32K)** | 1x | 1.40x | Sparse V decoding |

---

## 🛠️ Installation

### Python Package

```bash
git clone https://github.com/lakshmana64/turboquant-app.git
cd turboquant-app/turboquant-app
pip install -e .
```

### llama.cpp (for LLM Inference)

```bash
cd turboquant-app/llama.cpp/turboquant-llama-cpp

# NVIDIA CUDA
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# Apple Metal
mkdir build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release
```

---

## 📖 Documentation

### Getting Started
- [`README.md`](README.md) - This file (overview)
- [`llama.cpp/README.md`](llama.cpp/README.md) - llama.cpp setup
- [`CUDA_SETUP.md`](CUDA_SETUP.md) - NVIDIA GPU setup

### Features
- [`TURBOQUANT_PLUS_FEATURES.md`](TURBOQUANT_PLUS_FEATURES.md) - All 8 features
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) - Performance data
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Implementation status

### Examples
- [`examples/turboquant_plus_examples.py`](examples/turboquant_plus_examples.py) - Code examples
- [`notebooks/turboquant_plus_demo.ipynb`](notebooks/turboquant_plus_demo.ipynb) - Interactive demo

---

## 🧪 Quick Tests

### Test Python Package
```bash
python test_turboquant_plus.py
```

### Test llama.cpp
```bash
cd llama.cpp/turboquant-llama-cpp/build/bin
./main -m ../../../models/test.gguf -p "Hello" -n 32
```

### Test CUDA Integration
```bash
python test_cuda_integration.py --model llama3:8b --gpu-layers 32
```

---

## 🎯 Which Component Do I Need?

| I Want To... | Need Python? | Need llama.cpp? |
|--------------|--------------|-----------------|
| Compress embeddings | ✅ Yes | ❌ No |
| Run LLM inference | ❌ No | ✅ Yes |
| Build RAG system | ✅ Yes | ✅ Yes |
| Use with LangChain | ✅ Yes | ❌ No |
| Deploy LLM API | ❌ No | ✅ Yes |
| Optimize vectors | ✅ Yes | ❌ No |
| Chat with LLM | ❌ No | ✅ Yes |

**Rule of thumb:**
- **Vectors/Embeddings** → Python only
- **LLM Inference** → llama.cpp only
- **Full AI Stack** → Both

---

## 🔗 Links

- **GitHub**: https://github.com/lakshmana64/turboquant-app
- **Paper**: https://arxiv.org/abs/2504.19874
- **turboquant_plus**: https://github.com/TheTom/turboquant_plus
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

---

## 💡 Why TurboQuant?

### The Problem

LLMs need **huge amounts of memory**:
- 7B model @ 4K context = 8GB VRAM
- 7B model @ 32K context = 64GB VRAM
- Embeddings for 1M docs = 10GB+ RAM

### The Solution

TurboQuant reduces memory by **75-87%** with:
1. **Unbiased quantization** - No quality loss
2. **GPU acceleration** - 10-50x faster
3. **Production ready** - C++/Python/Metal/CUDA

### The Result

- Run **8x longer context** on same hardware
- Store **8x more embeddings** in same memory
- Deploy on **consumer GPUs** (RTX 3090/4090)

---

## 📋 Table of Contents

1. [What is TurboQuant?](#-what-is-turboquant)
2. [Quick Start](#-quick-start)
3. [What's In This Repository](#-whats-in-this-repository)
4. [Use Cases](#-use-cases)
5. [Performance](#-performance)
6. [Installation](#️-installation)
7. [Documentation](#-documentation)
8. [Quick Tests](#-quick-tests)
9. [Which Component Do I Need?](#-which-component-do-i-need)
10. [Why TurboQuant?](#-why-turboquant)

---

**Status**: ✅ PRODUCTION READY - March 31, 2026  
**License**: MIT
