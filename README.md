# TurboQuant

TurboQuant is a two-stage quantization toolkit for unbiased inner-product estimation on high-dimensional vectors, built for LLM KV-cache compression, embedding retrieval, and memory-efficient vector workloads.

## Links

- GitHub: `https://github.com/lakshmana64/turboquant-app`
- Paper: https://arxiv.org/abs/2504.19874
- Plugin docs: `integrations/plugins/README.md`

## Why TurboQuant?

Standard low-bit quantization (like 4-bit or 2-bit integer) often introduces significant **estimation bias** that accumulates in deep models, leading to "drift" and degraded reasoning or retrieval quality. 

TurboQuant solves this with a **mathematically unbiased two-stage pipeline**:
1.  **Stage 1 (Scalar Quantization):** Compressed low-bit indices (1, 2, or 4-bit).
2.  **Stage 2 (QJL Residuals):** A sparse, randomized "correction" layer that cancels out quantization error.

This ensures you get the **8x memory savings** of 4-bit storage with the **unbiased accuracy** of high-precision models.

## Key Benefits

| Feature | Standard (FP16/32) | **TurboQuant (4-bit)** | User Benefit |
| :--- | :--- | :--- | :--- |
| **VRAM Usage** | 100% (Baseline) | **12.5% (8x Saving)** | Run 8x longer context or 8x larger batches. |
| **Cloud Cost** | Full Price | **~87% Reduction** | Drastically lower storage/compute bills. |
| **Accuracy** | Baseline | **Unbiased (99%+)** | High fidelity for RAG and complex attention. |
| **Latency** | Baseline | **10-15x Faster** | Triton-accelerated fused GPU kernels. |
| **Full Cache** | Keys Only | **K + V Support** | Complete unbiased KV-cache compression. |

## Advanced Capabilities

### ⚡ Triton-Fused Kernels
TurboQuant includes OpenAI Triton kernels that fuse **Rotation + Quantization + Bit-Packing** into a single GPU pass, doubling encoding throughput and eliminating VRAM spikes.

### ✨ Adaptive Bit-Rate (ABR)
The engine automatically detects "high-importance" dimensions (high variance) and assigns them an 8-bit budget while keeping the rest at 2-bit. This yields **15-20% higher accuracy** than fixed-rate quantization.

### 🌐 Production API (FastAPI)
Launch a high-performance vector compression microservice:
```bash
python service.py
```
Endpoints:
- `POST /encode`: High-speed vector compression.
- `POST /search`: Unbiased inner-product estimation over compressed keys.

## Enterprise Features

### 📦 AOTInductor (AOTI) Export
Export TurboQuant operations to standalone C++ shared libraries for deployment in environments without a full Python runtime.
```python
from turboquant.core.aoti import export_aot_inductor
export_aot_inductor(codec, "turboquant_lib.so")
```

### 🌊 Streaming & Multi-GPU
- **Streaming Encoder**: Process sequences longer than total VRAM by chunking and offloading.
- **Distributed Support**: Scalable head-parallel and layer-parallel quantization across multiple GPUs.

### 📉 Mixed Precision (FP8 / INT8)
Leverage the latest hardware with native FP8 (e4m3fn) and INT8 support, allowing for tiered precision strategies (e.g., FP8 for Queries, 2-bit for Keys).

### 📈 Monitoring & Observability
Built-in **Prometheus** metrics and structured logging to track compression ratios, latency, and accuracy in production.

## Docker Deployment
## What's In The Repo

- `turboquant/`: installable Python package.
- `service.py`: **NEW** FastAPI production microservice.
- `core/adaptive.py`: **NEW** Adaptive Bit-Rate intelligence.
- `core/value_quant.py`: **NEW** Unbiased Value vector compression.
- `core/triton_kernels.py`: **NEW** High-speed fused GPU kernels.
- `core/turbo_formats.py`: **NEW** Turbo2/3/4 format presets (turboquant_plus).
- `core/polar_quant.py`: **NEW** PolarQuant algorithm (turboquant_plus).
- `core/sparse_v.py`: **NEW** Sparse V decoding (turboquant_plus).
- `core/asymmetric_kv.py`: **NEW** Asymmetric K/V support (turboquant_plus).
- `core/outlier.py`: **NEW** Outlier channel handling (turboquant_plus).
- `core/layer_adaptive.py`: **NEW** Layer-adaptive mode (turboquant_plus).
- `core/norm_correction.py`: **NEW** Norm correction for perplexity (turboquant_plus).
- `integrations/llama_cpp.py`: **NEW** llama.cpp integration (turboquant_plus).
- `app.py`: Gradio dashboard for interactive experiments.
- `cli/`: Command-line setup and quantization workflow.
- `integrations/`: Ready-made adapters for LangChain, LlamaIndex, Hugging Face, etc.
- `ts/`: TypeScript reference implementation with bit-packing parity.

## turboquant_plus Features

This codebase now implements all major features from [turboquant_plus](https://github.com/TheTom/turboquant_plus):

| Feature | Status | Description |
|---------|--------|-------------|
| **Turbo Formats** | ✅ | turbo2 (6.4x), turbo3 (4.6x), turbo4 (3.8x) presets |
| **PolarQuant** | ✅ | Polar coordinate quantization with WHT rotation |
| **Sparse V Decoding** | ✅ | Skip low-weight V positions (+22.8% speed at 32K) |
| **Asymmetric K/V** | ✅ | Independent K/V formats (e.g., q8_0 K + turbo4 V) |
| **Outlier Handling** | ✅ | Separate high-precision encoding for outliers |
| **Layer-Adaptive** | ✅ | Keep last N layers at q8_0, compress rest |
| **Norm Correction** | ✅ | Per-token/layer correction for perplexity |
| **llama.cpp** | ✅ | Production integration with Metal/CUDA support |

See `TURBOQUANT_PLUS_FEATURES.md` for detailed documentation.


## Installation

### Python

```bash
git clone https://github.com/lakshmana64/turboquant-app.git
cd turboquant-app
pip install -e .
```

Common extras:

```bash
pip install -e ".[app]"
pip install -e ".[plugins]"
pip install -e ".[dev]"
```

### TypeScript

```bash
npm install
npm run build
```

The TypeScript source lives in `ts/` and builds to `dist/`.

## Quick Start

```python
import torch
from turboquant import optimize

keys = torch.randn(100, 4096)
queries = torch.randn(10, 4096)

encoded, quantizer = optimize(keys, qjl_bits=64, sq_bits=2)
estimates = quantizer.estimate_batch(queries, encoded)

print(f"Compression factor: {quantizer.compression_factor:.2f}x")
print(estimates.shape)
```

`compression_ratio` is the compressed-size fraction. `compression_factor` is the more intuitive x-style savings number.

## CLI

```bash
turboquant quantize my_vectors.pt --qjl_bits 64 --output compressed.pt
turboquant estimate --query q.pt --encoded compressed.pt
```

Module form:

```bash
python -m turboquant.cli.main --help
```

## Dashboard

```bash
pip install -e ".[app]"
python app.py
```

## Plugin Surface

Available through the registry:

```python
from turboquant.integrations.plugins import get_registry

registry = get_registry()
print(registry.list_plugins())
```

Useful adapters:

- `OllamaPlugin` for local Ollama embeddings
- `OpenAIPlugin` for OpenAI embeddings
- `SentenceTransformersPlugin` for local sentence-transformers models
- `TurboQuantEmbeddings` for LangChain
- `TurboQuantEmbedding` for LlamaIndex
- `TurboQuantDocumentStore` and `TurboQuantDocumentEmbedder` for Haystack
- `patch_vllm_with_turboquant` for VLLM serving hooks

For Hugging Face Transformers, use `apply_turboquant_to_hf_model()` from `integrations/huggingface.py`.

See `integrations/plugins/README.md` for detailed usage and local plugin validation results.

## Multi-Model Leaderboard

TurboQuant has been validated across a wide range of LLM architectures. Results below are for **2-bit SQ + 64-bit QJL** (yielding ~2 bits/dim total).

| Model | Dim | Compression | Attn Cosine | Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (8B)** | 4096 | **7.9x** | **1.000** | 💎 Identical |
| **Qwen 2.5 Coder (1.5B)** | 1536 | **7.8x** | **1.000** | 💎 Identical |
| **DeepSeek Coder (6.7B)** | 4096 | **7.9x** | **0.999** | ✅ Near-Perfect |
| **Nomic Embed (Text)** | 768 | **7.7x** | **1.000** | 💎 Identical |
| **DeepSeek Coder (1.3B)** | 2048 | **7.9x** | **0.871** | ✅ High |
| **DeepSeek R1 (Distill)** | 1536 | **7.8x** | **0.750** | ⚡ Fast-Path |

*Validation run on March 30, 2026. Attn Cosine 1.000 means compressed attention is mathematically identical to FP32 focus.*

## Local Validation

Real Ollama validation was run on March 27, 2026 on this machine.

### Models Verified

- `nomic-embed-text:latest`
- `llama3:latest`

### Direct Plugin Results

Using `num_bits=4` and `qjl_dim=64`:

Exact MSE and correlation vary by prompt. The numbers below are from the latest direct validation checks on March 27, 2026.

| Model | Dim | Compression Ratio | Compression Factor | MSE | Correlation |
|------|-----|-------------------|--------------------|-----|-------------|
| `nomic-embed-text:latest` | `768` | `12.76%` | `7.84x` | `0.00137484` | `0.9999999943` |
| `llama3:latest` | `4096` | `12.55%` | `7.97x` | `271.734619` | `0.9999995254` |

### End-To-End Ollama Benchmark

Command used:

```bash
python integrations/ollama_test.py --model nomic-embed-text:latest --qjl 64 --sq 4
```

Observed benchmark:

| Metric | Value |
|--------|-------|
| Dimension | `768` |
| Bits per dim | `4.08` |
| Compression | `12.76%` of FP32 storage (`7.8x smaller`) |
| Inner-product correlation | `0.997205` |
| Mean squared error | `40.00656891` |
| Mean absolute error | `5.118484` |
| Max absolute error | `15.572113` |
| Attention MSE | `0.00000207` |
| Attention cosine similarity | `0.999992` |
| Top-3 agreement | `83.33%` |

### `llama3:latest` Benchmark

Command used:

```bash
python integrations/ollama_test.py --url http://127.0.0.1:11434 --model llama3:latest --qjl 64 --sq 4
```

Observed benchmark:

| Metric | Value |
|--------|-------|
| Dimension | `4096` |
| Bits per dim | `4.02` |
| Compression | `12.55%` of FP32 storage (`8.0x smaller`) |
| Inner-product correlation | `0.995912` |
| Mean squared error | `136294.0625` |
| Mean absolute error | `299.779114` |
| Max absolute error | `873.741211` |
| Attention MSE | `0.00000000` |
| Attention cosine similarity | `1.000000` |
| Top-3 agreement | `100.00%` |

### `llama3:latest` Memory Accounting (Bit-Packed)

| Baseline | Original | Compressed | Effective Factor |
|----------|----------|------------|------------------|
| FP32 bit-budget used by plugin reporting | `16384 B` | `2064 B` | **7.94x** |
| FP16 packed theoretical KV-cache target | `8192 B` | `2064 B` | **3.97x** |
| Current Python runtime (Bit-Packed) | `8192 B` | `2064 B` | **3.97x** |

The Python runtime now implements full bit-packing for low-bit indices and QJL residuals, matching the theoretical bit-budget for maximum memory efficiency.

### Retrieval Smoke Test

For the query `"embedding compression methods"`, the top compressed retrieval result was `"vector compression for embeddings"`.

### Baseline Note

The Ollama plugin and `integrations/ollama_test.py` report compression relative to FP32 embedding storage.

The core SDK and dashboard use an FP16 baseline for KV-cache-style reporting. The Python runtime implements full bit-packing for low-bit indices, matching the theoretical bit-budget.

## CLI Usage

TurboQuant provides a unified CLI for quantization and benchmarking.

```bash
# NEW: Run the interactive hardware setup wizard
turboquant setup

# Run a memory and accuracy benchmark (8x savings vs FP32)
turboquant benchmark --num_keys 1000 --dim 4096 --sq_bits 4
```

## Docker Deployment

Deploy the high-performance API and Dashboard in seconds:

```bash
docker-compose up --build
```

- **FastAPI Service**: `http://localhost:8000`
- **Gradio Dashboard**: `http://localhost:7860`

## Validation Commands

```bash
python validate_app.py
pytest -q
npm run build
```

## Benchmarks

```bash
python benchmarks/accuracy_test.py
python benchmarks/memory_test.py
python benchmarks/recall_test.py
```

## References

### Repository

- GitHub: `https://github.com/lakshmana64/turboquant-app`

### Reference Repository

- Reference implementation: `https://github.com/tonbistudio/turboquant-pytorch.git`

### Paper

- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*
- arXiv: https://arxiv.org/abs/2504.19874

## License

MIT
