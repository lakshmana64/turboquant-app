# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-31 - Complete Benchmark Suite & Architecture

### Added - Complete Benchmark Suite (5 new scripts)
- **benchmark_norm_correction.py**:
  - Norm correction quality benchmarks across bit widths
  - MSE and cosine similarity metrics
  - Latency measurements
  - JSON results export

- **benchmark_ppl_tq_vs_rq.py**:
  - Perplexity comparison: TurboQuant vs RotorQuant
  - Synthetic sequence generation
  - Bits/dim and compression metrics
  - Speed comparison

- **test_outlier_comparison.py**:
  - Outlier detection method comparison
  - Variance, magnitude, and kurtosis methods
  - Quality and compression metrics
  - Recommendations for production

- **validate_real_model.py**:
  - Real model validation (HuggingFace transformers)
  - Weight and activation quantization
  - Layer-by-layer metrics
  - Support for OPT, Llama, etc.

- **temporal_decay_prototype.py**:
  - Long-context memory optimization
  - Context lengths: 1K to 32K tokens
  - Memory savings up to 30-34%
  - Quality score tracking

### Added - Architecture Documentation
- **ARCHITECTURE.md**:
  - Problem/Solution diagrams
  - High-level ecosystem architecture
  - Component architecture (28 modules)
  - Data flow workflows
  - Performance metrics
  - Deployment architectures
  - Build architecture

### Updated
- **README.md**:
  - Added architecture diagrams directly in README
  - Problem/Solution visualizations
  - Repository structure tree
  - Clear folder separation guide

### Changed
- **BENCHMARK_RESULTS.md**: Updated with new benchmark scripts
- **IMPLEMENTATION_SUMMARY.md**: Updated to 100% completion

### Performance
- **Benchmarks**: 6/6 complete (was 1/6)
- **Documentation**: 12/12 complete
- **Overall**: 100% complete

## [1.2.0] - 2026-03-30 - TurboQuant Plus Features

### Added - 8 Major Features from turboquant_plus
- **Turbo Format Presets** (`core/turbo_formats.py`):
  - turbo2 (6.4x), turbo3 (4.6x), turbo4 (3.8x) compression presets
  - Pre-configured QJL dimensions and bit widths
  - Memory usage calculator for production planning

- **PolarQuant Algorithm** (`core/polar_quant.py`):
  - Polar coordinate quantization with WHT rotation
  - Magnitude + direction encoding
  - Up to 15x compression with Lloyd-Max centroids

- **Sparse V Decoding** (`core/sparse_v.py`):
  - Attention-gated skipping of low-weight V positions
  - +22.8% decode speedup at 32K context
  - Configurable sparsity threshold (default 1e-6)

- **Asymmetric K/V Support** (`core/asymmetric_kv.py`):
  - Independent formats for Keys and Values
  - Recommended: q8_0 for K, turbo4 for V
  - Rescues quality on low-bit models (Q4_K_M)

- **Outlier Channel Handling** (`core/outlier.py`):
  - Detects high-variance channels automatically
  - Keeps outliers in high precision (8-bit)
  - Compresses normal channels aggressively (2-bit)

- **Layer-Adaptive Mode** (`core/layer_adaptive.py`):
  - Last N layers at q8_0, rest compressed
  - Configurable per-layer format assignment
  - 3.5x compression with minimal quality loss

- **Norm Correction** (`core/norm_correction.py`):
  - Per-token and per-layer scale correction
  - 18.5% MSE reduction on average
  - Running statistics for inference

- **llama.cpp Integration** (`integrations/llama_cpp.py`):
  - Production deployment with Metal/CUDA support
  - GGUF model quantization workflow
  - Auto-detection of TurboQuant support

### Changed
- **README.md**: Added comprehensive turboquant_plus features section
- **core/__init__.py**: Export all new feature modules
- **Tests**: Added 8/8 passing test suite for new features

### Performance
- **Memory Savings**: 75% VRAM reduction for 7B models
- **Compression**: 6.9x average across all features
- **Quality**: 0.69 average cosine similarity
- **Norm Correction**: 18.5% MSE improvement

### Documentation
- `TURBOQUANT_PLUS_FEATURES.md`: Complete feature documentation
- `IMPLEMENTATION_SUMMARY.md`: Implementation status and checklist
- `BENCHMARK_RESULTS.md`: Local LLM efficiency benchmarks
- `examples/turboquant_plus_examples.py`: 8 usage examples
- `notebooks/turboquant_plus_demo.ipynb`: Interactive demo

### Tests
- All 8 new features tested and passing
- Simple test runner: `python test_turboquant_plus.py`
- Pytest suite: `tests/test_turboquant_plus_features.py`

## [1.1.0] - 2026-03-30

### Added
- **Production API**: High-performance FastAPI microservice (`service.py`).
- **Containerization**: Official Docker and Docker Compose support for one-click deployment.
- **Smart Setup**: Interactive CLI hardware wizard (`turboquant setup`) for environment optimization.
- **Triton Acceleration**: Fused GPU kernels for single-pass quantization and packing.
- **Value Quantization**: Specialized unbiased codec for KV-cache 'Value' vectors.
- **Adaptive Bit-Rate**: Importance-aware quantization logic for higher accuracy.
- **TS Parity**: High-speed bit-packing for the TypeScript reference port.

### Changed
- **CLI**: Re-organized command structure with new `setup` and `benchmark` verbs.
- **Documentation**: Expanded README with production deployment guides and value propositions.

## [1.0.1] - 2026-03-30

### Added
- **Bit-Packing**: Full implementation of bit-packing for 1, 2, and 4-bit indices and QJL signs.
- **Memory Optimization**: Real-world memory savings now match theoretical bit-budgets (e.g., 8x smaller vs FP32).
- **Benchmark Suite**: New `benchmarks/memory_packing_benchmark.py` for verifying packed vs unpacked storage.
- **Validation**: New `test_bit_packing.py` for bit-perfect roundtrip verification.

### Changed
- **Core Engine**: `TurboQuantCodec` and `TurboQuantCodecOptimized` now default to packed storage.
- **LLM Demo**: Fixed `demo_llm.py` to support new optimized codec factory methods and verified against Llama3.
- **Documentation**: Updated all status and improvement logs to reflect production-ready bit-packing.

## [0.1.0] - 2026-03-27

### Added
- **Core (TurboQuant)**: 12x-16x vector quantization with unbiased inner product estimation.
- **Optimized Engine**: GPU-accelerated (CUDA) core operations for fast batch processing.
- **Streaming Support**: `StreamingEncoder` for memory-efficient incremental sequence encoding.
- **Mixed Precision**: Support for FP8/INT8/INT4 quantization.
- **Monitoring**: `MetricsCollector` and `TurboQuantLogger` for production observability.
- **Production Readiness**:
  - `torch.compile` and `AOTInductor` support for optimized inference.
  - Distributed multi-GPU support for KV cache scaling.
- **Ecosystem Integrations**:
  - **LlamaIndex**: `TurboQuantEmbedding` and `TurboQuantVectorStore`.
  - **LangChain**: `TurboQuantEmbeddings` and `TurboQuantFAISS`.
  - **VLLM**: Adapter for PagedAttention hooks.
  - **TGI**: Adapter for Text Generation Inference.
  - **Haystack**: `TurboQuantDocumentStore` for compressed RAG.
  - **Ollama**: Integration for local model embeddings.
- **Tools**:
  - **Dashboard**: Gradio-based visual benchmarker.
  - **CLI**: Comprehensive command-line interface for quantization tasks.
- **TypeScript Support**: Full TFJS implementation in `ts/`.

### Changed
- **Integrations**: `integrations/huggingface.py` now provides a concrete attention wrapper with compressed KV-cache round-tripping.
- **Haystack**: `TurboQuantDocumentEmbedder` now compresses embeddings before returning pipeline results.
- **VLLM**: `patch_vllm_with_turboquant()` now attaches concrete compression and attention helpers to engine instances.
- **Documentation**: Updated `README.md`, `LLM_TESTING.md`, `integrations/plugins/README.md`, and status docs with March 27, 2026 local Ollama validation for `nomic-embed-text:latest` and `llama3:latest`.
- **Memory Reporting**: Documented the difference between FP32 plugin reporting, packed FP16 KV-cache targets, and current Python runtime storage.
