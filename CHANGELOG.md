# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
