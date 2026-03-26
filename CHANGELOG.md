# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
