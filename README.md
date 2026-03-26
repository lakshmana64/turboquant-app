# TurboQuant

TurboQuant is a two-stage quantization toolkit for unbiased inner-product estimation on high-dimensional vectors, built for LLM KV-cache compression, embedding retrieval, and memory-efficient vector workloads.

## Links

- GitHub: `https://github.com/lakshmana64/turboquant-app`
- Paper: https://arxiv.org/abs/2504.19874
- Plugin docs: `integrations/plugins/README.md`

## Why TurboQuant

- Unbiased inner-product estimation instead of plain low-bit approximation
- Practical compression for KV-cache and embedding-heavy systems
- Python package, CLI, dashboard, plugin adapters, and a TypeScript reference port in one repo
- Validated locally against real Ollama models on this machine

## What’s In The Repo

- `turboquant/`: installable Python package
- `app.py`: Gradio dashboard for interactive compression experiments
- `cli/` and `turboquant/cli/`: command-line quantization workflow
- `integrations/plugins/`: Ollama, OpenAI, SentenceTransformers, LangChain, LlamaIndex, VLLM, TGI, and Haystack adapters
- `ts/`: TypeScript reference implementation
- `benchmarks/`: accuracy, memory, recall, and Ollama-based validation scripts

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

See `integrations/plugins/README.md` for detailed usage and local plugin validation results.

## Local Validation

Real Ollama validation was run on March 27, 2026 on this machine.

### Models Verified

- `nomic-embed-text:latest`
- `llama3:latest`

### Direct Plugin Results

Using `num_bits=4` and `qjl_dim=64`:

| Model | Dim | Compression Ratio | Compression Factor | MSE | Correlation |
|------|-----|-------------------|--------------------|-----|-------------|
| `nomic-embed-text:latest` | `768` | `12.76%` | `7.84x` | `0.00137484` | `0.9999999943` |
| `llama3:latest` | `4096` | n/a | `7.97x` | n/a | `0.9999029384` |

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

### Retrieval Smoke Test

For the query `"embedding compression methods"`, the top compressed retrieval result was `"vector compression for embeddings"`.

### Baseline Note

The Ollama plugin and `integrations/ollama_test.py` report compression relative to FP32 embedding storage.

The core SDK and dashboard use an FP16 baseline for KV-cache-oriented reporting, so their `compression_factor` values will be lower for the same `num_bits` and `qjl_dim`. That difference is expected.

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
