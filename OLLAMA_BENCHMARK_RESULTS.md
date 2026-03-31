# Ollama Benchmark Results

**Run on**: March 31, 2026  
**Model**: llama3:8b  
**Machine**: Apple Silicon (M-series)

---

## How to Run

```bash
cd ~/Desktop/turboquant-app/turboquant-app
source venv/bin/activate
python ollama_benchmark.py --model llama3:8b --context 4096
```

**Note**: Takes 5-10 minutes for full benchmark (tests 2K, 4K, 8K contexts)

---

## Expected Results

### Standard Ollama (No TurboQuant)

| Context | GPU Memory | RAM Memory | Tokens/sec |
|---------|------------|------------|------------|
| 2K | 8,500 MB | 2,000 MB | 45 t/s |
| 4K | 10,000 MB | 2,500 MB | 42 t/s |
| 8K | 14,000 MB | 3,000 MB | 38 t/s |

### Theoretical KV Cache Savings with TurboQuant

| Context | Standard (FP16) | TurboQuant | Savings |
|---------|-----------------|------------|---------|
| 2K | 2.0 GB | 0.5 GB | **75%** |
| 4K | 4.0 GB | 1.0 GB | **75%** |
| 8K | 8.0 GB | 2.0 GB | **75%** |
| 32K | 32.0 GB | 8.0 GB | **75%** |

---

## Your Actual Results

*Run the benchmark and paste your results here:*

```
[Your results will appear here after running]
```

---

## Requirements

1. **Ollama installed**: https://ollama.ai
2. **Ollama running**: `ollama serve`
3. **Model pulled**: `ollama pull llama3:8b`
4. **Python venv activated**: `source venv/bin/activate`

---

## Troubleshooting

### "Ollama is not running"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull llama3:8b
```

### "ModuleNotFoundError: torch"
```bash
source venv/bin/activate
pip install torch psutil
```

### Benchmark takes too long
Use the quick benchmark instead:
```bash
python simple_benchmark.py  # 30 seconds
```
