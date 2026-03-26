"""
Multi-Model Ollama Benchmark for TurboQuant

Automatically detects all available Ollama models and benchmarks 
TurboQuant's compression accuracy and efficiency for each.
"""

import requests
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime

from turboquant.sdk.optimize import TurboQuantizer

def get_installed_models():
    """Fetch list of all models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json()["models"]]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def get_embeddings(prompt, model):
    """Fetch embeddings for a specific model."""
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": prompt}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return torch.tensor(response.json()["embedding"])
    except Exception:
        return None

def run_benchmark_for_model(model_name, prompts, qjl_bits=64, sq_bits=2):
    print(f"Testing model: {model_name}...", end=" ", flush=True)
    
    embeddings = []
    for p in prompts:
        vec = get_embeddings(p, model_name)
        if vec is not None:
            embeddings.append(vec)
    
    if len(embeddings) < 2:
        print("SKIP (Failed to fetch embeddings)")
        return None

    keys = torch.stack(embeddings)
    queries = keys.clone() # Self-attention proxy
    d = keys.shape[-1]
    
    # Quantize
    quantizer = TurboQuantizer(d, qjl_bits=qjl_bits, sq_bits=sq_bits)
    encoded = quantizer.encode(keys)
    
    # Estimate
    true_dots = queries @ keys.T
    turbo_dots = quantizer.estimate_batch(queries, encoded)
    
    # Metrics
    mse = torch.mean((true_dots - turbo_dots)**2).item()
    
    # Attention Accuracy (Cosine similarity of the result vectors)
    true_attn = F.softmax(true_dots, dim=-1)
    turbo_attn = F.softmax(turbo_dots, dim=-1)
    
    # Flat cosine similarity between probability distributions
    cosine_sim = F.cosine_similarity(true_attn.view(-1), turbo_attn.view(-1), dim=0).item()
    
    print(f"DONE (Dim: {d}, Cosine: {cosine_sim:.6f})")
    
    return {
        "Model": model_name,
        "Dimension": d,
        "MSE": mse,
        "Attn_Cosine": cosine_sim,
        "Compression": f"{quantizer.compression_factor:.1f}x",
        "Bits_per_dim": (sq_bits * d + qjl_bits) / d
    }

def main():
    models = get_installed_models()
    if not models:
        print("No models found in Ollama. Ensure 'ollama serve' is running.")
        return

    # Standard set of diverse prompts to exercise the embedding space
    prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        "Write a Python script to sort a list of numbers.",
        "The quick brown fox jumps over the lazy dog.",
        "A long time ago in a galaxy far, far away...",
        "Translate 'hello' to Spanish.",
        "Summarize the plot of Inception.",
        "How do I cook a perfect steak?"
    ]

    results = []
    print(f"--- TurboQuant Multi-Model Benchmark ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    print(f"Found {len(models)} models: {', '.join(models)}\n")

    for model in models:
        # Skip models known to not support embeddings if any, or just try all
        res = run_benchmark_for_model(model, prompts)
        if res:
            results.append(res)

    if not results:
        print("No benchmarks completed.")
        return

    # Display Summary Table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"{'OLLAMA MODEL COMPARISON':^80}")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print("Note: Attn_Cosine measures how well the attention distribution is preserved.")
    print("A score of 1.000000 indicates identical attention behavior to FP32.")

if __name__ == "__main__":
    main()
