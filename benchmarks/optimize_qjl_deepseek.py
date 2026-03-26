"""
QJL Optimization for DeepSeek-Coder:1.3b

Sweeps through QJL bit configurations to find the optimal balance 
between compression and attention accuracy for this specific model.
"""

import requests
import torch
import torch.nn.functional as F
import pandas as pd

from turboquant.sdk.optimize import TurboQuantizer

def get_embeddings(prompt, model="deepseek-coder:1.3b"):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": prompt}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return torch.tensor(response.json()["embedding"])
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return None

def run_sweep():
    model = "deepseek-coder:1.3b"
    print(f"--- Optimizing QJL Bits for {model} ---")
    
    prompts = [
        "Write a function to calculate Fibonacci numbers.",
        "How to use decorators in Python?",
        "Explain the difference between a list and a tuple.",
        "Implement a simple web server in Go.",
        "What is a pull request?",
        "Standard library functions for string manipulation.",
        "How to handle errors in JavaScript.",
        "Data structures for efficient searching."
    ]
    
    print(f"Fetching embeddings for {len(prompts)} prompts...")
    embeddings = []
    for p in prompts:
        vec = get_embeddings(p, model)
        if vec is not None:
            embeddings.append(vec)
            
    if len(embeddings) < 2:
        print("Failed to get enough data.")
        return

    keys = torch.stack(embeddings)
    queries = keys.clone()
    d = keys.shape[-1]
    
    # Sweep QJL bits
    qjl_options = [32, 64, 128, 256, 512, 1024]
    sq_bits = 2 # Keeping Stage 1 at 2 bits
    
    results = []
    
    print(f"{'QJL Bits':<10} | {'Bits/Dim':<10} | {'Compression':<12} | {'Attn Cosine':<12} | {'MSE':<10}")
    print("-" * 65)
    
    # Ground Truth
    true_dots = queries @ keys.T
    true_attn = F.softmax(true_dots, dim=-1)
    
    for m in qjl_options:
        quantizer = TurboQuantizer(d, qjl_bits=m, sq_bits=sq_bits)
        encoded = quantizer.encode(keys)
        
        # Estimate
        turbo_dots = quantizer.estimate_batch(queries, encoded)
        turbo_attn = F.softmax(turbo_dots, dim=-1)
        
        # Metrics
        mse = torch.mean((true_dots - turbo_dots)**2).item()
        cosine_sim = F.cosine_similarity(true_attn.view(-1), turbo_attn.view(-1), dim=0).item()
        
        bits_per_dim = (sq_bits * d + m) / d
        comp_ratio = 32 / bits_per_dim
        
        print(f"{m:<10} | {bits_per_dim:<10.3f} | {comp_ratio:<12.1f}x | {cosine_sim:<12.6f} | {mse:<10.2e}")
        
        results.append({
            "QJL_Bits": m,
            "Bits_per_dim": bits_per_dim,
            "Compression": comp_ratio,
            "Cosine": cosine_sim,
            "MSE": mse
        })
        
    df = pd.DataFrame(results)
    
    # Find the first QJL bit count that reaches > 0.95 cosine
    target = df[df['Cosine'] > 0.95]
    if not target.empty:
        recommendation = target.iloc[0]['QJL_Bits']
        print(f"\n✅ RECOMMENDATION: Use QJL_BITS = {int(recommendation)} for {model}")
        print(f"This achieves {target.iloc[0]['Cosine']:.4f} accuracy with {target.iloc[0]['Compression']:.1f}x compression.")
    else:
        print("\n⚠️ Note: Even with 1024 QJL bits, cosine didn't reach 0.95. Consider increasing Stage 1 bits (sq_bits).")

if __name__ == "__main__":
    run_sweep()
