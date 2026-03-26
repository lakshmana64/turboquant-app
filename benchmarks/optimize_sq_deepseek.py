"""
Stage 1 Optimization for DeepSeek-Coder:1.3b

Sweeps through Stage 1 (SQ) bits to find the optimal base 
for this high-precision model.
"""

import requests
import torch
import torch.nn.functional as F

from turboquant.sdk.optimize import TurboQuantizer

def get_embeddings(prompt, model="deepseek-coder:1.3b"):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": prompt}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return torch.tensor(response.json()["embedding"])
    except Exception:
        return None

def run_stage1_sweep():
    model = "deepseek-coder:1.3b"
    print(f"--- Stage 1 Optimization for {model} (QJL Bits Fixed at 256) ---")
    
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
    
    embeddings = []
    for p in prompts:
        vec = get_embeddings(p, model)
        if vec is not None:
            embeddings.append(vec)
            
    keys = torch.stack(embeddings)
    queries = keys.clone()
    d = keys.shape[-1]
    
    # Sweep SQ bits
    sq_options = [2, 3, 4]
    qjl_bits = 256
    
    # Ground Truth
    true_dots = queries @ keys.T
    true_attn = F.softmax(true_dots, dim=-1)
    
    print(f"{'SQ Bits':<10} | {'Bits/Dim':<10} | {'Compression':<12} | {'Attn Cosine':<12} | {'MSE':<10}")
    print("-" * 65)
    
    for sq in sq_options:
        quantizer = TurboQuantizer(d, qjl_bits=qjl_bits, sq_bits=sq)
        encoded = quantizer.encode(keys)
        
        turbo_dots = quantizer.estimate_batch(queries, encoded)
        turbo_attn = F.softmax(turbo_dots, dim=-1)
        
        mse = torch.mean((true_dots - turbo_dots)**2).item()
        cosine_sim = F.cosine_similarity(true_attn.view(-1), turbo_attn.view(-1), dim=0).item()
        
        bits_per_dim = (sq * d + qjl_bits) / d
        comp_ratio = 32 / bits_per_dim
        
        print(f"{sq:<10} | {bits_per_dim:<10.3f} | {comp_ratio:<12.1f}x | {cosine_sim:<12.6f} | {mse:<10.2e}")

if __name__ == "__main__":
    run_stage1_sweep()
