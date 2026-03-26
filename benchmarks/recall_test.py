"""
Recall@K Benchmark for TurboQuant

Measures how well TurboQuant preserves the relative ordering of vectors
for Top-K retrieval tasks.
"""

import torch
import time
from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig

def calculate_recall(true_indices, pred_indices):
    """Calculate Recall@K."""
    n_queries = true_indices.shape[0]
    recall = 0
    for i in range(n_queries):
        # Intersection of top-K results
        intersection = torch.isin(pred_indices[i], true_indices[i])
        recall += intersection.sum().item() / true_indices.shape[1]
    return recall / n_queries

def run_recall_benchmark():
    n_keys = 5000
    n_queries = 100
    dim = 128
    K = 10
    
    print(f"--- Recall@K Benchmark (N={n_keys}, Dim={dim}, K={K}) ---")
    
    # Setup data
    torch.manual_seed(42)
    keys = torch.randn(n_keys, dim)
    queries = torch.randn(n_queries, dim)
    
    # True Top-K
    true_dots = queries @ keys.T
    _, true_indices = torch.topk(true_dots, K, dim=1)
    
    configs = [
        ("TurboQuant (2b + 64 bits)", TurboQuantConfig(num_bits=2, qjl_dim=64)),
        ("TurboQuant (4b + 128 bits)", TurboQuantConfig(num_bits=4, qjl_dim=128)),
        ("Scalar Quant only (2b)", TurboQuantConfig(num_bits=2, qjl_dim=0)),
        ("Scalar Quant only (4b)", TurboQuantConfig(num_bits=4, qjl_dim=0)),
    ]
    
    print(f"{'Configuration':<30} | {'Recall@K':<10} | {'Encoding Time':<15}")
    print("-" * 65)
    
    for label, config in configs:
        codec = TurboQuantCodec(dim, config=config)
        
        # 1. Encoding
        start = time.time()
        encoded = codec.encode_keys_batch(keys)
        encode_time = time.time() - start
        
        # 2. Retrieval
        estimates = codec.estimate_inner_products(queries, encoded)
        _, pred_indices = torch.topk(estimates, K, dim=1)
        
        recall = calculate_recall(true_indices, pred_indices)
        
        print(f"{label:<30} | {recall:>10.4f} | {encode_time:>13.4f}s")

if __name__ == "__main__":
    run_recall_benchmark()
