"""
TurboQuant Basic Usage Example

This script demonstrates how to compress a tensor and estimate inner products.
"""

import torch
from turboquant.sdk.optimize import optimize

def main():
    # 1. Generate some random data (e.g., key and query vectors)
    # head_dim=128, num_keys=1000, num_queries=10
    keys = torch.randn(1000, 128)
    queries = torch.randn(10, 128)
    
    print(f"Original keys shape: {keys.shape}")
    print(f"Original queries shape: {queries.shape}")

    # 2. Compress the keys using TurboQuant
    # sq_bits=2 (Stage 1), qjl_bits=64 (Stage 2)
    print("\nCompressing keys...")
    encoded, quantizer = optimize(keys, sq_bits=2, qjl_bits=64)
    
    # 3. Estimate inner products
    print("Estimating inner products...")
    estimates = quantizer.estimate_batch(queries, encoded)
    
    # 4. Compare with ground truth
    true_dots = queries @ keys.T
    mse = torch.mean((true_dots - estimates)**2).item()
    
    print("\nResults:")
    print(f"Compression Factor: {quantizer.compression_factor:.2f}x")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Estimated Dot Product (first query, first key): {estimates[0, 0]:.4f}")
    print(f"True Dot Product (first query, first key):      {true_dots[0, 0]:.4f}")

if __name__ == "__main__":
    main()
