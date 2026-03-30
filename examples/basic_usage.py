"""
TurboQuant Basic Usage Example

This script demonstrates how to compress a tensor and estimate inner products
using the high-level SDK with bit-packing enabled.
"""

import torch
from turboquant import optimize

def get_memory_usage(encoded_data):
    """Estimate memory usage of tensors in an EncodedKey dictionary."""
    total_bytes = 0
    for key, value in encoded_data.items():
        if isinstance(value, torch.Tensor):
            total_bytes += value.element_size() * value.nelement()
        elif isinstance(value, dict):
            total_bytes += get_memory_usage(value)
    return total_bytes

def main():
    # 1. Generate some random data (e.g., key and query vectors)
    # head_dim=128, num_keys=1000, num_queries=10
    keys = torch.randn(1000, 128)
    queries = torch.randn(10, 128)
    
    print(f"Original keys shape: {keys.shape}")
    print(f"Original queries shape: {queries.shape}")

    # 2. Compress the keys using TurboQuant
    # sq_bits=2 (Stage 1), qjl_bits=64 (Stage 2)
    # Bit-packing is enabled by default in the SDK
    print("\nCompressing keys with 2-bit quantization + bit-packing...")
    encoded, quantizer = optimize(keys, sq_bits=2, qjl_bits=64)
    
    # 3. Estimate inner products
    print("Estimating inner products...")
    estimates = quantizer.estimate_batch(queries, encoded)
    
    # 4. Memory Reporting
    fp16_bytes = keys.nelement() * 2
    tq_bytes = get_memory_usage(encoded)
    
    # 5. Compare with ground truth
    true_dots = queries @ keys.T
    mse = torch.mean((true_dots - estimates)**2).item()
    
    print("\nResults:")
    print(f"Compression Factor: {fp16_bytes / tq_bytes:.2f}x (vs FP16)")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"True vs Estimate (example): {true_dots[0, 0]:.4f} vs {estimates[0, 0]:.4f}")

if __name__ == "__main__":
    main()
