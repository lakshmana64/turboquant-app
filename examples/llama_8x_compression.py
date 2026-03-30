"""
TurboQuant: 8x Memory Compression Example (Llama3-style)

This example demonstrates how TurboQuant achieves a real 8x reduction 
in memory usage for high-dimensional vectors (like Llama3 embeddings)
while maintaining unbiased inner product estimates.
"""

import torch
import time
from turboquant import optimize

def get_memory_usage(encoded_data):
    """Calculate the actual memory usage of the encoded tensors."""
    total_bytes = 0
    for key, value in encoded_data.items():
        if isinstance(value, torch.Tensor):
            total_bytes += value.element_size() * value.nelement()
        elif isinstance(value, dict):
            total_bytes += get_memory_usage(value)
    return total_bytes

def main():
    # 1. Setup parameters (Llama3 dimensions)
    num_keys = 5000
    dim = 4096  # Llama3-8B attention dimension
    sq_bits = 4 # Stage 1 bits
    qjl_bits = 64 # Stage 2 bits
    
    print(f"--- TurboQuant 8x Compression Magic ---")
    print(f"Vectors:   {num_keys}")
    print(f"Dimension: {dim}")
    print(f"Config:    {sq_bits}-bit SQ + {qjl_bits}-bit QJL")
    print("-" * 40)

    # 2. Generate synthetic embeddings
    # We use random Gaussian data, which is typical for normalized embeddings
    print("Generating synthetic embeddings...")
    keys = torch.randn(num_keys, dim)
    queries = torch.randn(10, dim)
    
    # 3. Baseline memory (FP32)
    # FP32: 4 bytes per element
    fp32_memory = keys.element_size() * keys.nelement()
    # FP16: 2 bytes per element (Standard KV cache baseline)
    fp16_memory = keys.nelement() * 2

    # 4. Compress using TurboQuant (with bit-packing)
    print("Compressing with TurboQuant...")
    start_time = time.time()
    encoded, quantizer = optimize(keys, sq_bits=sq_bits, qjl_bits=qjl_bits, pack_bits=True)
    encode_duration = time.time() - start_time
    
    # 5. Measure real memory
    tq_memory = get_memory_usage(encoded)
    
    # 6. Verify Accuracy
    print("Verifying estimation accuracy...")
    true_dots = queries @ keys.T
    est_dots = quantizer.estimate_batch(queries, encoded)
    
    # Calculate Mean Squared Error and Correlation
    mse = torch.mean((true_dots - est_dots)**2).item()
    correlation = torch.corrcoef(torch.stack([true_dots.flatten(), est_dots.flatten()]))[0, 1].item()

    # 7. Print Results
    print("\n" + "="*40)
    print(f"{'METRIC':<20} | {'VALUE':<15}")
    print("-" * 40)
    print(f"{'FP32 Memory':<20} | {fp32_memory / 1024**2:>8.2f} MB")
    print(f"{'FP16 Memory':<20} | {fp16_memory / 1024**2:>8.2f} MB")
    print(f"{'TurboQuant Memory':<20} | {tq_memory / 1024**2:>8.2f} MB (Bit-Packed)")
    print("-" * 40)
    print(f"{'Compression Factor':<20} | {fp32_memory / tq_memory:>8.2f}x (vs FP32)")
    print(f"{'Compression Factor':<20} | {fp16_memory / tq_memory:>8.2f}x (vs FP16)")
    print("-" * 40)
    print(f"{'Estimation MSE':<20} | {mse:>8.6f}")
    print(f"{'Correlation':<20} | {correlation:>8.6f}")
    print(f"{'Encoding Time':<20} | {encode_duration:>8.4f} seconds")
    print("="*40)

    print("\nNote: The 8x saving vs FP32 is achieved because 4-bit indices")
    print("are packed into bytes, using exactly 0.5 bytes per dimension.")

if __name__ == "__main__":
    main()
