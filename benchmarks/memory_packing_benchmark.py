"""
Benchmark for TurboQuant Bit-packing Memory Savings.
"""

import torch
import time
import sys
import os

# Add parent directory to path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import TurboQuantCodec, TurboQuantConfig

def get_tensor_memory(obj):
    """Estimate memory usage of tensors in an EncodedKey dictionary."""
    total_bytes = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                total_bytes += v.element_size() * v.nelement()
            elif isinstance(v, dict):
                total_bytes += get_tensor_memory(v)
    return total_bytes

def run_benchmark():
    # Parameters
    num_keys = 10000
    dim = 4096  # Llama3 dimension
    num_bits = 4
    qjl_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"--- TurboQuant Memory Benchmark ---")
    print(f"Num Keys:   {num_keys}")
    print(f"Dimension:  {dim}")
    print(f"Bits (SQ):  {num_bits}")
    print(f"QJL Dim:    {qjl_dim}")
    print(f"Device:     {device}")
    print("-" * 35)

    # Generate synthetic data
    keys = torch.randn(num_keys, dim, device=device)
    query = torch.randn(dim, device=device)
    
    # 1. Benchmark WITHOUT Packing
    config_u = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, pack_bits=False)
    codec_u = TurboQuantCodec(dim, config=config_u, device=device)
    
    start = time.time()
    encoded_u = codec_u.encode_keys_batch(keys)
    encode_time_u = time.time() - start
    
    mem_u = get_tensor_memory(encoded_u)
    
    # 2. Benchmark WITH Packing
    config_p = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, pack_bits=True)
    codec_p = TurboQuantCodec(dim, config=config_p, device=device)
    
    start = time.time()
    encoded_p = codec_p.encode_keys_batch(keys)
    encode_time_p = time.time() - start
    
    mem_p = get_tensor_memory(encoded_p)
    
    # 3. Accuracy check (ensure both give same results)
    est_u = codec_u.estimate_inner_products(query, encoded_u)
    est_p = codec_p.estimate_inner_products(query, encoded_p)
    max_diff = (est_u - est_p).abs().max().item()
    
    # Results
    baseline_fp16 = num_keys * dim * 2
    baseline_fp32 = num_keys * dim * 4
    
    print(f"Results for {num_keys} keys:")
    print(f"Baseline (FP32):      {baseline_fp32 / 1024**2:8.2f} MB")
    print(f"Baseline (FP16):      {baseline_fp16 / 1024**2:8.2f} MB")
    print(f"Unpacked Storage:     {mem_u / 1024**2:8.2f} MB")
    print(f"Packed Storage:       {mem_p / 1024**2:8.2f} MB")
    print("-" * 35)
    print(f"Effective Savings (vs FP16): {baseline_fp16 / mem_p:.2f}x")
    print(f"Effective Savings (vs FP32): {baseline_fp32 / mem_p:.2f}x")
    print(f"Packing Speedup:             {mem_u / mem_p:.2f}x less memory")
    print(f"Max Diff between P/U:        {max_diff:.2e}")
    print(f"Encode Time (Unpacked):      {encode_time_u:.4f}s")
    print(f"Encode Time (Packed):        {encode_time_p:.4f}s")
    print("-" * 35)

if __name__ == "__main__":
    run_benchmark()
