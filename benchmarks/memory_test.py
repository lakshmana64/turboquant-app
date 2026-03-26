"""
Memory Benchmark for TurboQuant

Measures actual memory footprint of TurboQuant's compressed keys.
"""

import torch
import sys
from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig

def get_obj_size(obj):
    """Deep size estimator for nested dictionaries of tensors."""
    size = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += sys.getsizeof(k) + get_obj_size(v)
    elif isinstance(obj, torch.Tensor):
        size += obj.element_size() * obj.nelement()
    else:
        size += sys.getsizeof(obj)
    return size

def run_memory_benchmark():
    n_keys = 100000
    dim = 128
    
    print(f"--- Memory Benchmark (N={n_keys}, Dim={dim}) ---")
    
    # 1. Baseline (FP16)
    keys_fp16 = torch.randn(n_keys, dim, dtype=torch.float16)
    size_fp16 = keys_fp16.element_size() * keys_fp16.nelement() / (1024**2)
    
    # 2. Baseline (FP32)
    keys_fp32 = torch.randn(n_keys, dim, dtype=torch.float32)
    size_fp32 = keys_fp32.element_size() * keys_fp32.nelement() / (1024**2)

    # 3. TurboQuant (2-bit SQ + 64-bit QJL)
    config = TurboQuantConfig(num_bits=2, qjl_dim=64)
    codec = TurboQuantCodec(dim, config=config)
    
    # Encode
    encoded = codec.encode_keys_batch(keys_fp32)
    
    # Calculate size of the 'encoded' dictionary
    # In practice, Stage 1 indices are stored as uint8
    # QJL signs are stored as bits (packed)
    size_tq = get_obj_size(encoded) / (1024**2)
    
    print(f"{'Method':<15} | {'Size (MB)':<12} | {'Bits/Dim':<12}")
    print("-" * 45)
    print(f"{'FP32':<15} | {size_fp32:>12.2f} | {32.0:>12.1f}")
    print(f"{'FP16':<15} | {size_fp16:>12.2f} | {16.0:>12.1f}")
    print(f"{'TurboQuant':<15} | {size_tq:>12.2f} | {(size_tq*1024*1024*8)/(n_keys*dim):>12.1f}")
    
    reduction = (size_fp16 - size_tq) / size_fp16
    print(f"\nReduction vs FP16: {reduction*100:.1f}%")

if __name__ == "__main__":
    run_memory_benchmark()
