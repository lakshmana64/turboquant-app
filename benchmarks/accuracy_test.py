"""
Accuracy Benchmark for TurboQuant

Compares TurboQuant against standard Scalar Quantization across various 
bit-rates and dimensions.
"""

import torch
import time
import pandas as pd
from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig

def run_accuracy_benchmark():
    dimensions = [128, 512, 1024]
    sq_bits_list = [2, 4, 8]
    qjl_dims = [32, 64, 128]
    
    results = []
    
    print(f"{'Dim':<6} | {'SQ Bits':<8} | {'QJL Bits':<8} | {'True IP':<10} | {'TQ MSE':<10} | {'SQ MSE':<10}")
    print("-" * 70)
    
    for d in dimensions:
        # Generate test data
        keys = torch.randn(1000, d)
        queries = torch.randn(100, d)
        true_dots = queries @ keys.T
        
        for sq_bits in sq_bits_list:
            for qjl_dim in qjl_dims:
                # 1. TurboQuant
                config = TurboQuantConfig(num_bits=sq_bits, qjl_dim=qjl_dim)
                codec = TurboQuantCodec(d, config=config)
                
                start = time.time()
                encoded = codec.encode_keys_batch(keys)
                estimates = codec.estimate_inner_products(queries, encoded)
                tq_time = time.time() - start
                
                tq_mse = torch.mean((true_dots - estimates)**2).item()
                
                # 2. Standard SQ (Stage 1 only)
                sq_recon = codec.decode_keys(encoded)
                sq_dots = queries @ sq_recon.T
                sq_mse = torch.mean((true_dots - sq_dots)**2).item()
                
                print(f"{d:<6} | {sq_bits:<8} | {qjl_dim:<8} | {true_dots.mean().item():>10.4f} | {tq_mse:>10.6f} | {sq_mse:>10.6f}")
                
                results.append({
                    'dim': d,
                    'sq_bits': sq_bits,
                    'qjl_dim': qjl_dim,
                    'tq_mse': tq_mse,
                    'sq_mse': sq_mse,
                    'reduction': (sq_mse - tq_mse) / sq_mse if sq_mse > 0 else 0
                })
                
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_accuracy_benchmark()
    print("\nSummary of Error Reduction:")
    print(df.groupby('sq_bits')['reduction'].mean())
