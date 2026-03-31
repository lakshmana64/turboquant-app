#!/usr/bin/env python3
"""
Simple Benchmark: TurboQuant Memory & Quality

Run this to get REAL numbers for your README.
Takes 30 seconds, produces credible results.

Usage:
    python simple_benchmark.py
"""

import torch
import time
from turboquant import optimize


def benchmark():
    print("="*70)
    print("TurboQuant Simple Benchmark")
    print("="*70)
    print()
    
    # Test parameters
    num_vectors = 1000
    dim = 4096
    
    print(f"Test Configuration:")
    print(f"  • Vectors: {num_vectors:,}")
    print(f"  • Dimension: {dim:,}")
    print(f"  • Total elements: {num_vectors * dim:,}")
    print()
    
    # Generate test data
    print("Generating test data...")
    vectors = torch.randn(num_vectors, dim)
    
    # Measure original size
    original_size_mb = vectors.element_size() * vectors.nelement() / (1024 * 1024)
    print(f"  Original size: {original_size_mb:.1f} MB (FP32)")
    print()
    
    # Import codec directly
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    # Create codec
    codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=4, qjl_dim=64))
    
    # Compress with TurboQuant
    print("Compressing with TurboQuant (4-bit)...")
    start_time = time.time()
    compressed = codec.encode_key(vectors)
    compress_time = (time.time() - start_time) * 1000
    
    # Measure compressed size (estimate from codec)
    memory_usage = codec.get_memory_usage(num_vectors)
    compressed_size_mb = memory_usage['compressed'] / (1024 * 1024)
    compression_ratio = memory_usage['factor']
    
    print(f"  Compressed size: {compressed_size_mb:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print(f"  Memory savings: {(1 - 1/compression_ratio)*100:.1f}%")
    print(f"  Compression time: {compress_time:.1f}ms")
    print()
    
    # Measure quality
    print("Measuring quality (reconstruction)...")
    start_time = time.time()
    reconstructed = codec.decode_key(compressed)
    decompress_time = (time.time() - start_time) * 1000
    
    # Calculate metrics
    mse = ((vectors - reconstructed) ** 2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        vectors.view(-1, dim), 
        reconstructed.view(-1, dim)
    ).mean().item()
    
    # Inner product test
    queries = torch.randn(10, dim)
    original_dots = queries @ vectors.T
    compressed_dots = codec.estimate_inner_products(queries, compressed)
    
    dot_correlation = torch.corrcoef(
        torch.stack([
            original_dots.flatten(),
            compressed_dots.flatten()
        ])
    )[0, 1].item()
    
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.4f} ({cosine_sim*100:.1f}%)")
    print(f"  Dot Product Correlation: {dot_correlation:.4f} ({dot_correlation*100:.1f}%)")
    print(f"  Decompression time: {decompress_time:.1f}ms")
    print()
    
    # Summary table
    print("="*70)
    print("RESULTS (Copy-Paste for README)")
    print("="*70)
    print()
    print("```markdown")
    print("### Benchmark Results")
    print()
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| **Vectors** | {num_vectors:,} |")
    print(f"| **Dimension** | {dim:,} |")
    print(f"| **Original Size** | {original_size_mb:.1f} MB |")
    print(f"| **Compressed Size** | {compressed_size_mb:.1f} MB |")
    print(f"| **Compression Ratio** | {compression_ratio:.1f}x |")
    print(f"| **Memory Savings** | {(1 - compressed_size_mb/original_size_mb)*100:.1f}% |")
    print(f"| **Compression Time** | {compress_time:.1f}ms |")
    print(f"| **Cosine Similarity** | {cosine_sim:.4f} ({cosine_sim*100:.1f}%) |")
    print(f"| **Dot Product Correlation** | {dot_correlation:.4f} ({dot_correlation*100:.1f}%) |")
    print("```")
    print()
    print("="*70)
    
    # Save results
    import json
    results = {
        "num_vectors": num_vectors,
        "dimension": dim,
        "original_size_mb": original_size_mb,
        "compressed_size_mb": compressed_size_mb,
        "compression_ratio": compression_ratio,
        "memory_savings_percent": (1 - compressed_size_mb/original_size_mb)*100,
        "compress_time_ms": compress_time,
        "decompress_time_ms": decompress_time,
        "mse": mse,
        "cosine_similarity": cosine_sim,
        "dot_product_correlation": dot_correlation
    }
    
    with open("simple_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: simple_benchmark_results.json")
    print()
    print("✅ Now add these REAL numbers to your README!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    benchmark()
