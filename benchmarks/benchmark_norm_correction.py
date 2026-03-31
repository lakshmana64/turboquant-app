#!/usr/bin/env python3
"""
Benchmark Norm Correction for TurboQuant.

Measures the quality improvement from norm correction across different
bit widths and model dimensions.

Reference: turboquant_plus/benchmarks/benchmark_norm_correction.py
"""

import torch
import time
from typing import Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Benchmark result for norm correction."""
    dim: int
    num_bits: int
    mse_before: float
    mse_after: float
    improvement_percent: float
    cosine_before: float
    cosine_after: float
    latency_ms: float


def benchmark_norm_correction(
    dim: int = 4096,
    num_bits: int = 4,
    num_samples: int = 100
) -> BenchmarkResult:
    """
    Benchmark norm correction for a specific configuration.
    
    Args:
        dim: Vector dimension
        num_bits: Quantization bits
        num_samples: Number of samples to test
    
    Returns:
        BenchmarkResult with metrics
    """
    from core.norm_correction import NormCorrectionConfig, NormCorrectedCodec
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    # Create codec with norm correction
    base_codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=num_bits))
    codec = NormCorrectedCodec(base_codec, NormCorrectionConfig(), calibrate=True)
    
    # Calibration data
    calibration_data = [torch.randn(5, dim) for _ in range(5)]
    codec.calibrate(calibration_data)
    
    # Test data
    test_data = [torch.randn(10, dim) for _ in range(num_samples)]
    
    # Benchmark without norm correction
    mse_before_list = []
    cosine_before_list = []
    
    for x in test_data:
        encoded = base_codec.encode_key(x)
        decoded = base_codec.decode_key(encoded)
        
        mse = ((x - decoded) ** 2).mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            x.view(-1, dim), decoded.view(-1, dim)
        ).mean().item()
        
        mse_before_list.append(mse)
        cosine_before_list.append(cosine)
    
    # Benchmark with norm correction
    mse_after_list = []
    cosine_after_list = []
    latencies = []
    
    for x in test_data:
        start = time.time()
        encoded = codec.encode_with_correction(x)
        decoded = codec.decode_with_correction(encoded)
        latency = (time.time() - start) * 1000
        
        mse = ((x - decoded) ** 2).mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            x.view(-1, dim), decoded.view(-1, dim)
        ).mean().item()
        
        mse_after_list.append(mse)
        cosine_after_list.append(cosine)
        latencies.append(latency)
    
    # Calculate metrics
    mse_before = sum(mse_before_list) / len(mse_before_list)
    mse_after = sum(mse_after_list) / len(mse_after_list)
    cosine_before = sum(cosine_before_list) / len(cosine_before_list)
    cosine_after = sum(cosine_after_list) / len(cosine_after_list)
    avg_latency = sum(latencies) / len(latencies)
    
    improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
    
    return BenchmarkResult(
        dim=dim,
        num_bits=num_bits,
        mse_before=mse_before,
        mse_after=mse_after,
        improvement_percent=improvement,
        cosine_before=cosine_before,
        cosine_after=cosine_after,
        latency_ms=avg_latency
    )


def run_full_benchmark() -> List[BenchmarkResult]:
    """
    Run full benchmark across multiple configurations.
    
    Returns:
        List of BenchmarkResult
    """
    results = []
    
    configs = [
        (512, 2), (512, 4),
        (2048, 2), (2048, 4),
        (4096, 2), (4096, 4),
    ]
    
    print("="*70)
    print("Norm Correction Benchmark")
    print("="*70)
    print()
    
    for dim, num_bits in configs:
        print(f"Testing dim={dim}, bits={num_bits}...")
        result = benchmark_norm_correction(dim, num_bits)
        results.append(result)
        
        print(f"  MSE Before: {result.mse_before:.6f}")
        print(f"  MSE After:  {result.mse_after:.6f}")
        print(f"  Improvement: {result.improvement_percent:.1f}%")
        print(f"  Cosine Before: {result.cosine_before:.4f}")
        print(f"  Cosine After:  {result.cosine_after:.4f}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print()
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    avg_improvement = sum(r.improvement_percent for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    print(f"Average Improvement: {avg_improvement:.1f}%")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print()
    
    print("Detailed Results:")
    print(f"{'Dim':<8} {'Bits':<8} {'MSE Before':<12} {'MSE After':<12} {'Improvement':<12} {'Cosine After':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r.dim:<8} {r.num_bits:<8} {r.mse_before:<12.6f} {r.mse_after:<12.6f} {r.improvement_percent:<12.1f} {r.cosine_after:<12.4f}")
    
    print()
    print("="*70)


def main():
    """Run norm correction benchmark."""
    results = run_full_benchmark()
    print_summary(results)
    
    # Save results to JSON
    output = {
        "benchmark": "norm_correction",
        "timestamp": time.time(),
        "results": [
            {
                "dim": r.dim,
                "num_bits": r.num_bits,
                "mse_before": r.mse_before,
                "mse_after": r.mse_after,
                "improvement_percent": r.improvement_percent,
                "cosine_before": r.cosine_before,
                "cosine_after": r.cosine_after,
                "latency_ms": r.latency_ms
            }
            for r in results
        ]
    }
    
    with open("benchmark_norm_correction_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: benchmark_norm_correction_results.json")


if __name__ == "__main__":
    main()
