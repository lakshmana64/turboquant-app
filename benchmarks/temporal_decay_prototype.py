#!/usr/bin/env python3
"""
Temporal Decay Prototype for Long-Context TurboQuant.

Experimental feature for reducing memory at long contexts by applying
decay factors to older KV cache entries.

Reference: turboquant_plus/benchmarks/temporal_decay_prototype.py
"""

import torch
from typing import Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class TemporalDecayResult:
    """Temporal decay benchmark result."""
    context_length: int
    original_memory_mb: float
    compressed_memory_mb: float
    memory_saved_mb: float
    savings_percent: float
    quality_score: float


def benchmark_temporal_decay(
    dim: int = 4096,
    context_lengths: List[int] = None
) -> List[TemporalDecayResult]:
    """
    Benchmark temporal decay for different context lengths.
    
    Args:
        dim: Hidden dimension
        context_lengths: List of context lengths to test
    
    Returns:
        List of TemporalDecayResult
    """
    from core.temporal_decay import TemporalDecayKVCache, TemporalDecayConfig
    
    if context_lengths is None:
        context_lengths = [1024, 4096, 8192, 16384, 32768]
    
    results = []
    
    print("="*70)
    print("Temporal Decay Benchmark")
    print("="*70)
    print()
    
    for ctx_len in context_lengths:
        print(f"Testing context length: {ctx_len:,}...")
        
        # Create cache with temporal decay
        config = TemporalDecayConfig(
            decay_rate=0.995,
            min_bits=2,
            max_bits=4,
            context_threshold=4096
        )
        
        cache = TemporalDecayKVCache(dim, config)
        
        # Simulate appending context in chunks
        chunk_size = 512
        for i in range(0, ctx_len, chunk_size):
            k = torch.randn(chunk_size, dim)
            v = torch.randn(chunk_size, dim)
            cache.append(k, v, i)
        
        # Get statistics
        stats = cache.get_stats()
        
        # Calculate memory
        original_memory_mb = ctx_len * dim * 4 / (1024 * 1024)  # FP32
        compressed_memory_mb = original_memory_mb / float(stats['compression_ratio'].replace('x', '')) if stats['compression_ratio'] != 'N/A' else original_memory_mb
        memory_saved_mb = original_memory_mb - compressed_memory_mb
        savings_percent = (memory_saved_mb / original_memory_mb) * 100
        
        # Quality score (approximate based on bit width)
        avg_bit_width = stats['avg_bit_width']
        quality_score = avg_bit_width / 4.0  # Normalize to 4-bit baseline
        
        result = TemporalDecayResult(
            context_length=ctx_len,
            original_memory_mb=original_memory_mb,
            compressed_memory_mb=compressed_memory_mb,
            memory_saved_mb=memory_saved_mb,
            savings_percent=savings_percent,
            quality_score=quality_score
        )
        
        results.append(result)
        
        print(f"  Original Memory: {original_memory_mb:.1f} MB")
        print(f"  Compressed Memory: {compressed_memory_mb:.1f} MB")
        print(f"  Memory Saved: {memory_saved_mb:.1f} MB ({savings_percent:.1f}%)")
        print(f"  Avg Bit Width: {avg_bit_width:.2f} bits")
        print(f"  Quality Score: {quality_score:.2f}")
        print()
    
    return results


def print_summary(results: List[TemporalDecayResult]):
    """Print benchmark summary."""
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    print("Memory Savings by Context Length:")
    print(f"{'Context':<12} {'Original':<12} {'Compressed':<12} {'Saved':<12} {'Savings':<12} {'Quality':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r.context_length:<12,} {r.original_memory_mb:<12.1f} {r.compressed_memory_mb:<12.1f} {r.memory_saved_mb:<12.1f} {r.savings_percent:<12.1f}% {r.quality_score:<10.2f}")
    
    print()
    
    # Overall statistics
    avg_savings = sum(r.savings_percent for r in results) / len(results)
    avg_quality = sum(r.quality_score for r in results) / len(results)
    
    print(f"Average Memory Savings: {avg_savings:.1f}%")
    print(f"Average Quality Score: {avg_quality:.2f}")
    print()
    
    # Find best context length for savings
    best_savings = max(results, key=lambda r: r.savings_percent)
    print(f"Best Savings: {best_savings.context_length:,} tokens ({best_savings.savings_percent:.1f}%)")
    print()
    print("="*70)
    print("Recommendation: Enable temporal decay for contexts >8K tokens")
    print("="*70)


def main():
    """Run temporal decay benchmark."""
    results = benchmark_temporal_decay(dim=4096)
    print_summary(results)
    
    # Save results
    output = {
        "benchmark": "temporal_decay",
        "timestamp": time.time(),
        "dim": 4096,
        "results": [
            {
                "context_length": r.context_length,
                "original_memory_mb": r.original_memory_mb,
                "compressed_memory_mb": r.compressed_memory_mb,
                "memory_saved_mb": r.memory_saved_mb,
                "savings_percent": r.savings_percent,
                "quality_score": r.quality_score
            }
            for r in results
        ]
    }
    
    with open("benchmark_temporal_decay_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: benchmark_temporal_decay_results.json")


if __name__ == "__main__":
    main()
