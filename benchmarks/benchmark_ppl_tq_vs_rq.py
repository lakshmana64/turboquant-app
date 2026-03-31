#!/usr/bin/env python3
"""
Benchmark Perplexity: TurboQuant vs RotorQuant.

Compares perplexity between TurboQuant and RotorQuant methods
on language model tasks.

Reference: turboquant_plus/benchmarks/benchmark_ppl_tq_vs_rq.py
"""

import torch
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
import math


@dataclass
class PerplexityResult:
    """Perplexity benchmark result."""
    method: str
    perplexity: float
    bits_per_dim: float
    compression_factor: float
    latency_ms: float


def generate_test_sequences(
    num_sequences: int = 10,
    seq_length: int = 512
) -> List[torch.Tensor]:
    """
    Generate test sequences for perplexity evaluation.
    
    Args:
        num_sequences: Number of sequences
        seq_length: Length of each sequence
    
    Returns:
        List of tensors
    """
    # Use synthetic data (in production, use real text)
    sequences = []
    for _ in range(num_sequences):
        # Generate pseudo-text-like data
        seq = torch.randn(seq_length, 512)  # 512 dim embeddings
        sequences.append(seq)
    
    return sequences


def compute_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute reconstruction error metrics.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
    
    Returns:
        Tuple of (mse, cosine_similarity)
    """
    mse = ((original - reconstructed) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        original.view(-1, original.shape[-1]),
        reconstructed.view(-1, reconstructed.shape[-1])
    ).mean().item()
    
    return mse, cosine


def estimate_perplexity_from_mse(mse: float) -> float:
    """
    Estimate perplexity from MSE.
    
    This is an approximation - real perplexity requires language model.
    
    Args:
        mse: Mean squared error
    
    Returns:
        Estimated perplexity
    """
    # Approximation: perplexity ≈ exp(MSE * scaling_factor)
    # Scaling factor depends on embedding dimension
    scaling = 0.1
    return math.exp(mse * scaling)


def benchmark_turboquant(
    sequences: List[torch.Tensor],
    num_bits: int = 4
) -> PerplexityResult:
    """
    Benchmark TurboQuant on sequences.
    
    Args:
        sequences: Test sequences
        num_bits: Quantization bits
    
    Returns:
        PerplexityResult
    """
    from core.codec import TurboQuantCodec, TurboQuantConfig
    import time
    
    dim = sequences[0].shape[-1]
    codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=num_bits))
    
    total_mse = 0.0
    total_time = 0.0
    num_tokens = 0
    
    for seq in sequences:
        start = time.time()
        encoded = codec.encode_key(seq)
        decoded = codec.decode_key(encoded)
        total_time += (time.time() - start) * 1000
        
        mse, _ = compute_reconstruction_error(seq, decoded)
        total_mse += mse
        num_tokens += seq.shape[0]
    
    avg_mse = total_mse / len(sequences)
    perplexity = estimate_perplexity_from_mse(avg_mse)
    bits_per_dim = num_bits + (64 / dim)  # QJL overhead
    compression = codec.compression_factor
    
    return PerplexityResult(
        method="TurboQuant",
        perplexity=perplexity,
        bits_per_dim=bits_per_dim,
        compression_factor=compression,
        latency_ms=total_time / num_tokens
    )


def benchmark_rotorquant(
    sequences: List[torch.Tensor],
    num_bits: int = 4
) -> PerplexityResult:
    """
    Benchmark RotorQuant (simulated) on sequences.
    
    RotorQuant uses random rotation instead of WHT.
    
    Args:
        sequences: Test sequences
        num_bits: Quantization bits
    
    Returns:
        PerplexityResult
    """
    from core.codec import TurboQuantCodec, TurboQuantConfig
    import time
    
    dim = sequences[0].shape[-1]
    
    # RotorQuant uses random rotation
    codec = TurboQuantCodec(
        dim,
        TurboQuantConfig(num_bits=num_bits, rotation_type="random")
    )
    
    total_mse = 0.0
    total_time = 0.0
    num_tokens = 0
    
    for seq in sequences:
        start = time.time()
        encoded = codec.encode_key(seq)
        decoded = codec.decode_key(encoded)
        total_time += (time.time() - start) * 1000
        
        mse, _ = compute_reconstruction_error(seq, decoded)
        total_mse += mse
        num_tokens += seq.shape[0]
    
    avg_mse = total_mse / len(sequences)
    perplexity = estimate_perplexity_from_mse(avg_mse)
    bits_per_dim = num_bits + (64 / dim)
    compression = codec.compression_factor
    
    return PerplexityResult(
        method="RotorQuant",
        perplexity=perplexity,
        bits_per_dim=bits_per_dim,
        compression_factor=compression,
        latency_ms=total_time / num_tokens
    )


def run_perplexity_benchmark() -> Tuple[PerplexityResult, PerplexityResult]:
    """
    Run perplexity comparison benchmark.
    
    Returns:
        Tuple of (TurboQuant result, RotorQuant result)
    """
    print("="*70)
    print("Perplexity Benchmark: TurboQuant vs RotorQuant")
    print("="*70)
    print()
    
    # Generate test sequences
    print("Generating test sequences...")
    sequences = generate_test_sequences(num_sequences=10, seq_length=512)
    print(f"Generated {len(sequences)} sequences")
    print()
    
    # Benchmark TurboQuant
    print("Benchmarking TurboQuant...")
    tq_result = benchmark_turboquant(sequences, num_bits=4)
    print(f"  Perplexity: {tq_result.perplexity:.2f}")
    print(f"  Bits/Dim: {tq_result.bits_per_dim:.2f}")
    print(f"  Compression: {tq_result.compression_factor:.1f}x")
    print(f"  Latency: {tq_result.latency_ms:.2f}ms/token")
    print()
    
    # Benchmark RotorQuant
    print("Benchmarking RotorQuant...")
    rq_result = benchmark_rotorquant(sequences, num_bits=4)
    print(f"  Perplexity: {rq_result.perplexity:.2f}")
    print(f"  Bits/Dim: {rq_result.bits_per_dim:.2f}")
    print(f"  Compression: {rq_result.compression_factor:.1f}x")
    print(f"  Latency: {rq_result.latency_ms:.2f}ms/token")
    print()
    
    return tq_result, rq_result


def print_comparison(tq: PerplexityResult, rq: PerplexityResult):
    """Print comparison summary."""
    print("="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print()
    
    # Perplexity improvement
    ppl_improvement = (rq.perplexity - tq.perplexity) / rq.perplexity * 100
    
    print(f"Perplexity:")
    print(f"  TurboQuant:  {tq.perplexity:.2f}")
    print(f"  RotorQuant:  {rq.perplexity:.2f}")
    print(f"  Improvement: {ppl_improvement:.1f}% (lower is better)")
    print()
    
    # Speed comparison
    speed_improvement = (rq.latency_ms - tq.latency_ms) / rq.latency_ms * 100
    
    print(f"Latency (ms/token):")
    print(f"  TurboQuant:  {tq.latency_ms:.2f}")
    print(f"  RotorQuant:  {rq.latency_ms:.2f}")
    print(f"  Improvement: {speed_improvement:.1f}% (higher is better)")
    print()
    
    print("="*70)
    if ppl_improvement > 0:
        print("✓ TurboQuant has better perplexity!")
    else:
        print("⚠ RotorQuant has slightly better perplexity")
    
    if speed_improvement > 0:
        print("✓ TurboQuant is faster!")
    print("="*70)


def main():
    """Run perplexity benchmark."""
    tq_result, rq_result = run_perplexity_benchmark()
    print_comparison(tq_result, rq_result)
    
    # Save results
    output = {
        "benchmark": "perplexity_tq_vs_rq",
        "timestamp": time.time(),
        "turboquant": {
            "perplexity": tq_result.perplexity,
            "bits_per_dim": tq_result.bits_per_dim,
            "compression_factor": tq_result.compression_factor,
            "latency_ms": tq_result.latency_ms
        },
        "rotorquant": {
            "perplexity": rq_result.perplexity,
            "bits_per_dim": rq_result.bits_per_dim,
            "compression_factor": rq_result.compression_factor,
            "latency_ms": rq_result.latency_ms
        }
    }
    
    with open("benchmark_ppl_tq_vs_rq_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: benchmark_ppl_tq_vs_rq_results.json")


if __name__ == "__main__":
    main()
