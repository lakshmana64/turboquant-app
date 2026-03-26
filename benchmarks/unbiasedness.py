"""
Unbiasedness Benchmark for TurboQuant

Validates that the inner product estimator is unbiased:
  E[estimate - true] ≈ 0

This is the key mathematical property of TurboQuant.
"""

import torch
from typing import Dict, Any


def benchmark_unbiasedness(
    dim: int = 128,
    num_bits: int = 4,
    qjl_dim: int = 64,
    num_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Benchmark unbiasedness of TurboQuant inner product estimator.
    
    Method:
    1. Generate random vectors x, q
    2. Quantize x to get x_hat
    3. Encode residual with QJL
    4. Estimate <q, x> and compare to true value
    5. Repeat with different QJL seeds to estimate expectation
    
    Args:
        dim: Vector dimension
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        num_samples: Number of Monte Carlo samples
        device: Device to run on
        
    Returns:
        Dict with mean_error, std_error, max_error, bias_ratio
    """
    from ..core.codec import TurboQuantCodec, TurboQuantConfig
    
    device = torch.device(device)
    
    # Generate test vectors
    torch.manual_seed(42)
    x = torch.randn(dim, device=device)
    q = torch.randn(dim, device=device)
    
    # True inner product
    true_dot = (q * x).sum().item()
    
    # Create base codec for Stage 1 (deterministic)
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=42)
    base_codec = TurboQuantCodec(dim, config, device)
    
    # Encode x (Stage 1 + Stage 2 with base seed)
    encoded = base_codec.encode_key(x)
    x_hat = base_codec.decode_key(encoded)
    
    # Stage 1 only estimate (baseline - biased)
    stage1_only = (q * x_hat).sum().item()
    stage1_bias = stage1_only - true_dot
    
    # Full estimate with multiple QJL seeds (Monte Carlo)
    errors = []
    full_errors = []
    
    for seed in range(num_samples):
        # Create codec with different QJL seed
        config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=seed)
        codec = TurboQuantCodec(dim, config, device)
        
        # Re-encode with this seed
        encoded = codec.encode_key(x)
        
        # Estimate
        estimate = codec.estimate_inner_product(q, encoded).item()
        
        # Error
        error = estimate - true_dot
        errors.append(error)
        
        # Also track full error (should be small)
        full_errors.append(abs(error))
    
    errors = torch.tensor(errors)
    full_errors = torch.tensor(full_errors)
    
    # Statistics
    mean_error = errors.mean().item()
    std_error = errors.std().item()
    max_error = full_errors.max().item()
    
    # Bias ratio: |mean| / std should be << 1 for unbiased estimator
    bias_ratio = abs(mean_error) / (std_error + 1e-8)
    
    # Compare to Stage 1 bias
    stage1_abs_bias = abs(stage1_bias)
    improvement = stage1_abs_bias / (abs(mean_error) + 1e-8)
    
    return {
        'true_dot': true_dot,
        'stage1_bias': stage1_bias,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'bias_ratio': bias_ratio,
        'stage1_abs_bias': stage1_abs_bias,
        'improvement': improvement,
        'num_samples': num_samples
    }


def benchmark_unbiasedness_batch(
    dim: int = 128,
    num_bits: int = 4,
    qjl_dim: int = 64,
    num_vectors: int = 100,
    num_queries: int = 10,
    num_seeds: int = 50,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Batch unbiasedness benchmark with multiple vectors and queries.
    
    Args:
        dim: Vector dimension
        num_bits: Scalar quantization bits
        qjl_dim: QJL output dimension
        num_vectors: Number of key vectors
        num_queries: Number of query vectors
        num_seeds: Number of QJL seeds to average over
        device: Device to run on
        
    Returns:
        Dict with aggregate statistics
    """
    from ..core.codec import TurboQuantCodec, TurboQuantConfig
    
    device = torch.device(device)
    
    # Generate test data
    torch.manual_seed(42)
    keys = torch.randn(num_vectors, dim, device=device)
    queries = torch.randn(num_queries, dim, device=device)
    
    # True inner products
    true_dots = queries @ keys.T  # (num_queries, num_vectors)
    
    all_errors = []
    
    for seed in range(num_seeds):
        config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=seed)
        codec = TurboQuantCodec(dim, config, device)
        
        # Encode all keys
        encoded = codec.encode_keys_batch(keys)
        
        # Estimate all inner products
        estimates = torch.zeros(num_queries, num_vectors, device=device)
        for i in range(num_queries):
            estimates[i] = codec.estimate_inner_products(queries[i], encoded)
        
        # Errors
        errors = estimates - true_dots
        all_errors.append(errors)
    
    all_errors = torch.stack(all_errors)  # (num_seeds, num_queries, num_vectors)
    
    # Statistics
    mean_errors = all_errors.mean(dim=[0, 1])  # Average over seeds, queries, vectors
    std_errors = all_errors.std(dim=[0, 1])
    max_abs_error = all_errors.abs().max().item()
    
    return {
        'mean_error': mean_errors.mean().item(),
        'std_error': std_errors.mean().item(),
        'max_error': max_abs_error,
        'num_pairs': num_vectors * num_queries * num_seeds
    }


def run_unbiasedness_suite(
    dim: int = 128,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run comprehensive unbiasedness test suite.
    
    Tests various configurations and reports aggregate results.
    
    Args:
        dim: Vector dimension
        device: Device to run on
        
    Returns:
        Dict with all benchmark results
    """
    results = {}
    
    configs = [
        (2, 32),
        (2, 64),
        (4, 64),
        (4, 128),
        (8, 64),
    ]
    
    print("Running unbiasedness benchmark suite...")
    print(f"  Dimension: {dim}")
    print(f"  Device: {device}")
    print()
    
    for num_bits, qjl_dim in configs:
        print(f"  Testing {num_bits}-bit + {qjl_dim}-bit QJL...")
        
        result = benchmark_unbiasedness(
            dim=dim,
            num_bits=num_bits,
            qjl_dim=qjl_dim,
            num_samples=500,
            device=device
        )
        
        key = f"{num_bits}b_{qjl_dim}d"
        results[key] = result
        
        print(f"    Mean error: {result['mean_error']:.6f}")
        print(f"    Std error:  {result['std_error']:.6f}")
        print(f"    Bias ratio: {result['bias_ratio']:.4f}")
        print()
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--qjl-dim', type=int, default=64)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    result = benchmark_unbiasedness(
        dim=args.dim,
        num_bits=args.bits,
        qjl_dim=args.qjl_dim,
        num_samples=args.samples,
        device=args.device
    )
    
    print("\nUnbiasedness Benchmark Results")
    print("=" * 40)
    print(f"  True dot product:   {result['true_dot']:.4f}")
    print(f"  Stage 1 bias:       {result['stage1_bias']:.4f}")
    print(f"  Mean error (bias):  {result['mean_error']:.6f}")
    print(f"  Std error:          {result['std_error']:.6f}")
    print(f"  Max error:          {result['max_error']:.6f}")
    print(f"  Bias ratio:         {result['bias_ratio']:.4f}")
    print(f"  Improvement vs S1:  {result['improvement']:.2f}x")
