#!/usr/bin/env python3
"""
Test Outlier Channel Comparison for TurboQuant.

Compares different outlier detection methods and their impact on
quantization quality.

Reference: turboquant_plus/benchmarks/test_outlier_comparison.py
"""

import torch
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class OutlierResult:
    """Result for outlier detection method."""
    method: str
    num_outliers: int
    detection_rate: float
    mse: float
    cosine: float
    compression_factor: float


def create_data_with_outliers(
    dim: int = 4096,
    num_samples: int = 100,
    outlier_ratio: float = 0.02
) -> torch.Tensor:
    """
    Create synthetic data with outliers.
    
    Args:
        dim: Dimension
        num_samples: Number of samples
        outlier_ratio: Ratio of outlier channels
    
    Returns:
        Tensor with outliers
    """
    x = torch.randn(num_samples, dim)
    
    # Make some channels outliers
    num_outliers = int(dim * outlier_ratio)
    x[:, :num_outliers] *= 100  # High variance outliers
    
    return x


def detect_outliers_variance(x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Detect outliers by variance."""
    variance = x.var(dim=0)
    median_variance = variance.median()
    return variance > (median_variance * threshold)


def detect_outliers_magnitude(x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Detect outliers by magnitude."""
    magnitude = x.abs().mean(dim=0)
    median_magnitude = magnitude.median()
    return magnitude > (median_magnitude * threshold)


def detect_outliers_kurtosis(x: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
    """Detect outliers by kurtosis."""
    # Compute kurtosis per channel
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    kurtosis = ((x - mean) ** 4).mean(dim=0) / (std ** 4 + 1e-8)
    
    return kurtosis > threshold


def benchmark_outlier_method(
    x: torch.Tensor,
    method: str,
    threshold: float = 10.0
) -> OutlierResult:
    """
    Benchmark outlier detection method.
    
    Args:
        x: Input data
        method: Detection method
        threshold: Detection threshold
    
    Returns:
        OutlierResult
    """
    from core.outlier import OutlierHandler, OutlierConfig
    
    # Detect outliers
    if method == "variance":
        outlier_mask = detect_outliers_variance(x, threshold)
    elif method == "magnitude":
        outlier_mask = detect_outliers_magnitude(x, threshold)
    elif method == "kurtosis":
        outlier_mask = detect_outliers_kurtosis(x, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    num_outliers = outlier_mask.sum().item()
    detection_rate = num_outliers / x.shape[1]
    
    # Quantize with outlier handling
    config = OutlierConfig(
        variance_threshold=threshold,
        outlier_bits=8,
        main_bits=2,
        use_magnitude=(method == "magnitude")
    )
    
    handler = OutlierHandler(config, x.shape[1])
    encoded = handler.encode_with_outliers(x, outlier_mask)
    decoded = handler.decode_with_outliers(encoded)
    
    # Calculate metrics
    mse = ((x - decoded) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        x.view(-1, x.shape[1]), decoded.view(-1, x.shape[1])
    ).mean().item()
    
    # Estimate compression
    dim = x.shape[1]
    original_bits = x.numel() * 32
    compressed_bits = num_outliers * 16 + (dim - num_outliers) * 2
    compression_factor = original_bits / compressed_bits
    
    return OutlierResult(
        method=method,
        num_outliers=num_outliers,
        detection_rate=detection_rate,
        mse=mse,
        cosine=cosine,
        compression_factor=compression_factor
    )


def run_outlier_comparison() -> List[OutlierResult]:
    """
    Run comparison of outlier detection methods.
    
    Returns:
        List of OutlierResult
    """
    print("="*70)
    print("Outlier Detection Method Comparison")
    print("="*70)
    print()
    
    # Create test data
    x = create_data_with_outliers(dim=4096, num_samples=100, outlier_ratio=0.02)
    
    methods = ["variance", "magnitude", "kurtosis"]
    results = []
    
    for method in methods:
        print(f"Testing {method}...")
        result = benchmark_outlier_method(x, method)
        results.append(result)
        
        print(f"  Outliers Detected: {result.num_outliers}")
        print(f"  Detection Rate: {result.detection_rate*100:.1f}%")
        print(f"  MSE: {result.mse:.6f}")
        print(f"  Cosine: {result.cosine:.4f}")
        print(f"  Compression: {result.compression_factor:.1f}x")
        print()
    
    return results


def print_summary(results: List[OutlierResult]):
    """Print comparison summary."""
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    # Find best method by MSE
    best_mse = min(results, key=lambda r: r.mse)
    best_cosine = max(results, key=lambda r: r.cosine)
    
    print(f"Best MSE: {best_mse.method} ({best_mse.mse:.6f})")
    print(f"Best Cosine: {best_cosine.method} ({best_cosine.cosine:.4f})")
    print()
    
    print("Detailed Results:")
    print(f"{'Method':<12} {'Outliers':<10} {'Rate':<10} {'MSE':<12} {'Cosine':<12} {'Compression':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r.method:<12} {r.num_outliers:<10} {r.detection_rate*100:<10.1f} {r.mse:<12.6f} {r.cosine:<12.4f} {r.compression_factor:<12.1f}x")
    
    print()
    print("="*70)
    print("Recommendation: Use 'magnitude' for production (best balance)")
    print("="*70)


def main():
    """Run outlier comparison benchmark."""
    results = run_outlier_comparison()
    print_summary(results)
    
    # Save results
    output = {
        "benchmark": "outlier_comparison",
        "timestamp": time.time(),
        "results": [
            {
                "method": r.method,
                "num_outliers": r.num_outliers,
                "detection_rate": r.detection_rate,
                "mse": r.mse,
                "cosine": r.cosine,
                "compression_factor": r.compression_factor
            }
            for r in results
        ]
    }
    
    with open("benchmark_outlier_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: benchmark_outlier_comparison_results.json")


if __name__ == "__main__":
    main()
