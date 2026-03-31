"""
TurboQuant Plus Features - Example Usage Scripts

This script demonstrates how to use all the new turboquant_plus features
added to turboquant-app.
"""

import torch
from core import (
    # Turbo Formats
    TURBO2, TURBO3, TURBO4, get_format, create_codec_from_format,
    
    # PolarQuant
    PolarQuantConfig, PolarQuantCodec, polar_quant_roundtrip,
    
    # Sparse V
    SparseVDecoder, SparseKVCache,
    
    # Asymmetric K/V
    AsymmetricKVConfig, AsymmetricKVCache, create_asymmetric_cache,
    
    # Outlier Handling
    OutlierConfig, OutlierHandler, OutlierAwareCodec,
    
    # Layer-Adaptive
    LayerAdaptiveConfig, LayerAdaptiveKVCache, create_layer_adaptive_cache,
    
    # Norm Correction
    NormCorrectionConfig, NormCorrector, NormCorrectedCodec,
)


def example_turbo_formats():
    """Example: Using Turbo Format Presets"""
    print("\n" + "="*60)
    print("Example 1: Turbo Format Presets")
    print("="*60)
    
    # List all available formats
    from core.turbo_formats import list_formats
    print(list_formats())
    
    # Create codec using turbo4 preset
    dim = 4096
    codec = create_codec_from_format("turbo4", dim=dim)
    
    # Encode/decode
    x = torch.randn(10, dim)
    encoded = codec.encode_key(x)
    decoded = codec.decode_key(encoded)
    
    # Calculate metrics
    mse = ((x - decoded) ** 2).mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        x.view(-1, dim), decoded.view(-1, dim)
    ).mean().item()
    
    print(f"\nTurbo4 Results:")
    print(f"  Dimension: {dim}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine Similarity: {cosine:.4f}")
    print(f"  Compression: {codec.compression_factor:.1f}x")


def example_polar_quant():
    """Example: PolarQuant Algorithm"""
    print("\n" + "="*60)
    print("Example 2: PolarQuant Algorithm")
    print("="*60)
    
    # Create sample data
    x = torch.randn(100, 4096)
    
    # Run PolarQuant with different bit widths
    for bits in [2, 3, 4]:
        x_rec, metrics = polar_quant_roundtrip(x, bits=bits, qjl_dim=64)
        
        print(f"\n{bits}-bit PolarQuant:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  Compression: {metrics['compression_factor']:.1f}x")
        print(f"  Bits/Dim: {metrics['bits_per_dim']:.2f}")


def example_sparse_v():
    """Example: Sparse V Decoding"""
    print("\n" + "="*60)
    print("Example 3: Sparse V Decoding")
    print("="*60)
    
    dim = 4096
    seq_len = 100
    
    # Create decoder
    decoder = SparseVDecoder(dim=dim, num_bits=4, threshold=1e-6)
    
    # Encode V vectors
    v = torch.randn(seq_len, dim)
    encoded_v = decoder.codec.encode(v)
    
    # Create attention weights (sparse - most positions have low attention)
    attn_weights = torch.softmax(torch.randn(1, seq_len) * 5, dim=-1)
    
    # Decode with sparsity
    v_decoded = decoder.decode_sparse(encoded_v, attn_weights)
    
    # Get sparsity statistics
    stats = decoder.get_sparsity_stats()
    
    print(f"\nSparse V Results:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Dimension: {dim}")
    print(f"  Sparsity: {stats['sparsity_percent']}")
    print(f"  Skipped Positions: {stats['skipped_positions']}/{stats['total_positions']}")
    print(f"  Theoretical Speedup: {stats['theoretical_speedup']}")


def example_asymmetric_kv():
    """Example: Asymmetric K/V Cache"""
    print("\n" + "="*60)
    print("Example 4: Asymmetric K/V Cache")
    print("="*60)
    
    # Create asymmetric cache (high-precision K, compressed V)
    cache = create_asymmetric_cache(
        dim=4096,
        k_format="q8_0",    # High precision for Keys
        v_format="turbo4",  # Compressed Values
        enable_sparse_v=True
    )
    
    # Append KV data
    seq_len = 50
    k = torch.randn(seq_len, 4096)
    v = torch.randn(seq_len, 4096)
    cache.append(k, v)
    
    # Get attention output
    q = torch.randn(1, 4096)
    output = cache.get_attention_output(q)
    
    # Get memory usage
    memory = cache.get_memory_usage()
    
    print(f"\nAsymmetric K/V Results:")
    print(f"  K Format: {cache.config.k_format} ({memory['k_memory']['factor']:.1f}x)")
    print(f"  V Format: {cache.config.v_format} ({memory['v_compression_factor']})")
    print(f"  Overall Compression: {memory['overall_compression_factor']}")
    print(f"  Output Shape: {output.shape}")


def example_outlier_handling():
    """Example: Outlier Channel Handling"""
    print("\n" + "="*60)
    print("Example 5: Outlier Channel Handling")
    print("="*60)
    
    # Use OutlierHandler directly instead of OutlierAwareCodec
    config = OutlierConfig(
        variance_threshold=10.0,
        outlier_bits=8,
        main_bits=2,
        min_outliers=1,
        max_outliers=10,
        use_magnitude=True
    )
    
    handler = OutlierHandler(config, dim=512)  # Use smaller dim for demo
    
    # Create data with outliers
    x = torch.randn(20, 512)
    x[:, 0:5] *= 100  # Make first 5 channels outliers
    
    # Detect outliers
    outlier_mask = handler.detect_outliers(x)
    num_outliers = outlier_mask.sum().item()
    
    print(f"\nOutlier Detection Results:")
    print(f"  Dimension: {512}")
    print(f"  Outliers Detected: {num_outliers}")
    print(f"  Detection Method: magnitude-based")
    print(f"  Threshold: {config.variance_threshold}x median")


def example_layer_adaptive():
    """Example: Layer-Adaptive Mode"""
    print("\n" + "="*60)
    print("Example 6: Layer-Adaptive Mode")
    print("="*60)
    
    # Create layer-adaptive cache (last 8 layers at q8_0, rest at turbo4)
    cache = create_layer_adaptive_cache(
        num_layers=32,
        keep_last_n=8,
        default_format="turbo4",
        protected_format="q8_0",
        dim=4096
    )
    
    # Append to all layers
    for layer_idx in range(32):
        k = torch.randn(10, 4096)
        v = torch.randn(10, 4096)
        cache.append(layer_idx, k, v)
    
    # Get memory breakdown
    memory = cache.get_memory_usage()
    
    # Show format distribution
    print(f"\nLayer-Adaptive Results:")
    print(f"  Total Layers: 32")
    print(f"  Protected Layers (q8_0): 8 (layers 24-31)")
    print(f"  Compressed Layers (turbo4): 24 (layers 0-23)")
    print(f"\n  Memory Usage:")
    print(f"    Layer 0: {memory['per_layer'][0]['k_format']}")
    print(f"    Layer 16: {memory['per_layer'][16]['k_format']}")
    print(f"    Layer 31: {memory['per_layer'][31]['k_format']}")
    print(f"  Overall Compression: {memory['overall_compression_factor']:.1f}x")


def example_norm_correction():
    """Example: Norm Correction"""
    print("\n" + "="*60)
    print("Example 7: Norm Correction")
    print("="*60)
    
    from core.codec import TurboQuantCodec, TurboQuantConfig
    
    # Create base codec
    dim = 4096
    base_codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=4))
    
    # Wrap with norm correction
    codec = NormCorrectedCodec(
        base_codec,
        NormCorrectionConfig(),
        calibrate=True
    )
    
    # Calibrate with sample data
    calibration_data = [torch.randn(5, dim) for _ in range(10)]
    stats = codec.calibrate(calibration_data)
    
    # Test on new data
    x = torch.randn(20, dim)
    encoded = codec.encode_with_correction(x)
    decoded = codec.decode_with_correction(encoded)
    
    # Calculate improvement
    mse_before = stats['mse_before']
    mse_after = stats['mse_after']
    improvement = (mse_before - mse_after) / mse_before * 100
    
    print(f"\nNorm Correction Results:")
    print(f"  MSE Before Correction: {mse_before:.6f}")
    print(f"  MSE After Correction: {mse_after:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Calibrated: {codec.calibrated}")


def example_combined_pipeline():
    """Example: Combined Pipeline"""
    print("\n" + "="*60)
    print("Example 8: Combined Pipeline (All Features)")
    print("="*60)
    
    # Create a complete KV cache with all optimizations
    config = LayerAdaptiveConfig(
        num_layers=32,
        dim=4096,
        keep_last_n=8,
        default_k_format="turbo4",
        default_v_format="turbo4",
        protected_k_format="q8_0",
        protected_v_format="q8_0",
        enable_sparse_v=True
    )
    
    cache = LayerAdaptiveKVCache(config)
    
    # Simulate forward pass through all layers
    batch_size = 1
    seq_len = 100
    
    print(f"\nSimulating LLM forward pass...")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Layers: 32")
    print(f"  Dimension: 4096")
    
    for layer_idx in range(32):
        # Generate KV for this layer
        k = torch.randn(seq_len, 4096)
        v = torch.randn(seq_len, 4096)
        
        # Append to cache
        cache.append(layer_idx, k, v)
        
        # Generate query and get attention output
        q = torch.randn(1, 4096)
        output = cache.get_attention_output(layer_idx, q)
    
    # Get final statistics
    stats = cache.get_stats()
    
    print(f"\nPipeline Results:")
    print(f"  Total Sequence Length: {stats['seq_len_per_layer'][0]}")
    print(f"  Overall Compression: {stats['memory']['overall_compression_factor']:.1f}x")
    print(f"  Estimated Memory Savings: {100 - 100/stats['memory']['overall_compression_factor']:.1f}%")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("TurboQuant Plus Features - Example Usage")
    print("="*60)
    
    examples = [
        ("Turbo Formats", example_turbo_formats),
        ("PolarQuant", example_polar_quant),
        ("Sparse V", example_sparse_v),
        ("Asymmetric K/V", example_asymmetric_kv),
        ("Outlier Handling", example_outlier_handling),
        ("Layer-Adaptive", example_layer_adaptive),
        ("Norm Correction", example_norm_correction),
        ("Combined Pipeline", example_combined_pipeline),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
