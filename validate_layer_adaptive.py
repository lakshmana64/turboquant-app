"""
Validation script for Layer-Adaptive KV Cache.
"""

import torch
from core.layer_adaptive import LayerAdaptiveKVCache, LayerAdaptiveConfig

def test_layer_adaptive_logic():
    print("Testing Layer-Adaptive Logic...")
    num_layers = 32
    keep_last_n = 8
    dim = 128
    
    config = LayerAdaptiveConfig(
        num_layers=num_layers,
        dim=dim,
        keep_last_n=keep_last_n,
        default_k_format="turbo4",
        protected_k_format="q8_0"
    )
    
    # Validate split point
    assert config.split_layer == 24, f"Expected split at 24, got {config.split_layer}"
    
    # Check layer protection
    assert not config.is_layer_protected(0), "Layer 0 should NOT be protected"
    assert not config.is_layer_protected(23), "Layer 23 should NOT be protected"
    assert config.is_layer_protected(24), "Layer 24 SHOULD be protected"
    assert config.is_layer_protected(31), "Layer 31 SHOULD be protected"
    
    # Check format routing
    config_low = config.get_config_for_layer(0)
    assert config_low.k_format == "turbo4", f"Expected turbo4 for layer 0, got {config_low.k_format}"
    
    config_high = config.get_config_for_layer(31)
    assert config_high.k_format == "q8_0", f"Expected q8_0 for layer 31, got {config_high.k_format}"
    
    print("✓ Logic validation passed!")

def test_cache_integration():
    print("\nTesting Cache Integration...")
    dim = 64
    cache = LayerAdaptiveKVCache(
        LayerAdaptiveConfig(num_layers=4, dim=dim, keep_last_n=2)
    )
    
    # Add some data
    k = torch.randn(10, dim)
    v = torch.randn(10, dim)
    
    # Layer 0 (Compressed)
    cache.append(0, k, v)
    # Layer 3 (Protected)
    cache.append(3, k, v)
    
    # Test attention
    q = torch.randn(1, dim)
    out0 = cache.get_attention_output(0, q)
    out3 = cache.get_attention_output(3, q)
    
    assert out0.shape == (1, dim)
    assert out3.shape == (1, dim)
    
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    assert stats["protected_layers"] == 2
    assert stats["compressed_layers"] == 2
    
    print("✓ Integration validation passed!")

if __name__ == "__main__":
    try:
        test_layer_adaptive_logic()
        test_cache_integration()
        print("\nALL LAYER-ADAPTIVE TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
