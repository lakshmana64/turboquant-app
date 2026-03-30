"""
Tests for Fast Walsh-Hadamard Transform (WHT).
"""

import torch
import pytest
import time
from core.wht import fast_walsh_hadamard_transform, apply_random_wht
from core.codec import TurboQuantCodec, TurboQuantConfig

def test_wht_is_orthogonal():
    """Test that WHT preserves the norm (is orthogonal)."""
    d = 1024
    x = torch.randn(10, d)
    norm_orig = torch.norm(x, dim=-1)
    
    x_wht = fast_walsh_hadamard_transform(x)
    norm_wht = torch.norm(x_wht, dim=-1)
    
    assert torch.allclose(norm_orig, norm_wht, atol=1e-5)

def test_wht_roundtrip():
    """Test that WHT is its own inverse (with normalization)."""
    d = 512
    x = torch.randn(1, d)
    
    # FWHT(FWHT(x)) should equal x
    x_hat = fast_walsh_hadamard_transform(fast_walsh_hadamard_transform(x))
    
    assert torch.allclose(x, x_hat, atol=1e-5)

def test_codec_with_wht():
    """Test TurboQuantCodec with Hadamard rotation."""
    dim = 1024
    # Use 8-bit to ensure SQ is very accurate, so we only test rotation logic
    config = TurboQuantConfig(num_bits=8, qjl_dim=64, rotation_type="hadamard", pack_bits=False)
    codec = TurboQuantCodec(dim, config=config)
    
    assert codec.rotation_matrix is None # Memory saved!
    
    x = torch.randn(5, dim)
    q = torch.randn(dim)
    
    # Encode/Estimate
    encoded = codec.encode_key(x)
    estimates = codec.estimate_inner_products(q, encoded)
    
    true_dots = q @ x.T
    correlation = torch.corrcoef(torch.stack([true_dots, estimates]))[0, 1].item()
    
    assert correlation > 0.8 # Should maintain high accuracy

def test_wht_speed_advantage():
    """Verify that WHT is faster than O(d^2) matrix multiply for large d."""
    d = 4096
    x = torch.randn(100, d)
    mat = torch.randn(d, d)
    
    # Matrix Multiply O(d^2)
    start = time.time()
    for _ in range(10):
        _ = x @ mat
    mm_time = time.time() - start
    
    # WHT O(d log d)
    start = time.time()
    for _ in range(10):
        _ = fast_walsh_hadamard_transform(x)
    wht_time = time.time() - start
    
    print(f"\nSpeed Benchmark (d={d}):")
    print(f"Matrix Multiply: {mm_time:.4f}s")
    print(f"Fast WHT:        {wht_time:.4f}s")
    print(f"WHT Speedup:     {mm_time / wht_time:.2f}x")
    
    assert wht_time < mm_time

if __name__ == "__main__":
    pytest.main([__file__])
