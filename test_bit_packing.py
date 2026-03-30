"""
Tests for TurboQuant Bit-packing.
"""

import torch
import pytest
from core.bit_packing import pack_bits, unpack_bits, pack_signs, unpack_signs
from core.codec import TurboQuantCodec, TurboQuantConfig

def test_pack_unpack_roundtrip():
    """Test that packing and unpacking returns original indices."""
    d = 128
    for bits in [1, 2, 4]:
        num_levels = 2 ** bits
        x = torch.randint(0, num_levels, (10, d), dtype=torch.int64)
        
        packed = pack_bits(x, bits)
        
        # Check expected shape
        expected_bytes = (d * bits + 7) // 8
        assert packed.shape == (10, expected_bytes)
        assert packed.dtype == torch.uint8
        
        unpacked = unpack_bits(packed, bits, d)
        assert torch.equal(x, unpacked)

def test_pack_unpack_signs():
    """Test that packing and unpacking signs returns original signs."""
    m = 64
    signs = torch.sign(torch.randn(10, m))
    # Ensure no zeros
    signs[signs == 0] = 1.0
    
    packed = pack_signs(signs)
    
    # Check shape: 64 bits = 8 bytes
    assert packed.shape == (10, 8)
    
    unpacked = unpack_signs(packed, m)
    assert torch.equal(signs, unpacked)

def test_codec_with_packing():
    """Test TurboQuantCodec with bit-packing enabled."""
    dim = 128
    config = TurboQuantConfig(num_bits=2, qjl_dim=64, pack_bits=True)
    codec = TurboQuantCodec(dim, config=config)
    
    x = torch.randn(5, dim)
    q = torch.randn(dim)
    
    # Encode
    encoded = codec.encode_key(x)
    
    # Check that it's packed
    assert encoded['indices'].dtype == torch.uint8
    assert encoded['indices'].shape == (5, 32) # 128 * 2 / 8 = 32 bytes
    assert encoded['r_signs'].shape == (5, 8)  # 64 / 8 = 8 bytes
    assert encoded['x_hat'] is None
    
    # Estimate
    estimates = codec.estimate_inner_products(q, encoded)
    assert estimates.shape == (5,)
    
    # True inner products (Stage 1 only for comparison)
    true_dots = q @ x.T
    correlation = torch.corrcoef(torch.stack([true_dots, estimates]))[0, 1].item()
    assert correlation > 0.8 # Should still be accurate

def test_memory_savings_real():
    """Verify memory savings calculations reflect packing."""
    dim = 4096 # Llama3 dim
    
    # Packed
    config_p = TurboQuantConfig(num_bits=4, qjl_dim=64, pack_bits=True)
    codec_p = TurboQuantCodec(dim, config=config_p)
    mem_p = codec_p.get_memory_usage(1)
    
    # Unpacked
    config_u = TurboQuantConfig(num_bits=4, qjl_dim=64, pack_bits=False)
    codec_u = TurboQuantCodec(dim, config=config_u)
    mem_u = codec_u.get_memory_usage(1)
    
    print(f"\nMemory for 1 key (dim={dim}, 4-bit):")
    print(f"Packed:   {mem_p['compressed']:.2f} bytes")
    print(f"Unpacked: {mem_u['compressed']:.2f} bytes")
    
    # Packed should be significantly smaller
    # Unpacked indices (4096 bytes) + x_hat (4096*4 bytes) + ...
    # Packed indices (4096*4/8 = 2048 bytes) + ...
    assert mem_p['compressed'] < mem_u['compressed'] / 4

if __name__ == "__main__":
    pytest.main([__file__])
