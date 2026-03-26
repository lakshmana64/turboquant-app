"""
Comprehensive Tests for TurboQuant

Tests cover:
1. Scalar quantization correctness
2. QJL projection properties
3. Unbiased inner product estimation
4. End-to-end codec functionality
5. Batch processing
"""

import torch
import pytest
import math


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_vectors(device) -> torch.Tensor:
    """Generate test vectors."""
    torch.manual_seed(42)
    return torch.randn(100, 128, device=device)


@pytest.fixture
def test_queries(device) -> torch.Tensor:
    """Generate test queries."""
    torch.manual_seed(123)
    return torch.randn(10, 128, device=device)


# ============================================================================
# Scalar Quantization Tests
# ============================================================================

class TestScalarQuantization:
    """Tests for scalar_quant.py"""
    
    def test_quantize_dequantize_roundtrip(self, device):
        """Test that quantize -> dequantize gives reasonable reconstruction."""
        from turboquant.core.scalar_quant import quantize_scalar, dequantize_scalar
        
        x = torch.randn(10, 64, device=device)
        
        indices, scales, norms, R = quantize_scalar(x, num_bits=4)
        x_hat = dequantize_scalar(indices, scales, num_bits=4, rotation_matrix=R)
        
        # Check shapes
        assert x_hat.shape == x.shape
        
        # Check reconstruction quality (should be reasonable for 4-bit)
        mse = ((x - x_hat) ** 2).mean().item()
        signal_var = x.var().item()
        nmse = mse / signal_var
        
        # 4-bit should give NMSE < 0.1
        assert nmse < 0.1, f"NMSE too high: {nmse}"
    
    def test_per_vector_scaling(self, device):
        """Test that per-vector scaling is applied correctly."""
        from turboquant.core.scalar_quant import quantize_scalar
        
        # Create vectors with different norms
        x = torch.randn(5, 64, device=device)
        scales = torch.linspace(0.5, 2.0, 5, device=device).view(-1, 1)
        x = x * scales
        
        indices, out_scales, norms, R = quantize_scalar(x, num_bits=4)
        
        # Scales should correlate with input norms
        input_norms = x.norm(dim=1, keepdim=True)
        correlation = torch.corrcoef(
            torch.stack([input_norms.squeeze(), out_scales.squeeze()])
        )[0, 1].item()
        
        assert correlation > 0.9, f"Scale correlation too low: {correlation}"
    
    def test_zero_vector_handling(self, device):
        """Test that zero vectors are handled without errors."""
        from turboquant.core.scalar_quant import quantize_scalar, dequantize_scalar
        
        x = torch.zeros(5, 64, device=device)
        
        indices, scales, norms, R = quantize_scalar(x, num_bits=4)
        x_hat = dequantize_scalar(indices, scales, num_bits=4, rotation_matrix=R)
        
        # Should not crash, reconstruction should be near zero
        assert x_hat.abs().max() < 0.5
    
    def test_determinism(self, device):
        """Test that quantization is deterministic with fixed seed."""
        from turboquant.core.scalar_quant import quantize_scalar
        
        x = torch.randn(10, 64, device=device)
        
        indices1, scales1, _, R1 = quantize_scalar(x, num_bits=4, rotation_seed=42)
        indices2, scales2, _, R2 = quantize_scalar(x, num_bits=4, rotation_seed=42)
        
        assert torch.equal(indices1, indices2)
        assert torch.equal(scales1, scales2)
        assert torch.allclose(R1, R2)
    
    def test_codebook_cache(self, device):
        """Test that codebook caching works."""
        from turboquant.core.scalar_quant import get_codebook
        
        centroids1, boundaries1 = get_codebook(4, device)
        centroids2, boundaries2 = get_codebook(4, device)
        
        assert torch.equal(centroids1, centroids2)
        assert torch.equal(boundaries1, boundaries2)


# ============================================================================
# QJL Projection Tests
# ============================================================================

class TestQJLProjection:
    """Tests for qjl_projection.py"""
    
    def test_projection_shape(self, device):
        """Test projection output shape."""
        from turboquant.core.qjl_projection import QJLProjection
        
        qjl = QJLProjection(input_dim=128, output_dim=64, device=device)
        x = torch.randn(10, 128, device=device)
        
        projected = qjl.project(x)
        assert projected.shape == (10, 64)
    
    def test_sign_quantization(self, device):
        """Test that sign quantization produces {-1, +1}."""
        from turboquant.core.qjl_projection import QJLProjection
        
        qjl = QJLProjection(input_dim=128, output_dim=64, device=device)
        r = torch.randn(10, 128, device=device)
        
        signs = qjl.project_and_quantize(r)
        
        assert signs.shape == (10, 64)
        assert ((signs == 1) | (signs == -1)).all()
    
    def test_determinism(self, device):
        """Test that projection is deterministic with fixed seed."""
        from turboquant.core.qjl_projection import QJLProjection
        
        x = torch.randn(10, 128, device=device)
        
        qjl1 = QJLProjection(input_dim=128, output_dim=64, seed=42, device=device)
        qjl2 = QJLProjection(input_dim=128, output_dim=64, seed=42, device=device)
        
        assert torch.allclose(qjl1.projection_matrix, qjl2.projection_matrix)
        
        p1 = qjl1.project(x)
        p2 = qjl2.project(x)
        assert torch.allclose(p1, p2)
    
    def test_inner_product_estimation_scaling(self, device):
        """Test QJL inner product estimation scaling."""
        from turboquant.core.qjl_projection import QJLProjection
        
        qjl = QJLProjection(input_dim=128, output_dim=64, device=device)
        
        q = torch.randn(128, device=device)
        r = torch.randn(128, device=device)
        
        q_proj = qjl.project(q.unsqueeze(0))
        r_signs = qjl.project_and_quantize(r.unsqueeze(0))
        r_norm = r.norm().unsqueeze(0).unsqueeze(0)
        
        estimate = qjl.estimate_inner_product(q_proj, r_signs, r_norm)
        
        # Estimate should be finite
        assert torch.isfinite(estimate).all()


# ============================================================================
# Residual Tests
# ============================================================================

class TestResidual:
    """Tests for residual.py"""
    
    def test_residual_computation(self, device):
        """Test residual r = x - x_hat."""
        from turboquant.core.residual import compute_residual
        
        x = torch.randn(10, 128, device=device)
        x_hat = torch.randn(10, 128, device=device)
        
        r = compute_residual(x, x_hat)
        expected = x - x_hat
        
        assert torch.allclose(r, expected)
    
    def test_residual_encoder(self, device):
        """Test ResidualEncoder end-to-end."""
        from turboquant.core.residual import ResidualEncoder
        
        encoder = ResidualEncoder(input_dim=128, output_dim=64, device=device)
        
        x = torch.randn(10, 128, device=device)
        x_hat = torch.randn(10, 128, device=device)
        
        signs, norms = encoder.encode(x, x_hat)
        
        assert signs.shape == (10, 64)
        assert norms.shape == (10, 1)


# ============================================================================
# Estimator Tests
# ============================================================================

class TestEstimator:
    """Tests for estimator.py"""
    
    def test_unbiasedness_single_vector(self, device):
        """Test that estimator is unbiased for single vector pair."""
        from turboquant.core.estimator import UnbiasedInnerProductEstimator
        
        dim = 128
        x = torch.randn(dim, device=device)
        q = torch.randn(dim, device=device)
        
        # True inner product
        true_dot = (q * x).sum().item()
        
        # Create codec for Stage 1
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        config = TurboQuantConfig(num_bits=4, qjl_dim=64, seed=42)
        codec = TurboQuantCodec(dim, config=config, device=device)
        
        encoded = codec.encode_key(x)
        x_hat = codec.decode_key(encoded)
        
        # Test with multiple QJL seeds
        errors = []
        for seed in range(500):
            estimator = UnbiasedInnerProductEstimator(dim, output_dim=64, seed=seed, device=device)
            _, r_signs, r_norm = estimator.encode_key(x, x_hat)
            estimate = estimator.estimate(q, x_hat, r_signs, r_norm).item()
            errors.append(estimate - true_dot)
        
        errors = torch.tensor(errors)
        mean_error = errors.mean().item()
        
        # Mean error should be close to zero (unbiased)
        assert abs(mean_error) < 0.2, f"Estimator appears biased: mean_error={mean_error}"
    
    def test_batch_estimation(self, device):
        """Test batch inner product estimation."""
        from turboquant.core.estimator import UnbiasedInnerProductEstimator
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        dim = 128
        num_keys = 50
        num_queries = 10
        
        keys = torch.randn(num_keys, dim, device=device)
        queries = torch.randn(num_queries, dim, device=device)
        
        # True inner products
        true_dots = queries @ keys.T
        
        # Compressed
        config = TurboQuantConfig(num_bits=4, qjl_dim=64, seed=42)
        codec = TurboQuantCodec(dim, config=config, device=device)
        encoded = codec.encode_keys_batch(keys)
        
        estimator = UnbiasedInnerProductEstimator(dim, output_dim=64, seed=42, device=device)
        
        # Estimate
        estimates = codec.estimate_inner_products(queries[0], encoded)
        
        assert estimates.shape == (num_keys,)
        
        # Correlation should be high
        correlation = torch.corrcoef(
            torch.stack([true_dots[0], estimates])
        )[0, 1].item()
        
        assert correlation > 0.9, f"Correlation too low: {correlation}"


# ============================================================================
# Codec Tests
# ============================================================================

class TestCodec:
    """Tests for codec.py"""
    
    def test_codec_encode_decode(self, device):
        """Test TurboQuantCodec encode/decode."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        x = torch.randn(10, 128, device=device)
        
        # Encode batch
        encoded = codec.encode_keys_batch(x)
        
        # Decode
        x_hat = codec.decode_keys(encoded)
        
        assert x_hat.shape == x.shape
        
        # Reconstruction quality
        mse = ((x - x_hat) ** 2).mean().item()
        assert mse < 0.5  # Reasonable for 4-bit
    
    def test_codec_inner_product(self, device):
        """Test TurboQuantCodec inner product estimation."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        x = torch.randn(50, 128, device=device)
        q = torch.randn(128, device=device)
        
        # True inner products
        true_dots = q @ x.T
        
        # Encode and estimate
        encoded = codec.encode_keys_batch(x)
        estimates = codec.estimate_inner_products(q, encoded)
        
        assert estimates.shape == (50,)
        
        # High correlation expected
        correlation = torch.corrcoef(torch.stack([true_dots, estimates]))[0, 1].item()
        assert correlation > 0.95
    
    def test_codec_compression_ratio(self, device):
        """Test compression ratio calculation."""
        from turboquant.core.codec import TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        
        # For dim=128: 
        #   Original: 128 * 16 = 2048 bits (FP16)
        #   Compressed: 128 * 4 + 64 = 576 bits
        #   Ratio: 576 / 2048 = 0.28125
        from turboquant.core.codec import TurboQuantCodec
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        expected_ratio = (128 * 4 + 64) / (128 * 16)
        assert abs(codec.compression_ratio - expected_ratio) < 1e-6
    
    def test_codec_attention_scores(self, device):
        """Test attention score computation."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        keys = torch.randn(100, 128, device=device)
        query = torch.randn(128, device=device)
        
        # Encode
        encoded = codec.encode_keys_batch(keys)
        
        # Compute scores with scaling
        scale = 1.0 / math.sqrt(128)
        scores = codec.compute_attention_scores(query, encoded, scale=scale)
        
        assert scores.shape == (100,)
        assert torch.isfinite(scores).all()

    def test_codec_llama3_dimensions(self, device):
        """Test with llama3 dimensions (dim=4096)."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        dim = 4096
        config = TurboQuantConfig(num_bits=4, qjl_dim=64, seed=42)
        codec = TurboQuantCodec(dim=dim, config=config, device=device)
        
        # Generate some synthetic embeddings
        torch.manual_seed(42)
        x = torch.randn(5, dim, device=device)
        # Normalize to simulate real embeddings
        x = x / x.norm(dim=1, keepdim=True)
        
        # Encode
        encoded = codec.encode_keys_batch(x)
        
        # Query
        query = x[0].unsqueeze(0)
        estimates = codec.estimate_inner_products(query, encoded)
        
        # Self-dot product should be close to 1
        assert abs(estimates[0, 0].item() - 1.0) < 0.15
        
        # Correlation should be very high
        true_dots = query @ x.T
        correlation = torch.corrcoef(torch.stack([true_dots.squeeze(), estimates.squeeze()]))[0, 1].item()
        assert correlation > 0.95


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, device):
        """Test complete TurboQuant pipeline."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        # Setup
        dim = 128
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim, config=config, device=device)
        
        # Generate data
        keys = torch.randn(100, dim, device=device)
        queries = torch.randn(10, dim, device=device)
        
        # True attention
        scale = 1.0 / math.sqrt(dim)
        true_scores = (queries @ keys.T) * scale
        true_attention = torch.softmax(true_scores, dim=-1)
        
        # Compressed
        encoded = codec.encode_keys_batch(keys)
        
        est_scores = torch.zeros_like(true_scores)
        for i in range(10):
            est_scores[i] = codec.estimate_inner_products(queries[i], encoded) * scale
        est_attention = torch.softmax(est_scores, dim=-1)
        
        # Metrics
        mse = ((true_attention - est_attention) ** 2).mean().item()
        cosine = (
            (true_attention.view(-1) @ est_attention.view(-1)) /
            (true_attention.view(-1).norm() * est_attention.view(-1).norm())
        ).item()
        
        assert mse < 0.01, f"MSE too high: {mse}"
        assert cosine > 0.95, f"Cosine too low: {cosine}"
    
    def test_memory_efficiency(self, device):
        """Test memory savings from compression."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        num_keys = 1000
        memory = codec.get_memory_usage(num_keys)
        
        assert memory['ratio'] < 0.6, f"Compression ratio too high: {memory['ratio']}"
        assert memory['compressed'] < memory['original']


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_small_dimension(self, device):
        """Test with small vector dimension."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=32)
        codec = TurboQuantCodec(dim=32, config=config, device=device)
        
        x = torch.randn(10, 32, device=device)
        encoded = codec.encode_keys_batch(x)
        
        assert encoded['indices'].shape == (10, 32)
    
    def test_large_batch(self, device):
        """Test with large batch size."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        x = torch.randn(1000, 128, device=device)
        encoded = codec.encode_keys_batch(x)
        
        assert encoded['indices'].shape[0] == 1000
    
    def test_unit_vectors(self, device):
        """Test with unit norm vectors."""
        from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig
        
        config = TurboQuantConfig(num_bits=4, qjl_dim=64)
        codec = TurboQuantCodec(dim=128, config=config, device=device)
        
        x = torch.randn(50, 128, device=device)
        x = x / x.norm(dim=1, keepdim=True)
        
        encoded = codec.encode_keys_batch(x)
        x_hat = codec.decode_keys(encoded)
        
        # Reconstructed should also be approximately unit norm
        norms = x_hat.norm(dim=1)
        assert (norms - 1).abs().mean().item() < 0.1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
