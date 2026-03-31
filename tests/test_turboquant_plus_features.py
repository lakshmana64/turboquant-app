"""
Comprehensive Tests for TurboQuant Plus Features.

Tests for:
- Turbo formats (turbo2, turbo3, turbo4)
- PolarQuant algorithm
- Sparse V decoding
- Asymmetric K/V support
- Outlier handling
- Layer-adaptive mode
- Norm correction
"""

import torch
import pytest
import math
from typing import Dict, Any


# ============================================================================
# Turbo Formats Tests
# ============================================================================

class TestTurboFormats:
    """Tests for turbo format presets."""
    
    def test_turbo_format_presets_exist(self):
        """Test that all turbo format presets are defined."""
        from core.turbo_formats import TURBO2, TURBO3, TURBO4
        
        assert TURBO2.name == "turbo2"
        assert TURBO2.sq_bits == 2
        assert TURBO2.compression_factor == 6.4
        
        assert TURBO3.name == "turbo3"
        assert TURBO3.sq_bits == 3
        assert TURBO3.compression_factor == 4.6
        
        assert TURBO4.name == "turbo4"
        assert TURBO4.sq_bits == 4
        assert TURBO4.compression_factor == 3.8
    
    def test_get_format(self):
        """Test format retrieval."""
        from core.turbo_formats import get_format, FORMAT_PRESETS
        
        for name in FORMAT_PRESETS.keys():
            fmt = get_format(name)
            assert fmt is not None
            assert fmt.name == name
    
    def test_get_format_invalid(self):
        """Test that invalid format raises error."""
        from core.turbo_formats import get_format
        
        with pytest.raises(ValueError):
            get_format("invalid_format")
    
    def test_calculate_memory_usage(self):
        """Test memory usage calculation."""
        from core.turbo_formats import calculate_memory_usage
        
        usage = calculate_memory_usage("turbo4", dim=4096, num_keys=1000)
        
        assert "format" in usage
        assert "compressed_bytes" in usage
        assert "compression_factor" in usage
        assert "turbo4" in usage["format"]
    
    def test_create_codec_from_format(self):
        """Test codec creation from format preset."""
        from core.turbo_formats import create_codec_from_format
        
        dim = 512
        codec = create_codec_from_format("turbo4", dim=dim)
        
        assert codec is not None
        assert codec.dim == dim
        assert codec.config.num_bits == 4


# ============================================================================
# PolarQuant Tests
# ============================================================================

class TestPolarQuant:
    """Tests for PolarQuant algorithm."""
    
    def test_polar_quant_encode_decode(self):
        """Test PolarQuant encode-decode roundtrip."""
        from core.polar_quant import polar_quant_roundtrip
        
        x = torch.randn(10, 512)
        
        for bits in [2, 3, 4]:
            x_reconstructed, metrics = polar_quant_roundtrip(
                x, bits=bits, qjl_dim=64
            )
            
            assert x_reconstructed.shape == x.shape
            assert metrics["cosine_similarity"] > 0.9
            assert metrics["compression_factor"] > 1.0
    
    def test_polar_quant_with_wht(self):
        """Test PolarQuant with WHT rotation."""
        from core.polar_quant import polar_quant_roundtrip
        
        x = torch.randn(10, 512)
        
        x_rot, metrics_rot = polar_quant_roundtrip(x, bits=2, use_wht=True)
        x_no_rot, metrics_no_rot = polar_quant_roundtrip(x, bits=2, use_wht=False)
        
        # WHT should improve quality for Gaussian-like data
        assert metrics_rot["cosine_similarity"] >= metrics_no_rot["cosine_similarity"]
    
    def test_polar_quant_codec_class(self):
        """Test PolarQuantCodec class."""
        from core.polar_quant import PolarQuantConfig, PolarQuantCodec
        
        config = PolarQuantConfig(bits=2, qjl_dim=64)
        codec = PolarQuantCodec(config, dim=512)
        
        x = torch.randn(5, 512)
        encoded = codec.encode(x)
        decoded = codec.decode(encoded)
        
        assert decoded.shape == x.shape
        assert 'mag_indices' in encoded
        assert 'dir_indices' in encoded
        assert 'qjl_signs' in encoded


# ============================================================================
# Sparse V Decoding Tests
# ============================================================================

class TestSparseVDecoding:
    """Tests for Sparse V decoding."""
    
    def test_sparse_v_decoder_basic(self):
        """Test basic sparse V decoding."""
        from core.sparse_v import SparseVDecoder
        
        dim = 512
        decoder = SparseVDecoder(dim, num_bits=4, threshold=1e-6)
        
        # Create sample V data
        v = torch.randn(10, dim)
        encoded_v = decoder.codec.encode(v)
        
        # Create attention weights with sparsity
        attn_weights = torch.softmax(torch.randn(1, 10), dim=-1)
        
        # Decode with sparsity
        v_decoded = decoder.decode_sparse(encoded_v, attn_weights)
        
        assert v_decoded.shape == (1, 10, dim)
        
        # Check sparsity stats
        stats = decoder.get_sparsity_stats()
        assert "sparsity" in stats
    
    def test_sparse_v_threshold(self):
        """Test sparse V threshold effect."""
        from core.sparse_v import SparseVDecoder
        
        dim = 256
        decoder_loose = SparseVDecoder(dim, threshold=1e-3)
        decoder_tight = SparseVDecoder(dim, threshold=1e-8)
        
        v = torch.randn(5, dim)
        encoded_v = decoder_loose.codec.encode(v)
        attn_weights = torch.softmax(torch.randn(1, 5), dim=-1)
        
        # Decode with different thresholds
        decoder_loose.decode_sparse_optimized(encoded_v, attn_weights)
        decoder_tight.decode_sparse_optimized(encoded_v, attn_weights)
        
        stats_loose = decoder_loose.get_sparsity_stats()
        stats_tight = decoder_tight.get_sparsity_stats()
        
        # Looser threshold should skip more positions
        assert stats_loose["sparsity"] >= stats_tight["sparsity"]
    
    def test_sparse_kv_cache(self):
        """Test SparseKVCache integration."""
        from core.sparse_v import SparseKVCache
        
        cache = SparseKVCache(
            dim=512,
            k_format="q8_0",
            v_format="turbo4",
            sparse_threshold=1e-6
        )
        
        # Append some data
        k = torch.randn(1, 4, 10, 512)  # [batch, heads, seq, dim]
        v = torch.randn(1, 4, 10, 512)
        cache.append(k, v)
        
        # Get attention output
        q = torch.randn(1, 512)
        output = cache.get_attention_output(q)
        
        assert output.shape == (1, 512)


# ============================================================================
# Asymmetric K/V Support Tests
# ============================================================================

class TestAsymmetricKV:
    """Tests for asymmetric K/V cache support."""
    
    def test_asymmetric_config(self):
        """Test asymmetric KV configuration."""
        from core.asymmetric_kv import AsymmetricKVConfig
        
        config = AsymmetricKVConfig(
            dim=512,
            k_format="q8_0",
            v_format="turbo4"
        )
        
        assert config.k_format == "q8_0"
        assert config.v_format == "turbo4"
        assert config.k_format_obj.compression_factor == 2.0
        assert config.v_format_obj.compression_factor == 3.8
    
    def test_asymmetric_cache_basic(self):
        """Test basic asymmetric KV cache."""
        from core.asymmetric_kv import AsymmetricKVCache, AsymmetricKVConfig
        
        config = AsymmetricKVConfig(
            dim=256,
            k_format="q8_0",
            v_format="turbo4",
            enable_sparse_v=False
        )
        
        cache = AsymmetricKVCache(config)
        
        # Append data
        k = torch.randn(5, 256)
        v = torch.randn(5, 256)
        cache.append(k, v)
        
        # Get attention output
        q = torch.randn(1, 256)
        output = cache.get_attention_output(q)
        
        assert output.shape == (1, 256)
    
    def test_asymmetric_memory_usage(self):
        """Test asymmetric cache memory calculation."""
        from core.asymmetric_kv import AsymmetricKVCache, AsymmetricKVConfig
        
        config = AsymmetricKVConfig(
            dim=512,
            k_format="q8_0",
            v_format="turbo2"
        )
        
        cache = AsymmetricKVCache(config)
        
        # Append some data
        k = torch.randn(10, 512)
        v = torch.randn(10, 512)
        cache.append(k, v)
        
        memory = cache.get_memory_usage()
        
        assert "k_memory" in memory
        assert "v_memory_bytes" in memory
        assert "overall_compression_factor" in memory
    
    def test_recommend_asymmetric_config(self):
        """Test asymmetric config recommendation."""
        from core.asymmetric_kv import recommend_asymmetric_config
        
        # For 4-bit model with balanced priority
        k_fmt, v_fmt = recommend_asymmetric_config(
            model_bits=4,
            quality_priority="balanced"
        )
        
        assert k_fmt == "q8_0"  # High precision for K
        assert v_fmt == "turbo4"  # Compressed V


# ============================================================================
# Outlier Handling Tests
# ============================================================================

class TestOutlierHandling:
    """Tests for outlier channel handling."""
    
    def test_outlier_detection_variance(self):
        """Test outlier detection by variance."""
        from core.outlier import OutlierConfig, OutlierHandler
        
        config = OutlierConfig(
            variance_threshold=10.0,
            use_magnitude=False
        )
        
        handler = OutlierHandler(config, dim=256)
        
        # Create data with outliers
        x = torch.randn(100, 256)
        x[:, 0:5] *= 100  # Make first 5 channels outliers
        
        outlier_mask = handler.detect_outliers(x)
        
        # Should detect the outlier channels
        assert outlier_mask[0:5].sum() > 0
    
    def test_outlier_detection_magnitude(self):
        """Test outlier detection by magnitude."""
        from core.outlier import OutlierConfig, OutlierHandler
        
        config = OutlierConfig(
            variance_threshold=10.0,
            use_magnitude=True
        )
        
        handler = OutlierHandler(config, dim=256)
        
        x = torch.randn(100, 256)
        x[:, 10:15] *= 100  # Make channels 10-15 outliers
        
        outlier_mask = handler.detect_outliers(x)
        
        assert outlier_mask[10:15].sum() > 0
    
    def test_outlier_aware_encoding(self):
        """Test outlier-aware encoding and decoding."""
        from core.outlier import OutlierHandler, OutlierConfig
        
        config = OutlierConfig(
            variance_threshold=5.0,
            outlier_bits=8,
            main_bits=2
        )
        
        handler = OutlierHandler(config, dim=128)
        
        x = torch.randn(10, 128)
        x[:, 0:3] *= 50  # Add outliers
        
        outlier_mask = handler.detect_outliers(x)
        encoded = handler.encode_with_outliers(x, outlier_mask)
        decoded = handler.decode_with_outliers(encoded)
        
        assert decoded.shape == x.shape
        
        # Check reconstruction quality
        mse = ((x - decoded) ** 2).mean().item()
        assert mse < 1.0  # Should have reasonable MSE
    
    def test_outlier_aware_codec(self):
        """Test OutlierAwareCodec class."""
        from core.outlier import OutlierAwareCodec
        
        codec = OutlierAwareCodec(
            dim=256,
            main_bits=2,
            outlier_bits=8,
            variance_threshold=10.0
        )
        
        x = torch.randn(5, 256)
        encoded = codec.encode(x)
        decoded = codec.decode(encoded)
        
        assert decoded.shape == x.shape
        
        stats = codec.get_stats()
        assert "main_bits" in stats
        assert "outlier_bits" in stats


# ============================================================================
# Layer-Adaptive Mode Tests
# ============================================================================

class TestLayerAdaptive:
    """Tests for layer-adaptive quantization."""
    
    def test_layer_adaptive_config(self):
        """Test layer-adaptive configuration."""
        from core.layer_adaptive import LayerAdaptiveConfig
        
        config = LayerAdaptiveConfig(
            total_layers=32,
            high_precision_layers=8,
            high_precision_format="q8_0",
            compressed_format="turbo4"
        )
        
        # Last 8 layers should be high precision
        assert config.get_layer_format(31) == ("q8_0", "q8_0")
        assert config.get_layer_format(24) == ("q8_0", "q8_0")
        
        # First 24 layers should be compressed
        assert config.get_layer_format(0) == ("turbo4", "turbo4")
        assert config.get_layer_format(23) == ("turbo4", "turbo4")
    
    def test_layer_adaptive_cache(self):
        """Test layer-adaptive KV cache."""
        from core.layer_adaptive import LayerAdaptiveKVCache, LayerAdaptiveConfig
        
        config = LayerAdaptiveConfig(
            total_layers=4,
            high_precision_layers=2,
            high_precision_format="q8_0",
            compressed_format="turbo4",
            dim=256
        )
        
        cache = LayerAdaptiveKVCache(config)
        
        # Append to different layers
        for layer_idx in range(4):
            k = torch.randn(5, 256)
            v = torch.randn(5, 256)
            cache.append(layer_idx, k, v)
        
        # Get output for specific layer
        q = torch.randn(1, 256)
        output = cache.get_layer_output(0, q)
        
        assert output.shape == (1, 256)
    
    def test_layer_adaptive_memory(self):
        """Test layer-adaptive memory calculation."""
        from core.layer_adaptive import LayerAdaptiveKVCache, LayerAdaptiveConfig
        
        config = LayerAdaptiveConfig(
            total_layers=4,
            high_precision_layers=2,
            dim=512
        )
        
        cache = LayerAdaptiveKVCache(config)
        
        # Append data
        for layer_idx in range(4):
            k = torch.randn(10, 512)
            v = torch.randn(10, 512)
            cache.append(layer_idx, k, v)
        
        memory = cache.get_memory_usage()
        
        assert "per_layer" in memory
        assert "total_compressed_bytes" in memory
        assert "overall_compression_factor" in memory
    
    def test_layer_adaptive_recommendation(self):
        """Test layer config recommendation."""
        from core.layer_adaptive import recommend_layer_config
        
        config = recommend_layer_config(
            model_size="7b",
            target_compression=3.5
        )
        
        assert config.total_layers == 32
        assert config.high_precision_layers >= 8


# ============================================================================
# Norm Correction Tests
# ============================================================================

class TestNormCorrection:
    """Tests for norm correction."""
    
    def test_norm_corrector_basic(self):
        """Test basic norm correction."""
        from core.norm_correction import NormCorrectionConfig, NormCorrector
        
        config = NormCorrectionConfig()
        corrector = NormCorrector(config, dim=256)
        
        x_original = torch.randn(10, 256)
        x_quantized = x_original * 0.9  # Simulate quantization
        
        scale = corrector.compute_norm_correction(x_original, x_quantized)
        x_corrected = corrector.apply_correction(x_quantized, scale)
        
        assert x_corrected.shape == x_quantized.shape
        
        # Correction should improve similarity
        sim_before = torch.nn.functional.cosine_similarity(
            x_original.view(-1), x_quantized.view(-1)
        ).mean()
        sim_after = torch.nn.functional.cosine_similarity(
            x_original.view(-1), x_corrected.view(-1)
        ).mean()
        
        assert sim_after >= sim_before
    
    def test_norm_corrector_running_stats(self):
        """Test running statistics update."""
        from core.norm_correction import NormCorrectionConfig, NormCorrector
        
        config = NormCorrectionConfig(use_running_stats=True)
        corrector = NormCorrector(config, dim=128)
        
        # Update with multiple batches
        for _ in range(10):
            x_original = torch.randn(5, 128)
            x_quantized = x_original * 0.95
            corrector.update_running_stats(x_original, x_quantized)
        
        assert corrector.num_updates == 10
        assert corrector.get_inference_scale() > 0
    
    def test_norm_corrected_codec(self):
        """Test norm-corrected codec."""
        from core.norm_correction import NormCorrectionConfig, NormCorrectedCodec
        from core.codec import TurboQuantConfig, TurboQuantCodec
        
        dim = 256
        base_codec = TurboQuantCodec(dim, TurboQuantConfig(num_bits=4))
        
        codec = NormCorrectedCodec(
            base_codec,
            NormCorrectionConfig(),
            calibrate=True
        )
        
        # Calibrate
        calibration_data = [torch.randn(5, dim) for _ in range(5)]
        stats = codec.calibrate(calibration_data)
        
        assert codec.calibrated
        assert "mse_before" in stats
        assert "mse_after" in stats
        
        # Encode/decode with correction
        x = torch.randn(3, dim)
        encoded = codec.encode_with_correction(x)
        decoded = codec.decode_with_correction(encoded)
        
        assert decoded.shape == x.shape
    
    def test_apply_norm_correction_function(self):
        """Test convenience function for norm correction."""
        from core.norm_correction import apply_norm_correction
        
        x_original = torch.randn(10, 256)
        x_quantized = x_original * 0.9
        
        x_corrected, scale, improvement = apply_norm_correction(
            x_original, x_quantized, correction_type="per_token"
        )
        
        assert x_corrected.shape == x_quantized.shape
        assert isinstance(scale, float)
        assert isinstance(improvement, float)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_turbo_format_with_sparse_v(self):
        """Test turbo formats with sparse V decoding."""
        from core.turbo_formats import get_format
        from core.sparse_v import SparseVDecoder
        
        fmt = get_format("turbo4")
        decoder = SparseVDecoder(dim=512, num_bits=fmt.sq_bits)
        
        v = torch.randn(10, 512)
        encoded = decoder.codec.encode(v)
        attn_weights = torch.softmax(torch.randn(1, 10), dim=-1)
        
        v_decoded = decoder.decode_sparse_optimized(encoded, attn_weights)
        
        assert v_decoded.shape == (1, 10, 512)
    
    def test_asymmetric_with_layer_adaptive(self):
        """Test asymmetric K/V with layer-adaptive mode."""
        from core.layer_adaptive import LayerAdaptiveKVCache, LayerAdaptiveConfig
        
        # Use asymmetric formats per layer
        config = LayerAdaptiveConfig(
            total_layers=4,
            high_precision_layers=2,
            high_precision_format="q8_0",  # High precision for last layers
            compressed_format="turbo4",
            dim=256
        )
        
        cache = LayerAdaptiveKVCache(config)
        
        for layer_idx in range(4):
            k = torch.randn(5, 256)
            v = torch.randn(5, 256)
            cache.append(layer_idx, k, v)
        
        stats = cache.get_stats()
        
        # Verify different layers have different formats
        memory = stats["memory"]["per_layer"]
        assert memory[0]["k_format"] == "turbo4"  # Early layer
        assert memory[3]["k_format"] == "q8_0"  # Late layer
    
    def test_outlier_with_norm_correction(self):
        """Test outlier handling with norm correction."""
        from core.outlier import OutlierAwareCodec
        from core.norm_correction import NormCorrectionConfig, NormCorrectedCodec
        
        # Create codec with outlier handling
        outlier_codec = OutlierAwareCodec(
            dim=256,
            main_bits=2,
            outlier_bits=8
        )
        
        x = torch.randn(10, 256)
        x[:, 0:5] *= 50  # Add outliers
        
        encoded = outlier_codec.encode(x)
        decoded = outlier_codec.decode(encoded)
        
        # Apply additional norm correction
        x_final, scale, improvement = apply_norm_correction(
            x, decoded, correction_type="per_token"
        )
        
        assert x_final.shape == x.shape


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_turbo_format_compression(self):
        """Test that turbo formats achieve expected compression."""
        from core.turbo_formats import calculate_memory_usage
        
        dim = 4096
        num_keys = 1000
        
        for format_name, expected_factor in [
            ("turbo2", 6.4),
            ("turbo3", 4.6),
            ("turbo4", 3.8)
        ]:
            usage = calculate_memory_usage(format_name, dim, num_keys)
            actual_factor = float(usage["compression_factor"].replace("x", ""))
            
            # Allow 10% tolerance
            assert abs(actual_factor - expected_factor) / expected_factor < 0.1
    
    def test_sparse_v_speedup(self):
        """Test that sparse V provides theoretical speedup."""
        from core.sparse_v import SparseVDecoder
        
        decoder = SparseVDecoder(dim=512, threshold=1e-3)
        
        v = torch.randn(100, 512)
        encoded = decoder.codec.encode(v)
        
        # Create sparse attention (most weights near zero)
        attn_weights = torch.zeros(1, 100)
        attn_weights[0, 0:5] = 1.0  # Only 5 positions have attention
        attn_weights = attn_weights / attn_weights.sum()
        
        decoder.decode_sparse_optimized(encoded, attn_weights)
        
        stats = decoder.get_sparsity_stats()
        
        # Should have high sparsity (>90% skipped)
        assert stats["sparsity"] > 0.9
        assert float(stats["theoretical_speedup"].replace("x", "")) > 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
