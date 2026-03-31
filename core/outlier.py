"""
Outlier Channel Strategy for TurboQuant.

Handles outlier dimensions (high-variance channels) separately to prevent
quantization error from dominating the reconstruction.

Strategy:
1. Detect outlier channels with variance > threshold * median_variance
2. Keep outliers in higher precision (FP16/FP8)
3. Quantize remaining channels with aggressive compression
4. Combine during reconstruction

Benefits:
- Prevents quality degradation from outlier-dominated error
- Allows more aggressive quantization on non-outlier channels
- Essential for maintaining quality on low-bit models
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
from .scalar_quant import quantize_scalar, dequantize_scalar
from .turbo_formats import get_format


class OutlierConfig:
    """Configuration for outlier handling."""
    
    def __init__(
        self,
        variance_threshold: float = 10.0,
        outlier_bits: int = 8,  # FP16 equivalent
        main_bits: int = 2,
        min_outliers: int = 1,
        max_outliers: int = 128,
        use_magnitude: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize outlier config.
        
        Args:
            variance_threshold: Channels with variance > threshold * median are outliers
            outlier_bits: Bits for outlier channels (8 = FP16-like)
            main_bits: Bits for non-outlier channels
            min_outliers: Minimum number of outlier channels to detect
            max_outliers: Maximum number of outlier channels
            use_magnitude: Use magnitude-based detection instead of variance
            device: Torch device
        """
        self.variance_threshold = variance_threshold
        self.outlier_bits = outlier_bits
        self.main_bits = main_bits
        self.min_outliers = min_outliers
        self.max_outliers = max_outliers
        self.use_magnitude = use_magnitude
        self.device = device or torch.device('cpu')


class OutlierHandler:
    """
    Detects and handles outlier channels in KV cache vectors.
    """
    
    def __init__(self, config: OutlierConfig, dim: int):
        """
        Initialize outlier handler.
        
        Args:
            config: OutlierConfig instance
            dim: Vector dimension
        """
        self.config = config
        self.dim = dim
        self.device = config.device
        
        # Statistics tracking
        self.outlier_indices: Optional[Tensor] = None
        self.variance_profile: Optional[Tensor] = None
    
    def detect_outliers(self, x: Tensor) -> Tensor:
        """
        Detect outlier channels in input tensor.
        
        Args:
            x: Input tensor [batch, dim] or [batch, seq, dim]
        
        Returns:
            Boolean mask [dim] where True indicates outlier channel
        """
        if x.dim() == 3:
            # [batch, seq, dim] -> [batch*seq, dim]
            x = x.view(-1, x.shape[-1])
        
        if self.config.use_magnitude:
            # Use magnitude-based detection
            outlier_mask = self._detect_by_magnitude(x)
        else:
            # Use variance-based detection
            outlier_mask = self._detect_by_variance(x)
        
        # Enforce min/max constraints
        num_outliers = outlier_mask.sum().item()
        
        if num_outliers < self.config.min_outliers:
            # Promote highest variance channels to outliers
            if self.variance_profile is not None:
                _, top_indices = self.variance_profile.topk(self.config.min_outliers)
                outlier_mask[top_indices] = True
        
        if num_outliers > self.config.max_outliers:
            # Demote lowest variance outliers
            if self.variance_profile is not None:
                outlier_indices = outlier_mask.nonzero(as_tuple=True)[0]
                outlier_variances = self.variance_profile[outlier_indices]
                _, demote_indices = outlier_variances.topk(
                    num_outliers - self.config.max_outliers,
                    largest=False
                )
                outlier_mask[outlier_indices[demote_indices]] = False
        
        self.outlier_indices = outlier_mask
        return outlier_mask
    
    def _detect_by_variance(self, x: Tensor) -> Tensor:
        """Detect outliers by variance across batch."""
        # Compute variance per channel
        variance = x.var(dim=0)  # [dim]
        self.variance_profile = variance
        
        # Compute threshold
        median_variance = variance.median()
        threshold = median_variance * self.config.variance_threshold
        
        # Mark outliers
        outlier_mask = variance > threshold
        
        return outlier_mask
    
    def _detect_by_magnitude(self, x: Tensor) -> Tensor:
        """Detect outliers by average magnitude."""
        # Compute mean absolute value per channel
        magnitude = x.abs().mean(dim=0)  # [dim]
        self.variance_profile = magnitude
        
        # Compute threshold
        median_magnitude = magnitude.median()
        threshold = median_magnitude * self.config.variance_threshold
        
        # Mark outliers
        outlier_mask = magnitude > threshold
        
        return outlier_mask
    
    def encode_with_outliers(
        self,
        x: Tensor,
        outlier_mask: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """
        Encode with outlier handling.
        
        Args:
            x: Input tensor [batch, dim]
            outlier_mask: Pre-computed outlier mask [dim]
        
        Returns:
            Dictionary with encoded data
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Detect or use provided outliers
        if outlier_mask is None:
            if self.outlier_indices is None:
                outlier_mask = self.detect_outliers(x)
            else:
                outlier_mask = self.outlier_indices
        
        # Separate outlier and non-outlier channels
        outlier_channels = outlier_mask.nonzero(as_tuple=True)[0]  # [num_outliers]
        normal_channels = (~outlier_mask).nonzero(as_tuple=True)[0]  # [num_normal]
        
        num_outliers = len(outlier_channels)
        num_normal = len(normal_channels)
        
        # Encode outliers with high precision
        if num_outliers > 0:
            x_outliers = x[:, outlier_channels]  # [batch, num_outliers]
            # Keep in FP16 or use 8-bit quantization
            if self.config.outlier_bits >= 8:
                outliers_encoded = {
                    'data': x_outliers,
                    'mode': 'fp16'
                }
            else:
                indices, scales, norms, _ = quantize_scalar(
                    x_outliers, self.config.outlier_bits
                )
                outliers_encoded = {
                    'indices': indices,
                    'scales': scales,
                    'norms': norms,
                    'mode': 'quantized'
                }
        else:
            outliers_encoded = {'mode': 'none'}
        
        # Encode normal channels with aggressive quantization
        if num_normal > 0:
            x_normal = x[:, normal_channels]  # [batch, num_normal]
            indices, scales, norms, _ = quantize_scalar(
                x_normal, self.config.main_bits
            )
            normal_encoded = {
                'indices': indices,
                'scales': scales,
                'norms': norms
            }
        else:
            normal_encoded = {'mode': 'none'}
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_channels': outlier_channels,
            'normal_channels': normal_channels,
            'outliers': outliers_encoded,
            'normal': normal_encoded,
            'original_shape': x.shape
        }
    
    def decode_with_outliers(self, encoded: Dict[str, Any]) -> Tensor:
        """
        Decode with outlier handling.
        
        Args:
            encoded: Dictionary from encode_with_outliers
        
        Returns:
            Reconstructed tensor [batch, dim]
        """
        batch_size = encoded['original_shape'][0]
        dim = encoded['original_shape'][1]
        
        # Initialize output
        x_reconstructed = torch.zeros(
            batch_size, dim, device=self.device, dtype=torch.float32
        )
        
        outlier_mask = encoded['outlier_mask']
        outlier_channels = encoded['outlier_channels']
        normal_channels = encoded['normal_channels']
        
        # Decode outliers
        if encoded['outliers'] is not None and encoded['outliers'].get('mode') == 'fp16':
            x_outliers = encoded['outliers']['data']
            x_reconstructed[:, outlier_channels] = x_outliers
        elif encoded['outliers'] is not None and encoded['outliers'].get('mode') == 'quantized':
            x_outliers = dequantize_scalar(
                encoded['outliers']['indices'],
                encoded['outliers']['scales'],
                encoded['outliers'].get('norms'),
                self.config.outlier_bits
            )
            x_reconstructed[:, outlier_channels] = x_outliers

        # Decode normal channels
        if encoded.get('normal') is not None and encoded['normal'].get('mode') != 'none':
            x_normal = dequantize_scalar(
                encoded['normal']['indices'],
                encoded['normal']['scales'],
                encoded['normal'].get('norms'),
                self.config.main_bits
            )
            x_reconstructed[:, normal_channels] = x_normal

        return x_reconstructed


class OutlierAwareCodec:
    """
    Complete codec with outlier handling integrated.
    
    Wraps standard TurboQuant codec with outlier detection
    and separate encoding for outlier channels.
    """
    
    def __init__(
        self,
        dim: int,
        main_bits: int = 2,
        outlier_bits: int = 8,
        variance_threshold: float = 10.0,
        qjl_dim: int = 64,
        device: Optional[torch.device] = None
    ):
        """
        Initialize outlier-aware codec.
        
        Args:
            dim: Vector dimension
            main_bits: Bits for non-outlier channels
            outlier_bits: Bits for outlier channels
            variance_threshold: Outlier detection threshold
            qjl_dim: QJL dimension for residuals
            device: Torch device
        """
        from .codec import TurboQuantConfig, TurboQuantCodec
        
        self.dim = dim
        self.device = device or torch.device('cpu')
        
        # Outlier handler
        self.outlier_config = OutlierConfig(
            variance_threshold=variance_threshold,
            outlier_bits=outlier_bits,
            main_bits=main_bits
        )
        self.outlier_handler = OutlierHandler(self.outlier_config, dim)
        
        # Main codec for non-outlier channels
        self.main_codec = TurboQuantCodec(
            dim=dim,
            config=TurboQuantConfig(
                num_bits=main_bits,
                qjl_dim=qjl_dim
            ),
            device=self.device
        )
        
        # Statistics
        self.num_outliers = 0
        self.compression_stats = []
    
    def encode(self, x: Tensor) -> Dict[str, Any]:
        """
        Encode with outlier handling.
        
        Args:
            x: Input tensor [batch, dim]
        
        Returns:
            Encoded dictionary
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Detect outliers
        outlier_mask = self.outlier_handler.detect_outliers(x)
        self.num_outliers = outlier_mask.sum().item()
        
        # Split into outlier and normal components
        outlier_channels = outlier_mask.nonzero(as_tuple=True)[0]
        normal_channels = (~outlier_mask).nonzero(as_tuple=True)[0]
        
        encoded = {
            'outlier_mask': outlier_mask,
            'outlier_channels': outlier_channels,
            'normal_channels': normal_channels
        }
        
        # Encode outliers separately
        if len(outlier_channels) > 0:
            x_outliers = x[:, outlier_channels]
            # Use higher precision for outliers
            encoded['outliers'] = {
                'data': x_outliers,
                'bits': self.outlier_config.outlier_bits
            }
        else:
            encoded['outliers'] = None
        
        # Encode normal channels
        if len(normal_channels) > 0:
            x_normal = x[:, normal_channels]
            encoded['normal'] = self.main_codec.encode_key(x_normal)
        else:
            encoded['normal'] = None
        
        encoded['original_shape'] = x.shape
        
        # Track compression stats
        original_bits = x.numel() * 32
        compressed_bits = self._estimate_compressed_bits(encoded)
        self.compression_stats.append({
            'original': original_bits,
            'compressed': compressed_bits,
            'factor': original_bits / compressed_bits
        })
        
        return encoded
    
    def decode(self, encoded: Dict[str, Any]) -> Tensor:
        """
        Decode with outlier handling.
        
        Args:
            encoded: Dictionary from encode()
        
        Returns:
            Reconstructed tensor
        """
        batch_size = encoded['original_shape'][0]
        dim = encoded['original_shape'][1]
        
        x_reconstructed = torch.zeros(
            batch_size, dim, device=self.device, dtype=torch.float32
        )
        
        outlier_channels = encoded['outlier_channels']
        normal_channels = encoded['normal_channels']
        
        # Decode outliers
        if encoded['outliers'] is not None:
            x_outliers = encoded['outliers']['data']
            x_reconstructed[:, outlier_channels] = x_outliers
        
        # Decode normal channels
        if encoded['normal'] is not None:
            x_normal = self.main_codec.decode_key(encoded['normal'])
            x_reconstructed[:, normal_channels] = x_normal
        
        return x_reconstructed
    
    def _estimate_compressed_bits(self, encoded: Dict[str, Any]) -> int:
        """Estimate compressed size in bits."""
        bits = 0
        
        # Outlier mask
        bits += self.dim
        
        # Outliers (FP16)
        if encoded['outliers'] is not None:
            num_outliers = len(encoded['outlier_channels'])
            batch_size = encoded['original_shape'][0]
            bits += batch_size * num_outliers * 16
        
        # Normal channels
        if encoded['normal'] is not None:
            num_normal = len(encoded['normal_channels'])
            batch_size = encoded['original_shape'][0]
            bits += batch_size * num_normal * self.outlier_config.main_bits
            # Plus QJL overhead
            bits += batch_size * self.main_codec.config.qjl_dim
        
        return bits
    
    def get_stats(self) -> Dict[str, Any]:
        """Get codec statistics."""
        avg_factor = sum(s['factor'] for s in self.compression_stats) / len(self.compression_stats) if self.compression_stats else 0
        
        return {
            'dim': self.dim,
            'main_bits': self.outlier_config.main_bits,
            'outlier_bits': self.outlier_config.outlier_bits,
            'avg_outliers': self.num_outliers,
            'avg_compression_factor': f"{avg_factor:.1f}x"
        }


def apply_outlier_aware_quantization(
    x: Tensor,
    main_bits: int = 2,
    outlier_bits: int = 8,
    variance_threshold: float = 10.0,
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Convenience function for outlier-aware quantization.
    
    Args:
        x: Input tensor [batch, dim]
        main_bits: Bits for non-outlier channels
        outlier_bits: Bits for outlier channels
        variance_threshold: Outlier detection threshold
        device: Torch device
    
    Returns:
        (reconstructed, stats)
    """
    codec = OutlierAwareCodec(
        dim=x.shape[-1],
        main_bits=main_bits,
        outlier_bits=outlier_bits,
        variance_threshold=variance_threshold,
        device=device
    )
    
    encoded = codec.encode(x)
    x_reconstructed = codec.decode(encoded)
    
    # Compute quality metrics
    mse = ((x - x_reconstructed) ** 2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        x.view(-1, x.shape[-1]),
        x_reconstructed.view(-1, x_reconstructed.shape[-1])
    ).mean().item()
    
    stats = {
        'mse': mse,
        'cosine_similarity': cosine_sim,
        'num_outliers': codec.num_outliers,
        'compression_factor': codec.compression_stats[-1]['factor'] if codec.compression_stats else 0
    }
    
    return x_reconstructed, stats
