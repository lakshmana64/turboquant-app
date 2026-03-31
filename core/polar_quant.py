"""
PolarQuant Algorithm for TurboQuant.

PolarQuant is an advanced quantization technique that combines:
1. Walsh-Hadamard Transform (WHT) for rotation
2. Polar coordinate representation for efficient encoding
3. Optimized centroid computation for minimal distortion

This implementation matches the approach used in turboquant_plus
for KV cache compression with 4.6-6.4x compression ratios.

Reference:
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- ICLR 2026 implementation
"""

import torch
from torch import Tensor
from typing import Tuple, Dict, Any, Optional
import math

from .wht import apply_random_wht, fast_walsh_hadamard_transform
from .turbo_formats import TurboFormat, TURBO2, TURBO3, TURBO4


class PolarQuantConfig:
    """Configuration for PolarQuant."""
    
    def __init__(
        self,
        bits: int = 2,
        qjl_dim: int = 64,
        use_wht: bool = True,
        wht_seed: int = 42,
        qjl_seed: int = 42,
        optimize_centroids: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize PolarQuant config.
        
        Args:
            bits: Number of bits for scalar quantization (2, 3, 4)
            qjl_dim: QJL projection dimension for residuals
            use_wht: Enable Walsh-Hadamard rotation
            wht_seed: Seed for WHT random signs
            qjl_seed: Seed for QJL projection
            optimize_centroids: Use optimized centroids vs uniform
            device: Torch device
        """
        self.bits = bits
        self.qjl_dim = qjl_dim
        self.use_wht = use_wht
        self.wht_seed = wht_seed
        self.qjl_seed = qjl_seed
        self.optimize_centroids = optimize_centroids
        self.device = device or torch.device('cpu')
        
        # Compute number of levels
        self.num_levels = 2 ** bits


class PolarQuantCodec:
    """
    PolarQuant codec for KV cache compression.
    
    Combines WHT rotation, polar coordinate quantization,
    and QJL residuals for near-optimal distortion rate.
    """
    
    def __init__(self, config: PolarQuantConfig, dim: int):
        """
        Initialize PolarQuant codec.
        
        Args:
            config: PolarQuantConfig instance
            dim: Input dimension
        """
        self.config = config
        self.dim = dim
        self.device = config.device
        
        # Pre-compute centroids
        if config.optimize_centroids:
            self.centroids = self._compute_optimal_centroids()
        else:
            self.centroids = self._compute_uniform_centroids()
        
        # QJL projection matrix
        self.qjl_matrix = self._init_qjl_matrix()
    
    def _compute_uniform_centroids(self) -> Tensor:
        """Compute uniform quantization centroids."""
        num_levels = self.config.num_levels
        # Uniform centroids in [-1, 1]
        centroids = torch.linspace(-1 + 1/num_levels, 1 - 1/num_levels, num_levels, device=self.device)
        return centroids
    
    def _compute_optimal_centroids(self) -> Tensor:
        """
        Compute optimal centroids using Lloyd-Max algorithm.
        
        For Gaussian-distributed data (after WHT rotation),
        optimal centroids are non-uniformly spaced.
        """
        num_levels = self.config.num_levels
        
        # For Gaussian distribution, use optimal Lloyd-Max centroids
        # These are pre-computed for common bit widths
        if num_levels == 4:  # 2-bit
            # Optimal for standard normal
            boundaries = torch.tensor([-float('inf'), -0.9816, 0, 0.9816, float('inf')], device=self.device)
            centroids = torch.tensor([-1.5104, -0.4529, 0.4529, 1.5104], device=self.device)
        elif num_levels == 8:  # 3-bit
            boundaries = torch.tensor([
                -float('inf'), -1.7480, -1.0595, -0.5723, 0, 0.5723, 1.0595, 1.7480, float('inf')
            ], device=self.device)
            centroids = torch.tensor([
                -2.1542, -1.3666, -0.8169, -0.2887, 0.2887, 0.8169, 1.3666, 2.1542
            ], device=self.device)
        elif num_levels == 16:  # 4-bit
            # 16-level optimal Gaussian centroids
            centroids = torch.tensor([
                -2.6906, -2.1542, -1.7480, -1.3980, -1.0795, -0.7820, -0.4973, -0.2199,
                0.2199, 0.4973, 0.7820, 1.0795, 1.3980, 1.7480, 2.1542, 2.6906
            ], device=self.device)
        else:
            # Fallback to uniform
            return self._compute_uniform_centroids()
        
        # Normalize to [-1, 1] range
        centroids = centroids / centroids.abs().max()
        
        return centroids
    
    def _init_qjl_matrix(self) -> Tensor:
        """Initialize QJL projection matrix."""
        qjl_dim = self.config.qjl_dim
        # Gaussian projection matrix
        matrix = torch.randn(qjl_dim, self.dim, device=self.device)
        matrix = matrix / math.sqrt(qjl_dim)
        return matrix
    
    def encode(self, x: Tensor) -> Dict[str, Any]:
        """
        Encode using PolarQuant.
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Dictionary with encoded data
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)
        
        # Step 1: Apply WHT rotation (Gaussianizes the distribution)
        if self.config.use_wht:
            x_rotated = apply_random_wht(x_flat, seed=self.config.wht_seed)
        else:
            x_rotated = x_flat
        
        # Step 2: Compute magnitude (radius) and direction
        magnitudes = x_rotated.norm(dim=-1, keepdim=True)  # [..., 1]
        directions = x_rotated / (magnitudes + 1e-8)  # [..., dim]
        
        # Step 3: Quantize magnitude
        mag_quantized, mag_indices, mag_scales = self._quantize_magnitude(magnitudes)
        
        # Step 4: Quantize direction using scalar quantization
        dir_quantized, dir_indices = self._quantize_direction(directions)
        
        # Step 5: Compute residual for QJL
        residual = x_rotated - dir_quantized * mag_quantized
        
        # Step 6: Apply QJL to residual
        qjl_signs, qjl_norms = self._apply_qjl(residual)
        
        return {
            'mag_indices': mag_indices,
            'mag_scales': mag_scales,
            'dir_indices': dir_indices,
            'qjl_signs': qjl_signs,
            'qjl_norms': qjl_norms,
            'original_shape': orig_shape
        }
    
    def decode(self, encoded: Dict[str, Any]) -> Tensor:
        """
        Decode PolarQuant representation.
        
        Args:
            encoded: Dictionary from encode()
        
        Returns:
            Reconstructed tensor
        """
        # Dequantize magnitude
        mag_quantized = self._dequantize_magnitude(
            encoded['mag_indices'],
            encoded['mag_scales']
        )
        
        # Dequantize direction
        dir_quantized = self._dequantize_direction(encoded['dir_indices'])
        
        # Reconstruct main component
        x_reconstructed = dir_quantized * mag_quantized
        
        # Add QJL residual correction
        qjl_correction = self._estimate_qjl_correction(
            encoded['qjl_signs'],
            encoded['qjl_norms']
        )
        
        x_reconstructed = x_reconstructed + qjl_correction
        
        # Reshape to original
        orig_shape = encoded['original_shape']
        return x_reconstructed.view(orig_shape)
    
    def _quantize_magnitude(
        self,
        magnitudes: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize magnitude values.
        
        Args:
            magnitudes: [..., 1] tensor
        
        Returns:
            (quantized, indices, scales)
        """
        # Use 2-bit quantization for magnitude
        mag_bits = 2
        num_levels = 2 ** mag_bits
        
        # Compute per-sample scales
        max_mag = magnitudes.max(dim=-1, keepdim=True)[0]
        scales = max_mag / (num_levels - 1) + 1e-8
        
        # Quantize to indices
        normalized = magnitudes / scales
        indices = torch.clamp(torch.round(normalized), 0, num_levels - 1).long()
        
        # Dequantize
        quantized = indices.to(magnitudes.dtype) * scales
        
        return quantized, indices, scales
    
    def _dequantize_magnitude(
        self,
        indices: Tensor,
        scales: Tensor
    ) -> Tensor:
        """Dequantize magnitude values."""
        return indices.to(scales.dtype) * scales
    
    def _quantize_direction(
        self,
        directions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Quantize direction vector.
        
        Uses scalar quantization per dimension with shared scale.
        
        Args:
            directions: [..., dim] normalized directions
        
        Returns:
            (quantized, indices)
        """
        # Compute scale based on max absolute value
        max_abs = directions.abs().max(dim=-1, keepdim=True)[0]
        scales = torch.clamp(max_abs, min=1e-8)
        
        # Normalize to [-1, 1]
        normalized = directions / scales
        
        # Quantize using centroids
        # Find nearest centroid for each dimension
        indices = self._find_nearest_centroid(normalized)
        quantized = self.centroids[indices] * scales
        
        return quantized, indices
    
    def _dequantize_direction(self, indices: Tensor) -> Tensor:
        """Dequantize direction from indices."""
        return self.centroids[indices]
    
    def _find_nearest_centroid(self, x: Tensor) -> Tensor:
        """Find nearest centroid index for each value."""
        # x: [..., dim]
        # centroids: [num_levels]
        # Compute distance to each centroid
        x_expanded = x.unsqueeze(-1)  # [..., dim, 1]
        centroids_expanded = self.centroids.view(1, -1)  # [1, num_levels]
        
        distances = (x_expanded - centroids_expanded).abs()
        indices = distances.argmin(dim=-1)
        
        return indices
    
    def _apply_qjl(
        self,
        residual: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply QJL (Quantized Johnson-Lindenstrauss) to residual.
        
        Args:
            residual: [..., dim] residual vector
        
        Returns:
            (signs, norms) where:
                signs: [..., qjl_dim] binary signs
                norms: [..., 1] residual norms
        """
        # Compute norms
        norms = residual.norm(dim=-1, keepdim=True)
        
        # Project to lower dimension
        projected = residual @ self.qjl_matrix.T  # [..., qjl_dim]
        
        # Extract signs
        signs = torch.sign(projected)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        return signs, norms
    
    def _estimate_qjl_correction(
        self,
        signs: Tensor,
        norms: Tensor
    ) -> Tensor:
        """
        Estimate residual correction from QJL data.
        
        Uses the QJL signs and norms to approximate the residual.
        """
        # Scale factor for QJL estimation
        m = self.config.qjl_dim
        scale = math.sqrt(math.pi / 2) / m
        
        # Project signs back to original space
        correction = signs @ self.qjl_matrix  # [..., dim]
        
        # Scale by norms
        correction = correction * norms * scale
        
        return correction


def polar_quant(
    x: Tensor,
    bits: int = 2,
    qjl_dim: int = 64,
    use_wht: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[Dict[str, Any], 'PolarQuantCodec']:
    """
    Convenience function for PolarQuant encoding.
    
    Args:
        x: Input tensor [..., dim]
        bits: Quantization bits (2, 3, 4)
        qjl_dim: QJL dimension
        use_wht: Enable WHT rotation
        device: Torch device
    
    Returns:
        (encoded_data, codec)
    """
    config = PolarQuantConfig(
        bits=bits,
        qjl_dim=qjl_dim,
        use_wht=use_wht,
        device=device
    )
    codec = PolarQuantCodec(config, x.shape[-1])
    encoded = codec.encode(x)
    return encoded, codec


def polar_quant_roundtrip(
    x: Tensor,
    bits: int = 2,
    qjl_dim: int = 64,
    use_wht: bool = True
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Perform PolarQuant encode-decode roundtrip.
    
    Args:
        x: Input tensor
        bits: Quantization bits
        qjl_dim: QJL dimension
        use_wht: Enable WHT
    
    Returns:
        (reconstructed, metrics)
    """
    encoded, codec = polar_quant(x, bits, qjl_dim, use_wht)
    x_reconstructed = codec.decode(encoded)
    
    # Compute metrics
    mse = ((x - x_reconstructed) ** 2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        x.view(-1, x.shape[-1]),
        x_reconstructed.view(-1, x.shape[-1])
    ).mean().item()
    
    # Compression ratio
    original_bits = x.numel() * 32  # FP32
    compressed_bits = (
        x.numel() * bits +  # Direction indices
        x.shape[:-1].numel() * 2 +  # Magnitude indices
        x.shape[:-1].numel() * 32 +  # Scales
        x.shape[:-1].numel() * qjl_dim +  # QJL signs
        x.shape[:-1].numel() * 32  # QJL norms
    )
    compression_factor = original_bits / compressed_bits
    
    metrics = {
        "mse": mse,
        "cosine_similarity": cosine_sim,
        "compression_factor": compression_factor,
        "bits_per_dim": compressed_bits / x.numel()
    }
    
    return x_reconstructed, metrics
