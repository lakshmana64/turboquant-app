"""
Scalar Quantization Module for TurboQuant

Implements MSE-optimal Lloyd-Max scalar quantization with random rotation.
The rotation ensures coordinates follow a concentrated distribution (approx. Gaussian),
making scalar quantization near-optimal.

Key insight: After rotating by a random orthogonal matrix, vector coordinates
concentrate around N(0, ||x||^2/d), enabling efficient per-coordinate quantization.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import math


def _generate_rotation_matrix(d: int, seed: int = 42, device: Optional[torch.device] = None) -> Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of Gaussian matrix.
    
    This rotation is data-oblivious and ensures coordinate concentration.
    
    Args:
        d: Dimension of the space
        seed: Random seed for reproducibility
        device: Target device
        
    Returns:
        Orthogonal matrix Q of shape (d, d)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # Generate random Gaussian matrix
    A = torch.randn(d, d, generator=generator, device=device)
    
    # QR decomposition to get orthogonal matrix
    Q, _ = torch.linalg.qr(A)
    
    return Q


def _compute_lloyd_max_codebook(
    num_bits: int,
    num_samples: int = 100000,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Tensor]:
    """
    Compute Lloyd-Max optimal quantization codebook via numerical optimization.
    
    For a standard normal distribution N(0, 1), finds centroids and boundaries
    that minimize MSE.
    
    Args:
        num_bits: Number of bits per coordinate (determines 2^num_bits levels)
        num_samples: Number of samples for Monte Carlo optimization
        seed: Random seed
        device: Target device
        
    Returns:
        centroids: Tensor of shape (2^num_bits,) - optimal reconstruction values
        boundaries: Tensor of shape (2^num_bits - 1,) - decision boundaries
    """
    num_levels = 2 ** num_bits
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # Sample from standard normal
    samples = torch.randn(num_samples, generator=generator, device=device)
    
    # Initialize centroids uniformly in [-4, 4] (covers ~99.99% of N(0,1))
    centroids = torch.linspace(-4, 4, num_levels + 1, device=device)
    centroids = (centroids[:-1] + centroids[1:]) / 2  # Midpoints
    
    # Iterative Lloyd-Max optimization
    for _ in range(50):
        # Assign samples to nearest centroid
        # distances: (num_samples, num_levels)
        distances = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = distances.argmin(dim=1)
        
        # Vectorized update using scatter_reduce
        new_centroids = torch.zeros(num_levels, device=device)
        counts = torch.zeros(num_levels, device=device)
        
        # counts: [num_levels]
        ones = torch.ones_like(assignments, dtype=torch.float32)
        counts = counts.scatter_add(0, assignments, ones)
        
        # sum of samples per centroid
        sums = torch.zeros(num_levels, device=device)
        sums = sums.scatter_add(0, assignments, samples)
        
        # Update centroids: mean = sum / count
        # Update only if count > 0 to avoid NaNs
        mask = (counts > 0)
        new_centroids = centroids.clone()
        new_centroids[mask] = sums[mask] / counts[mask]
        centroids = new_centroids
    
    # Compute boundaries as midpoints between centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    
    return centroids, boundaries


# Cache for precomputed codebooks (shared across quantization operations)
_codebook_cache: dict = {}


def get_codebook(
    num_bits: int,
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Tensor]:
    """
    Get or compute Lloyd-Max codebook for given bit width.
    
    Uses caching to avoid recomputation. Codebooks are device-specific.
    
    Args:
        num_bits: Bits per coordinate
        device: Target device
        
    Returns:
        centroids: Optimal reconstruction values
        boundaries: Decision boundaries
    """
    cache_key = (num_bits, str(device) if device else 'cpu')
    
    if cache_key not in _codebook_cache:
        centroids, boundaries = _compute_lloyd_max_codebook(num_bits, device=device)
        _codebook_cache[cache_key] = (centroids, boundaries)
    
    return _codebook_cache[cache_key]


def quantize_scalar(
    x: Tensor,
    num_bits: int,
    rotation_matrix: Optional[Tensor] = None,
    rotation_seed: int = 42,
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Apply MSE-optimal scalar quantization to input vectors.
    
    Pipeline:
    1. Optionally rotate input (for coordinate concentration)
    2. Scale per-vector to unit variance
    3. Quantize each coordinate to nearest codebook level
    4. Store scale for dequantization
    
    Args:
        x: Input tensor of shape (..., d) - last dim is feature dimension
        num_bits: Bits per coordinate
        rotation_matrix: Precomputed rotation matrix (d, d), or None
        rotation_seed: Seed for generating rotation if not provided
        eps: Numerical stability constant
        
    Returns:
        indices: Quantized indices (..., d), values in [0, 2^num_bits - 1]
        scales: Per-vector scale factors (..., 1)
        norms: Original vector norms (..., 1) for residual computation
        rotation_matrix: The rotation matrix used (returned for consistency)
    """
    # Preserve original shape
    orig_shape = x.shape
    device = x.device
    
    # Flatten to (N, d) for processing
    x_flat = x.view(-1, x.shape[-1])
    N, d = x_flat.shape
    
    # Generate or use provided rotation matrix
    if rotation_matrix is None:
        rotation_matrix = _generate_rotation_matrix(d, rotation_seed, device)
    
    # Rotate: induces coordinate concentration
    x_rotated = x_flat @ rotation_matrix
    
    # Compute per-vector norm for scaling
    # After rotation, E[x_i^2] ≈ ||x||^2 / d
    norms = x_rotated.norm(dim=1, keepdim=True)  # (N, 1)
    
    # Handle zero vectors
    is_zero = norms < eps
    scales = torch.where(is_zero, torch.ones_like(norms), norms / math.sqrt(d))
    
    # Normalize to unit variance
    x_normalized = x_rotated / (scales + eps)
    
    # Get codebook
    centroids, boundaries = get_codebook(num_bits, device)
    
    # Quantize: find nearest centroid for each coordinate
    # Using searchsorted for efficiency
    # boundaries is sorted, so we can use digitize
    indices = torch.searchsorted(boundaries, x_normalized.contiguous())
    indices = indices.clamp(0, len(centroids) - 1)
    
    # Restore original shape
    indices = indices.view(orig_shape)
    scales = scales.view(*orig_shape[:-1], 1)
    norms = norms.view(*orig_shape[:-1], 1)
    
    return indices, scales, norms, rotation_matrix


def dequantize_scalar(
    indices: Tensor,
    scales: Tensor,
    num_bits: int,
    rotation_matrix: Optional[Tensor] = None,
    rotation_seed: int = 42
) -> Tensor:
    """
    Reconstruct vectors from scalar quantization indices.
    
    Args:
        indices: Quantized indices (..., d)
        scales: Per-vector scale factors (..., 1)
        num_bits: Bits per coordinate
        rotation_matrix: Rotation matrix used during encoding
        rotation_seed: Seed for rotation if not provided
        
    Returns:
        x_hat: Reconstructed vectors (..., d)
    """
    orig_shape = indices.shape
    device = indices.device
    
    # Flatten
    indices_flat = indices.view(-1, indices.shape[-1])
    scales_flat = scales.view(-1, 1)
    d = indices_flat.shape[-1]
    
    # Get codebook
    centroids, _ = get_codebook(num_bits, device)
    
    # Lookup centroids
    x_normalized = centroids[indices_flat]
    
    # Rescale
    x_scaled = x_normalized * scales_flat
    
    # Inverse rotation
    if rotation_matrix is None:
        rotation_matrix = _generate_rotation_matrix(d, rotation_seed, device)
    
    x_rotated = x_scaled @ rotation_matrix.T
    
    # Restore shape
    return x_rotated.view(orig_shape)


def quantize_and_reconstruct(
    x: Tensor,
    num_bits: int,
    rotation_seed: int = 42
) -> Tuple[Tensor, dict]:
    """
    Convenience function for quantization and immediate reconstruction.
    
    Args:
        x: Input tensor (..., d)
        num_bits: Bits per coordinate
        rotation_seed: Random seed for rotation
        
    Returns:
        x_hat: Reconstructed tensor
        metadata: Dict with indices, scales, norms, rotation_matrix
    """
    indices, scales, norms, rotation_matrix = quantize_scalar(
        x, num_bits, rotation_seed=rotation_seed
    )
    
    x_hat = dequantize_scalar(indices, scales, num_bits, rotation_matrix)
    
    metadata = {
        'indices': indices,
        'scales': scales,
        'norms': norms,
        'rotation_matrix': rotation_matrix
    }
    
    return x_hat, metadata
