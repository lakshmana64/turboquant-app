"""
Scalar Quantization Module for TurboQuant

Implements MSE-optimal Lloyd-Max scalar quantization with random rotation.
The rotation ensures coordinates follow a concentrated distribution (approx. Gaussian),
making scalar quantization near-optimal.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import math
from .wht import apply_random_wht


def _generate_rotation_matrix(d: int, seed: int = 42, device: Optional[torch.device] = None) -> Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of Gaussian matrix.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    A = torch.randn(d, d, generator=generator, device=device)
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
    """
    num_levels = 2 ** num_bits
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    samples = torch.randn(num_samples, generator=generator, device=device)
    
    centroids = torch.linspace(-4, 4, num_levels + 1, device=device)
    centroids = (centroids[:-1] + centroids[1:]) / 2
    
    for _ in range(50):
        distances = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = distances.argmin(dim=1)
        
        counts = torch.zeros(num_levels, device=device)
        ones = torch.ones_like(assignments, dtype=torch.float32)
        counts = counts.scatter_add(0, assignments, ones)
        
        sums = torch.zeros(num_levels, device=device)
        sums = sums.scatter_add(0, assignments, samples)
        
        mask = (counts > 0)
        new_centroids = centroids.clone()
        new_centroids[mask] = sums[mask] / counts[mask]
        centroids = new_centroids
    
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return centroids, boundaries


_codebook_cache: dict = {}


def get_codebook(
    num_bits: int,
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Tensor]:
    """
    Get or compute Lloyd-Max codebook for given bit width.
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
    rotation_type: str = "random",
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Apply MSE-optimal scalar quantization to input vectors.
    """
    orig_shape = x.shape
    device = x.device
    x_flat = x.view(-1, x.shape[-1])
    N, d = x_flat.shape
    
    if rotation_type == "hadamard":
        x_rotated = apply_random_wht(x_flat, seed=rotation_seed)
    else:
        if rotation_matrix is None:
            rotation_matrix = _generate_rotation_matrix(d, rotation_seed, device)
        x_rotated = x_flat @ rotation_matrix
    
    norms = x_rotated.norm(dim=1, keepdim=True)
    is_zero = norms < eps
    scales = torch.where(is_zero, torch.ones_like(norms), norms / math.sqrt(d))
    x_normalized = x_rotated / (scales + eps)
    
    centroids, boundaries = get_codebook(num_bits, device)
    indices = torch.searchsorted(boundaries, x_normalized.contiguous())
    indices = indices.clamp(0, len(centroids) - 1)
    
    indices = indices.view(orig_shape)
    scales = scales.view(*orig_shape[:-1], 1)
    norms = norms.view(*orig_shape[:-1], 1)
    
    return indices, scales, norms, rotation_matrix


def dequantize_scalar(
    indices: Tensor,
    scales: Tensor,
    num_bits: int,
    rotation_matrix: Optional[Tensor] = None,
    rotation_seed: int = 42,
    rotation_type: str = "random"
) -> Tensor:
    """
    Reconstruct vectors from scalar quantization indices.
    """
    orig_shape = indices.shape
    device = indices.device
    indices_flat = indices.view(-1, indices.shape[-1])
    scales_flat = scales.view(-1, 1)
    
    centroids, _ = get_codebook(num_bits, device)
    x_normalized = centroids[indices_flat]
    x_scaled = x_normalized * scales_flat
    
    if rotation_type == "hadamard":
        x_hat = apply_random_wht(x_scaled, seed=rotation_seed)
    else:
        if rotation_matrix is None:
            rotation_matrix = _generate_rotation_matrix(indices_flat.shape[-1], rotation_seed, device)
        x_hat = x_scaled @ rotation_matrix.T
    
    return x_hat.view(orig_shape)


def quantize_and_reconstruct(
    x: Tensor,
    num_bits: int,
    rotation_seed: int = 42
) -> Tuple[Tensor, dict]:
    """
    Convenience function for quantization and immediate reconstruction.
    """
    indices, scales, norms, rotation_matrix = quantize_scalar(
        x, num_bits, rotation_seed=rotation_seed
    )
    x_hat = dequantize_scalar(indices, scales, num_bits, rotation_matrix)
    metadata = {'indices': indices, 'scales': scales, 'norms': norms, 'rotation_matrix': rotation_matrix}
    return x_hat, metadata
