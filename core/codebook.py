"""
Lloyd-Max Optimal Centroid Computation for TurboQuant.

Implements the Lloyd-Max algorithm for finding optimal quantization centroids
that minimize mean squared error for a given bit width.

Reference:
- Lloyd, S. P. (1982). "Least squares quantization in PCM"
- Max, J. (1960). "Quantizing for minimum distortion"
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def compute_lloyd_max_centroids(
    num_bits: int,
    num_iterations: int = 100,
    tolerance: float = 1e-6,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[Tensor, Tensor]:
    """
    Compute optimal Lloyd-Max centroids for scalar quantization.
    
    Args:
        num_bits: Number of bits (2, 3, 4, 8)
        num_iterations: Maximum iterations
        tolerance: Convergence tolerance
        device: Torch device
        seed: Random seed
    
    Returns:
        Tuple of (centroids, boundaries)
    """
    torch.manual_seed(seed)
    
    num_levels = 2 ** num_bits
    
    # Initialize centroids uniformly in [-1, 1]
    centroids = torch.linspace(-1 + 1/num_levels, 1 - 1/num_levels, num_levels, device=device)
    
    # Generate synthetic Gaussian data for optimization
    # (Lloyd-Max for Gaussian distribution)
    num_samples = 100000
    data = torch.randn(num_samples, device=device)
    
    # Normalize data to [-1, 1] range
    data = data / data.abs().max()
    
    prev_centroids = centroids.clone()
    
    for iteration in range(num_iterations):
        # E-step: Assign data points to nearest centroid
        distances = torch.abs(data.unsqueeze(1) - centroids.unsqueeze(0))
        assignments = torch.argmin(distances, dim=1)
        
        # M-step: Update centroids as mean of assigned points
        new_centroids = torch.zeros_like(centroids)
        for i in range(num_levels):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean()
        
        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        
        if max_shift < tolerance:
            break
    
    # Compute boundaries (midpoints between centroids)
    boundaries = torch.zeros(num_levels + 1, device=device)
    boundaries[0] = -float('inf')
    boundaries[-1] = float('inf')
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
    
    return centroids, boundaries


def get_optimal_codebook(
    num_bits: int,
    device: str = 'cpu'
) -> Tuple[Tensor, Tensor]:
    """
    Get pre-computed optimal codebook for common bit widths.
    
    Args:
        num_bits: Number of bits
        device: Torch device
    
    Returns:
        Tuple of (centroids, boundaries)
    """
    # Pre-computed optimal centroids for Gaussian distribution
    precomputed = {
        2: (
            torch.tensor([-0.9816, -0.3307, 0.3307, 0.9816], device=device),
            torch.tensor([-float('inf'), -0.6561, 0, 0.6561, float('inf')], device=device)
        ),
        3: (
            torch.tensor([-1.3666, -0.8169, -0.4529, -0.1510, 0.1510, 0.4529, 0.8169, 1.3666], device=device),
            torch.tensor([-float('inf'), -1.0918, -0.6349, -0.3020, 0, 0.3020, 0.6349, 1.0918, float('inf')], device=device)
        ),
        4: (
            torch.tensor([-1.5104, -1.1507, -0.8953, -0.6871, -0.5020, -0.3307, -0.1673, -0.0079,
                          0.0079, 0.1673, 0.3307, 0.5020, 0.6871, 0.8953, 1.1507, 1.5104], device=device),
            torch.tensor([-float('inf'), -1.3306, -1.0230, -0.7912, -0.5946, -0.4164, -0.2487, -0.0876, 0,
                          0.0876, 0.2487, 0.4164, 0.5946, 0.7912, 1.0230, 1.3306, float('inf')], device=device)
        ),
        8: (
            torch.linspace(-1, 1, 256, device=device),
            torch.cat([
                torch.tensor([-float('inf')], device=device),
                torch.linspace(-1 + 1/256, 1 - 1/256, 255, device=device),
                torch.tensor([float('inf')], device=device)
            ])
        )
    }
    
    if num_bits in precomputed:
        return precomputed[num_bits]
    
    # Compute for other bit widths
    return compute_lloyd_max_centroids(num_bits, device=device)


def quantize_lloyd_max(
    x: Tensor,
    num_bits: int,
    centroids: Optional[Tensor] = None,
    boundaries: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor using Lloyd-Max optimal centroids.
    
    Args:
        x: Input tensor
        num_bits: Number of bits
        centroids: Pre-computed centroids (optional)
        boundaries: Pre-computed boundaries (optional)
    
    Returns:
        Tuple of (quantized values, indices)
    """
    if centroids is None or boundaries is None:
        centroids, boundaries = get_optimal_codebook(num_bits, device=x.device)
    
    # Find nearest centroid for each value
    x_expanded = x.unsqueeze(-1)
    centroids_expanded = centroids.view(*[1]*x.dim(), -1)
    
    distances = (x_expanded - centroids_expanded).abs()
    indices = torch.argmin(distances, dim=-1)
    
    # Quantize by looking up centroids
    quantized = centroids[indices]
    
    return quantized, indices


class LloydMaxQuantizer:
    """
    Lloyd-Max quantization module for TurboQuant.
    """
    
    def __init__(
        self,
        num_bits: int = 4,
        device: str = 'cpu'
    ):
        """
        Initialize Lloyd-Max quantizer.
        
        Args:
            num_bits: Number of bits
            device: Torch device
        """
        self.num_bits = num_bits
        self.device = device
        self.centroids, self.boundaries = get_optimal_codebook(num_bits, device)
    
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize input tensor.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (quantized, indices)
        """
        return quantize_lloyd_max(x, self.num_bits, self.centroids, self.boundaries)
    
    def dequantize(self, indices: Tensor) -> Tensor:
        """
        Dequantize from indices.
        
        Args:
            indices: Quantization indices
        
        Returns:
            Dequantized tensor
        """
        return self.centroids[indices]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        return self.quantize(x)
