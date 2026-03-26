"""
Quantized Johnson-Lindenstrauss (QJL) Projection Module

Implements the 1-bit QJL transform for residual encoding.
The QJL projection enables unbiased inner product estimation by capturing
the residual information lost during scalar quantization.

Key insight: For residual r, storing sign(R @ r) where R is a random
Gaussian matrix allows estimating <r, q> via <R@q, sign(R@r)>.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import math


class QJLProjection:
    """
    Quantized Johnson-Lindenstrauss projection for residual encoding.
    
    Uses a shared random Gaussian projection matrix R to project residuals
    into a lower-dimensional space, then stores only the sign (1 bit per dim).
    
    The projection is:
      - Data-oblivious (random, not learned)
      - Deterministic (fixed seed)
      - Reusable across vectors
      - Batch-efficient
    
    Attributes:
        input_dim: Input dimension d
        output_dim: Output dimension m (number of projection bits)
        seed: Random seed for reproducibility
        projection_matrix: The random matrix R of shape (m, d)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize QJL projection.
        
        Args:
            input_dim: Dimension of input vectors (d)
            output_dim: Dimension of projected space (m)
            seed: Random seed for reproducibility
            device: Target device
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.device = device
        
        # Generate projection matrix
        # R has entries ~ N(0, 1) - normalization handled in projection
        self.projection_matrix = self._generate_projection_matrix()
    
    def _generate_projection_matrix(self) -> Tensor:
        """
        Generate random Gaussian projection matrix.
        
        Uses fixed seed for determinism. Matrix is registered as buffer
        for potential GPU residency.
        
        Returns:
            Projection matrix R of shape (output_dim, input_dim)
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        
        # Standard Gaussian entries
        R = torch.randn(
            self.output_dim,
            self.input_dim,
            generator=generator,
            device=self.device
        )
        
        # NOTE: We do NOT normalize rows here as the scaling factor sqrt(pi/2)/m 
        # in the estimator assumes standard normal entries N(0, 1).
        
        return R
    
    def project(self, x: Tensor) -> Tensor:
        """
        Apply QJL projection to input vectors.
        
        Computes R @ x and returns the result (not quantized).
        Used for queries in inner product estimation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Projected values of shape (..., output_dim)
        """
        # Handle batch dimensions
        orig_shape = x.shape
        x_flat = x.view(-1, self.input_dim)
        
        # Project: (batch, m) = (batch, d) @ (d, m)
        projected = x_flat @ self.projection_matrix.T
        
        return projected.view(*orig_shape[:-1], self.output_dim)
    
    def project_and_quantize(self, r: Tensor) -> Tensor:
        """
        Apply QJL projection and 1-bit quantization (sign).
        
        This is the encoding operation for residuals.
        Stores only the sign of each projection (1 bit per dimension).
        
        Args:
            r: Residual tensor of shape (..., input_dim)
            
        Returns:
            Sign vector of shape (..., output_dim), values in {-1, +1}
        """
        projected = self.project(r)
        
        # 1-bit quantization: sign
        # Handle zero case (map to +1 for consistency)
        signs = torch.sign(projected)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        
        return signs
    
    def estimate_inner_product(
        self,
        q_projected: Tensor,
        r_signs: Tensor,
        residual_norm: Tensor
    ) -> Tensor:
        """
        Estimate inner product <q, r> from QJL encoding.
        
        Uses the formula:
          <q, r> ≈ sqrt(π/2) * ||r|| / m * <R@q, sign(R@r)>
        
        This is an unbiased estimator with variance O(1/m).
        
        Args:
            q_projected: Projected query R@q of shape (..., output_dim)
            r_signs: Sign vector sign(R@r) of shape (..., output_dim)
            residual_norm: Norm ||r|| of shape (..., 1)
            
        Returns:
            Estimated inner product of shape (...)
        """
        m = self.output_dim
        
        # Scaling factor from Gaussian projection theory
        # E[<Rq, sign(Rr)>] = sqrt(2/π) * m * <q, r> / ||r||
        # So: <q, r> ≈ sqrt(π/2) * ||r|| / m * <Rq, sign(Rr)>
        scale = math.sqrt(math.pi / 2) * residual_norm / m
        
        # Inner product in projected space
        dot_product = (q_projected * r_signs).sum(dim=-1, keepdim=True)
        
        # Scaled estimate
        estimate = scale * dot_product
        
        return estimate.squeeze(-1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Alias for project()."""
        return self.project(x)


def create_qjl_projection(
    input_dim: int,
    output_dim: int,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> QJLProjection:
    """
    Factory function to create QJL projection.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (number of bits)
        seed: Random seed
        device: Target device
        
    Returns:
        QJLProjection instance
    """
    return QJLProjection(input_dim, output_dim, seed, device)


def batch_encode_residuals(
    residuals: Tensor,
    qjl: QJLProjection
) -> Tuple[Tensor, Tensor]:
    """
    Batch encode multiple residuals using QJL.
    
    Args:
        residuals: Tensor of shape (batch, input_dim)
        qjl: QJLProjection instance
        
    Returns:
        signs: Sign vectors (batch, output_dim)
        norms: Residual norms (batch, 1)
    """
    norms = residuals.norm(dim=1, keepdim=True)
    signs = qjl.project_and_quantize(residuals)
    return signs, norms


def batch_decode_inner_products(
    queries: Tensor,
    signs: Tensor,
    norms: Tensor,
    qjl: QJLProjection
) -> Tensor:
    """
    Batch estimate inner products between queries and residual-encoded vectors.
    
    Args:
        queries: Query vectors (batch_q, input_dim)
        signs: Encoded residuals (batch_r, output_dim)
        norms: Residual norms (batch_r, 1)
        qjl: QJLProjection instance
        
    Returns:
        Estimated inner products (batch_q, batch_r)
    """
    # Project all queries
    q_projected = qjl.project(queries)  # (batch_q, output_dim)
    
    # Compute estimates for all pairs
    # This can be done efficiently via matrix multiplication
    # <Rq_i, sign_j> for all i, j
    
    # q_projected: (batch_q, m)
    # signs: (batch_r, m)
    # Result: (batch_q, batch_r)
    dot_products = q_projected @ signs.T  # (batch_q, batch_r)
    
    # Apply scaling per residual
    m = qjl.output_dim
    scale = math.sqrt(math.pi / 2) / m  # scalar
    scaled = scale * dot_products  # (batch_q, batch_r)
    
    # Multiply by norm per residual
    # norms: (batch_r, 1) -> (1, batch_r)
    result = scaled * norms.T  # (batch_q, batch_r)
    
    return result
