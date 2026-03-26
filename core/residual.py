"""
Residual Computation and Encoding Module

Handles the residual r = x - x_hat between original vectors and their
Stage 1 (scalar quantization) reconstruction. The residual is encoded
using 1-bit QJL for the correction term in inner product estimation.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

from .qjl_projection import QJLProjection


def compute_residual(
    x: Tensor,
    x_hat: Tensor
) -> Tensor:
    """
    Compute residual between original and reconstructed vectors.
    
    Args:
        x: Original vectors (..., d)
        x_hat: Reconstructed vectors (..., d)
        
    Returns:
        r: Residual vectors (..., d) where r = x - x_hat
    """
    return x - x_hat


def compute_residual_norm(
    x: Tensor,
    x_hat: Tensor,
    eps: float = 1e-8
) -> Tensor:
    """
    Compute L2 norm of residual vectors.
    
    Args:
        x: Original vectors (..., d)
        x_hat: Reconstructed vectors (..., d)
        eps: Numerical stability constant
        
    Returns:
        norms: Residual norms (..., 1)
    """
    r = compute_residual(x, x_hat)
    norms = r.norm(dim=-1, keepdim=True)
    return norms


def encode_residual_qjl(
    x: Tensor,
    x_hat: Tensor,
    qjl_projection: 'QJLProjection'  # type: ignore
) -> Tuple[Tensor, Tensor]:
    """
    Encode residual using QJL 1-bit encoding.
    
    Computes r = x - x_hat, then encodes as sign(R @ r).
    
    Args:
        x: Original vectors (..., d)
        x_hat: Reconstructed vectors (..., d)
        qjl_projection: QJLProjection instance
        
    Returns:
        signs: Sign vectors (..., output_dim)
        norms: Residual norms (..., 1)
    """
    r = compute_residual(x, x_hat)
    
    # Get residual norm before quantization
    norms = r.norm(dim=-1, keepdim=True)
    
    # Encode with QJL
    signs = qjl_projection.project_and_quantize(r)
    
    return signs, norms


def decode_residual_correction(
    q: Tensor,
    signs: Tensor,
    norms: Tensor,
    qjl_projection: 'QJLProjection'  # type: ignore
) -> Tensor:
    """
    Decode residual correction term for inner product.
    
    Estimates <q, r> from QJL encoding.
    
    Args:
        q: Query vectors (..., d)
        signs: Encoded residuals (..., output_dim)
        norms: Residual norms (..., 1)
        qjl_projection: QJLProjection instance
        
    Returns:
        correction: Estimated <q, r> (...)
    """
    # Project query
    q_projected = qjl_projection.project(q)
    
    # Estimate inner product
    correction = qjl_projection.estimate_inner_product(
        q_projected, signs, norms
    )
    
    return correction


class ResidualEncoder:
    """
    Stateful residual encoder for TurboQuant Stage 2.
    
    Encapsulates the QJL projection and provides a clean interface
    for encoding/decoding residuals.
    
    Attributes:
        qjl: QJLProjection instance
        input_dim: Input dimension
        output_dim: QJL output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize residual encoder.
        
        Args:
            input_dim: Input vector dimension
            output_dim: QJL output dimension (number of bits)
            seed: Random seed for QJL
            device: Target device
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.qjl = QJLProjection(input_dim, output_dim, seed, device)
    
    def encode(
        self,
        x: Tensor,
        x_hat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode residual between x and its reconstruction.
        
        Args:
            x: Original vectors (..., d)
            x_hat: Reconstructed vectors (..., d)
            
        Returns:
            signs: QJL sign vectors (..., output_dim)
            norms: Residual norms (..., 1)
        """
        return encode_residual_qjl(x, x_hat, self.qjl)
    
    def decode_correction(
        self,
        q: Tensor,
        signs: Tensor,
        norms: Tensor
    ) -> Tensor:
        """
        Decode residual correction for inner product estimation.
        
        Args:
            q: Query vectors (..., d)
            signs: Encoded residuals (..., output_dim)
            norms: Residual norms (..., 1)
            
        Returns:
            correction: Estimated <q, r> (...)
        """
        return decode_residual_correction(q, signs, norms, self.qjl)
    
    def estimate_inner_product(
        self,
        q: Tensor,
        x_hat: Tensor,
        signs: Tensor,
        norms: Tensor
    ) -> Tensor:
        """
        Estimate full inner product <q, x> using reconstruction + correction.
        
        <q, x> = <q, x_hat> + <q, r>
               ≈ <q, x_hat> + correction_from_qjl
        
        Args:
            q: Query vectors (..., d)
            x_hat: Reconstructed key vectors (..., d)
            signs: Encoded residuals for keys (..., output_dim)
            norms: Residual norms for keys (..., 1)
            
        Returns:
            Estimated inner products (...)
        """
        # Stage 1 contribution
        base_dot = (q * x_hat).sum(dim=-1)
        
        # Stage 2 correction
        correction = self.decode_correction(q, signs, norms)
        
        return base_dot + correction
