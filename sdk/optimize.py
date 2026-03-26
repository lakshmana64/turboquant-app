"""
TurboQuant SDK - High-level API for Unbiased Quantization

This module provides a simple, unified interface for the two-stage TurboQuant
quantization scheme.
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, Tuple

from turboquant.core.scalar_quant import (
    _generate_rotation_matrix,
    dequantize_scalar,
    quantize_scalar,
)
from turboquant.core.estimator import UnbiasedInnerProductEstimator


class TurboQuantizer:
    """
    High-level API for TurboQuant quantization and estimation.
    
    Manages both Stage 1 (Scalar Quantization) and Stage 2 (QJL Residuals).
    """
    
    def __init__(
        self,
        input_dim: int,
        qjl_bits: int = 64,
        sq_bits: int = 2,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the quantizer.
        
        Args:
            input_dim: Dimension of the vectors (d)
            qjl_bits: Number of bits for residual encoding (m)
            sq_bits: Bits per coordinate for scalar quantization
            seed: Random seed for reproducibility
            device: Target device (CPU/CUDA)
        """
        self.input_dim = input_dim
        self.qjl_bits = qjl_bits
        self.sq_bits = sq_bits
        self.device = device
        
        # Initialize core components
        self.rotation_matrix = _generate_rotation_matrix(input_dim, seed, device)
        self.estimator = UnbiasedInnerProductEstimator(input_dim, qjl_bits, seed, device)

    @property
    def compression_ratio(self) -> float:
        """
        Return compressed size / original size.

        A smaller value is better. For example, ``0.25`` means the
        representation uses 25% of the original FP16 storage.
        """
        compressed_bits = self.sq_bits * self.input_dim + self.qjl_bits
        original_bits = 16 * self.input_dim
        return compressed_bits / original_bits

    @property
    def compression_factor(self) -> float:
        """Return original size / compressed size for x-style reporting."""
        return 1.0 / self.compression_ratio

    def encode(self, x: Tensor) -> Dict[str, Any]:
        """
        Encode (quantize) a tensor using the two-stage scheme.
        
        Args:
            x: Input tensor of shape (..., d)
            
        Returns:
            Dictionary containing quantized components:
                - indices: Stage 1 indices
                - scales: Stage 1 scale factors
                - r_signs: Stage 2 residual signs
                - r_norm: Stage 2 residual norms
                - x_hat: Reconstructed Stage 1 vector (cached for speed)
        """
        # Stage 1: Scalar Quantization with Random Rotation
        indices, scales, _, _ = quantize_scalar(
            x, self.sq_bits, rotation_matrix=self.rotation_matrix
        )
        
        # Reconstruction for residual computation
        x_hat = dequantize_scalar(
            indices, scales, self.sq_bits, rotation_matrix=self.rotation_matrix
        )
        
        # Stage 2: QJL Residual Encoding
        _, r_signs, r_norm = self.estimator.encode_key(x, x_hat)
        
        return {
            'indices': indices,
            'scales': scales,
            'r_signs': r_signs,
            'r_norm': r_norm,
            'x_hat': x_hat
        }

    def estimate(self, q: Tensor, encoded: Dict[str, Any]) -> Tensor:
        """
        Estimate the inner product <q, x> using encoded components.
        
        Args:
            q: Query tensor (..., d)
            encoded: Output from self.encode()
            
        Returns:
            Estimated inner product
        """
        return self.estimator.estimate(
            q, encoded['x_hat'], encoded['r_signs'], encoded['r_norm']
        )

    def estimate_batch(self, queries: Tensor, encoded_keys: Dict[str, Any]) -> Tensor:
        """
        Estimate batch of inner products (queries x keys).
        
        Args:
            queries: Queries (n_q, d)
            encoded_keys: Encoded keys (n_k, d)
            
        Returns:
            Estimated inner product matrix (n_q, n_k)
        """
        return self.estimator.estimate_batch(
            queries, 
            encoded_keys['x_hat'], 
            encoded_keys['r_signs'], 
            encoded_keys['r_norm']
        )


def optimize(
    x: Tensor, 
    qjl_bits: int = 64, 
    sq_bits: int = 2, 
    seed: int = 42
) -> Tuple[Dict[str, Any], TurboQuantizer]:
    """
    One-line API to quantize a tensor and return the quantizer instance.
    
    Args:
        x: Tensor to quantize
        qjl_bits: Number of QJL bits
        sq_bits: Bits per coordinate
        seed: Random seed
        
    Returns:
        tuple of (encoded_data, quantizer_instance)
    """
    d = x.shape[-1]
    quantizer = TurboQuantizer(d, qjl_bits, sq_bits, seed, x.device)
    encoded = quantizer.encode(x)
    return encoded, quantizer
