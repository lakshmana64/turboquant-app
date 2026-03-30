"""
TurboQuant SDK - High-level API for Unbiased Quantization

This module provides a simple, unified interface for the two-stage TurboQuant
quantization scheme, leveraging the optimized codec under the hood.
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, Tuple

from turboquant.core.codec import TurboQuantCodec, TurboQuantConfig


class TurboQuantizer:
    """
    High-level API for TurboQuant quantization and estimation.
    
    Wraps TurboQuantCodec to provide a simplified interface for 
    encoding and inner product estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        qjl_bits: int = 64,
        sq_bits: int = 2,
        seed: int = 42,
        device: Optional[torch.device] = None,
        pack_bits: bool = True
    ):
        """
        Initialize the quantizer.
        
        Args:
            input_dim: Dimension of the vectors (d)
            qjl_bits: Number of bits for residual encoding (m)
            sq_bits: Bits per coordinate for scalar quantization
            seed: Random seed for reproducibility
            device: Target device (CPU/CUDA)
            pack_bits: Whether to enable bit-packing for memory efficiency
        """
        self.input_dim = input_dim
        self.config = TurboQuantConfig(
            num_bits=sq_bits,
            qjl_dim=qjl_bits,
            seed=seed,
            pack_bits=pack_bits
        )
        self.codec = TurboQuantCodec(input_dim, config=self.config, device=device)

    @property
    def compression_ratio(self) -> float:
        """Return compressed size / original size (vs FP16)."""
        return self.codec.compression_ratio

    @property
    def compression_factor(self) -> float:
        """Return original size / compressed size (vs FP16)."""
        return self.codec.compression_factor

    def encode(self, x: Tensor) -> Dict[str, Any]:
        """
        Encode (quantize) a tensor using the two-stage scheme.
        
        Args:
            x: Input tensor of shape (..., d)
            
        Returns:
            Dictionary containing quantized and (optionally) packed components.
        """
        return self.codec.encode_keys_batch(x)

    def estimate(self, q: Tensor, encoded: Dict[str, Any]) -> Tensor:
        """
        Estimate the inner product <q, x> using encoded components.
        
        Args:
            q: Query tensor (..., d)
            encoded: Output from self.encode()
            
        Returns:
            Estimated inner product
        """
        return self.codec.estimate_inner_products(q, encoded)

    def estimate_batch(self, queries: Tensor, encoded_keys: Dict[str, Any]) -> Tensor:
        """
        Estimate batch of inner products (queries x keys).
        
        Args:
            queries: Queries (n_q, d)
            encoded_keys: Encoded keys (n_k, d)
            
        Returns:
            Estimated inner product matrix (n_q, n_k)
        """
        return self.codec.estimate_inner_products(queries, encoded_keys)


def optimize(
    x: Tensor, 
    qjl_bits: int = 64, 
    sq_bits: int = 2, 
    seed: int = 42,
    pack_bits: bool = True
) -> Tuple[Dict[str, Any], TurboQuantizer]:
    """
    One-line API to quantize a tensor and return the quantizer instance.
    
    Args:
        x: Tensor to quantize
        qjl_bits: Number of QJL bits
        sq_bits: Bits per coordinate
        seed: Random seed
        pack_bits: Whether to enable bit-packing
        
    Returns:
        tuple of (encoded_data, quantizer_instance)
    """
    d = x.shape[-1]
    quantizer = TurboQuantizer(d, qjl_bits, sq_bits, seed, x.device, pack_bits=pack_bits)
    encoded = quantizer.encode(x)
    return encoded, quantizer
