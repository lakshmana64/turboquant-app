"""
TurboQuant Codec - Unified Interface for Two-Stage Quantization

The Codec coordinates Stage 1 (Scalar Quantization) and Stage 2 (QJL)
to provide a seamless experience for compressing keys and estimating
inner products in transformer KV caches.
"""

import torch
import math
from torch import Tensor
from typing import Dict, Any, Optional

from .scalar_quant import quantize_scalar, dequantize_scalar, _generate_rotation_matrix
from .estimator import UnbiasedInnerProductEstimator
from .bit_packing import pack_bits, unpack_bits, pack_signs, unpack_signs


class EncodedKey(Dict[str, Any]):
    """Type hint for encoded key data."""
    pass


def create_codec(
    dim: int,
    num_bits: int = 2,
    qjl_dim: int = 64,
    seed: int = 42,
    device: Optional[torch.device] = None,
    pack_bits: bool = True
) -> 'TurboQuantCodec':
    """Factory function for TurboQuantCodec."""
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim, seed=seed, pack_bits=pack_bits)
    return TurboQuantCodec(dim, config=config, device=device)


class TurboQuantConfig:
    """Configuration for TurboQuant Codec."""
    def __init__(
        self,
        num_bits: int = 2,
        qjl_dim: int = 64,
        seed: int = 42,
        rotation_seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        pack_bits: bool = True
    ):
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.seed = seed
        self.rotation_seed = seed if rotation_seed is None else rotation_seed
        self.dtype = dtype
        self.pack_bits = pack_bits


class TurboQuantCodec:
    """
    Unified Codec for TurboQuant quantization.
    
    Handles both Stage 1 (SQ) and Stage 2 (QJL) and provides 
    utilities for attention mechanism integration.
    """
    
    def __init__(
        self,
        dim: int,
        config: TurboQuantConfig = TurboQuantConfig(),
        device: Optional[torch.device] = None
    ):
        self.dim = dim
        self.config = config
        self.device = device if device else torch.device('cpu')
        
        # Initialize components
        self.rotation_matrix = _generate_rotation_matrix(dim, config.seed, self.device)
        self.estimator = UnbiasedInnerProductEstimator(
            dim, config.qjl_dim, config.seed, self.device
        )

    @property
    def compression_ratio(self) -> float:
        """
        Return compressed size / original size.

        A smaller value is better. For example, ``0.25`` means the
        representation uses 25% of the original FP16 storage.
        """
        compressed_bits = self.config.num_bits * self.dim + self.config.qjl_dim
        original_bits = 16 * self.dim
        return compressed_bits / original_bits

    @property
    def compression_factor(self) -> float:
        """Return original size / compressed size for x-style reporting."""
        return 1.0 / self.compression_ratio

    def encode_key(self, x: Tensor) -> EncodedKey:
        """Encode a single key or a batch of keys."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Stage 1
        indices, scales, norms, _ = quantize_scalar(
            x, self.config.num_bits, rotation_matrix=self.rotation_matrix
        )
        
        # Recon for residual
        x_hat = dequantize_scalar(
            indices, scales, self.config.num_bits, rotation_matrix=self.rotation_matrix
        )
        
        # Stage 2
        _, r_signs, r_norm = self.estimator.encode_key(x, x_hat)
        
        # Bit-packing
        if self.config.pack_bits:
            indices = pack_bits(indices, self.config.num_bits)
            r_signs = pack_signs(r_signs)
            
        return EncodedKey({
            'indices': indices,
            'scales': scales,
            'r_signs': r_signs,
            'r_norm': r_norm,
            # We don't store x_hat in the packed format to save memory
            # It will be reconstructed during estimation if needed
            'x_hat': x_hat if not self.config.pack_bits else None
        })

    def encode_keys_batch(self, x: Tensor) -> EncodedKey:
        """Alias for encode_key."""
        return self.encode_key(x)

    def decode_key(self, encoded: Dict[str, Any]) -> Tensor:
        """Reconstruct the Stage 1 version (x_hat)."""
        indices = encoded['indices']
        if self.config.pack_bits:
            indices = unpack_bits(indices, self.config.num_bits, self.dim)
            
        return dequantize_scalar(
            indices, 
            encoded['scales'], 
            self.config.num_bits, 
            rotation_matrix=self.rotation_matrix
        )

    def decode_keys(self, encoded: Dict[str, Any]) -> Tensor:
        """Alias for decode_key."""
        return self.decode_key(encoded)

    def estimate_inner_products(self, q: Tensor, encoded: Dict[str, Any]) -> Tensor:
        """
        Estimate inner products between query q and batch of encoded keys.
        
        Args:
            q: Query vector (dim,) or (n_q, dim)
            encoded: Dictionary from encode_key
            
        Returns:
            Inner product estimates (n_keys,) or (n_q, n_keys)
        """
        # Handle packed data if needed
        x_hat = encoded.get('x_hat')
        r_signs = encoded['r_signs']
        
        if x_hat is None:
            # Data was packed, reconstruct Stage 1 and Stage 2
            x_hat = self.decode_key(encoded)
            r_signs = unpack_signs(r_signs, self.config.qjl_dim)
            
        if q.dim() == 1:
            # Single query vs batch of keys
            return self.estimator.estimate_batch(
                q.unsqueeze(0),
                x_hat,
                r_signs,
                encoded['r_norm']
            ).squeeze(0)
        else:
            # Batch queries vs batch keys
            return self.estimator.estimate_batch(
                q,
                x_hat,
                r_signs,
                encoded['r_norm']
            )

    def compute_attention_scores(
        self, 
        q: Tensor, 
        encoded: Dict[str, Any], 
        scale: float = 1.0
    ) -> Tensor:
        """
        Compute attention scores: softmax(estimate_inner_products * scale).
        """
        dots = self.estimate_inner_products(q, encoded)
        return dots * scale

    def get_memory_usage(self, num_keys: int) -> Dict[str, float]:
        """Estimate memory usage in bytes."""
        # Baseline (FP16)
        original = num_keys * self.dim * 2
        
        # Compressed
        if self.config.pack_bits:
            # indices: bits / 8 bytes per dimension
            indices_bytes = num_keys * math.ceil(self.dim * self.config.num_bits / 8)
            # r_signs: 1 bit per qjl_dim
            r_signs_bytes = num_keys * math.ceil(self.config.qjl_dim / 8)
        else:
            # indices: uint8 (if num_bits <= 8)
            indices_bytes = num_keys * self.dim * (1 if self.config.num_bits <= 8 else 2)
            # r_signs: float32 (before packing)
            r_signs_bytes = num_keys * self.config.qjl_dim * 4
            
        # scales: float32
        scales_bytes = num_keys * 4
        # r_norm: float32
        r_norm_bytes = num_keys * 4
        
        # x_hat overhead (if not packed)
        x_hat_bytes = num_keys * self.dim * 4 if not self.config.pack_bits else 0
        
        compressed = indices_bytes + scales_bytes + r_signs_bytes + r_norm_bytes + x_hat_bytes
        
        return {
            'original': original,
            'compressed': compressed,
            'ratio': compressed / original,
            'factor': original / compressed
        }
