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
from .wht import apply_random_wht


class EncodedKey(Dict[str, Any]):
    """Type hint for encoded key data."""
    pass


def create_codec(
    dim: int,
    num_bits: int = 2,
    qjl_dim: int = 64,
    seed: int = 42,
    device: Optional[torch.device] = None,
    pack_bits: bool = True,
    rotation_type: str = "hadamard"
) -> 'TurboQuantCodec':
    """Factory function for TurboQuantCodec."""
    config = TurboQuantConfig(
        num_bits=num_bits, 
        qjl_dim=qjl_dim, 
        seed=seed, 
        pack_bits=pack_bits,
        rotation_type=rotation_type
    )
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
        pack_bits: bool = True,
        rotation_type: str = "hadamard" # "random" or "hadamard"
    ):
        self.num_bits = num_bits
        self.qjl_dim = qjl_dim
        self.seed = seed
        self.rotation_seed = seed if rotation_seed is None else rotation_seed
        self.dtype = dtype
        self.pack_bits = pack_bits
        self.rotation_type = rotation_type


class TurboQuantCodec:
    """
    Unified Codec for TurboQuant quantization.
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
        if config.rotation_type == "random":
            self.rotation_matrix = _generate_rotation_matrix(dim, config.seed, self.device)
        else:
            self.rotation_matrix = None # WHT is computed on-the-fly, saves memory!
            
        self.estimator = UnbiasedInnerProductEstimator(
            dim, config.qjl_dim, config.seed, self.device
        )

    @property
    def compression_ratio(self) -> float:
        """Return compressed size / original size."""
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
            x, self.config.num_bits, 
            rotation_matrix=self.rotation_matrix,
            rotation_type=self.config.rotation_type,
            rotation_seed=self.config.rotation_seed
        )
        
        # Recon for residual
        x_hat = dequantize_scalar(
            indices, scales, self.config.num_bits, 
            rotation_matrix=self.rotation_matrix,
            rotation_type=self.config.rotation_type,
            rotation_seed=self.config.rotation_seed
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
            'x_hat': x_hat if not self.config.pack_bits else None
        })

    def encode_keys_batch(self, x: Tensor) -> EncodedKey:
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
            rotation_matrix=self.rotation_matrix,
            rotation_type=self.config.rotation_type,
            rotation_seed=self.config.rotation_seed
        )

    def decode_keys(self, encoded: Dict[str, Any]) -> Tensor:
        return self.decode_key(encoded)

    def estimate_inner_products(self, q: Tensor, encoded: Dict[str, Any]) -> Tensor:
        """
        Estimate inner products between query q and batch of encoded keys.
        """
        # 1. Handle packed data and reconstruction
        x_hat = encoded.get('x_hat')
        r_signs = encoded['r_signs']
        
        if x_hat is None:
            # Data was packed, reconstruct Stage 1 and Stage 2
            x_hat = self.decode_key(encoded)
            r_signs = unpack_signs(r_signs, self.config.qjl_dim)
        
        # 2. Estimate
        if q.dim() == 1:
            return self.estimator.estimate_batch(
                q.unsqueeze(0), x_hat, r_signs, encoded['r_norm']
            ).squeeze(0)
        else:
            return self.estimator.estimate_batch(
                q, x_hat, r_signs, encoded['r_norm']
            )

    def compute_attention_scores(self, q: Tensor, encoded: Dict[str, Any], scale: float = 1.0) -> Tensor:
        dots = self.estimate_inner_products(q, encoded)
        return dots * scale

    def get_memory_usage(self, num_keys: int) -> Dict[str, float]:
        """Estimate memory usage in bytes."""
        original = num_keys * self.dim * 2
        if self.config.pack_bits:
            indices_bytes = num_keys * math.ceil(self.dim * self.config.num_bits / 8)
            r_signs_bytes = num_keys * math.ceil(self.config.qjl_dim / 8)
        else:
            indices_bytes = num_keys * self.dim * (1 if self.config.num_bits <= 8 else 2)
            r_signs_bytes = num_keys * self.config.qjl_dim * 4
            
        scales_bytes = num_keys * 4
        r_norm_bytes = num_keys * 4
        x_hat_bytes = num_keys * self.dim * 4 if not self.config.pack_bits else 0
        compressed = indices_bytes + scales_bytes + r_signs_bytes + r_norm_bytes + x_hat_bytes
        
        if original == 0:
            return {'original': 0, 'compressed': 0, 'ratio': 1.0, 'factor': 1.0}
            
        return {'original': original, 'compressed': compressed, 'ratio': compressed / original, 'factor': original / compressed}
