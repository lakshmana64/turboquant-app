"""
TurboQuant Optimized Core

High-performance implementations with:
  - GPU acceleration (CUDA)
  - Memory-efficient batch operations
  - Vectorized inner product estimation
  - Kernel fusion for reduced overhead
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
import math

from .scalar_quant import _generate_rotation_matrix, get_codebook
from .qjl_projection import QJLProjection
from .codec import TurboQuantConfig
from .bit_packing import pack_bits, unpack_bits, pack_signs, unpack_signs
from .config import load_user_config
from .triton_kernels import run_fused_quantize, HAS_TRITON
from .wht import apply_random_wht


class QJLProjectionOptimized(QJLProjection):
    """
    Optimized QJL projection with GPU support and batch efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(input_dim, output_dim, seed, device)
        self.dtype = dtype
        self.projection_matrix = self.projection_matrix.to(device=device, dtype=dtype)
    
    def project(self, x: Tensor) -> Tensor:
        x = x.to(device=self.projection_matrix.device, dtype=self.dtype)
        orig_shape = x.shape
        x_flat = x.view(-1, self.input_dim)
        projected = x_flat @ self.projection_matrix.T
        return projected.view(*orig_shape[:-1], self.output_dim)
    
    def project_and_quantize_fused(self, r: Tensor) -> Tuple[Tensor, Tensor]:
        r = r.to(device=self.projection_matrix.device, dtype=self.dtype)
        orig_shape = r.shape
        r_flat = r.view(-1, self.input_dim)
        norms = r_flat.norm(dim=1, keepdim=True)
        projected = r_flat @ self.projection_matrix.T
        signs = torch.sign(projected)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return (signs.view(*orig_shape[:-1], self.output_dim), norms.view(*orig_shape[:-1], 1))
    
    def estimate_inner_product_batch_optimized(self, q_projected: Tensor, r_signs: Tensor, r_norms: Tensor) -> Tensor:
        m = self.output_dim
        scale = math.sqrt(math.pi / 2) / m
        dots = q_projected @ r_signs.T
        return scale * dots * r_norms.T


class TurboQuantCodecOptimized:
    """
    High-performance TurboQuant codec with GPU acceleration.
    """
    
    def __init__(
        self,
        dim: int,
        config: Optional[TurboQuantConfig] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.dim = dim
        self.config = config or TurboQuantConfig()
        self.dtype = dtype
        self.user_config = load_user_config()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.use_triton = (self.device.type == 'cuda' and self.user_config.get("use_triton", False) and HAS_TRITON)
        
        self.qjl = QJLProjectionOptimized(dim, self.config.qjl_dim, seed=self.config.seed, device=self.device, dtype=self.dtype)
        
        if self.config.rotation_type == "random":
            self._rotation_matrix = _generate_rotation_matrix(dim, seed=self.config.rotation_seed, device=self.device).to(dtype=self.dtype)
        else:
            self._rotation_matrix = None
        
        centroids, boundaries = get_codebook(self.config.num_bits, device=self.device)
        self._centroids = centroids.to(dtype=self.dtype)
        self._boundaries = boundaries.to(dtype=self.dtype)
    
    def encode_keys_batch_optimized(self, keys: Tensor, return_x_hat: Optional[bool] = None) -> Dict[str, Tensor]:
        if return_x_hat is None:
            return_x_hat = not self.config.pack_bits
        keys = keys.to(device=self.device, dtype=self.dtype)
        
        if self.config.rotation_type == "hadamard":
            x_rotated = apply_random_wht(keys, seed=self.config.rotation_seed)
        else:
            x_rotated = keys @ self._rotation_matrix
        
        norms = x_rotated.norm(dim=1, keepdim=True)
        scales = torch.where(norms < 1e-8, torch.ones_like(norms), norms / math.sqrt(self.dim))
        x_normalized = x_rotated / (scales + 1e-8)
        
        if self.use_triton and self.config.num_bits == 4:
            indices = run_fused_quantize(x_normalized, torch.ones_like(scales))
        else:
            indices = torch.searchsorted(self._boundaries, x_normalized.contiguous()).clamp(0, len(self._centroids) - 1)
        
        x_quantized = self._centroids[indices.to(torch.long)]
        x_scaled = x_quantized * scales
        
        if self.config.rotation_type == "hadamard":
            x_hat = apply_random_wht(x_scaled, seed=self.config.rotation_seed)
        else:
            x_hat = x_scaled @ self._rotation_matrix.T
            
        residual = keys - x_hat
        r_signs, r_norms = self.qjl.project_and_quantize_fused(residual)
        if self.config.pack_bits:
            indices = pack_bits(indices, self.config.num_bits)
            r_signs = pack_signs(r_signs)
            
        result = {'indices': indices, 'scales': scales, 'r_signs': r_signs, 'r_norms': r_norms, 'original_norms': norms}
        if return_x_hat:
            result['x_hat'] = x_hat
        return result
    
    def estimate_inner_products_vectorized(self, queries: Tensor, encoded: Dict[str, Tensor]) -> Tensor:
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
            single_query = True
        else:
            single_query = False
        queries = queries.to(device=self.device, dtype=self.dtype)
        
        x_hat = encoded.get('x_hat')
        r_signs = encoded['r_signs']
        if x_hat is None:
            x_hat = self.decode_keys_vectorized(encoded)
            r_signs = unpack_signs(r_signs, self.config.qjl_dim).to(device=self.device, dtype=self.dtype)
            
        base_dots = queries @ x_hat.T
        q_projected = self.qjl.project(queries)
        correction = self.qjl.estimate_inner_product_batch_optimized(q_projected, r_signs, encoded['r_norms'])
        estimates = base_dots + correction
        return estimates.squeeze(0) if single_query else estimates
    
    def decode_keys_vectorized(self, encoded: Dict[str, Tensor]) -> Tensor:
        indices = encoded['indices']
        scales = encoded['scales']
        if self.config.pack_bits:
            indices = unpack_bits(indices, self.config.num_bits, self.dim).to(self.device)
        x_quantized = self._centroids[indices.to(torch.long)]
        x_scaled = x_quantized * scales
        
        if self.config.rotation_type == "hadamard":
            x_hat = apply_random_wht(x_scaled, seed=self.config.rotation_seed)
        else:
            x_hat = x_scaled @ self._rotation_matrix.T
        return x_hat
    
    def encode_and_estimate(self, keys: Tensor, queries: Tensor, scale: Optional[float] = None) -> Tensor:
        encoded = self.encode_keys_batch_optimized(keys, return_x_hat=True)
        scores = self.estimate_inner_products_vectorized(queries, encoded)
        if scale is not None:
            scores = scores * scale
        return scores
    
    def get_memory_usage(self, num_keys: int) -> Dict[str, Any]:
        original_bytes = num_keys * self.dim * 4
        index_bits = num_keys * self.dim * self.config.num_bits
        sign_bits = num_keys * self.config.qjl_dim
        scale_bits = num_keys * 32
        compressed_bytes = (index_bits + sign_bits + scale_bits) // 8
        return {'original': original_bytes, 'compressed': compressed_bytes, 'ratio': compressed_bytes / original_bytes, 'savings_mb': (original_bytes - compressed_bytes) / 1e6}
    
    @property
    def is_cuda(self) -> bool:
        return self.device.type == 'cuda'
    
    def to(self, device: str) -> 'TurboQuantCodecOptimized':
        self.device = torch.device(device)
        if self._rotation_matrix is not None:
            self._rotation_matrix = self._rotation_matrix.to(self.device)
        self._centroids = self._centroids.to(self.device)
        self._boundaries = self._boundaries.to(self.device)
        self.qjl = QJLProjectionOptimized(self.dim, self.config.qjl_dim, seed=self.config.seed, device=self.device, dtype=self.dtype)
        return self


def create_optimized_codec(dim: int, num_bits: int = 4, qjl_dim: int = 64, device: Optional[str] = None, dtype: torch.dtype = torch.float32) -> TurboQuantCodecOptimized:
    config = TurboQuantConfig(num_bits=num_bits, qjl_dim=qjl_dim)
    return TurboQuantCodecOptimized(dim, config, device, dtype)
