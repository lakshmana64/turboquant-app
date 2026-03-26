from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple, Union, List
import math

from .codec import TurboQuantConfig, TurboQuantCodec

"""
TurboQuant Mixed Precision

Support for FP8, INT8, and other low-precision quantization formats.

Usage:
    from turboquant.core.mixed_precision import MixedPrecisionQuantizer
    
    # FP8 quantization
    quantizer = MixedPrecisionQuantizer(dtype='fp8')
    quantized = quantizer.quantize(tensor)
    
    # INT8 quantization
    quantizer = MixedPrecisionQuantizer(dtype='int8')
    quantized = quantizer.quantize(tensor)
    
    # Mixed precision (FP8 for keys, FP16 for queries)
    from turboquant.core.mixed_precision import MixedPrecisionCodec
    
    codec = MixedPrecisionCodec(
        key_dtype='fp8',
        query_dtype='fp16'
    )
"""


# FP8 data types (available in PyTorch 2.1+)
try:
    float8_e4m3fn = torch.float8_e4m3fn
    float8_e5m2 = torch.float8_e5m2
    FP8_AVAILABLE = True
except AttributeError:
    FP8_AVAILABLE = False
    float8_e4m3fn = None
    float8_e5m2 = None


class MixedPrecisionQuantizer:
    """
    Mixed precision quantizer supporting multiple data types.
    
    Supported dtypes:
      - fp32: Full precision (baseline)
      - fp16: Half precision
      - bf16: BFloat16
      - fp8: FP8 (e4m3fn) - requires PyTorch 2.1+
      - int8: 8-bit integer
      - int4: 4-bit integer (simulated)
    """
    
    def __init__(
        self,
        dtype: str = 'fp16',
        scale_per_tensor: bool = True,
        clip_range: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize quantizer.
        
        Args:
            dtype: Target data type
            scale_per_tensor: Use per-tensor scaling
            clip_range: Optional clipping range (min, max)
        """
        self.dtype_str = dtype
        self.scale_per_tensor = scale_per_tensor
        self.clip_range = clip_range
        
        # Map string to torch dtype
        self.dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp8': float8_e4m3fn if FP8_AVAILABLE else torch.float16,
            'int8': torch.int8,
            'int4': torch.int8,  # Simulated int4
        }
        
        self.dtype = self.dtype_map.get(dtype, torch.float16)
        self.is_integer = dtype.startswith('int')
    
    def quantize(
        self,
        x: Tensor,
        return_scale: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Quantize tensor to target dtype.
        
        Args:
            x: Input tensor
            return_scale: Return scale factor
            
        Returns:
            Quantized tensor, optionally with scale
        """
        # Clip if specified
        if self.clip_range:
            x = x.clamp(self.clip_range[0], self.clip_range[1])
        
        if self.is_integer:
            # Integer quantization with scaling
            if self.scale_per_tensor:
                scale = x.abs().max() / 127.0  # For int8
                x_scaled = x / scale
            else:
                scale = x.abs().max(dim=-1, keepdim=True)[0] / 127.0
                x_scaled = x / scale
            
            # Quantize
            if self.dtype_str == 'int4':
                # Simulated int4 (pack into int8)
                x_int = x_scaled.clamp(-8, 7).round().to(torch.int8)
            else:
                x_int = x_scaled.clamp(-128, 127).round().to(self.dtype)
            
            if return_scale:
                return x_int, scale
            return x_int
        
        else:
            # Floating point quantization
            x_quant = x.to(self.dtype)
            
            if return_scale:
                # Scale for FP8
                if self.dtype_str == 'fp8' and FP8_AVAILABLE:
                    scale = torch.tensor(1.0, device=x.device)
                else:
                    scale = torch.tensor(1.0, device=x.device)
                return x_quant, scale
            
            return x_quant
    
    def dequantize(
        self,
        x_quant: Tensor,
        scale: Optional[Tensor] = None
    ) -> Tensor:
        """
        Dequantize tensor to FP32.
        
        Args:
            x_quant: Quantized tensor
            scale: Scale factor from quantization
            
        Returns:
            Dequantized tensor (FP32)
        """
        # Convert to FP32
        x = x_quant.to(torch.float32)
        
        # Apply scale
        if scale is not None:
            x = x * scale
        
        return x


class MixedPrecisionCodec:
    """
    TurboQuant codec with mixed precision support.
    
    Allows different precisions for:
      - Keys (stored in cache)
      - Queries (computed on-the-fly)
      - Values (optional)
    
    Use cases:
      - FP8 keys + FP16 queries: Maximum compression
      - INT8 keys + FP16 queries: Integer arithmetic
      - FP16 keys + FP32 queries: Balanced approach
    """
    
    def __init__(
        self,
        dim: int,
        key_dtype: str = 'fp8',
        query_dtype: str = 'fp16',
        value_dtype: str = 'fp16',
        config: Optional[TurboQuantConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize mixed precision codec.
        
        Args:
            dim: Input dimension
            key_dtype: Data type for keys
            query_dtype: Data type for queries
            value_dtype: Data type for values
            config: TurboQuant configuration
            device: Target device
        """
        self.dim = dim
        self.key_dtype = key_dtype
        self.query_dtype = query_dtype
        self.value_dtype = value_dtype
        self.config = config or TurboQuantConfig()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create base TurboQuant codec
        self.base_codec = TurboQuantCodec(dim, config, self.device)
        
        # Create quantizers
        self.key_quantizer = MixedPrecisionQuantizer(key_dtype)
        self.query_quantizer = MixedPrecisionQuantizer(query_dtype)
        self.value_quantizer = MixedPrecisionQuantizer(value_dtype)
        
        # Statistics
        self._compression_stats = {
            'keys_original_mb': 0,
            'keys_compressed_mb': 0,
        }
    
    def encode_keys_mixed(
        self,
        keys: Tensor
    ) -> Dict[str, Any]:
        """
        Encode keys with mixed precision.
        
        Stage 1: TurboQuant scalar quantization
        Stage 2: QJL residual encoding
        Final: Convert to target dtype
        
        Args:
            keys: Key tensor (n, d)
            
        Returns:
            Encoded data dict
        """
        # Base TurboQuant encoding
        encoded = self.base_codec.encode_keys_batch(keys)
        
        # Additional quantization for keys
        if self.key_dtype in ['fp8', 'int8', 'int4']:
            # Quantize indices (already integers, just change dtype if needed)
            if self.key_dtype == 'int8':
                encoded['indices'] = encoded['indices'].to(torch.int8)
            
            # Quantize scales
            encoded['scales'], scale_scale = self.key_quantizer.quantize(
                encoded['scales'].squeeze(-1)
            )
            encoded['scale_scale'] = scale_scale
            
            # Quantize residual norms
            encoded['r_norms'], norm_scale = self.key_quantizer.quantize(
                encoded['r_norms'].squeeze(-1)
            )
            encoded['norm_scale'] = norm_scale
        
        # Track compression
        original_bytes = keys.numel() * 4  # FP32
        compressed_bytes = sum(
            v.numel() * v.element_size()
            for v in encoded.values()
            if isinstance(v, Tensor)
        )
        
        self._compression_stats['keys_original_mb'] = original_bytes / 1e6
        self._compression_stats['keys_compressed_mb'] = compressed_bytes / 1e6
        
        return encoded
    
    def decode_keys_mixed(
        self,
        encoded: Dict[str, Any]
    ) -> Tensor:
        """
        Decode keys from mixed precision.
        
        Args:
            encoded: Encoded data from encode_keys_mixed
            
        Returns:
            Decoded keys (n, d)
        """
        # Dequantize if needed
        if 'scale_scale' in encoded:
            encoded['scales'] = self.key_quantizer.dequantize(
                encoded['scales'],
                encoded.get('scale_scale')
            ).unsqueeze(-1)
            
            encoded['r_norms'] = self.key_quantizer.dequantize(
                encoded['r_norms'],
                encoded.get('norm_scale')
            ).unsqueeze(-1)
        
        # Base decoding
        return self.base_codec.decode_keys(encoded)
    
    def estimate_inner_products_mixed(
        self,
        queries: Tensor,
        encoded: Dict[str, Any]
    ) -> Tensor:
        """
        Estimate inner products with mixed precision.
        
        Args:
            queries: Query tensor (n_q, d)
            encoded: Encoded keys
            
        Returns:
            Inner products (n_q, n_k)
        """
        # Convert queries to target dtype
        queries = self.query_quantizer.quantize(queries, return_scale=False)
        
        # Dequantize encoded data if needed
        if 'scale_scale' in encoded:
            encoded['scales'] = self.key_quantizer.dequantize(
                encoded['scales'],
                encoded.get('scale_scale')
            ).unsqueeze(-1)
            
            encoded['r_norms'] = self.key_quantizer.dequantize(
                encoded['r_norms'],
                encoded.get('norm_scale')
            ).unsqueeze(-1)
        
        # Base estimation
        return self.base_codec.estimate_inner_products(queries, encoded)
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio for keys."""
        if self._compression_stats['keys_original_mb'] == 0:
            return 1.0
        
        return (
            self._compression_stats['keys_compressed_mb'] /
            self._compression_stats['keys_original_mb']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get codec statistics."""
        return {
            'dim': self.dim,
            'key_dtype': self.key_dtype,
            'query_dtype': self.query_dtype,
            'value_dtype': self.value_dtype,
            'compression_ratio': self.get_compression_ratio(),
            'fp8_available': FP8_AVAILABLE,
        }


class LowPrecisionAttention:
    """
    Low-precision attention mechanism using mixed precision codec.
    
    Optimized for:
      - KV cache compression
      - Fast attention score computation
      - Memory efficiency
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        key_dtype: str = 'fp8',
        query_dtype: str = 'fp16',
        config: Optional[TurboQuantConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize low-precision attention.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            key_dtype: Data type for KV cache
            query_dtype: Data type for queries
            config: TurboQuant configuration
            device: Target device
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create per-head codecs
        self.codecs = [
            MixedPrecisionCodec(
                head_dim,
                key_dtype,
                query_dtype,
                config=config,
                device=self.device
            )
            for _ in range(num_heads)
        ]
        
        # KV cache storage
        self.kv_cache: List[Optional[Dict[str, Any]]] = [None] * num_heads
    
    def append_kv(
        self,
        keys: Tensor,
        values: Tensor,
    ):
        """
        Append keys and values to cache.
        
        Args:
            keys: Key tensor (batch, num_heads, seq_len, head_dim)
            values: Value tensor (batch, num_heads, seq_len, head_dim)
        """
        batch, num_heads, seq_len, head_dim = keys.shape
        
        for head_idx in range(num_heads):
            head_keys = keys[:, head_idx, :, :].view(-1, head_dim)
            head_values = values[:, head_idx, :, :].view(-1, head_dim)
            
            # Encode keys
            encoded = self.codecs[head_idx].encode_keys_mixed(head_keys)
            
            # Store values (compressed)
            encoded['values'] = head_values
            
            # Append to cache
            if self.kv_cache[head_idx] is None:
                self.kv_cache[head_idx] = encoded
            else:
                for key in encoded:
                    if key in self.kv_cache[head_idx]:
                        self.kv_cache[head_idx][key] = torch.cat(
                            [self.kv_cache[head_idx][key], encoded[key]],
                            dim=0
                        )
    
    def compute_attention(
        self,
        queries: Tensor,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Compute attention with compressed KV cache.
        
        Args:
            queries: Query tensor (batch, num_heads, seq_len, head_dim)
            scale: Scaling factor
            
        Returns:
            Attention output (batch, num_heads, seq_len, head_dim)
        """
        batch, num_heads, seq_len, head_dim = queries.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        outputs = []
        
        for head_idx in range(num_heads):
            head_queries = queries[:, head_idx, :, :].view(-1, head_dim)
            
            if self.kv_cache[head_idx] is None:
                continue
            
            # Get attention scores
            scores = self.codecs[head_idx].estimate_inner_products_mixed(
                head_queries,
                self.kv_cache[head_idx]
            ) * scale
            
            # Attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply to values
            values = self.kv_cache[head_idx]['values']
            output = attn_weights @ values
            
            outputs.append(output.view(batch, seq_len, head_dim))
        
        return torch.stack(outputs, dim=1)  # (batch, num_heads, seq_len, head_dim)
    
    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache = [None] * self.num_heads
    
    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size in bytes."""
        total_bytes = 0
        
        for head_cache in self.kv_cache:
            if head_cache is not None:
                for v in head_cache.values():
                    if isinstance(v, Tensor):
                        total_bytes += v.numel() * v.element_size()
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / 1e6,
        }
