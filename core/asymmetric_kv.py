"""
Asymmetric K/V Cache Support for TurboQuant.

Enables independent cache types for Keys and Values, allowing:
- Different quantization formats for K vs V
- Higher precision for K (controls attention routing)
- More aggressive compression for V (less quality-sensitive)

Use Cases:
- Rescue quality on low-bit models (e.g., Q4_K_M)
- K precision is dominant for quality
- V can be compressed more aggressively

Example:
    # Qwen2.5-7B Q4_K_M validated with q8_0 K + turbo4 V
    cache = AsymmetricKVCache(
        dim=4096,
        k_format="q8_0",  # High precision for K
        v_format="turbo4"  # Aggressive compression for V
    )
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List
from .turbo_formats import get_format, create_codec_from_format, TurboFormat
from .value_quant import TurboValueCodec
from .sparse_v import SparseVDecoder


class AsymmetricKVConfig:
    """Configuration for asymmetric K/V cache."""
    
    def __init__(
        self,
        dim: int,
        k_format: str = "q8_0",
        v_format: str = "turbo4",
        enable_sparse_v: bool = True,
        sparse_threshold: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize asymmetric KV config.
        
        Args:
            dim: Hidden dimension
            k_format: Format for Keys (e.g., "q8_0", "turbo4", "turbo3")
            v_format: Format for Values (e.g., "turbo4", "turbo2")
            enable_sparse_v: Enable sparse V decoding
            sparse_threshold: Threshold for sparse V skipping
            seed: Random seed
        """
        self.dim = dim
        self.k_format = k_format
        self.v_format = v_format
        self.enable_sparse_v = enable_sparse_v
        self.sparse_threshold = sparse_threshold
        self.seed = seed
        
        # Validate formats
        k_fmt = get_format(k_format)
        v_fmt = get_format(v_format)
        
        self.k_format_obj = k_fmt
        self.v_format_obj = v_fmt
    
    def __repr__(self) -> str:
        return (
            f"AsymmetricKVConfig(\n"
            f"  dim={self.dim},\n"
            f"  k_format={self.k_format} ({self.k_format_obj.compression_factor}x),\n"
            f"  v_format={self.v_format} ({self.v_format_obj.compression_factor}x),\n"
            f"  sparse_v={self.enable_sparse_v}\n"
            f")"
        )


class AsymmetricKVCache:
    """
    Asymmetric KV Cache with independent K/V formats.
    
    Provides optimal quality by using high-precision K for attention
    routing and compressed V for memory efficiency.
    """
    
    def __init__(self, config: AsymmetricKVConfig, device: Optional[torch.device] = None):
        """
        Initialize asymmetric KV cache.
        
        Args:
            config: AsymmetricKVConfig instance
            device: Torch device
        """
        self.config = config
        self.device = device or torch.device('cpu')
        self.dim = config.dim
        
        # Initialize K codec
        self.k_codec = create_codec_from_format(
            config.k_format,
            config.dim,
            device=self.device,
            seed=config.seed
        )
        
        # Initialize V codec
        v_fmt = get_format(config.v_format)
        self.v_codec = TurboValueCodec(
            config.dim,
            num_bits=v_fmt.sq_bits,
            device=self.device
        )
        
        # Initialize sparse V decoder if enabled
        if config.enable_sparse_v:
            self.v_decoder = SparseVDecoder(
                config.dim,
                num_bits=v_fmt.sq_bits,
                threshold=config.sparse_threshold,
                device=self.device
            )
        else:
            self.v_decoder = None
        
        # Storage
        self.encoded_keys: List[Dict[str, Any]] = []
        self.encoded_values: List[Dict[str, Any]] = []
        self.seq_len = 0
        
        # Statistics
        self.stats = {
            "k_compression": config.k_format_obj.compression_factor,
            "v_compression": config.v_format_obj.compression_factor,
            "total_appends": 0
        }
    
    def append(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Append new K, V to the cache.
        
        Args:
            k: Key tensor [seq, dim] or [batch, seq, dim] or [batch, heads, seq, dim]
            v: Value tensor matching k shape
            layer_idx: Optional layer index for tracking
        
        Returns:
            Metadata about the append operation
        """
        # Normalize input shape to [total_tokens, dim]
        orig_shape = k.shape
        if k.dim() == 4:
            # [batch, heads, seq, dim]
            k = k.transpose(1, 2).reshape(-1, self.dim)
            v = v.transpose(1, 2).reshape(-1, self.dim)
        elif k.dim() == 3:
            # [batch, seq, dim]
            k = k.reshape(-1, self.dim)
            v = v.reshape(-1, self.dim)
        elif k.dim() == 1:
            # [dim]
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        
        # Encode K
        encoded_k = self.k_codec.encode_key(k)
        
        # Encode V
        encoded_v = self.v_codec.encode(v)
        
        # Store
        self.encoded_keys.append(encoded_k)
        self.encoded_values.append(encoded_v)
        self.seq_len += k.shape[0]
        self.stats["total_appends"] += 1
        
        return {
            "seq_added": k.shape[0],
            "total_seq": self.seq_len,
            "layer_idx": layer_idx
        }
    
    def get_attention_output(
        self,
        q: Tensor,
        scale: float = 1.0,
        use_sparse_v: Optional[bool] = None
    ) -> Tensor:
        """
        Compute attention output with asymmetric K/V.
        
        Args:
            q: Query tensor [batch, dim] or [batch, heads, dim]
            scale: Attention scale factor
            use_sparse_v: Override sparse V setting
        
        Returns:
            Attention output [batch, dim]
        """
        if len(self.encoded_keys) == 0:
            return torch.zeros_like(q)
        
        use_sparse = use_sparse_v if use_sparse_v is not None else self.config.enable_sparse_v
        batch_size = q.shape[0]
        
        # Compute attention scores from K
        all_scores = []
        for encoded_k in self.encoded_keys:
            scores = self.k_codec.compute_attention_scores(q, encoded_k, scale=scale)
            all_scores.append(scores)
        
        # Concatenate scores [batch, total_seq]
        all_scores = torch.cat(all_scores, dim=-1)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(all_scores, dim=-1)
        
        # Decode V with or without sparsity
        if use_sparse and self.v_decoder is not None:
            v_output = self._decode_v_sparse(attn_weights)
        else:
            v_output = self._decode_v_dense(batch_size)
        
        # Apply attention weights to V
        # attn_weights: [batch, seq_len]
        # v_output: [batch, seq_len, dim]
        output = (attn_weights.unsqueeze(-1) * v_output).sum(dim=1)
        
        return output
    
    def _decode_v_sparse(self, attn_weights: Tensor) -> Tensor:
        """Decode V with sparse optimization."""
        combined_v = self._combine_encoded_values()
        v_output = self.v_decoder.decode_sparse_optimized(combined_v, attn_weights)
        return v_output
    
    def _decode_v_dense(self, batch_size: int) -> Tensor:
        """Decode all V positions densely."""
        # Decode all V values
        all_v = []
        for encoded_v in self.encoded_values:
            v_decoded = self.v_codec.decode(encoded_v)
            all_v.append(v_decoded)
        
        # Concatenate [seq, dim] -> [1, seq, dim]
        v_tensor = torch.cat(all_v, dim=0).unsqueeze(0)
        
        # Expand for batch
        return v_tensor.expand(batch_size, -1, -1)
    
    def _combine_encoded_values(self) -> Dict[str, Any]:
        """Combine all encoded V batches."""
        if len(self.encoded_values) == 0:
            return {}
        
        return {
            'indices': torch.cat([ev['indices'] for ev in self.encoded_values], dim=0),
            'scales': torch.cat([ev['scales'] for ev in self.encoded_values], dim=0),
            'bias': torch.cat([
                ev.get('bias', torch.zeros(1, 1, device=self.device))
                for ev in self.encoded_values
            ], dim=0),
            'dim': self.dim,
            'batch_size': 1
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Calculate memory usage for K and V separately."""
        k_usage = self.k_codec.get_memory_usage(self.seq_len)
        
        # V usage estimation
        v_fmt = get_format(self.config.v_format)
        v_bits_per_key = v_fmt.sq_bits * self.dim + 32  # indices + scale
        v_bytes = self.seq_len * v_bits_per_key // 8
        
        total_compressed = k_usage['compressed'] + v_bytes
        total_original = k_usage['original'] * 2  # K + V original
        
        return {
            "k_memory": k_usage,
            "v_memory_bytes": v_bytes,
            "v_compression_factor": f"{v_fmt.compression_factor}x",
            "total_compressed_bytes": total_compressed,
            "total_original_bytes": total_original,
            "overall_compression_factor": f"{total_original / total_compressed:.1f}x"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "config": str(self.config),
            "seq_len": self.seq_len,
            "num_blocks": len(self.encoded_keys),
            "memory": self.get_memory_usage()
        }
        
        if self.v_decoder is not None:
            stats["sparse_v"] = self.v_decoder.get_sparsity_stats()
        
        return stats
    
    def clear(self):
        """Clear the cache."""
        self.encoded_keys = []
        self.encoded_values = []
        self.seq_len = 0
        if self.v_decoder is not None:
            self.v_decoder.reset_stats()


def create_asymmetric_cache(
    dim: int,
    k_format: str = "q8_0",
    v_format: str = "turbo4",
    enable_sparse_v: bool = True,
    device: Optional[torch.device] = None
) -> AsymmetricKVCache:
    """
    Factory function for asymmetric KV cache.
    
    Args:
        dim: Hidden dimension
        k_format: Format for Keys
        v_format: Format for Values
        enable_sparse_v: Enable sparse V decoding
        device: Torch device
    
    Returns:
        Configured AsymmetricKVCache
    """
    config = AsymmetricKVConfig(
        dim=dim,
        k_format=k_format,
        v_format=v_format,
        enable_sparse_v=enable_sparse_v
    )
    return AsymmetricKVCache(config, device=device)


def recommend_asymmetric_config(
    model_bits: int = 4,
    target_compression: float = 4.0,
    quality_priority: str = "balanced"
) -> Tuple[str, str]:
    """
    Recommend K/V formats based on model and requirements.
    
    Args:
        model_bits: Base model quantization (4, 8, etc.)
        target_compression: Target overall compression factor
        quality_priority: "quality", "balanced", or "compression"
    
    Returns:
        Tuple of (k_format, v_format)
    """
    recommendations = {
        "quality": {
            4: ("q8_0", "q8_0"),
            8: ("q8_0", "q8_0"),
        },
        "balanced": {
            4: ("q8_0", "turbo4"),
            8: ("q8_0", "turbo3"),
        },
        "compression": {
            4: ("turbo4", "turbo2"),
            8: ("turbo4", "turbo3"),
        }
    }
    
    priority = quality_priority if quality_priority in recommendations else "balanced"
    model_key = min(recommendations[priority].keys(), key=lambda x: abs(x - model_bits))
    
    return recommendations[priority][model_key]
